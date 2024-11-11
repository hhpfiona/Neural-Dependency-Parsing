import typing as T
from math import inf

import torch
from torch.nn.functional import pad
from torch import Tensor
import einops


def is_projective(heads: T.Iterable[int]) -> bool:
    """
    Determines whether the dependency tree for a sentence is projective.

    Args:
        heads: The indices of the heads of the words in sentence. Since ROOT
          has no head, it is not expected to be part of the input, but the
          index values in heads are such that ROOT is assumed in the
          starting (zeroth) position. See the examples below.

    Returns:
        True if and only if the tree represented by the input is
          projective.

    Examples:
        >>> is_projective([2, 5, 4, 2, 0, 7, 5, 7])
        True

        >>> is_projective([2, 0, 2, 2, 6, 3, 6])
        False
    """
    projective = True
    
    heads_copy = list(heads)
    heads_copy.insert(0, 0)
    n = len(heads_copy)

    for i in range(n):
        for j in range(i, n):
            # Get the head of each word
            hi = heads_copy[i]
            hj = heads_copy[j]

            if (i < j < hi < hj) or (hi < hj < i < j):
                projective = False

    return projective


def is_single_root(heads: Tensor, lengths: Tensor) -> Tensor:
    """
    Determines whether the selected arcs for a sentence constitute a tree with
    a single root word.

    Remember that index 0 indicates the ROOT node. A tree with "a single root
    word" has exactly one outgoing edge from ROOT.

    Args:
        heads (Tensor): a Tensor of dimensions (batch_sz, sent_len) and dtype
            int where the entry at index (b, i) indicates the index of the
            predicted head for vertex i for input b in the batch

        lengths (Tensor): a Tensor of dimensions (batch_sz,) and dtype int
            where each element indicates the number of words (this doesn't
            include ROOT) in the corresponding sentence.

    Returns:
        A Tensor of dtype bool and dimensions (batch_sz,) where the value
        for each element is True if and only if the corresponding arcs
        constitute a single-root-word tree as defined above

    Examples:
        Valid trees:
        >>> is_single_root(torch.tensor([[2, 5, 4, 2, 0, 7, 5, 7],\
                                              [2, 0, 2, 2, 6, 3, 6, 0]]),\
                                torch.tensor([8, 7]))
        tensor([True, True])

        Invalid trees (the first has a cycle; the second has multiple roots):
        >>> is_single_root(torch.tensor([[2, 5, 4, 2, 0, 8, 6, 7],\
                                              [2, 0, 2, 2, 6, 3, 6, 0]]),\
                                torch.tensor([8, 8]))
        tensor([False, False])
    """
    
    batch_size = heads.size(0)
    result = torch.zeros(batch_size, dtype=torch.bool)

    for i in range(batch_size):
        n = lengths[i].item()
        sentence_heads = heads[i, :n]

        # Find the nodes that have ROOT (index 0) as their head
        root_children = (sentence_heads == 0).nonzero(as_tuple=False).squeeze(-1).tolist()
        if len(root_children) != 1:
            # There must be exactly one root
            result[i] = False
            continue

        # Build an adjacency list (children for each node)
        children = [[] for _ in range(n)]
        for child in range(n):
            head = sentence_heads[child].item()
            if head == 0:
                # ROOT node (no parent in the adjacency list)
                continue
            parent = head - 1  # Adjust for 0-based indexing
            if parent < 0 or parent >= n:
                # Invalid parent index
                result[i] = False
                break
            children[parent].append(child)
        else:
            # Perform DFS to check for cycles and connectivity
            visited = [False] * n
            stack = [root_children[0]]
            acyclic = True

            while stack:
                node = stack.pop()
                if visited[node]:
                    # Cycle detected
                    acyclic = False
                    break
                visited[node] = True
                stack.extend(children[node])

            # The tree is valid if it's acyclic and all nodes are visited
            result[i] = acyclic and all(visited)

    return result


def mst_single_root(arc_tensor: Tensor, lengths: Tensor) -> Tensor:
    """
    Finds the maximum spanning tree (more technically, arborescence) for the
    given sentences such that each tree has a single root word.

    Remember that index 0 indicates the ROOT node. A tree with "a single root
    word" has exactly one outgoing edge from ROOT.

    Args:
        arc_tensor (Tensor): a Tensor of dimensions (batch_sz, x, y) and dtype
            float where x=y and the entry at index (b, i, j) indicates the
            score for a candidate arc from vertex j to vertex i.

        lengths (Tensor): a Tensor of dimensions (batch_sz,) and dtype int
            where each element indicates the number of words (this doesn't
            include ROOT) in the corresponding sentence.

    Returns:
        A Tensor of dtype int and dimensions (batch_sz, x) where the value at
        index (b, i) indicates the head for vertex i according to the
        maximum spanning tree for the input graph.

    Examples:
        >>> mst_single_root(torch.tensor(\
            [[[0, 0, 0, 0],\
              [12, 0, 6, 5],\
              [4, 5, 0, 7],\
              [4, 7, 8, 0]],\
             [[0, 0, 0, 0],\
              [1.5, 0, 4, 0],\
              [2, 0.1, 0, 0],\
              [0, 0, 0, 0]],\
             [[0, 0, 0, 0],\
              [4, 0, 3, 1],\
              [6, 2, 0, 1],\
              [1, 1, 8, 0]]]),\
            torch.tensor([3, 2, 3]))
        tensor([[0, 0, 3, 1],
                [0, 2, 0, 0],
                [0, 2, 0, 2]])
    """
    
    batch_size, max_len, _ = arc_tensor.size()
    output = torch.zeros(batch_size, max_len, dtype=torch.long)

    for b in range(batch_size):
        n = lengths[b].item()
        scores = arc_tensor[b, :n + 1, :n + 1].clone()
        best_total_score = -inf
        best_heads = None

        # Iterate over each possible root (excluding ROOT at index 0)
        for root in range(1, n + 1):
            temp_scores = scores.clone()
            temp_scores[root, :] = -inf
            temp_scores[root, 0] = scores[root, 0]

            # Exclude ROOT node (index 0)
            sub_scores = temp_scores[1:, 1:]

            # Run the iterative Chu-Liu/Edmonds algorithm
            sub_heads = chu_liu_edmonds(sub_scores)
            sub_heads += 1  # Adjust indices

            heads = torch.zeros(n + 1, dtype=torch.long)
            heads[1:] = sub_heads
            heads[root] = 0

            indices = torch.arange(1, n + 1)
            total_score = torch.sum(scores[indices, heads[indices]])
            if total_score > best_total_score:
                best_total_score = total_score
                best_heads = heads.clone()

        output[b, :n + 1] = best_heads

    return output


def chu_liu_edmonds(scores: Tensor) -> Tensor:
    n_nodes = scores.size(0)
    # Initialize the graph
    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edges.append((i, j, scores[i, j].item()))
    # Sort edges in descending order
    edges.sort(key=lambda x: x[2], reverse=True)

    parent = [-1] * n_nodes
    rank = [0] * n_nodes
    UF = UnionFind(n_nodes)

    # For each node, pick the maximum incoming edge
    for i in range(n_nodes):
        max_score = -inf
        max_j = -1
        for j in range(n_nodes):
            if i != j and scores[i, j] > max_score:
                max_score = scores[i, j]
                max_j = j
        parent[i] = max_j

    # Detect cycles and contract them
    while True:
        cycles = find_cycles(parent)
        if not cycles:
            break
        for cycle in cycles:
            cycle_nodes = set(cycle)
            # Find edges entering the cycle
            enter_edges = []
            for i in range(n_nodes):
                if i not in cycle_nodes and parent[i] in cycle_nodes:
                    enter_edges.append((i, parent[i], scores[i, parent[i]].item()))
            # Remove one node from the cycle
            removed_node = cycle[0]
            parent[removed_node] = -1
            # Update the parents of nodes in the cycle
            for node in cycle[1:]:
                parent[node] = removed_node
            # Update the Union-Find structure
            for node in cycle:
                UF.union(removed_node, node)
    return torch.tensor(parent, dtype=torch.long)


def find_cycles(parent: T.List[int]) -> T.List[T.List[int]]:
    n = len(parent)
    index = 0
    indices = [-1] * n
    lowlinks = [-1] * n
    on_stack = [False] * n
    stack = []
    cycles = []

    def strongconnect(v):
        nonlocal index
        indices[v] = index
        lowlinks[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        h = parent[v]
        if h != -1:
            if indices[h] == -1:
                strongconnect(h)
                lowlinks[v] = min(lowlinks[v], lowlinks[h])
            elif on_stack[h]:
                lowlinks[v] = min(lowlinks[v], indices[h])

        if lowlinks[v] == indices[v]:
            # Start a new SCC
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                cycles.append(scc)

    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)

    return cycles


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, u):
        while u != self.parent[u]:
            self.parent[u] = self.parent[self.parent[u]]  # Path compression
            u = self.parent[u]
        return u

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            self.parent[pu] = pv


if __name__ == '__main__':
    import doctest

    doctest.testmod()
