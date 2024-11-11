"""Functions and classes that handle parsing"""

from itertools import chain

from nltk.parse import DependencyGraph


class PartialParse(object):
    """A PartialParse is a snapshot of an arc-standard dependency parse

    It is fully defined by a quadruple (sentence, stack, next, arcs).

    sentence is a tuple of ordered pairs of (word, tag), where word
    is a a word string and tag is its part-of-speech tag.

    Index 0 of sentence refers to the special "root" node
    (None, self.root_tag). Index 1 of sentence refers to the sentence's
    first word, index 2 to the second, etc.

    stack is a list of indices referring to elements of
    sentence. The 0-th index of stack should be the bottom of the stack,
    the (-1)-th index is the top of the stack (the side to pop from).

    next is the next index that can be shifted from the buffer to the
    stack. When next == len(sentence), the buffer is empty.

    arcs is a list of triples (idx_head, idx_dep, deprel) signifying the
    dependency relation `idx_head ->_deprel idx_dep`, where idx_head is
    the index of the head word, idx_dep is the index of the dependant,
    and deprel is a string representing the dependency relation label.
    """

    left_arc_id = 0
    """An identifier signifying a left arc transition"""

    right_arc_id = 1
    """An identifier signifying a right arc transition"""

    shift_id = 2
    """An identifier signifying a shift transition"""

    root_tag = "TOP"
    """A POS-tag given exclusively to the root"""

    def __init__(self, sentence):
        # the initial PartialParse of the arc-standard parse
        self.sentence = ((None, self.root_tag),) + tuple(sentence)
        self.stack = [0]
        self.next = 1
        self.arcs = []

    @property
    def complete(self):
        """bool: return true iff the PartialParse is complete

        Assume that the PartialParse is valid
        """

        empty_stack = [0]
        return self.stack == empty_stack and self.next == len(self.sentence)



    def parse_step(self, transition_id, deprel=None):
        """Update the PartialParse with a transition

        Args:
            transition_id : int
                One of left_arc_id, right_arc_id, or shift_id. You
                should check against `self.left_arc_id`,
                `self.right_arc_id`, and `self.shift_id` rather than
                against the values 0, 1, and 2 directly.
            deprel : str or None
                The dependency label to assign to an arc transition
                (either a left-arc or right-arc). Ignored if
                transition_id == shift_id

        Raises:
            ValueError if transition_id is an invalid id or is illegal
                given the current state
        """
        if transition_id == self.shift_id:
            if self.next >= len(self.sentence):
                raise ValueError("No element left in buffer.")
            self.stack.append(self.next)
            self.next += 1
        elif transition_id == self.left_arc_id:
            if len(self.stack) > 1:
                self.arcs.append((self.stack[-1], self.stack.pop(-2), deprel))
            else:
                raise ValueError("Adding arcs when there are less than 2 values left in stack.")
        elif transition_id == self.right_arc_id:
            if len(self.stack) > 1:
                self.arcs.append((self.stack[-2], self.stack.pop(-1), deprel))
            else:
                raise ValueError("Adding arcs when there are less than 2 values left in stack.")
        else:
            raise ValueError("Unknown transition type")


    def get_nleftmost(self, sentence_idx, n=None):
        """Returns a list of n leftmost dependants of word

        Leftmost means closest to the beginning of the sentence.

        Note that only the direct dependants of the word on the stack
        are returned (i.e. no dependants of dependants).

        Args:
            sentence_idx : refers to word at self.sentence[sentence_idx]
            n : the number of dependants to return. "None" refers to all
                dependants

        Returns:
            dep_list : The n leftmost dependants as sentence indices.
                If fewer than n, return all dependants. Return in order
                with the leftmost @ 0, immediately right of leftmost @
                1, etc.
        """

        target_arcs = list(filter(lambda x : x[0] == sentence_idx, self.arcs))
        sorted_arcs = sorted(target_arcs, key = lambda x : x[1])[:n]
        dep_list = [x[1] for x in sorted_arcs]

        return dep_list

    def get_nrightmost(self, sentence_idx, n=None):
        """Returns a list of n rightmost dependants of word on the stack @ idx

        Rightmost means closest to the end of the sentence.

        Note that only the direct dependants of the word on the stack
        are returned (i.e. no dependants of dependants).

        Args:
            sentence_idx : refers to word at self.sentence[sentence_idx]
            n : the number of dependants to return. "None" refers to all
                dependants

        Returns:
            dep_list : The n rightmost dependants as sentence indices. If
                fewer than n, return all dependants. Return in order
                with the rightmost @ 0, immediately left of rightmost @
                1, etc.
        """

        target_arcs = list(filter(lambda x : x[0] == sentence_idx, self.arcs))
        sorted_arcs = sorted(target_arcs, key = lambda x : x[1], reverse = True)[:n]
        dep_list = [x[1] for x in sorted_arcs]

        return dep_list

    def get_oracle(self, graph: DependencyGraph):
        """Given a projective dependency graph, determine an appropriate
        transition

        This method chooses either a left-arc, right-arc, or shift so
        that, after repeated calls to pp.parse_step(*pp.get_oracle(graph)),
        the arc-transitions this object models matches the
        DependencyGraph "graph". For arcs, it also has to pick out the
        correct dependency relationship.
        graph is projective: informally, this means no crossed lines in the
        dependency graph. More formally, if i -> j and j -> k, then:
             if i > j (left-arc), i > k
             if i < j (right-arc), i < k

        *IMPORTANT* if left-arc and shift operations are both valid and
        can lead to the same graph, always choose the left-arc
        operation.

        *ALSO IMPORTANT* make sure to use the values `self.left_arc_id`,
        `self.right_arc_id`, `self.shift_id` for the transition rather than
        0, 1, and 2 directly

        Args:
            graph : nltk.parse.dependencygraph.DependencyGraph
                A projective dependency graph to head towards

        Returns:
            transition, deprel_label : the next transition to take, along
                with the correct dependency relation label; if transition
                indicates shift, deprel_label should be None

        Raises:
            ValueError if already completed. Otherwise you can always
            assume that a valid move exists that heads towards the
            target graph
        """
        if self.complete:
            raise ValueError('PartialParse already completed')
        transition, dep_rel_label = -1, None

        if len(self.stack) == 1:
            transition, dep_rel_label = self.shift_id, None
        else:
            stack_last_word = self.stack[-1]
            stack_second_last_word = self.stack[-2]

            # check left_arc
            if get_head(stack_second_last_word, graph) == stack_last_word:
                transition = self.left_arc_id
                dep_rel_label = get_dep_rel(stack_second_last_word, graph)

            # check right_arc
            elif get_head(stack_last_word, graph) == stack_second_last_word:

                # two case to do right_arc
                right_deps_graph = list(get_dep_right(stack_last_word, graph))
                left_deps_graph = list(get_dep_left(stack_last_word, graph))

                # first case: there is no dependants of stack_last_word
                if right_deps_graph == 0:
                    transition = self.right_arc_id

                # second case: the dependants of stack_last_word are all in the arcs
                dependents_in_arcs = [i[1] for i in self.arcs if i[0] == stack_last_word]
                if len(dependents_in_arcs) == len(right_deps_graph) + len(left_deps_graph):
                    transition = self.right_arc_id
                    dep_rel_label = get_dep_rel(stack_last_word, graph)

        if transition == -1:
            transition, dep_rel_label = self.shift_id, None

        return transition, dep_rel_label

    def parse(self, td_pairs):
        """Applies the provided transitions/deprels to this PartialParse

        Simply reapplies parse_step for every element in td_pairs

        Args:
            td_pairs:
                The list of (transition_id, deprel) pairs in the order
                they should be applied
        Returns:
            The list of arcs produced when parsing the sentence.
            Represented as a list of tuples where each tuple is of
            the form (head, dependent)
        """
        for transition_id, deprel in td_pairs:
            self.parse_step(transition_id, deprel)
        return self.arcs


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    Note that parse_step may raise a ValueError if your model predicts an
    illegal (transition, label) pair. Remove any such "stuck" partial-parses
    from the list unfinished_parses.

    Args:
        sentences:
            A list of "sentences", where each element is itself a list
            of pairs of (word, pos)
        model:
            The model that makes parsing decisions. It is assumed to
            have a function model.predict(partial_parses) that takes in
            a list of PartialParse as input and returns a list of
            pairs of (transition_id, deprel) predicted for each parse.
            That is, after calling
                td_pairs = model.predict(partial_parses)
            td_pairs[i] will be the next transition/deprel pair to apply
            to partial_parses[i].
        batch_size:
            The number of PartialParse to include in each minibatch
    Returns:
        arcs:
            A list where each element is the arcs list for a parsed
            sentence. Ordering should be the same as in sentences (i.e.,
            arcs[i] should contain the arcs for sentences[i]).
    """
    partial_parses = []
    for i in range(len(sentences)):
        partial_parses.append(PartialParse(sentences[i]))

    unfinished_parses = partial_parses[:]

    while len(unfinished_parses) > 0:
        minibatch = unfinished_parses[:batch_size]
        td_pairs = model.predict(minibatch)
        for i in range(len(td_pairs)):
            try:
                minibatch[i].parse_step(td_pairs[i][0], td_pairs[i][1])
                if minibatch[i].complete:
                    unfinished_parses.remove(minibatch[i])
            except ValueError:
                unfinished_parses.remove(minibatch[i])

    arcs = []
    for i in range(len(partial_parses)):
        arcs.append(partial_parses[i].arcs)

    return arcs


# *** HELPER FUNCTIONS *** #


def get_dep_rel(sentence_idx: int, graph: DependencyGraph):
    """Get the dependency relation label for the word at index sentence_idx
    from the provided DependencyGraph"""
    return graph.nodes[sentence_idx]['rel']


def get_head(sentence_idx: int, graph: DependencyGraph):
    """Get the index of the head of the word at index sentence_idx from the
    provided DependencyGraph"""
    return graph.nodes[sentence_idx]['head']


def get_deps(sentence_idx: int, graph: DependencyGraph):
    """Get the indices of the dependants of the word at index sentence_idx
    from the provided DependencyGraph"""
    return list(chain(*graph.nodes[sentence_idx]['deps'].values()))


def get_dep_left(sentence_idx: int, graph: DependencyGraph):
    """Get the arc-left dependants of the word at index sentence_idx from
    the provided DependencyGraph"""
    return (dep for dep in get_deps(sentence_idx, graph)
            if dep < graph.nodes[sentence_idx]['address'])


def get_dep_right(sentence_idx: int, graph: DependencyGraph):
    """Get the arc-right dependants of the word at index sentence_idx from
    the provided DependencyGraph"""
    return (dep for dep in get_deps(sentence_idx, graph)
            if dep > graph.nodes[sentence_idx]['address'])


def get_sentence(graph, include_root=False):
    """Get the associated sentence from a DependencyGraph"""
    sentence_w_addresses = [(node['address'], node['word'], node['ctag'])
                            for node in graph.nodes.values()
                            if include_root or node['word'] is not None]
    sentence_w_addresses.sort()
    return tuple(t[1:] for t in sentence_w_addresses)

