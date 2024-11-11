# Neural-Dependency-Parsing

Welcome to my **Neural Dependency Parsing** project! This personal endeavor combines concepts from computational linguistics and machine learning to implement and experiment with dependency parsing using transition-based and graph-based approaches. By the end of the project, you’ll have a fully functional parser capable of predicting dependency trees for sentences using neural networks, with implementations designed to handle both projective and non-projective trees.

## Project Overview

This project is a comprehensive exploration of two types of dependency parsers:
1. **Transition-Based Dependency Parser**: A parser that builds dependency trees incrementally, using a neural network to decide transitions between parsing states.
2. **Graph-Based Dependency Parser**: A more flexible approach that scores all possible edges in a dependency graph, allowing for the construction of non-projective trees.

Both parsers are trained on Universal Dependencies data, with evaluations focusing on their performance in terms of attachment scores.

## Learning Goals

Working on this project helped me:
- Understand and implement dependency parsing algorithms, both transition-based and graph-based.
- Build and train neural networks using PyTorch, focusing on custom architectures for parsing.
- Apply algorithms like the maximum spanning tree to ensure well-formed dependency trees.
- Evaluate the effectiveness of parsing models using metrics like LAS (Labelled Attachment Score) and UAS (Unlabelled Attachment Score).
- Deepen my understanding of natural language processing (NLP) concepts, especially in the context of syntactic parsing.

## Components and How They Work

### 1. **Transition-Based Dependency Parsing**
- **Goal**: Build a dependency tree for a sentence incrementally using transitions: SHIFT, LEFT-ARC, and RIGHT-ARC.
- **Steps**:
  - Implement transition operations to manage a parsing stack, buffer, and dependency list.
  - Use a neural network classifier to predict the next transition given the current parse state.
  - Optimize parsing efficiency with minibatch processing, leveraging the power of neural networks.
- **Evaluation**: Use attachment scores to assess parsing quality and measure the model’s accuracy in predicting correct transitions.

### 2. **Graph-Based Dependency Parsing**
- **Goal**: Handle non-projective trees by scoring possible dependency edges using a neural network.
- **Steps**:
  - Create scoring layers for arcs and dependency labels using multi-layer perceptrons (MLPs) in PyTorch.
  - Implement an edge-factored model, calculating probabilities with softmax functions and optimizing using cross-entropy loss.
  - Use a maximum spanning tree algorithm to construct the final dependency tree, ensuring well-formedness.
- **Challenges**: Understanding and implementing gap degrees and dealing with the complexity of non-projective structures.

## Tools and Libraries

- **Programming Language**: Python
- **Framework**: PyTorch (v2.3.0)
- **Additional Packages**: `conllu` for processing annotated data and `transformers` for working with embeddings.

## How to Run

### Prerequisites
1. **Install Dependencies**:
   ```bash
   pip install torch==2.3.0 conllu transformers
   ```
2. **Set Up Your Environment**:
   - Make sure you have Python 3 and PyTorch installed.

### Running the Code
1. **Transition-Based Parser**:
   ```bash
   python3 train.py q1
   ```
   - Runs the training for the transition-based parser and evaluates its performance.

2. **Graph-Based Parser**:
   ```bash
   python3 train.py q2
   ```
   - Trains the graph-based parser and computes attachment scores on test data.

3. **Debugging**:
   - Use the `--debug` flag for faster, small-scale training during development:
     ```bash
     python3 train.py --debug q1
     python3 train.py --debug q2
     ```

### GPU Training
- For faster training, use a GPU:
  ```bash
  ./gpu-train.sh q1
  ./gpu-train.sh q2
  ```
- Ensure you have access to GPU-equipped machines, especially if working on university servers.

## Key Concepts and Algorithms

1. **Dependency Parsing**:
   - Models syntactic relationships in a sentence where words are linked to a "head" word.
   - Transition-based parsing uses a state-based mechanism to incrementally construct these links.
   - Graph-based parsing evaluates all potential links simultaneously and selects the best configuration.

2. **Neural Networks**:
   - Feature vectors are generated using pre-trained embeddings and transformed using neural network layers.
   - The parsing models employ ReLU activations and dropout for regularization.

3. **Maximum Spanning Tree (MST)**:
   - Used in the graph-based parser to ensure the predicted dependency tree is well-formed.
   - The Chu-Liu/Edmonds algorithm is used to extract the MST from scored edges.

## Challenges and Insights

- **Handling Non-Projective Trees**: Transition-based parsers struggle with non-projective structures, making graph-based approaches necessary.
- **Model Training and Evaluation**: Achieving high LAS requires careful implementation of the scoring mechanisms and regularization techniques.

## Future Work

- **Performance Tuning**: Experiment with different neural architectures and hyperparameters to improve parsing accuracy.
- **Real-World Applications**: Integrate parsers into NLP pipelines for tasks like information extraction or machine translation.
- **Model Comparison**: Conduct more detailed analyses of the differences between transition-based and graph-based approaches.

---

This project was an enlightening journey into dependency parsing, providing practical insights into how linguistic theories translate into computational models. I hope you find the implementation and exploration as rewarding as I did!
