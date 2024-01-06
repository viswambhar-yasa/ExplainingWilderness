# Concept Relevance Propagation (CRP) Method README

## Introduction
The Concept Relevance Propagation (CRP) method addresses challenges related to interpreting attribution scores in neural networks, particularly focusing on hidden layer representations. This README outlines the key components and principles of the CRP method.

## Shortcoming in other XAI methods
- **Interpretability of Latent Representations**: Neural network latent representations are difficult to interpret, limiting insights from attribution scores of hidden units.
- **Information Loss in Backpropagation**: Traditional backpropagation methods result in coarse attribution maps, losing detailed information about latent representation roles.

## CRP Method Overview [1]
- **Understanding Latent Filters and Neurons**: CRP aims to enhance the interpretability of model decisions in the input space by assuming knowledge of distinct roles of latent filters and neurons.
- **Class-Conditional Relevance Maps (R(x|y))**: Introduces class-conditional relevance maps to focus on specific network outputs (y) for attributing relevance to input features (x).
- **Disentangling Attribution Scores for Latent Representations**: Strategies proposed by CRP disentangle attribution scores for latent representations, improving the semantic fidelity of explaining heatmaps.
- **Variable θ for Multi-Concept-Conditional Computation**: Utilizes the variable θ for multi-concept-conditional computation of relevance attributions (R(x|θ)) based on conditions (cl) tied to representations of concepts across network layers.
- **Relevance Decomposition Formula**: Defines a relevance decomposition to propagate relevant information based on specific conditions, filtering the flow of backpropagated quantities through the model.

## Characteristics of CRP Method
- CRP emphasizes understanding latent representations.
- It introduces class-conditional relevance maps for explaining model decisions.
- The method disentangles attribution scores for latent representations to improve interpretability.
- CRP utilizes conditions to compute relevance attributions based on representations of concepts across network layers.


## Class-Conditional Relevance Maps [1]
The Concept Relevance Propagation (CRP) method introduces the notion of class-conditional relevance maps R(x|θ) to enhance interpretability based on specified conditions θ tied to representations of concepts across neural network layers.

### Formula for Class-Conditional Relevance Maps [1]

The relevance decomposition formula is extended to compute class-conditional relevance attributions (R(l−1,l)i←j (x|θ ∪ θl)) as follows:
```
R(l−1,l)i←j (x|θ ∪ θl) = z_ij / z_j * Σ_cl∈θ_l δ_jcl * R_lj(x|θ)
```
Where:
- θ represents a set of conditions tied to concepts applied to layers.
- δjcl selects the relevance quantity Rlj of layer l and neuron j for further propagation if neuron j meets the condition(s) cl tied to the concepts of interest.
- z_ij represents a relevance score between layer l-1 and l.
- z_j denotes the total relevance of neuron j.


## Significance of Relevant Concepts  [1]

Understanding the most relevant concepts from lower layers of a neural network is crucial in constructing an ontology-like semantic composition that aids in explanations and predictions. The composition of concept c in layer l is determined by the most relevant lower layer concepts, denoted as b in layer l−1, and can be obtained via a sorting process, such as:

```python
B = {b1, . . . , bn} = argsortdesc_i R_{l-1i}(x|θ_c) 
```

## Hierarchical Composition of Relevant Concepts [1]

The Concept Relevance Propagation (CRP) method allows for the identification and hierarchical composition of relevant concepts within a neural network, surpassing the limitations of existing methods that primarily focus on static or per-class interactions.

### Utilizing CRP for Conceptual Composition [1]

Assuming the identification of a significant concept c within layer l via concept-conditional relevance attributions (Rl(p,q,j)(x|θc)), the interest lies in analyzing the composition of concept c in terms of its sub-concepts in preceding network layers. CRP enables us to explore the interactions encoded by the model, specifically concerning the prediction of a particular sample x and the corresponding model outcome f(x), considering conditions θc.

### Extension of Equations for Concept Interaction Analysis [1]

For a convolutional DNN used in image categorization, considering activation and attribution tensors with three axes (spatial and channel axes), we extend Equation (9) to calculate downstream relevance messages from upstream activations:
```
R(l−1,l)(u,v,i)←(p,q,k)(x|θc) = z(u,v,i)(p,q,k) / z(p,q,k) * Rl(p,q,k)(x|θc) 
```

Here, Rl(p,q,k)(x|θc) represents the already masked upstream relevance for concept c at layer l, with channel k being unmasked. The resulting R(l,l+1)(u,v,i)←(p,q,j)(x|θc) signifies the flow of downstream relevance messages from upstream to downstream voxels.

### Aggregation of Relevant Concepts and Visual Representation [1]

To identify the most relevant lower layer concepts in layer l−1 for a specific sample x and conditions θc, we aggregate the downstream relevance into per-channel/concept relevances at layer l−1 over spatial coordinates u and v:
```
Rl−1_i(x|θc) = Σu,v Σp,q,j R(l−1,l)(u,v,i)←(p,q,j)(x|θc)  
```
Optionally, to analyze localized layer- and concept interactions, the spatial coordinates (p, q) or (u, v) can be selectively chosen over specific regions I.


## Reference
- [1] [From “Where” to “What”: Towards Human-Understandable Explanations through Concept Relevance Propagation](https://arxiv.org/abs/INSERT_PAPER_NUMBER)
Reduan Achtibat, Maximilian Dreyer, Ilona Eisenbraun, Sebastian Bosse, Thomas Wiegand, Wojciech Samek, Sebastian Lapuschkin
Fraunhofer Heinrich-Hertz-Institute, 10587 Berlin, Germany, Technische Universität Berlin, 10587 Berlin, Germany, BIFOLD – Berlin Institute for the Foundations of Learning and Data, 10587 Berlin, Germany
∗ contributed equally
† corresponding authors: {sebastian.lapuschkin,wojciech.samek}@hhi.fraunhofer.d