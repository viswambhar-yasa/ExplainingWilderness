# Layer-Wise Relevance Propagation (LRP) Overview

![Layer-Wise Relevance Propagation Procedure](https://www.researchgate.net/publication/342593824/figure/fig3/AS:11431281172869518@1688696619683/Illustration-of-the-layerwise-relevance-propagation-LRP-procedure-used-in-this-study.png)

#### Figure 1: Illustration of Layer-Wise Relevance Propagation Procedure [3]

## Introduction [1]

LRP is an explanation technique designed for neural network models. It facilitates the understanding of model predictions by propagating them backward through the network via purpose-designed local propagation rules. This README provides an overview of LRP, its rules for deep rectifier networks, implementation details, and applications.

## Key Features [1]

- **Explanation Technique**: LRP operates by propagating predictions backward through neural networks, applicable to inputs like images, videos, or text.

- **Rules for Deep Rectifier Networks**: LRP-0, LRP-ε, and LRP-γ, offering different enhancements for relevance redistribution.

## Applications [1]

LRP has found applications in various domains:

- **Bias Discovery**: Identifying biases in ML models and datasets.
- **Insight Extraction**: Gaining new insights from ML models, like wilderness classification.


## LRP Rules for Deep Rectifier Networks [1]

### Basic Rules

- **LRP-0**: Redistributes relevance based on contributions to neuron activation.
- **LRP-ε**: Enhanced version of LRP-0, incorporating a small positive term in the denominator.
- **LRP-γ**: Favors positive contributions over negative ones by adjusting parameter γ.
- **LRP-z+**: Favors positive contributions of weight and relevance.

## Efficient Implementation [1]

The structure of LRP-0 allows for efficient implementation in neural networks:

- **Forward Pass**: Computing z from the layer's weights and biases.
- **Element-wise Division**: Calculation of s = R / (z + ε).
- **Backward Pass**: Computation of c using gradients.
- **Element-wise Product**: Calculation of R = a * c.

![Layer-Wise Relevance Propagation Procedure [4]](https://miro.medium.com/v2/resize:fit:1400/0*Fg0u4MmcQ0lu3S0C.png
)

#### Efficient implementation of LRP-0 in 4 steps [1] 


## References

- [1] Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K. R. (Add the publication year). "Layer-Wise Relevance Propagation: An Overview."

- [2] Ullah, I.; Rios, A.; Gala, V.; Mckeever, S. Explaining Deep Learning Models for Tabular Data Using Layer-Wise Relevance Propagation. Appl. Sci. 2022, 12, 136. https://doi.org/10.3390/app12010136 


- [3] Toms, Benjamin & Barnes, Elizabeth & Ebert-Uphoff, Imme. (2020). Physically Interpretable Neural Networks for the Geosciences: Applications to Earth System Variability. Journal of Advances in Modeling Earth Systems. 12. 10.1029/2019MS002002. 