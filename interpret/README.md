# XAI Relevance Propagation Methods: LRP and CRP

## Overview

This repository provides implementations and explanations for Layer-wise Relevance Propagation (LRP) and concept Relevance Propagation (CRP), two eXplainable AI (XAI) techniques. The goal of these methods is to interpret the decision-making processes of complex machine learning models by attributing relevance to input features.

LRP focuses on understanding the contribution of individual features or neurons to the model's output, while CRP extends this by comparing the relevance between two different instances, aiding in understanding divergent predictions.

![CRP Path Visualization](https://www.hhi.fraunhofer.de/fileadmin/_processed_/4/4/csm_crp-path-visualization_66704e94d2.png)


## Features

- **LRP Implementation:** Includes code and resources for applying Layer-wise Relevance Propagation to neural network models.
- **CRP Implementation:** Provides tools and code for Concept Relevance Propagation for comparative analysis.
- **Visualization:** Tools for generating heatmaps and visualizations to interpret the relevance scores assigned by LRP and CRP.
- **Explanation Evaluation:** Tools for generate evaluation metric like robustness, faithfulness, complexity and randomization.

- **Example Notebooks:** Jupyter notebooks demonstrating the usage and application of LRP and CRP on sample datasets are provide in experiments.

## Requirements

- Python 3.x
- Libraries specified in `requirements.txt`


## References

- List of relevant research papers, articles, and documentation that served as the basis for these implementations are presented in [bibtex](../liteature_study.bib/liteature_study.bib.bib).



## Acknowledgments

- Standard XAI methods like Integrated Gradients, Grad CAM, Grad SHAP, LIME have been implemented using captum library and has been used within the code of conduct.

- CRP, LRP have beed implemented using zennit library which are referred below
    ```bibtex
    @article{anders2021software,
      author  = {Anders, Christopher J. and
                 Neumann, David and
                 Samek, Wojciech and
                 Müller, Klaus-Robert and
                 Lapuschkin, Sebastian},
      title   = {Software for Dataset-wide XAI: From Local Explanations to Global Insights with {Zennit}, {CoRelAy}, and {ViRelAy}},
      journal = {CoRR},
      volume  = {abs/2106.13200},
      year    = {2021},
    }
    ```
    ```bibtex
    @article{achtibat2023attribution,
  title={From attribution maps to human-understandable explanations through Concept Relevance Propagation},
  author={Achtibat, Reduan and Dreyer, Maximilian and Eisenbraun, Ilona and Bosse, Sebastian and Wiegand, Thomas and Samek, Wojciech and Lapuschkin, Sebastian},
  journal={Nature Machine Intelligence},
  volume={5},
  number={9},
  pages={1006–1019},
  year={2023},
  doi={10.1038/s42256-023-00711-8},
  url={https://doi.org/10.1038/s42256-023-00711-8},
  issn={2522-5839},
  publisher={Nature Publishing Group UK London}
    }
    ```

- Explanation evaluation metrics are build using quantus which referred below
    ```bibtex
    @article{hedstrom2023quantus,
      author  = {Anna Hedstr{\"{o}}m and Leander Weber and Daniel   Krakowczyk and Dilyara Bareeva and Franz Motzkus and Wojciech Samek and   Sebastian Lapuschkin and Marina Marina M.{-}C. H{\"{o}}hne},
      title   = {Quantus: An Explainable AI Toolkit for Responsible     Evaluation of Neural Network Explanations and Beyond},
      journal = {Journal of Machine Learning Research},
      year    = {2023},
      volume  = {24},
      number  = {34},
      pages   = {1--11},
      url     = {http://jmlr.org/papers/v24/22-0142.html}
    }


