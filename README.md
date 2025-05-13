
# Hillclimb-Causal Inference

**A Data-Driven Method to Explore the Causal Pathways Between Parental Behavior, Polygenic Risk Score (PRS), and Externalizing Behaviors in Children**  
*Mengman Wei, Qian Peng*

---

## 🧠 Overview

This repository provides scripts and configuration files used in our causal inference project exploring how parental behaviors and children's genetic predispositions interact to influence externalizing behaviors.  
We employ a combination of factor analysis, regression techniques, polygenic risk score (PRS) calculation, causal discovery algorithms (hill climbing search), structural equation modeling (SEM), and bootstrap validation to identify and verify key causal pathways.

---

## 📁 Repository Contents

| File/Folder                    | Description |
|-------------------------------|-------------|
| `Factoranalysis.R`            | Performs exploratory and factor analysis on the ABCD dataset and generates visualizations. |
| `Factoranalysis_regression.R` | Applies Lasso, Elastic Net, and Ridge regression to examine variable associations. |
| `Factoranalysis_heatmap.R`    | Computes a correlation matrix and produces a heatmap for visual inspection. |
| `Cal_PRS.sh`                  | Calculates children's polygenic risk scores (PRS) using PRSice. **Important:** Make sure your genetic data is properly quality-controlled and reference build (e.g., GRCh37/GRCh38) is correctly aligned. |
| `Hillclimb_Causal.py`         | Implements the main causal discovery algorithm using a hill climbing approach. |
| `SEM_pathanalysis.R`          | Performs structural equation modeling (SEM) and pathway analysis to validate causal relationships. |
| `bootstrapping.py`            | Conducts bootstrap validation to test the stability of inferred causal paths. |
| `environment.yml`             | Specifies all dependencies required to run this project via conda/mamba. |

---

## ⚙️ Setup & Installation

### 📦 Prerequisites

Ensure you have one of the following environment managers installed:
- [Micromamba](https://mamba.readthedocs.io/en/latest/) (recommended for speed and lightweight setup)
- [Conda](https://docs.conda.io/en/latest/)

### 🛠 Install Environment

```bash
# Create environment from the environment.yml file
micromamba create -f environment.yml

# Or with conda
conda env create -f environment.yml

# Activate the environment
micromamba activate my_env
# or
conda activate my_env

---

### References

**Hillclimb-Causal Inference: A Data-Driven Approach to Identify Causal Pathways Among Parental Behaviors, Genetic Risk, and Externalizing Behaviors in Children**  
Mengman Wei*, Qian Peng*  

**Subjects**: Quantitative Methods (q-bio.QM)

**Cite as**:  
Wei, M., & Peng, Q. (2025). *Hillclimb-Causal Inference: A Data-Driven Approach to Identify Causal Pathways Among Parental Behaviors, Genetic Risk, and Externalizing Behaviors in Children*. arXiv:2505.06784 [q-bio.QM].  
(You can also cite this version as arXiv:2505.06784v1 [q-bio.QM]).

**DOI**: [https://doi.org/10.48550/arXiv.2505.06784](https://doi.org/10.48550/arXiv.2505.06784)


