# Sparsity and Superposition in Mixture of Experts

This repository explores how superposition emerges in Mixture-of-Experts (MoE) architectures through theoretical analysis and empirical experiments with toy models.

## Overview

Building on [Anthropic's Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html), this project investigates the mechanistic differences between MoE and dense networks. We find that network sparsity (the ratio of active to total experts), rather than feature sparsity or importance, characterizes MoE behavior. Models with greater network sparsity exhibit greater monosemanticity, suggesting that interpretability and capability need not be fundamentally at odds.

## Key Findings

- **Network sparsity drives MoE behavior**: Unlike dense networks, neither feature sparsity nor feature importance cause discontinuous phase changes. Network sparsity (active/total experts) better characterizes MoEs.
- **Greater monosemanticity with sparsity**: Models with greater network sparsity exhibit greater monosemanticity, showing that experts naturally organize around coherent feature combinations.
- **New metrics for MoE superposition**: We develop specialized metrics for measuring superposition across experts, enabling mechanistic understanding of MoE behavior.
- **Expert specialization defined by monosemanticity**: Rather than load balancing, we propose defining expert specialization based on monosemantic feature representation, leading to more interpretable models without sacrificing performance.

## Repository Structure

### Notebooks

- **`demo-superposition.ipynb`** - Main demonstration notebook
  - Compares features per dimension between dense and MoE architectures
  - Visualizes expert weight matrices and superposition patterns
  - Generates publication figures showing the efficiency gains of MoE models

- **`phase_change.ipynb`** - Phase diagram visualizations
  - Creates comprehensive phase change diagrams from experiment data
  - Generates box-and-whisker plots comparing different architectures
  - Visualizes how expert weight norms and superposition scores change with sparsity
  - Renders publication-quality PDF/PGF figures for LaTeX integration

- **`phase_change_data/phase_change_experiment.ipynb`** - Phase change experiments
  - Runs grid search over sparsity and importance parameters (computationally intensive)
  - Generates/stores the .npz data files that phase_change.ipynb visualizes
  - Explores different architectures (2x1, 3x1, 3x2) with varying numbers of experts

- **`expert_specialization.ipynb`** - Expert specialization analysis
  - Reproduces Figures 5 and 6 from the paper
  - Analyzes how different initialization strategies (Xavier vs. K-hot) affect expert specialization
  - Compares expert usage patterns when features are activated
  - Generates Table 1 showing initialization-dependent specialization

- **`simple-testing-ground.ipynb`** - Testing and debugging
  - Simple experiments to verify model functionality
  - Testing MoE routing and expert selection

### Core Model Implementation

- **`model/model.py`** - MoE architecture implementation
  - Configurable mixture-of-experts model
  - Training routines with support for load balancing loss
  - Expert routing via learned gating mechanism
  - Feature importance weighting

### Data

- **`phase_change_data/*.npz`** - Pre-computed phase change experiment results
  - Contains trained model weights, loss values, and configuration parameters
  - Format: XYZ.npz where X=input dims, Y=hidden dims, Z=num experts
  - Used to generate the phase diagrams in phase_change.ipynb

### Helper Functions

- **`helpers/helpers.py`** - Utility functions for initialization and analysis

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### LaTeX Prerequisites (for figure generation)

To generate publication-quality figures with LaTeX rendering:

1. **Install TinyTeX**:
```bash
brew install --cask basictex
echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

2. **Install required packages**:
```bash
sudo tlmgr update --self --all
sudo tlmgr install underscore type1cm type1ec latexmk pgf xcolor xkeyval etoolbox geometry amsmath amsfonts lm cm-super courier helvetic fontaxes siunitx ulem graphics
```

**Alternative**: Set `plt.rcParams['text.usetex'] = False` in the notebook to disable LaTeX.

## Usage

### Reproducing Main Results

1. **Run the main demonstration**:
```bash
jupyter notebook demo-superposition.ipynb
```

2. **Generate phase diagrams**:
```bash
jupyter notebook phase_change.ipynb
```

3. **Analyze expert specialization**:
```bash
jupyter notebook expert_specialization.ipynb
```

### Running Experiments from Scratch

To regenerate the phase change data:

```bash
jupyter notebook phase_change_data/phase_change_experiment.ipynb
```

This runs extensive experiments over parameter grids and saves results to .npz files. Depending on the hardware, it can take 10+ hours.

## Key Concepts

### Network Sparsity in MoE Models

Unlike dense networks, **network sparsity** (the ratio of active experts to total experts) is the primary driver of MoE behavior, not feature sparsity or feature importance. This means:

- **Dense networks**: Phase changes driven by feature sparsity and importance
- **MoE models**: Behavior characterized by network sparsity (`n_active_experts / n_experts`)

### Monosemanticity vs. Polysemanticity

- **Monosemantic experts**: Each expert handles a small, coherent set of features (low superposition)
- **Polysemantic experts**: Each expert handles many features simultaneously (high superposition)

Our work shows that **greater network sparsity leads to greater monosemanticity**, enabling more interpretable models without sacrificing performance.

### Expert Specialization

We define expert specialization based on **monosemantic feature representation** rather than load balancing. Experts naturally organize around coherent feature combinations when initialized appropriately (e.g., K-hot initialization vs. Xavier initialization).

## References

### Core Paper
- [Anthropic's Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)

### Related Notebooks
- [Modeling a "phase diagram" of superposition for exact toy models](https://github.com/wattenberg/superposition/blob/main/Exploring_Exact_Toy_Models.ipynb)
- [Notebooks accompanying Anthropic's "Toy Models of Superposition" paper](https://github.com/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb)

## Citation

If you use this work, please cite:

```bibtex
@article{moe-superposition2025,
  title={Superposition in Mixture-of-Experts Models},
  author={Chaudhari, hMarmik, Nuer, Jeremi, Thorstenson, Rome},
  journal={NeurIPS ML Interpretability Workshop},
  year={2025}
}
```

**Accepted at NeurIPS ML Interpretability Workshop 2025**

## License

MIT License
