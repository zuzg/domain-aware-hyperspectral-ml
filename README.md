# Domain-Aware Machine Learning Architectures for Hyperspectral Remote Sensing

This repository contains the implementation code for the thesis:  
**"Domain-Aware Machine Learning Architectures for Hyperspectral Remote Sensing"**.

The project aims to design and evaluate machine learning models that incorporate domain-specific knowledge for improved analysis of hyperspectral data. This includes benchmarks, custom models, and tools for prediction and evaluation.

---

## 📁 Repository Structure
```bash
├── configs/ # Experiment and benchmark configuration files
├── notebooks/ # Jupyter notebooks for EDA and modeling
├── src/ # Core source code for training, evaluation, prediction
│ ├── data/ # Data loading and preprocessing
│ ├── models/ # Model architectures
│ ├── eval/ # Evaluation scripts
│ ├── predict/ # Prediction pipeline
│ ├── train/ # Training logic
│ └── soil_params/ # Domain-specific soil parameter estimation
├── env.yml # Conda environment dependencies
├── .gitignore
└── README.md
```
---

## 🚀 Getting Started

### 1. Environment Setup

```bash
conda env create -f env.yml
conda activate hyperview-env
```

### 2. Notebooks

### 3. Experiments

```bash
python src/entrypoint.py --config configs/benchmark.yaml
```