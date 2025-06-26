[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?logo=anaconda&logoColor=white)

# Domain-Aware Machine Learning Architectures for Hyperspectral Remote Sensing

This repository contains the implementation code for the thesis:  
**"Domain-Aware Machine Learning Architectures for Hyperspectral Remote Sensing"**.

The project aims to design and evaluate machine learning models that incorporate domain-specific knowledge for improved analysis of hyperspectral data. This includes benchmarks, custom models, and tools for prediction and evaluation.

---

## 📁 Repository Structure
```bash
├── configs/ # Experiment and benchmark configuration files
├── notebooks/ # Jupyter notebooks for EDA and XAI
├── src/ # Core source code for training, evaluation, prediction
│ ├── benchmark/ # Code for benchmarking on hyperspectral scenes
│ ├── data/ # Data loading and preprocessing
│ ├── eval/ # Evaluation scripts
│ ├── models/ # Model architectures
│ ├── predict/ # Prediction pipeline
│ ├── soil_params/ # Downstream task
│ ├── train/ # Training logic
│ ├── config.py
│ ├── consts.py
│ ├── entrypoint.py
│ └──experiment.py # Full experiment logic
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
