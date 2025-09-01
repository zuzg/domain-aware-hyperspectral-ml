[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![arXiv](https://img.shields.io/badge/arXiv-2508.21618-00ff00.svg)](https://arxiv.org/abs/2508.21618)


# Physics-Informed Spectral Modeling for Hyperspectral Imaging
<p align="center">
<img width="600" height="400" alt="phism" src="https://github.com/user-attachments/assets/403ba79b-d159-4483-90eb-84c124d44025" />
</p>

This repository contains the implementation code for the paper: **"Physics-Informed Spectral Modeling for Hyperspectral Imaging"**.

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
### 2. Data
* Pavia University / Salinas Valley / Indian Pines **disjoint patches**: [paper](https://arxiv.org/pdf/1811.03707), [download link](https://tinyurl.com/ieee-grsl)
* Pavia University / Salinas Valley / Indian Pines **full scenes**: [website](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
* HYPERVIEW 1: [paper](https://ieeexplore.ieee.org/document/9897443), [website](https://platform.ai4eo.eu/seeing-beyond-the-visible-permanent/data)
* HYPERVIEW 2: [website](https://www.eotdl.com/datasets/HYPERVIEW2)


## 🧪 Experiments
### 1. Classification on Pavia University / Salinas Valley / Indian Pines

Spatially disjoint folds:
```bash
python src/benchmark/entrypoint.py --config configs/benchmark.yaml
```

Small data experiment:
```bash
python src/benchmark/entrypoint.py --config configs/small_data.yaml
```

Change between datasets and models in config files.

### 2. Regression on HYPERVIEW
For HYPERVIEW 2 see the branch `hyperview2`. For HYPERVIEW 1 run:
```bash
python src/entrypoint.py --config configs/hyperview.yaml
```

### 3. Notebooks
See `notebooks/grsl_viz.ipynb` for visualization code.
