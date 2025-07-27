# HYPERVIEW 2: PUTential

## 🚀 Getting Started

### 1. Environment Setup

```bash
conda env create -f env.yml
conda activate hyperview-env
```

### 2. Experiments
Run the following command. Full training should take around 10 minutes on cpu.

```bash
python src/entrypoint.py --config configs/hyperview2.yaml
```

The submission file will be saved in `output/submissions` directory.