# HYPERVIEW 2: PUTential

## 🚀 Getting Started

### 1. Environment Setup
Create conda environment.

```bash
conda env create -f env.yml
conda activate hyperview2
```

Data files should be copied to `data/HYPERVIEW2` directory with following structure:

```bash
├── data/
│   └── HYPERVIEW2/
│       ├── stats/                     # directory with precomputed channelwise statistics
│       ├── test/
│       │   ├── hsi_satellite/
│       │   └── msi_satellite/
│       ├── train/
│       │   ├── hsi_airborne/
│       │   ├── hsi_satellite/
│       │   └── msi_satellite/
└──     └── train_gt.csv
```

### 2. Experiments
Run the following command. Full training should take around 10 minutes using cpu.

```bash
python src/entrypoint.py --model_config configs/hyperview2.yaml
```

The submission file will be saved in `output/submissions` directory.
