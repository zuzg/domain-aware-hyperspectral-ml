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

### 2. Inference
Run the following command. This script uses weights saved in `output\models`. Note that `features_putential.pth` is used to generate features which are then inputed to `predictor_putential.pickle`. The are five separate regressor models (random forest) for B, Cu, Zn, Fe, and S + Mn.

```bash
python src/predict/predict_ml.py --model_config configs/hyperview2.yaml
```

The submission file will be saved in `output/submissions` directory.


### 3. Training
Run the following command. Full training should take around 10 minutes using cpu.

```bash
python src/entrypoint.py --model_config configs/hyperview2.yaml
```

The submission file will be saved in `output/submissions` directory.
