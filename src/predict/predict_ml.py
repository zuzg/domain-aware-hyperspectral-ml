import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator

from src.consts import GT_DIM, GT_NAMES, MAX_PATH, TEST_IDS
from src.data.dataset import HyperviewDataset
from src.models.modeller import Modeller
from src.soil_params.data import prepare_datasets


@dataclass
class PredictionConfig:
    dataset_path: Path | str
    modeller_path: Path | str
    regressor_path: Path | str
    single_model: bool
    img_size: int
    channels: int
    max_val: int
    k: int
    batch_size: int
    device: str


def parse_args() -> PredictionConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/hyperview/test_data")
    parser.add_argument(
        "--modeller_path", type=str, default="output/models/modeller_var=GaussianRenderer_bias=Mean_k=5_20_k=5_full_soil.pth"
    )
    parser.add_argument("--regressor_path", type=str, default="output/models/xgb_5.pickle")
    parser.add_argument("--single_model", type=bool, default=True)
    parser.add_argument("--img_size", type=int, default=100)
    parser.add_argument("--channels", type=int, default=150)
    parser.add_argument("--max_val", type=int, default=6000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    cfg = PredictionConfig(**vars(args))
    return cfg


def predict_params(
    model: BaseEstimator, features: np.ndarray) -> np.ndarray:
    preds = model.predict(features)
    return preds


class Prediction:
    def __init__(self, cfg: PredictionConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        modeller = Modeller(self.cfg.img_size, self.cfg.channels, self.cfg.k, 4)
        modeller.load_state_dict(torch.load(self.cfg.modeller_path))
        modeller.to(self.cfg.device)

        with open(MAX_PATH, "rb") as f:
            maxx = np.load(f)
        maxx[maxx > self.cfg.max_val] = self.cfg.max_val

        dataset = HyperviewDataset(self.cfg.dataset_path, TEST_IDS, self.cfg.img_size, self.cfg.max_val, 0, maxx, mask=True)
        features = prepare_datasets(dataset, modeller, self.cfg.k, GT_DIM, self.cfg.batch_size, self.cfg.device)
        features_agg = np.sum(features, axis=(2, 3)) / np.count_nonzero(features, axis=(2, 3))

        with open(self.cfg.regressor_path, "rb") as f:
            regressor = pickle.load(f)

        preds = predict_params(regressor, features_agg)
        submission = pd.DataFrame(data=preds, columns=GT_NAMES)
        submission.to_csv("output/submission_ml_agg_5_new.csv", index_label="sample_index")


def main() -> None:
    cfg = parse_args()
    experiment = Prediction(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
