import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator

from src.config import ExperimentConfig
from src.consts import GT_NAMES, MAX_PATH, MEAN_PATH, TEST_IDS, TEST_PATH
from src.data.dataset import HyperviewDataset
from src.models.modeller import Modeller
from src.options import parse_args
from src.soil_params.data import prepare_datasets


def predict_params(
    model: BaseEstimator, features: np.ndarray) -> np.ndarray:
    preds = model.predict(features)
    return preds


class Prediction:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        modeller = Modeller(self.cfg.img_size, self.cfg.channels, self.cfg.k, 5)
        modeller.load_state_dict(torch.load(self.cfg.modeller_path))
        modeller.to(self.cfg.device)

        with open(MAX_PATH, "rb") as f:
            maxx = np.load(f)
        maxx[maxx > self.cfg.max_val] = self.cfg.max_val

        dataset = HyperviewDataset(TEST_PATH, TEST_IDS, self.cfg.img_size, self.cfg.max_val, 0, maxx, mask=True, bias_path=MEAN_PATH)
        features = prepare_datasets(dataset, modeller, self.cfg.k, self.cfg.channels, 5, self.cfg.batch_size, self.cfg.device)
        features_agg = np.sum(features, axis=(2, 3)) / np.count_nonzero(features, axis=(2, 3))

        with open(self.cfg.predictor_path, "rb") as f:
            regressor = pickle.load(f)

        preds = predict_params(regressor, features_agg)
        submission = pd.DataFrame(data=preds, columns=GT_NAMES)
        submission.to_csv(self.cfg.submission_path, index_label="sample_index")


def main() -> None:
    cfg = parse_args()
    experiment = Prediction(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
