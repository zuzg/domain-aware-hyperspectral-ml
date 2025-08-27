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
from src.soil_params.data import aggregate_features, compute_indices_hsi, prepare_datasets


def predict_params(model: BaseEstimator, features: np.ndarray) -> np.ndarray:
    preds = model.predict(features)
    return preds


class Prediction:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        modeller = Modeller(self.cfg.channels, self.cfg.k, 5)
        modeller.load_state_dict(torch.load(self.cfg.modeller_path))
        modeller.to(self.cfg.device)

        with open(MAX_PATH, "rb") as f:
            maxx = np.load(f)
        maxx[maxx > self.cfg.max_val] = self.cfg.max_val

        dataset = HyperviewDataset(TEST_PATH, TEST_IDS, self.cfg.max_val, 0, maxx, mask=True, bias_path=MEAN_PATH)
        preds, avg_refls = prepare_datasets(dataset, modeller, self.cfg.k, self.cfg.channels, 5, 1, self.cfg.device)
        preds_agg = aggregate_features(preds)
        indices = compute_indices_hsi(TEST_PATH)
        features = np.concatenate([preds_agg, indices], axis=1)
        print(features.shape)
        print(self.cfg.predictor_path)

        with open(f"{self.cfg.predictor_path}_small", "rb") as f:
            regressor_small = pickle.load(f)

        with open(f"{self.cfg.predictor_path}_large", "rb") as f:
            regressor_large = pickle.load(f)

        preds = []
        for img, feat in zip(dataset, features):
            feat = np.expand_dims(feat, axis=0)
            if img.shape[1] <= 11:
                preds.append(regressor_small.predict(feat)[0])
            else:
                preds.append(regressor_large.predict(feat)[0])

        # preds = predict_params(regressor, preds_agg)
        submission = pd.DataFrame(data=preds, columns=GT_NAMES)
        submission.to_csv(self.cfg.submission_path, index_label="sample_index")


def main() -> None:
    cfg = parse_args()
    experiment = Prediction(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
