import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator

from src.config import ExperimentConfig
from src.consts import GT_NAMES, MAX_PATH, MEAN_PATH, MSI_TEST_PATH, TEST_IDS, TEST_PATH
from src.data.dataset import HyperviewDataset
from src.models.modeller import Modeller
from src.options import parse_args
from src.soil_params.data import (
    aggregate_features,
    load_msi_images,
    prepare_datasets,
)


def predict_params(
    model: BaseEstimator, features: np.ndarray) -> np.ndarray:
    preds = model.predict(features)
    return preds


class Prediction:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        num_params = 5
        modeller = Modeller(self.cfg.img_size, 230, self.cfg.k, num_params)
        modeller.load_state_dict(torch.load(self.cfg.modeller_path))
        modeller.to(self.cfg.device)

        with open(MAX_PATH, "rb") as f:
            maxx = np.load(f)
        maxx[maxx > self.cfg.max_val] = self.cfg.max_val

        dataset = HyperviewDataset(TEST_PATH, TEST_IDS, self.cfg.img_size, self.cfg.max_val, 0, maxx, mask=False, bias_path=MEAN_PATH)
        preds, _ = prepare_datasets(dataset, modeller, self.cfg.k, self.cfg.channels, num_params, 1, self.cfg.device)
        print(preds.shape)
        preds_agg = aggregate_features(preds)
        # preds_agg = preds.reshape(preds.shape[0], -1)

        msi_means, _, _ = load_msi_images(MSI_TEST_PATH)
        # minerals = load_minerals(TEST_PATH)

        # sizes = np.array(dataset.size_list)
        # sizes = np.expand_dims(sizes, axis=1)

        # mask_info = np.array(dataset.is_fully_masked)
        # mask_info = np.expand_dims(mask_info, axis=1)
        # hsi = load_hsi_airborne_images(TEST_PATH, pred=True)

        # with open("output/models/airborne.pickle", "rb") as f:
        #     hsi_model = pickle.load(f)

        # hsi = hsi_model.predict(msi_means)
        # hsi = np.expand_dims(hsi, axis=1)

        preds_agg = np.concatenate([preds_agg, msi_means], axis=1)

        # with open("output/models/scaler.pickel", "rb") as f:
        #     scaler = pickle.load(f)
        # preds_agg = scaler.transform(preds_agg)

        with open(self.cfg.predictor_path, "rb") as f:
            regressor = pickle.load(f)

        # with open(f"{self.cfg.predictor_path}_cu", "rb") as f:
        #     regressor_cu = pickle.load(f)


        preds = predict_params(regressor, preds_agg)

        for i in range(4):  # B, CU, ZN, FE  
            with open(f"{self.cfg.predictor_path}_soil_{i}", "rb") as f:
                model = pickle.load(f)
            y_pred = model.predict(preds_agg)
            preds[:, i] = y_pred


        # preds_cu = predict_params(regressor_cu, preds_agg)
        # preds[:, 1] = preds_cu.astype(float)

        submission = pd.DataFrame(data=preds, columns=GT_NAMES)
        submission.to_csv(self.cfg.submission_path, index_label="sample_index")


def main() -> None:
    cfg = parse_args()
    experiment = Prediction(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
