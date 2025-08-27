from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

from src.config import ExperimentConfig
from src.consts import GT_MAX, GT_NAMES, MAX_PATH, MEAN_PATH, TEST_IDS, TEST_PATH
from src.data.dataset import HyperviewDataset
from src.models.modeller import Modeller
from src.options import parse_args
from src.soil_params.data import prepare_datasets
from src.soil_params.pred import MultiRegressionCNN


class PredDataset(Dataset):
    def __init__(self, image_list: list[Tensor]) -> None:
        super().__init__()
        self.images = image_list

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tensor:
        image = self.images[index]
        return image


def compute_masks(img: Tensor, mask: Tensor, gt_div_tensor: Tensor) -> tuple[Tensor]:
    expanded_mask = mask.unsqueeze(1)
    crop_mask = expanded_mask.expand(-1, gt_div_tensor.shape[1], -1, -1)
    masked_pred = torch.where(crop_mask == 0, img, torch.zeros_like(img))
    return masked_pred * gt_div_tensor


def predict_params(model: nn.Module, testloader: DataLoader, gt_div: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    gt_div_tensor = torch.tensor(gt_div, device=device).reshape(1, len(gt_div), 1, 1)
    preds = []

    with torch.no_grad():
        # TODO add tqdm
        for img in testloader:
            img = img.to(device)
            mask = img[:, 0] == 0
            div = (img[:, 0] != 0).sum().item()  # Count of non-zero elements in channel 0
            pred = model(img)
            masked_pred = compute_masks(pred, mask, gt_div_tensor)
            masked_pred_mean = masked_pred.sum(dim=(0, 2, 3)) / div
            pred_arr = masked_pred_mean.cpu().detach().numpy()
            if len(gt_div) == 1:
                pred_arr = pred_arr[0]
            preds.append(pred_arr)
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
        features = prepare_datasets(dataset, modeller, self.cfg.k, self.cfg.channels, 5, 1, self.cfg.device)
        pred_dataset = PredDataset(features)
        dataloader = DataLoader(pred_dataset, batch_size=1, shuffle=False)
        single_model = True
        if single_model:
            regressor = MultiRegressionCNN(25)
            regressor.load_state_dict(torch.load(self.cfg.predictor_path))
            regressor.to(self.cfg.device)
            preds = predict_params(regressor, dataloader, GT_MAX, self.cfg.device)
            submission = pd.DataFrame(data=preds, columns=GT_NAMES)
        else:
            submission = pd.DataFrame(columns=GT_NAMES)
            base_path = Path("output")
            models = [
                "regressor_full_325.0.pth",
                "regressor_full_625.0.pth",
                "regressor_full_400.0.pth",
                "regressor_full_7.8.pth",
            ]
            for gt_name, gt_max, model_path in zip(GT_NAMES, GT_MAX, models):
                regressor = MultiRegressionCNN(20, output_channels=1)
                regressor.load_state_dict(torch.load(base_path / model_path))
                regressor.to(self.cfg.device)
                preds = predict_params(regressor, dataloader, [gt_max], self.cfg.device)
                submission[gt_name] = preds

        submission.to_csv(self.cfg.submission_path, index_label="sample_index")


def main() -> None:
    cfg = parse_args()
    experiment = Prediction(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
