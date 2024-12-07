import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

from src.consts import GT_DIM, GT_MAX, GT_NAMES, MAX_PATH, TEST_IDS
from src.data.dataset import HyperviewDataset
from src.models.modeller import Modeller
from src.soil_params.data import prepare_datasets
from src.soil_params.pred import MultiRegressionCNN


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
        "--modeller_path", type=str, default="output/modeller_var=GaussianRenderer_bias=Mean_k=5_20_k=5_full_soil.pth"
    )
    parser.add_argument("--regressor_path", type=str, default="output/regressor_full_single.pth")
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


def predict_params(
    model: nn.Module, testloader: DataLoader, gt_div: np.ndarray, device: str) -> np.ndarray:
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
        pred_dataset = PredDataset(features)
        dataloader = DataLoader(pred_dataset, batch_size=1, shuffle=False)

        if self.cfg.single_model:
            regressor = MultiRegressionCNN(20)
            regressor.load_state_dict(torch.load(self.cfg.regressor_path))
            regressor.to(self.cfg.device)
            preds = predict_params(regressor, dataloader, GT_MAX, self.cfg.device)
            submission = pd.DataFrame(data=preds, columns=GT_NAMES)
        else:
            submission = pd.DataFrame(columns=GT_NAMES)
            base_path = Path("output")
            models = ["regressor_full_325.0.pth", "regressor_full_625.0.pth", "regressor_full_400.0.pth", "regressor_full_7.8.pth"]
            for gt_name, gt_max, model_path in zip(GT_NAMES, GT_MAX, models):
                regressor = MultiRegressionCNN(20, output_channels=1)
                regressor.load_state_dict(torch.load(base_path / model_path))
                regressor.to(self.cfg.device)
                preds = predict_params(regressor, dataloader, [gt_max], self.cfg.device)
                submission[gt_name] = preds

        submission.to_csv("output/submission_single_NEW_small_16.csv", index_label="sample_index")


def main() -> None:
    cfg = parse_args()
    experiment = Prediction(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
