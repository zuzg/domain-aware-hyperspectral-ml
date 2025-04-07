import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

from src.config import ExperimentConfig
from src.consts import GT_DIM, GT_MAX, GT_NAMES, MAX_PATH, MEAN_PATH, TEST_IDS, TEST_PATH
from src.data.dataset import HyperviewDataset
from src.models.modeller import Modeller
from src.options import parse_args
from src.soil_params.end_to_end import EndToEndModel, MultiRegressionCNN


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
        modeller = Modeller(self.cfg.img_size, self.cfg.channels, self.cfg.k, 5)
        regressor = MultiRegressionCNN(25)
        model = EndToEndModel(modeller, regressor)
        model.load_state_dict(torch.load(self.cfg.predictor_path))
        model.to(self.cfg.device)

        with open(MAX_PATH, "rb") as f:
            maxx = np.load(f)
        maxx[maxx > self.cfg.max_val] = self.cfg.max_val
        dataset = HyperviewDataset(
            TEST_PATH, TEST_IDS, self.cfg.img_size, self.cfg.max_val, 0, maxx, mask=True, bias_path=MEAN_PATH
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        preds = predict_params(model, dataloader, GT_MAX, self.cfg.device)
        submission = pd.DataFrame(data=preds, columns=GT_NAMES)
        submission.to_csv(self.cfg.submission_path, index_label="sample_index")


def main() -> None:
    cfg = parse_args()
    experiment = Prediction(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
