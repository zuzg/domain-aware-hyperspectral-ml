import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.consts import CHANNELS, GT_PATH
from src.models.modeller import Modeller


def apply_baseline_mask(data: np.ndarray, mask: np.ndarray, channels: int) -> np.ndarray:
    expanded_mask = np.expand_dims(mask, axis=1)
    crop_mask = np.repeat(expanded_mask, repeats=channels, axis=1)
    return np.where(crop_mask == 0, data, 0)


def apply_non_baseline_mask(pred: np.ndarray, mask: np.ndarray, k: int, num_params: int, ae: bool) -> np.ndarray:
    if ae:
        expanded_mask = np.expand_dims(mask, axis=1)
        crop_mask = np.repeat(expanded_mask, repeats=k * num_params, axis=1)
    else:
        shift = pred[:, 0, num_params - 1 :]
        pred[:, :, num_params - 1] = shift
        expanded_mask = np.expand_dims(np.expand_dims(mask, axis=1), axis=2)
        crop_mask = np.repeat(np.repeat(expanded_mask, repeats=k, axis=1), repeats=num_params, axis=2)
    return np.where(crop_mask == 0, pred, 0)


def prepare_datasets(
    dataset: Dataset,
    model: Modeller,
    k: int,
    num_params: int,
    batch_size: int,
    device: str,
    ae: bool,
    baseline: bool,
) -> np.ndarray:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    features = []
    for data in dataloader:
        mask = data[:, 0] == 0
        if baseline:
            masked_pred = apply_baseline_mask(data, mask, CHANNELS)
        else:
            data = data.to(device)
            pred = model(data)
            pred = pred.cpu().detach().numpy()
            data = data.cpu().detach().numpy()
            masked_pred = apply_non_baseline_mask(pred, mask, k, num_params, ae)
        features.append(masked_pred)

    features = np.array(features)
    f_dim = CHANNELS if baseline else k * num_params
    return features.reshape(features.shape[0] * batch_size, f_dim, features.shape[-2], features.shape[-1])


def prepare_gt(ids: np.ndarray) -> pd.DataFrame:
    gt = pd.read_csv(GT_PATH)
    gt = gt.loc[gt["sample_index"].isin(ids)]
    gt = gt.drop(["sample_index"], axis=1)
    gt = gt.reset_index(drop=True)
    return gt


class SoilDataset(Dataset):
    def __init__(self, image_list: list[Tensor], size: int, gt: pd.DataFrame, gt_div: np.ndarray) -> None:
        super().__init__()
        self.size = size
        self.images = image_list
        self.gt = gt
        self.gt_div = gt_div

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor]:
        image = self.images[index]
        soil = torch.from_numpy(self.gt.loc[index].values / self.gt_div).float()
        gt_unsqueezed = soil.unsqueeze(1).unsqueeze(2)
        gt_repeated = gt_unsqueezed.repeat(1, self.size, self.size)
        return image, gt_repeated
