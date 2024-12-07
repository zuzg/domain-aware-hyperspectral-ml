import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.consts import GT_PATH
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
    channels: int,
    num_params: int,
    batch_size: int,
    device: str,
    ae: bool = False,
    baseline: bool = False,
) -> np.ndarray:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    features = []
    model.eval()
    for data in dataloader:
        mask = data[:, 0] == 0
        if baseline:
            masked_pred = apply_baseline_mask(data, mask, channels)
        else:
            data = data.to(device)
            pred = model(data)
            pred = pred.cpu().detach().numpy()
            data = data.cpu().detach().numpy()
            shift = pred[:,0,num_params - 1:]
            pred[:,:, num_params - 1] = shift
            masked_pred = apply_non_baseline_mask(pred, mask, k, num_params, ae)
        features.append(masked_pred)

    features = np.array(features)
    f_dim = channels if baseline else k * num_params
    return features.reshape(features.shape[0] * batch_size, f_dim, features.shape[-2], features.shape[-1])


def prepare_gt(ids: np.ndarray) -> pd.DataFrame:
    gt = pd.read_csv(GT_PATH)
    gt = gt.loc[gt["sample_index"].isin(ids)]
    gt = gt.drop(["sample_index"], axis=1)
    gt = gt.reset_index(drop=True)
    return gt

class SoilDataset(Dataset):
    def __init__(
        self, image_list: np.ndarray, size: int, gt: pd.DataFrame, gt_div: np.ndarray, shuffle: bool = True
    ) -> None:
        super().__init__()
        self.size = size
        self.images = torch.from_numpy(image_list)
        self.gt = gt
        self.gt_div = gt_div
        self.gt_dim = len(gt_div)

        if shuffle:
            self.images, self.gt_values = self._shuffle_pixels_between_images()
        else:
            self.gt_values = self._prepare_gt_values()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        return self.images[index], self.gt_values[index]

    def _prepare_gt_values(self) -> torch.Tensor:
        num_images = len(self.images)
        gt_values = torch.zeros((num_images, self.gt_dim, self.size, self.size), dtype=torch.float32)

        for i in range(num_images):
            soil = torch.from_numpy(self.gt.loc[i].values / self.gt_div).float()
            gt_unsqueezed = soil.unsqueeze(1).unsqueeze(2)
            gt_values[i] = gt_unsqueezed.repeat(1, self.size, self.size)

        return gt_values

    def _shuffle_pixels_between_images(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_images, channels, height, width = self.images.shape
        num_pixels = height * width

        # Flatten images and ground truth tensors
        flat_images = self.images.view(num_images, channels, num_pixels)
        flat_gt = self._prepare_gt_values().view(num_images, self.gt_dim, num_pixels)

        # Shuffle pixels across all images using random indices
        shuffled_indices = torch.randperm(num_pixels)
        shuffled_images = flat_images[:, :, shuffled_indices]
        shuffled_gt = flat_gt[:, :, shuffled_indices]

        # Reshape back to original dimensions
        shuffled_images = shuffled_images.view(num_images, channels, height, width)
        shuffled_gt = shuffled_gt.view(num_images, self.gt_dim, height, width)

        return shuffled_images, shuffled_gt
