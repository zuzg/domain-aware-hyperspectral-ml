from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.consts import GT_PATH, MEAN_PATH
from src.models.modeller import Modeller


def collate_fn_pad(batch):
    max_h = 200  # max(img.shape[1] for img in batch)  # Find max height in batch
    max_w = 200  # max(img.shape[2] for img in batch)  # Find max width in batch

    padded_batch = []
    for img in batch:
        padded_img = torch.nn.functional.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))  # Pad to max size
        padded_batch.append(padded_img)

    return torch.stack(padded_batch)  # Stack into a single tensor


def apply_baseline_mask(data: np.ndarray, mask: np.ndarray, channels: int) -> np.ndarray:
    expanded_mask = np.expand_dims(mask, axis=1)
    crop_mask = np.repeat(expanded_mask, repeats=channels, axis=1)
    return np.where(crop_mask == 0, data, 0)


def apply_non_baseline_mask(pred: np.ndarray, mask: np.ndarray, k: int, num_params: int, ae: bool) -> np.ndarray:
    if ae:
        expanded_mask = np.expand_dims(mask, axis=1)
        crop_mask = np.repeat(expanded_mask, repeats=k * num_params, axis=1)
    else:
        shift = pred[:, 0, num_params - 2 : num_params - 1]
        pred[:, :, num_params - 2] = shift
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn_pad)
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
            masked_pred = apply_non_baseline_mask(pred, mask, k, num_params, ae)
        features.append(masked_pred)

    features = np.array(features)
    # sort by mu
    sort_key = features[:, :, :, 0, :, :]
    sort_indices = np.argsort(sort_key, axis=2)
    sorted_features = np.take_along_axis(features, np.expand_dims(sort_indices, axis=3), axis=2)

    f_dim = channels if baseline else k * num_params
    return sorted_features.reshape(features.shape[0] * batch_size, f_dim, features.shape[-2], features.shape[-1])


def prepare_gt(ids: np.ndarray) -> pd.DataFrame:
    gt = pd.read_csv(GT_PATH)
    gt = gt.loc[gt["sample_index"].isin(ids)]
    gt = gt.drop(["sample_index"], axis=1)
    gt = gt.reset_index(drop=True)
    return gt


def aggregate_features(features: np.ndarray) -> np.ndarray:
    features[features == 0] = np.nan
    preds_mean = np.nanmean(features, axis=(2, 3))
    preds_max = np.nanmax(features, axis=(2, 3))
    preds_var = np.nanvar(features, axis=(2, 3))
    preds_min = np.nanmin(features, axis=(2, 3))
    # preds_median = np.nanmedian(features, axis=(2, 3))
    # preds_skew = skew(features, axis=(2, 3), nan_policy="omit")
    # preds_kurt = kurtosis(features, axis=(2, 3), nan_policy="omit")
    # q75, q25 = np.nanpercentile(features, [75, 25], axis=(2, 3))
    # preds_iqr = q75 - q25

    features_agg = np.concatenate(
        [preds_mean, preds_max, preds_var, preds_min], axis=1
    )  # , preds_median, preds_skew, preds_kurt, preds_iqr], axis=1)
    return features_agg


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


class ImgGtDataset(Dataset):
    def __init__(self, directory: str, ids: np.ndarray, std, max_val, gt: pd.DataFrame, gt_div: np.ndarray, size: int):
        super().__init__()
        self.gt = gt
        self.ids = ids
        self.gt_div = gt_div
        self.size = size
        self.max_val = max_val
        self.bias = self.load_bias(MEAN_PATH)
        self.images = self.load_images(directory)
        self.transform = transforms.Compose([transforms.Normalize(0, std)])
        self.gt_dim = len(gt_div)
        self.gt_values = self._prepare_gt_values()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        return image, self.gt_values[index]

    def _prepare_gt_values(self) -> torch.Tensor:
        num_images = len(self.images)
        gt_values = torch.zeros((num_images, self.gt_dim, self.size, self.size), dtype=torch.float32)

        for i in range(num_images):
            soil = torch.from_numpy(self.gt.loc[i].values / self.gt_div).float()
            gt_unsqueezed = soil.unsqueeze(1).unsqueeze(2)
            gt_values[i] = gt_unsqueezed.repeat(1, self.size, self.size)

        return gt_values

    def load_bias(self, bias_path: str) -> np.ndarray:
        with open(bias_path, "rb") as f:
            bias = np.load(f)
        return bias.reshape(bias.shape[0], 1, 1)

    def load_images(self, directory: str) -> list[Tensor]:
        filenames = list(Path(directory).rglob("*.npz"))
        filenames = sorted(filenames, key=lambda i: int(i.name.split(".")[0]))
        image_list = []
        for filename in filenames:
            if int(filename.stem) in self.ids:
                with np.load(filename) as npz:
                    arr = np.ma.MaskedArray(**npz)
                    img = arr.data - self.bias
                    img[arr.mask] = 0
                    img[img > self.max_val] = self.max_val  # clip outliers to max_val
                    image_list.append(torch.from_numpy(img).float())
        return image_list
