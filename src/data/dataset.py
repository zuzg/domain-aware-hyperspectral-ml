from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from src.consts import water_bands


def pad_to_length(arr: np.ndarray, target_length: int = 9) -> np.ndarray:
    return np.pad(arr, (0, target_length - len(arr)), mode="constant")


class HyperviewDataset(Dataset):
    def __init__(
        self,
        directory: str,
        ids: np.ndarray,
        size: int,
        max_val: int,
        mean: float,
        std: float,
        mask: bool = True,
        bias_path: str | None = None,
    ) -> None:
        super().__init__()
        self.ids = ids
        self.max_val = max_val
        self.mask = mask
        self.size_list = []
        self.fully_masked_ids = []
        self.flat_masks = []
        self.is_fully_masked = []
        self.bias = self.load_bias(bias_path)
        self.images = self.load_images(directory)
        # self.bias = self.bias[self.water_mask]
        self.transform = transforms.Compose([transforms.Normalize(mean, std)])#[self.water_mask]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tensor:
        image = self.transform(self.images[index])
        return image

    def load_bias(self, bias_path: str) -> np.ndarray:
        with open(bias_path, "rb") as f:
            bias = np.load(f)
        return bias.reshape(bias.shape[0], 1, 1)
    
    def delete_water_bands(self, img: np.ndarray) -> np.ndarray:
        channel_indices = np.arange(img.shape[0])
        keep_mask = np.ones(img.shape[0], dtype=bool)
        for start, end in water_bands:
            keep_mask[(channel_indices > start) & (channel_indices < end)] = False
        self.water_mask = keep_mask
        filtered_img = img[keep_mask]
        return filtered_img

    def load_images(self, directory: str) -> list[Tensor]:
        filenames = [f for f in Path(directory).rglob("*.npz") if "hsi_satellite" in str(f)]
        filenames = sorted(filenames, key=lambda i: int(i.name.split(".")[0]))
        image_list = []
        for filename in filenames:
            if int(filename.stem) in self.ids:
                with np.load(filename) as npz:
                    arr = np.ma.MaskedArray(**npz)
                    img = arr.data - self.bias
                    img_raw = arr.data

                    # if self.mask and np.sum(img_raw) > 0:
                    #     img[arr.mask] = 0

                    if np.sum(img_raw) == 0:
                        self.fully_masked_ids.append(int(filename.stem))
                        self.is_fully_masked.append(1)
                    else:
                        self.is_fully_masked.append(0)
                    # img = self.delete_water_bands(img)
                    # padded_flat_mask = pad_to_length(arr.mask[0].flatten().astype(int))
                    # self.flat_masks.append(padded_flat_mask)

                    img[img > self.max_val] = self.max_val  # clip outliers to max_val
                    self.size_list.append(img.shape[1]*img.shape[2])
                    image_list.append(torch.from_numpy(img).float())
        self.channels = img.shape[0]
        return image_list


class HyperspectralScene(Dataset):
    def __init__(self, img: np.ndarray, mean: float, std: float) -> None:
        super().__init__()
        self.img = self.prepare_arr(img)
        self.transform = transforms.Normalize(mean, std)

    def __len__(self) -> int:
        return len(self.img)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = self.transform(self.img)
        return image

    def prepare_arr(self, img: np.ndarray) -> Tensor:
        img_r = img.transpose(2, 0, 1)
        return torch.from_numpy(img_r.astype(np.float32))
