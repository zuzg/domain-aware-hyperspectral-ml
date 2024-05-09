from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class HyperviewDataset(Dataset):
    def __init__(self, directory: str, size: int, max_val: int) -> None:
        super().__init__()
        self.max_val = max_val
        self.images = self.load_images(directory)
        self.transform = transforms.Compose([transforms.CenterCrop(size), transforms.Normalize(0, max_val)])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tensor:
        image = self.transform(self.images[index])
        return image

    def load_images(self, directory: str) -> list[Tensor]:
        filenames = Path(directory).rglob("*.npz")
        image_list = []
        for filename in filenames:
            with np.load(filename) as npz:
                arr = np.ma.MaskedArray(**npz)
                img = arr.data
                img[img > self.max_val] = self.max_val  # clip outliers to max_val
                image_list.append(torch.from_numpy(arr.data).float())
        return image_list
