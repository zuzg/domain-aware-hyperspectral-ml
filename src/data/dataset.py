from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class HyperviewDataset(Dataset):
    def __init__(self, directory: str, size: int, max_val: int, mean: float, std: float, mask: bool = True) -> None:
        super().__init__()
        self.max_val = max_val
        self.mask = mask
        self.images = self.load_images(directory)
        self.transform = transforms.Compose([transforms.CenterCrop(size), transforms.Normalize(mean, std)])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tensor:
        image = self.transform(self.images[index])
        return image

    def load_images(self, directory: str) -> list[Tensor]:
        filenames = list(Path(directory).rglob("*.npz"))
        filenames = sorted(filenames, key=lambda i: int(i.name.split(".")[0]))
        image_list = []
        for filename in filenames:
            with np.load(filename) as npz:
                arr = np.ma.MaskedArray(**npz)
                img = arr.data
                if self.mask:
                    img[arr.mask] = 0
                img[img > self.max_val] = self.max_val  # clip outliers to max_val
                image_list.append(torch.from_numpy(img).float())
        return image_list
