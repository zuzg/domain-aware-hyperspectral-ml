from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


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
        self.bias = self.load_bias(bias_path)
        self.images = self.load_images(directory)
        self.transform = transforms.Compose([transforms.Normalize(mean, std)])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tensor:
        image = self.transform(self.images[index])
        return image

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

                    if self.mask:
                        img[arr.mask] = 0
                    img[img > self.max_val] = self.max_val  # clip outliers to max_val
                    image_list.append(torch.from_numpy(img).float())
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
