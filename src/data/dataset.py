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
        self.size_list = []
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
                    self.size_list.append(img.shape[1])
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


class HyperpectralPatch(Dataset):
    def __init__(
        self,
        fold_dir: Path,
        mean: float = 0,
        std: float = 1,
        test: bool = False,
        extra: bool = False,
        ch: int = 0,
        h: int = 0,
        w: int = 0,
    ):
        super().__init__()
        if test:
            self.patches, self.gt = self.load_patches_gt_test(fold_dir)
            self.channel_max = std
            self.ch = ch
            self.h = h
            self.w = w
        else:
            self.patches, self.channel_max, self.channel_mean = self.load_patches(fold_dir)
            self.ch = self.patches[0].shape[1]
            self.h = self.patches[0].shape[1]
            self.w = self.patches[0].shape[w]
            if extra:
                test_patches = self.load_patches_test(fold_dir)
                self.patches = torch.cat([self.patches, test_patches], dim=0)
            mean = self.channel_mean
            self.gt = self.load_gt(fold_dir)
        self.transform = transforms.Compose([transforms.Normalize(mean, self.channel_max)])

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> Tensor:
        image = self.transform(self.patches[index])
        return image

    def load_patches(self, fold_dir: Path) -> list[Tensor]:
        filenames = [f for f in fold_dir.rglob("patch_*.npy") if "gt" not in f.name]
        filenames = sorted(filenames, key=lambda i: int(i.name.split(".")[0].split("_")[1]))
        img_tensors = []
        channel_max = None
        channel_sum = None

        for f in filenames:
            img = np.load(f)
            img_chw = img.transpose(2, 0, 1)
            tensor = torch.from_numpy(img_chw.astype(np.float32))
            img_tensors.append(tensor)

            per_tensor_max = tensor.amax(dim=(1, 2))
            if channel_max is None:
                channel_max = per_tensor_max
                channel_sum = torch.zeros_like(tensor.sum(dim=(1, 2)), dtype=torch.float32)
            else:
                channel_max = torch.maximum(channel_max, per_tensor_max)
            channel_sum += tensor.sum(dim=(1, 2))
        num_pixels = len(img_tensors) * img_tensors[0].shape[1] * img_tensors[0].shape[2]
        channel_mean = channel_sum / num_pixels
        return torch.stack(img_tensors), channel_max, channel_mean

    def load_patches_test(self, fold_dir: Path) -> list[Tensor]:
        img = np.load(fold_dir / "test.npy")
        gt = np.load(fold_dir / "test_gt.npy")
        flat_img = img.reshape(-1, img.shape[2])
        flat_gt = gt.reshape(-1)
        valid_mask = flat_gt != 0
        valid_pixels = flat_img[valid_mask].astype(np.float32)
        patches = [torch.from_numpy(p).view(img.shape[-1], 1, 1) for p in valid_pixels]

        spatial_size = self.h * self.w
        num_patches = len(patches) // spatial_size
        all_patches = torch.stack(patches[: num_patches * spatial_size])
        grouped = all_patches.view(num_patches, spatial_size, self.ch, 1, 1)

        grouped = grouped.permute(0, 2, 1, 3, 4)
        final_patches = grouped.view(num_patches, self.ch, self.h, self.w)

        return final_patches

    def load_patches_gt_test(self, fold_dir: Path) -> list[Tensor]:
        img = np.load(fold_dir / "test.npy")
        gt = np.load(fold_dir / "test_gt.npy")
        flat_img = img.reshape(-1, img.shape[2])
        flat_gt = gt.reshape(-1)
        valid_mask = flat_gt != 0
        valid_pixels = flat_img[valid_mask].astype(np.float32)
        valid_gt = flat_gt[valid_mask]
        patches = [torch.from_numpy(p).view(img.shape[-1], 1, 1) for p in valid_pixels]
        return patches, valid_gt

    def load_gt(self, fold_dir: Path) -> list[np.ndarray]:
        filenames = list(fold_dir.rglob("patch_*gt.npy"))
        filenames = sorted(filenames, key=lambda i: int(i.name.split(".")[0].split("_")[1]))
        gts = []
        for f in filenames:
            gt = np.load(f)
            gt_flat = gt.flatten()
            gts.append(gt_flat)
        return gts
