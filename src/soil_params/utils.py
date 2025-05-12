from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn, Tensor
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


def compute_masks(img: Tensor, gt: Tensor, mask: Tensor, gt_div_tensor: Tensor) -> tuple[Tensor]:
    expanded_mask = mask.unsqueeze(1)
    crop_mask = expanded_mask.expand(-1, gt.shape[1], -1, -1)
    masked_gt = torch.where(crop_mask == 0, gt, torch.zeros_like(gt))
    masked_pred = torch.where(crop_mask == 0, img, torch.zeros_like(img))
    return masked_gt * gt_div_tensor, masked_pred * gt_div_tensor


def collate_fn_pad_full(batch: Tensor):
    max_h = max(img.shape[1] for img, gt in batch)  # Find max height in batch
    max_w = max(img.shape[2] for img, gt in batch)  # Find max width in batch

    padded_imgs = []
    padded_gts = []
    for img, gt in batch:
        padded_img = nn.functional.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))
        padded_gt = nn.functional.pad(gt, (0, max_w - gt.shape[2], 0, max_h - gt.shape[1]))
        padded_imgs.append(padded_img)
        padded_gts.append(padded_gt)

    return torch.stack(padded_imgs), torch.stack(padded_gts)


@dataclass
class ModelConfig:
    name: str
    model: BaseEstimator
    default_params: dict[str, Any]
    hyperparameters: dict[str, list[Any]] | None = field(default_factory=dict)


MODELS_CONFIG: dict[str, ModelConfig] = {
    "RandomForest": ModelConfig(
        name="RandomForestRegressor",
        model=RandomForestRegressor,
        default_params={"n_estimators": 200},
        hyperparameters={
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [None, 10, 20, 30, 50, 100],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "criterion": ["squared_error", "absolute_error", "poisson"],
            "random_state": [42],
        },
    ),
    "KNeighbors": ModelConfig(
        name="KNeighborsRegressor",
        model=KNeighborsRegressor,
        default_params={"n_neighbors": 3},
        hyperparameters={
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [20, 30, 40, 50],
            "p": [1, 2],
        },
    ),
    "XGB": ModelConfig(
        name="XGBRegressor",
        model=XGBRegressor,
        default_params={"n_estimators": 100, "learning_rate": 1e-2},
        hyperparameters={
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
        },
    ),
}
