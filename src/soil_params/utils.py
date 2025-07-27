from dataclasses import dataclass, field
from typing import Any

import torch
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from torch import Tensor, nn
from xgboost import XGBRegressor


def compute_masks(
    img: Tensor, gt: Tensor, mask: Tensor, gt_div_tensor: Tensor
) -> tuple[Tensor]:
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
        padded_img = nn.functional.pad(
            img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])
        )
        padded_gt = nn.functional.pad(
            gt, (0, max_w - gt.shape[2], 0, max_h - gt.shape[1])
        )
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
        name="RandomForest",
        model=RandomForestRegressor,
        default_params={"n_estimators": 200},
        hyperparameters={
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 10, 20, 30, 50, 100],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "criterion": ["squared_error"],  # "absolute_error", "poisson"],
        },
    ),
    "AdaBoost": ModelConfig(
        name="AdaBoost",
        model=AdaBoostRegressor,
        default_params={"n_estimators": 100},
        hyperparameters={},
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
        name="XGB",
        model=XGBRegressor,
        default_params={"n_estimators": 100, "learning_rate": 1e-2},
        hyperparameters={
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
        },
    ),
    "LGBM": ModelConfig(
        name="LGBM",
        model=LGBMRegressor,
        default_params={"boosting_type": "gbdt", "n_estimators": 500},
        hyperparameters={
            "n_estimators": [100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            # "max_depth": [-1, 5, 10, 20],
            # "num_leaves": [31, 50, 100],
            # "min_child_samples": [10, 20, 30],
            # "subsample": [0.6, 0.8, 1.0],
            # "colsample_bytree": [0.6, 0.8, 1.0],
            # "reg_alpha": [0.0, 0.1, 0.5],
            # "reg_lambda": [0.0, 0.1, 0.5],
        },
    ),
    "MLP": ModelConfig(
        name="MLPRegressor",
        model=MLPRegressor,
        default_params={"hidden_layer_sizes": (100,), "max_iter": 500},
        hyperparameters={
            "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100)],
            "activation": ["relu", "tanh", "logistic"],
            "solver": ["adam", "lbfgs", "sgd"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "max_iter": [200, 500, 1000],
            "random_state": [42],
        },
    ),
    "LinearRegression": ModelConfig(
        name="LinearRegression",
        model=LinearRegression,
        default_params={"fit_intercept": True},
        hyperparameters={},
    ),
    "Lasso": ModelConfig(
        name="Lasso",
        model=Lasso,
        default_params={"fit_intercept": True},
        hyperparameters={},
    ),
    "Ridge": ModelConfig(
        name="Ridge",
        model=Ridge,
        default_params={"fit_intercept": True},
        hyperparameters={},
    ),
    "SVM": ModelConfig(
        name="SVR",
        model=SVR,
        default_params={"kernel": "rbf"},
        hyperparameters={},
    ),
    "ElasticNet": ModelConfig(
        name="ElasticNet",
        model=ElasticNet,
        default_params={"alpha": 1.0},
        hyperparameters={},
    ),
}
