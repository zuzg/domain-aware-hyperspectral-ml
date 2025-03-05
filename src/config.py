from dataclasses import dataclass
from typing import Literal

from src.models.renderers.base_renderer import BaseRenderer

_mu_type = None | Literal["equal_interval", "fixed_reference", "unconstrained"]


@dataclass
class ExperimentConfig:
    # data
    img_size: int
    channels: int
    max_val: int
    # model
    dual_mode: bool
    bias_renderer: str
    variance_renderer: str
    mu_type: _mu_type
    k: int
    # training
    batch_size: int
    epochs: int
    lr: float
    device: str
    # wandb
    wandb: bool
    tags: list[str]
    # misc
    save_model: bool
    predict_soil: bool
    # paths
    modeller_path: str
    predictor_path: str
    submission_path: str


@dataclass
class RendererConfig:
    model: BaseRenderer
    num_params: int
