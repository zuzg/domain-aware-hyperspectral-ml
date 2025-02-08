from dataclasses import dataclass
from typing import Literal

from src.models.renderers.base_renderer import BaseRenderer

_mu_type = None | Literal["equal_interval", "fixed_reference", "unconstrained"]


@dataclass
class ExperimentConfig:
    img_size: int
    channels: int
    max_val: int
    dual_mode: bool
    bias_renderer: str
    variance_renderer: str
    k: int
    batch_size: int
    epochs: int
    lr: float
    device: str
    wandb: bool
    save_model: bool
    predict_soil: bool
    mu_type: _mu_type


@dataclass
class RendererConfig:
    model: BaseRenderer
    num_params: int
