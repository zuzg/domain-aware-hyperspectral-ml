from dataclasses import dataclass

from src.models.renderers.base_renderer import BaseRenderer


@dataclass
class ExperimentConfig:
    img_size: int
    max_val: int
    bias_renderer: str
    variance_renderer: str
    k: int
    batch_size: int
    epochs: int
    lr: float
    device: str
    wandb: bool
    save_model: bool


@dataclass
class RendererConfig:
    model: BaseRenderer
    num_params: int
