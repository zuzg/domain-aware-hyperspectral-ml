import argparse

from src.config import ExperimentConfig
from src.models.renderers.base_renderer import BaseRenderer
from src.models.renderers.gaussian_renderer import GaussianRenderer


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_size", type=int, default=10)
    parser.add_argument("--max_val", type=int, default=6000)
    parser.add_argument("--renderer", type=BaseRenderer, default=GaussianRenderer)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    cfg = ExperimentConfig(**vars(args))
    return cfg
