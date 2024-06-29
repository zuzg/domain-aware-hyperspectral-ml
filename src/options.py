import argparse

from src.config import ExperimentConfig


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_size", type=int, default=10)
    parser.add_argument("--max_val", type=int, default=6000)
    parser.add_argument("--renderer", type=str, default="PolynomialDegreeRenderer")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", type=bool, default=True)

    args = parser.parse_args()

    cfg = ExperimentConfig(**vars(args))
    return cfg
