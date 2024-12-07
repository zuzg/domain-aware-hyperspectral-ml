import argparse

from src.config import ExperimentConfig


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--channels", type=int, default=103)
    parser.add_argument("--img_size", type=int, default=170)
    parser.add_argument("--max_val", type=int, default=6000)
    parser.add_argument("--bias_renderer", type=str, default="Mean")
    parser.add_argument(
        "--variance_renderer",
        type=str,
        default="GaussianRenderer",
        choices=[
            "GaussianRenderer",
            "PolynomialDegreeRenderer",
            "PolynomialRenderer",
            "SplineRenderer",
            "Autoencoder",
            "None",
        ],
    )
    parser.add_argument(
        "--mu_type", type=str, default="unconstrained", choices=["equal_interval", "fixed_reference", "unconstrained"]
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--predict_soil", type=bool, default=True)

    args = parser.parse_args()

    cfg = ExperimentConfig(**vars(args))
    return cfg
