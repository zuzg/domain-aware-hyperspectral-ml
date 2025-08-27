import argparse
from dataclasses import dataclass

import yaml

from src.config import ExperimentConfig


@dataclass
class Args:
    config: str

    @classmethod
    def from_cli(cls) -> "Args":
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="configs/dummy.yaml")
        args = parser.parse_args()
        return Args(
            config=args.config,
        )


def parse_args() -> ExperimentConfig:
    args = Args.from_cli()
    with open(args.config) as file:
        config_data = yaml.safe_load(file)
        cfg = ExperimentConfig(**config_data)

    return cfg
