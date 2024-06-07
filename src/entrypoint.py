from src.experiment import Experiment
from src.options import parse_args


def main() -> None:
    cfg = parse_args()
    experiment = Experiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
