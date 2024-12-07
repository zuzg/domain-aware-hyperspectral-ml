from src.benchmark.experiment import Experiment
from src.benchmark.options import parse_args


def main() -> None:
    cfg = parse_args()
    experiment = Experiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
