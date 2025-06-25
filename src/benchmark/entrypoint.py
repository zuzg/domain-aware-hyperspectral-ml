from src.benchmark.experiment import BenchmarkExperiment
from src.options import parse_args


def main() -> None:
    cfg = parse_args()
    experiment = BenchmarkExperiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
