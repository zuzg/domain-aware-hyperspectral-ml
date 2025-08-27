from src.benchmark.experiment import BenchmarkExperiment
from src.benchmark.experiment_small_data import SmallDataExperiment
from src.options import parse_args


def main() -> None:
    cfg = parse_args()
    if "SMALL_DATA" in cfg.tags:
        experiment = SmallDataExperiment(cfg=cfg)
    else:
        experiment = BenchmarkExperiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
