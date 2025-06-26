from src.experiment import Experiment
from src.options import parse_args
from src.predict.predict_ml import Prediction


def main() -> None:
    cfg = parse_args()
    experiment = Experiment(cfg=cfg)
    experiment.run()
    pred = Prediction(cfg=cfg)
    # pred.run()


if __name__ == "__main__":
    main()
