import numpy as np
import wandb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import Dataset
from xgboost import XGBRegressor

from src.config import ExperimentConfig
from src.consts import CHANNELS, MSE_BASE_K, MSE_BASE_MG, MSE_BASE_P, MSE_BASE_PH
from src.models.modeller import Modeller
from src.soil_params.data import prepare_datasets, prepare_gt


def predict_params(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=1e-3))
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds, multioutput="raw_values")
    return mse


def samples_number_experiment(
    x: np.ndarray, y: np.ndarray, sample_nums: list[int], n_runs: int = 10
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    mses_mean = []
    mses_std = []
    wandb.define_metric("soil/step")
    wandb.define_metric("soil/*", step_metric="soil/step")

    for sn in sample_nums:
        mses_for_sample = []

        for run in range(n_runs):
            x_train_base, x_test, y_train_base, y_test = train_test_split(x, y, test_size=0.2, random_state=run)
            x_train, y_train = x_train_base[:sn], y_train_base[:sn]

            mse = predict_params(x_train, x_test, y_train, y_test)
            mses_for_sample.append(mse / [MSE_BASE_P, MSE_BASE_K, MSE_BASE_MG, MSE_BASE_PH])

        mses_for_sample = np.array(mses_for_sample)
        mses_sample_mean = mses_for_sample.mean(axis=0)
        mses_mean.append(mses_sample_mean)
        mses_std.append(mses_for_sample.std(axis=0))
        wandb.log(
            {
                "soil/step": -sn,
                "soil/P": mses_sample_mean[0],
                "soil/K": mses_sample_mean[1],
                "soil/Mg": mses_sample_mean[2],
                "soil/pH": mses_sample_mean[3],
                "soil/score": 0.25 * np.sum(mses_sample_mean),
            }
        )
    return np.array(mses_mean), np.array(mses_std)


def predict_soil_parameters(
    dataset: Dataset,
    model: Modeller,
    num_params: int,
    cfg: ExperimentConfig,
    ae: bool,
) -> None:
    imgs_agg, preds_agg = prepare_datasets(dataset, model, cfg.k, num_params, cfg.batch_size, CHANNELS, cfg.device, ae)
    gt = prepare_gt(dataset.ids)
    samples = [500, 250, 200, 150, 100, 50, 25, 10]
    mses_mean_pred, mses_std_pred = samples_number_experiment(preds_agg, gt, samples)
