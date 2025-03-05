import pickle

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import Dataset
from xgboost import XGBRegressor

import wandb
from src.config import ExperimentConfig
from src.consts import MSE_BASE_K, MSE_BASE_MG, MSE_BASE_P, MSE_BASE_PH
from src.models.modeller import Modeller
from src.soil_params.data import prepare_datasets, prepare_gt


def predict_params(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, model_path: str, save_model: bool=False) -> np.ndarray:
    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=1e-2))
    model.fit(x_train, y_train)
    if save_model:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds, multioutput="raw_values")
    return mse


def samples_number_experiment(
    x: np.ndarray, y: np.ndarray, sample_nums: list[int], model_path: str, n_runs: int = 1, 
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    mses_mean = []
    mses_std = []
    wandb.define_metric("soil/step")
    wandb.define_metric("soil/*", step_metric="soil/step")
    save_model = True

    for sn in sample_nums:
        mses_for_sample = []

        for run in range(n_runs):
            print(run)
            # x_train_base, x_test, y_train_base, y_test = train_test_split(x, y, test_size=0.2, random_state=run)
            # x_train, y_train = x_train_base[:sn], y_train_base[:sn]

            mse = predict_params(x, x, y, y, model_path, save_model)
            save_model = False
            print(mse)
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
    preds = prepare_datasets(dataset, model, cfg.k, cfg.channels, num_params, cfg.batch_size, cfg.device, ae)
    preds_agg = np.sum(preds, axis=(2, 3)) / np.count_nonzero(preds, axis=(2, 3))
    gt = prepare_gt(dataset.ids)
    gt = gt[:1728]
    samples  = [1728]# [500, 250, 200, 150, 100, 50, 25, 10]
    mses_mean_pred, mses_std_pred = samples_number_experiment(preds_agg, gt, samples, cfg.predictor_path)
