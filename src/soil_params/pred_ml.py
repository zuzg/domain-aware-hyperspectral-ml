import pickle

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import Dataset
from xgboost import XGBRegressor

from src.config import ExperimentConfig
from src.consts import GT_NAMES, MSE_BASE_K, MSE_BASE_MG, MSE_BASE_P, MSE_BASE_PH
from src.models.modeller import Modeller
from src.soil_params.data import aggregate_features, prepare_datasets, prepare_gt
from src.soil_params.utils import ModelConfig, MODELS_CONFIG


def predict_params(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_config: ModelConfig,
    model_path: str,
    save_model: bool = False,
) -> np.ndarray:
    tune_hp = False
    if tune_hp:
        search = HalvingRandomSearchCV(
            model_config.model(), model_config.hyperparameters, scoring="neg_mean_squared_error", refit=True, n_jobs=-1
        )
        model = MultiOutputRegressor(search)
        model.fit(x_train, y_train)

        print(f"Best params for {model_path}")
        for i in range(4):
            print(model.estimators_[i].best_params_)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        model = MultiOutputRegressor(model_config.model(**model_config.default_params))
        model.fit(x_train, y_train)
    if save_model:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds, multioutput="raw_values")
    return mse


def predict_params_stratified(x: np.ndarray, y: np.ndarray, model_config: ModelConfig, num_bins: int = 5) -> np.ndarray:
    x_array = np.array(x)
    y_array = np.array(y)
    models = []
    mse_scores = []
    mse_stds = []

    for target_idx in range(y_array.shape[1]):
        print(f"Training model for target {GT_NAMES[target_idx]}")
        y_binned = pd.qcut(y_array[:, target_idx], q=num_bins, labels=False, duplicates="drop")
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        target_mse = []

        for train_idx, test_idx in skf.split(x_array, y_binned):
            x_train, x_test = x_array[train_idx], x_array[test_idx]
            y_train, y_test = y_array[train_idx, target_idx], y_array[test_idx, target_idx]

            model = model_config.model(**model_config.default_params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            target_mse.append(mse)

        models.append(model)
        print("Target MSE")
        print(target_mse)
        mse_scores.append(np.mean(target_mse))
        mse_stds.append(np.std(target_mse))
    print("-" * 10)
    print(mse_scores)
    print(mse_stds)
    mse_scores = np.array(mse_scores)
    return mse_scores


def samples_number_experiment(
    x: np.ndarray,
    y: np.ndarray,
    sample_nums: list[int],
    model_config: ModelConfig,
    model_path: str,
    n_runs: int = 5,
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
            x_train_base, x_test, y_train_base, y_test = train_test_split(x, y, test_size=0.2, random_state=run)
            x_train, y_train = x_train_base[:sn], y_train_base[:sn]

            # mse = predict_params(x_train, x_test, y_train, y_test, model_config, model_path, save_model)
            mse = predict_params_stratified(x, y, model_config)
            print(mse)
            # model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200))
            # model.fit(x, y)
            # if save_model:
            #     with open(model_path, "wb") as f:
            #         pickle.dump(model, f)
            save_model = False
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
                "soil/std": 0.25 * np.sum(mses_for_sample.std(axis=0)),
            }
        )
    return np.array(mses_mean), np.array(mses_std)


def predict_soil_parameters(
    dataset: Dataset, model: Modeller, num_params: int, cfg: ExperimentConfig, ae: bool, split: bool = False
) -> None:
    preds, avg_refls = prepare_datasets(
        dataset, model, cfg.k, cfg.channels, num_params, cfg.batch_size, cfg.device, ae, baseline=False
    )
    preds_agg = aggregate_features(preds, extended=False)
    features = preds_agg

    gt = prepare_gt(dataset.ids)
    max_samples = 1732
    gt = gt[:max_samples].values
    samples = [
        max_samples,
        # int(max_samples * 0.5),
        # int(max_samples * 0.25),
        # int(max_samples * 0.1),
        # int(max_samples * 0.05),
        # int(max_samples * 0.01),
    ]  # [500, 250, 200, 150, 100, 50, 25, 10]

    if split:
        sizes = dataset.size_list
        indices_small = [i for i, size in enumerate(sizes) if size <= 11 and i < max_samples]
        indices_large = [i for i, size in enumerate(sizes) if size > 11 and i < max_samples]
        preds_small = [features[i] for i in indices_small]
        gt_small = [gt[i] for i in indices_small]
        preds_large = [features[i] for i in indices_large]
        gt_large = [gt[i] for i in indices_large]

        mses_mean_pred, mses_std_pred = samples_number_experiment(
            preds_small, gt_small, samples, MODELS_CONFIG["RandomForest"], f"{cfg.predictor_path}_small"
        )
        mses_mean_pred, mses_std_pred = samples_number_experiment(
            preds_large, gt_large, samples, MODELS_CONFIG["RandomForest"], f"{cfg.predictor_path}_large"
        )
    else:
        mses_mean_pred, mses_std_pred = samples_number_experiment(
            features, gt, samples, MODELS_CONFIG["RandomForest"], f"{cfg.predictor_path}"
        )
