import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import Dataset
from xgboost import XGBRegressor

import wandb
from src.config import ExperimentConfig
from src.consts import MSE_BASE_K, MSE_BASE_MG, MSE_BASE_P, MSE_BASE_PH
from src.models.modeller import Modeller
from src.soil_params.data import aggregate_features, prepare_datasets, prepare_gt


def predict_params(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, model_path: str, save_model: bool=False) -> np.ndarray:
    tune_hp = False
    if tune_hp:
        hp_dict = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
        }
        reg = XGBRegressor()
        search = GridSearchCV(reg, hp_dict, scoring=["neg_mean_squared_error"],
            refit="neg_mean_squared_error", n_jobs=-1)

        model = MultiOutputRegressor(search)
        model.fit(x_train, y_train)

        print("Best params")
        for i in range(4):
            print(model.estimators_[i].best_params_)
    else:
        model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=1e-2))
        model.fit(x_train, y_train)
    if save_model:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds, multioutput="raw_values")
    return mse


def predict_params_stratified(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    num_bins = 5  
    y_array = y.to_numpy()  # Ensure y is a NumPy array

    # Initialize storage for results
    models = []
    mse_scores = []

    # Loop through each target separately
    for target_idx in range(y_array.shape[1]):
        print(f"Training model for target {target_idx+1}")

        # Bin the current target
        y_binned = pd.qcut(y_array[:, target_idx], q=num_bins, labels=False, duplicates='drop')

        # Stratified K-Fold for the current target
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        target_mse = []

        # Train separate model for this target
        for train_idx, test_idx in skf.split(x, y_binned):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y_array[train_idx, target_idx], y_array[test_idx, target_idx]

            model = XGBRegressor(n_estimators=100, learning_rate=1e-2)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            target_mse.append(mse)

        # Store results
        models.append(model)
        print("Target MSE")
        print(target_mse)
        mse_scores.append(np.mean(target_mse))
    print("-"*10)
    print(mse_scores)
    mse_scores = np.array(mse_scores)
    return mse_scores


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
            x_train_base, x_test, y_train_base, y_test = train_test_split(x, y, test_size=0.2, random_state=run)
            x_train, y_train = x_train_base[:sn], y_train_base[:sn]

            mse = predict_params_stratified(x, y)
            print(mse)
            model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=1e-2))
            model.fit(x, y)
            if save_model:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
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
    preds_agg = aggregate_features(preds)
    gt = prepare_gt(dataset.ids)
    gt = gt[:1728]
    samples  = [1728]# [500, 250, 200, 150, 100, 50, 25, 10]
    mses_mean_pred, mses_std_pred = samples_number_experiment(preds_agg, gt, samples, cfg.predictor_path)
