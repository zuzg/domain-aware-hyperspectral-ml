import logging
import pickle

import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
from torch.utils.data import Dataset

import wandb
from src.config import ExperimentConfig
from src.consts import (
    GT_NAMES,
    MSE_BASE_B,
    MSE_BASE_CU,
    MSE_BASE_FE,
    MSE_BASE_MN,
    MSE_BASE_S,
    MSE_BASE_ZN,
    MSI_TRAIN_PATH,
)
from src.models.modeller import Modeller
from src.soil_params.data import (
    aggregate_features,
    load_msi_images,
    prepare_datasets,
    prepare_gt,
)
from src.soil_params.utils import MODELS_CONFIG, ModelConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)



def get_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci = stats.t.interval(confidence, df=n - 1, loc=mean, scale=std_err)
    return ci


def objective(
    trial: optuna.trial.Trial,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    target_index: int,
):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "criterion": trial.suggest_categorical(
            "criterion", ["squared_error", "absolute_error", "poisson"]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    gt = y_train[:, target_index]
    weights = compute_sample_weight("balanced", gt)

    scores = cross_val_score(
        model,
        x_train,
        gt,
        scoring="neg_mean_squared_error",
        cv=5,
        params={"sample_weight": weights},
    )
    return -np.mean(scores)


def train_models_with_optuna(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_config: ModelConfig,
    model_path: str,
):
    preds = np.zeros_like(y_test)

    for i in range(6):
        log.info(f"Tuning model for param {i}")

        def optuna_objective(trial):
            return objective(trial, x_train, y_train, x_test, y_test, i)

        study = optuna.create_study(direction="minimize")
        study.optimize(optuna_objective, n_trials=100, show_progress_bar=True)

        log.info(f"Best params for target {i}: {study.best_params}")

        best_model = model_config.model(**study.best_params)
        gt = y_train[:, i]
        weights = compute_sample_weight("balanced", gt)

        best_model.fit(x_train, gt, sample_weight=weights)

        with open(f"{model_path}_soil_{i}", "wb") as f:
            pickle.dump(best_model, f)

        preds[:, i] = best_model.predict(x_test)

    mse = mean_squared_error(y_test, preds, multioutput="raw_values")
    return mse


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
        search = RandomizedSearchCV(
            model_config.model(),
            model_config.hyperparameters,
            scoring="neg_mean_squared_error",
            refit=True,
            n_jobs=-1,
        )
        model = MultiOutputRegressor(search)
        model.fit(x_train, y_train)

        log.info(f"Best params for {model_path}")
        for i, gt_name in enumerate(GT_NAMES):
            log.info(gt_name)
            log.info(model.estimators_[i].best_params_)
    else:
        model = MultiOutputRegressor(model_config.model(**model_config.default_params))
        model.fit(x_train, y_train)

    if save_model:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    preds = model.predict(x_test)

    for i in range(4):  # B, CU, ZN, FE
        log.info(f"param {i}")
        model = model_config.model(**model_config.default_params)
        gt = y_train[:, i]

        weights = compute_sample_weight("balanced", gt)

        model.fit(x_train, gt, sample_weight=weights)
        with open(f"{model_path}_soil_{i}", "wb") as f:
            pickle.dump(model, f)

        y_pred = model.predict(x_test)
        preds[:, i] = y_pred

    mse = mean_squared_error(y_test, preds, multioutput="raw_values")
    return mse


def predict_params_stratified(
    x: np.ndarray, y: np.ndarray, model_config: ModelConfig, num_bins: int = 5
) -> np.ndarray:
    x_array = np.array(x)
    y_array = np.array(y)
    models = []
    mse_scores = []

    for target_idx in range(y_array.shape[1]):
        log.info(f"Training model for target {GT_NAMES[target_idx]}")
        y_binned = pd.qcut(
            y_array[:, target_idx], q=num_bins, labels=False, duplicates="drop"
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        target_mse = []

        for train_idx, test_idx in skf.split(x_array, y_binned):
            x_train, x_test = x_array[train_idx], x_array[test_idx]
            y_train, y_test = (
                y_array[train_idx, target_idx],
                y_array[test_idx, target_idx],
            )

            model = model_config.model(**model_config.default_params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            target_mse.append(mse)

        models.append(model)
        log.info("Target MSE")
        log.info(target_mse)
        mse_scores.append(np.mean(target_mse))
    log.info("-" * 10)
    log.info(mse_scores)
    mse_scores = np.array(mse_scores)
    return mse_scores


def samples_number_experiment(
    x: np.ndarray,
    y: np.ndarray,
    sample_nums: list[int],
    model_config: ModelConfig,
    model_path: str,
    n_runs: int = 5,
    wdb: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    mses_mean = []
    mses_std = []
    if wdb:
        wandb.define_metric("soil/step")
        wandb.define_metric("soil/*", step_metric="soil/step")
    save_model = True

    if model_config.name not in ["RandomForest", "XGB", "LGBM"]:
        log.info("Scaling")
        scaler = StandardScaler().fit(x)

        x = scaler.transform(x)

    for sn in sample_nums:
        mses_for_sample = []

        for run in range(n_runs):
            x_train_base, x_test, y_train_base, y_test = train_test_split(
                x, y, test_size=0.2, random_state=run
            )
            x_train, y_train = x_train_base, y_train_base
            mse = predict_params(
                x_train, x_test, y_train, y_test, model_config, model_path, save_model
            )
            save_model = False
            mses_for_sample.append(
                mse
                / [
                    MSE_BASE_B,
                    MSE_BASE_CU,
                    MSE_BASE_ZN,
                    MSE_BASE_FE,
                    MSE_BASE_S,
                    MSE_BASE_MN,
                ]
            )

        mses_for_sample = np.array(mses_for_sample)
        mses_sample_mean = mses_for_sample.mean(axis=0)
        mses_mean.append(mses_sample_mean)
        mses_std.append(mses_for_sample.std(axis=0))
        low, high = get_confidence_interval(mses_for_sample.mean(axis=1))

        log.info(f"score = {1 / len(GT_NAMES) * np.sum(mses_sample_mean)}")
        print("int", (high - low) / 2)
        if wdb:
            wandb.log(
                {
                    # "soil/step": -sn,
                    f"soil/{GT_NAMES[0]}": mses_sample_mean[0],
                    f"soil/{GT_NAMES[1]}": mses_sample_mean[1],
                    f"soil/{GT_NAMES[2]}": mses_sample_mean[2],
                    f"soil/{GT_NAMES[3]}": mses_sample_mean[3],
                    f"soil/{GT_NAMES[4]}": mses_sample_mean[4],
                    f"soil/{GT_NAMES[5]}": mses_sample_mean[5],
                    "soil/score": 1 / len(GT_NAMES) * np.sum(mses_sample_mean),
                    "soil/std": (high - low) / 2,
                }
            )
    return np.array(mses_mean), np.array(mses_std)


def airborne_feature(x: np.ndarray, y: np.ndarray):
    model = MODELS_CONFIG["RandomForest"].model()
    model.fit(x, y)
    with open("output/models/airborne.pickle", "wb") as f:
        pickle.dump(model, f)


def predict_soil_parameters(
    dataset: Dataset,
    model: Modeller,
    num_params: int,
    cfg: ExperimentConfig,
    ae: bool,
    aug: bool = False,
) -> None:
    preds, avg_refls = prepare_datasets(
        dataset, model, cfg.k, cfg.channels, num_params, 4, cfg.device, ae
    )
    log.info("Dataset for regressors prepped")
    preds_agg = aggregate_features(preds)
    log.info(preds.shape)

    msi_means, msi_means_aug, msi_ids = load_msi_images(MSI_TRAIN_PATH, aug=aug)

    # minerals = load_minerals(TRAIN_PATH)
    # derivatives = load_derivatives(TRAIN_PATH)
    # hsi = load_hsi_airborne_images(AIRBORNE_TRAIN_PATH)
    # airborne_feature(msi_means, hsi)

    features = preds_agg #np.concatenate([preds_agg], axis=1)
    gt = prepare_gt(dataset.ids)
    gt = gt.values

    if aug:
        preds_agg_aug = preds_agg[msi_ids]
        features_aug = np.concatenate([preds_agg_aug, msi_means_aug], axis=1)
        gt_aug = gt[msi_ids]

        features = np.concatenate([features, features_aug], axis=0)
        gt = np.concatenate([gt, gt_aug], axis=0)

    log.info("Training regressors")
    mses_mean_pred, mses_std_pred = samples_number_experiment(
        features, gt, [len(gt)], MODELS_CONFIG["RandomForest"], f"{cfg.predictor_path}"
    )
