import pickle

import numpy as np
import pandas as pd

# import sklearn
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    HalvingRandomSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# from sklearn.utils import compute_sample_weight
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

# sklearn.set_config(enable_metadata_routing=True)


def get_weights(target: np.ndarray, q: float) -> np.ndarray:
    vals, counts = np.unique(target, return_counts=True)
    val_to_weight = dict(zip(vals, q / counts))
    sample_weight = np.vectorize(val_to_weight.get)(target)
    return sample_weight


def compute_sample_weights(y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    # Compute quantile-based bins (ensures ~equal-sized bins)
    bins = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-6  # ensure lowest value is included
    y_binned = np.digitize(y, bins[1:], right=True)
    bin_counts = np.bincount(y_binned, minlength=n_bins)
    weights = 1. / (bin_counts[y_binned] + 1e-6)
    weights *= len(weights) / np.sum(weights)
    return weights


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
        for i, gt_name in enumerate(GT_NAMES):
            print(gt_name)
            print(model.estimators_[i].best_params_)
    else:
        # model = RegressorChain(model_config.model(**model_config.default_params), order=[5, 4, 3, 0, 1, 2])
        model = MultiOutputRegressor(model_config.model(**model_config.default_params))
        # model = model_config.model(**model_config.default_params)
        model.fit(x_train, y_train)
    
    if save_model:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    preds = model.predict(x_test)
    # preds = np.zeros_like(y_test)

    for i in range(4):  # B, CU, ZN, FE
        print(f"param {i}")
        rf = model_config.model(**model_config.default_params)
        # model = rf
        # model = TransformedTargetRegressor(rf, func=np.log1p, inverse_func=np.expm1)
        model = RFECV(rf, n_jobs=-1, min_features_to_select=60, cv=2)

        gt = y_train[:, i]

        # df = pd.DataFrame(x_train)
        # df["y"] = gt

        # meth = "balance"# if i == 3 else "balanced"
        # oversampled = iblr.ro(data=df, y="y", samp_method=meth)

        # x_sampled = oversampled.iloc[:,:-1]
        # y_sampled = oversampled["y"]
  
        # weights = compute_sample_weight("balanced", gt)
        model.fit(x_train, gt)#, sample_weight=weights)

        print("cv results")
        print(model.cv_results_)

        with open(f"{model_path}_soil_{i}", "wb") as f:
            pickle.dump(model, f)
        y_pred = model.predict(x_test)
        preds[:, i] = y_pred
    
    
    # rounded_preds = preds.copy()

    # for col in range(rounded_preds.shape[1]):
    #     if col != 4:
    #         rounded_preds[:, col] = np.round(rounded_preds[:, col], 1)

    # y_train_cu = y_train[:, 1].astype(str)
    # model_cu = RandomForestClassifier(class_weight="balanced")
    # model_cu.fit(x_train, y_train_cu)

    # if save_model:
    #     with open(f"{model_path}_cu", "wb") as f:
    #         pickle.dump(model_cu, f)
    
    # preds_cu = model_cu.predict(x_test)
    # preds[:, 1] = preds_cu.astype(float)

    # last_col = x_test[:, -1]
    # unique_vals = np.unique(last_col)
    # grouped_mse = {}

    # for val in unique_vals:
    #     indices = np.where(last_col == val)[0]
    #     mse_group = mean_squared_error(y_test[indices], preds[indices], multioutput='raw_values')
    #     grouped_mse[val] = mse_group

    # for val, mse in grouped_mse.items():
    #     print(f"Size: {val:.0f} -> MSE: {np.mean(mse / [MSE_BASE_B, MSE_BASE_CU, MSE_BASE_ZN, MSE_BASE_FE, MSE_BASE_S, MSE_BASE_MN])}")
    
    mse = mean_squared_error(y_test, preds, multioutput="raw_values")
    return mse


def predict_params_stratified(x: np.ndarray, y: np.ndarray, model_config: ModelConfig, num_bins: int = 5) -> np.ndarray:
    x_array = np.array(x)
    y_array = np.array(y)
    models = []
    mse_scores = []

    for target_idx in range(y_array.shape[1]):
        print(f"Training model for target {GT_NAMES[target_idx]}")
        y_binned = pd.qcut(y_array[:, target_idx], q=num_bins, labels=False, duplicates="drop")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
    print("-" * 10)
    print(mse_scores)
    mse_scores = np.array(mse_scores)
    return mse_scores


def samples_number_experiment(
    x: np.ndarray,
    y: np.ndarray,
    sample_nums: list[int],
    model_config: ModelConfig,
    model_path: str,
    n_runs: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    mses_mean = []
    mses_std = []
    wandb.define_metric("soil/step")
    wandb.define_metric("soil/*", step_metric="soil/step")
    save_model = True

    if model_config.name not in ["RandomForest", "XGB", "LGBM"]:
        print("Scaling")
        scaler = StandardScaler().fit(x)
        # with open("output/models/scaler.pickel", "wb") as f:
        #     pickle.dump(scaler, f)
        x = scaler.transform(x)

    for sn in sample_nums:
        mses_for_sample = []

        for run in range(n_runs):
            x_train_base, x_test, y_train_base, y_test = train_test_split(x, y, test_size=0.2, random_state=run)
            x_train, y_train = x_train_base, y_train_base

            # for i in range(6):
            #     print(GT_NAMES[i])
            #     y_gt = y[:, i]
            #     estimator = model_config.model(**model_config.default_params)
            #     selector = RFE(estimator, n_features_to_select=15, step=1)
            #     selector = selector.fit(x, y_gt)
            #     print(selector.support_)
            #     print(selector.ranking_)

            # x_aug, y_aug = mix_batch(x_train, y_train, num_samples=1000)
            # x_train = np.concatenate([x_train, x_aug], axis=0)
            # y_train = np.concatenate([y_train, y_aug], axis=0)

            mse = predict_params(x_train, x_test, y_train, y_test, model_config, model_path, save_model)
            # mse = predict_params_stratified(x, y, model_config)
            # print(mse)
            # model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200))
            # model.fit(x, y)
            # if save_model:
            #     with open(model_path, "wb") as f:
            #         pickle.dump(model, f)
            save_model = False
            mses_for_sample.append(mse / [MSE_BASE_B, MSE_BASE_CU, MSE_BASE_ZN, MSE_BASE_FE, MSE_BASE_S, MSE_BASE_MN])

        mses_for_sample = np.array(mses_for_sample)
        mses_sample_mean = mses_for_sample.mean(axis=0)
        mses_mean.append(mses_sample_mean)
        mses_std.append(mses_for_sample.std(axis=0))

        print(f"score = {1 / len(GT_NAMES) * np.sum(mses_sample_mean)}")

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
            }
        )
    return np.array(mses_mean), np.array(mses_std)


def airborne_feature(x: np.ndarray, y: np.ndarray):
    model = MODELS_CONFIG["RandomForest"].model()
    model.fit(x, y)
    with open("output/models/airborne.pickle", "wb") as f:
        pickle.dump(model, f)


def predict_soil_parameters(
    dataset: Dataset, model: Modeller, num_params: int, cfg: ExperimentConfig, ae: bool, split: bool = False
) -> None:
    preds, avg_refls = prepare_datasets(dataset, model, cfg.k, cfg.channels, num_params, 4, cfg.device, ae)
    # TODO augment 2x2 from 3x3 and 3x2 before aggs
    print("Dataset prepped")
    preds_agg = aggregate_features(preds)
    # features = preds_agg
    print(preds.shape)
    # preds_flat = preds.reshape(preds.shape[0], -1)
    # features = preds_flat

    aug = False

    msi_means, msi_means_aug, msi_ids = load_msi_images(MSI_TRAIN_PATH, aug=aug)
    mask_info = np.array(dataset.is_fully_masked)
    mask_info = np.expand_dims(mask_info, axis=1)
    # features = np.concatenate([preds_flat, msi_means], axis=1)

    # minerals = load_minerals(TRAIN_PATH)
    
    # sizes = np.array(dataset.size_list)
    # sizes = np.expand_dims(sizes, axis=1)
    # hsi = load_hsi_airborne_images(AIRBORNE_TRAIN_PATH)
    features = np.concatenate([preds_agg, msi_means], axis=1)
    gt = prepare_gt(dataset.ids)
    max_samples = 1876
    gt = gt[:max_samples].values
    samples = [max_samples]

    # airborne_feature(msi_means, hsi)

    if aug:
        preds_agg_aug = preds_agg[msi_ids]
        features_aug = np.concatenate([preds_agg_aug, msi_means_aug], axis=1)
        gt_aug = gt[msi_ids]

        features = np.concatenate([features, features_aug], axis=0)
        gt = np.concatenate([gt, gt_aug], axis=0)


    if split:
        sizes = dataset.size_list
        possible_sizes = [1, 2, 4, 6, 9]

        for s in possible_sizes:
            # TODO train on all and check per size?
            # train on and bigger
            # if s == 4 or s == 6:
            #     ids = [i for i, size in enumerate(sizes) if size >= s]
            #     # features_size = np.array([features[i][features[i] != 0][:s*17] for i in ids])
            #     features_size = np.array([
            #         features[i][features[i] != 0][
            #             max(0, (len(features[i][features[i] != 0]) - s*17) // 2) :
            #             max(0, (len(features[i][features[i] != 0]) - s*17) // 2) + s*17
            #         ]
            #         for i in ids
            #     ])
            #     print(features_size.shape)
            # else:
            #     ids = [i for i, size in enumerate(sizes) if size == s]
            #     features_size = np.array([features[i][features[i] != 0] for i in ids])
            
            ids = [i for i, size in enumerate(sizes) if size == s]
            features_size = np.array([features[i][features[i] != 0] for i in ids])
            gt_size = [gt[i] for i in ids]

            print(f"size: {s}, number of samples: {features_size.shape[0]}")
            mses_mean_pred, mses_std_pred = samples_number_experiment(
            features_size, gt_size, samples, MODELS_CONFIG["RandomForest"], f"{cfg.predictor_path}"
        )



    # mask = np.array(dataset.fully_masked_ids)
    # features = features[~mask]
    # gt = gt[~mask]

    # features_permuted = permute_spatial_pixels(preds).reshape(preds.shape[0], -1)
    # features = np.concatenate([preds_flat, features_permuted], axis=0)
    # gt = np.concatenate([gt, gt], axis=0)
    else:
        print("Training")
        mses_mean_pred, mses_std_pred = samples_number_experiment(
            features, gt, samples, MODELS_CONFIG["RandomForest"], f"{cfg.predictor_path}"
        )
