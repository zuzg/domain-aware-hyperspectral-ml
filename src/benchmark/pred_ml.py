import pickle

import numpy as np
import wandb
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def predict_params(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    model = XGBClassifier(n_estimators=100, learning_rate=1e-2)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    bacc = balanced_accuracy_score(y_test, preds)
    return acc, bacc


def samples_number_experiment(
    x: np.ndarray, y: np.ndarray, sample_nums: list[int], n_runs: int = 1
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    mses_mean = []
    save_model = True

    for sn in sample_nums:
        mses_for_sample = []

        for run in range(n_runs):
            x_train_base, x_test, y_train_base, y_test = train_test_split(x, y, test_size=0.2, random_state=run)
            x_train, y_train = x_train_base[:sn], y_train_base[:sn]

            acc, bacc = predict_params(x, x, y, y)
            save_model = False
            print(acc)
            print(bacc)
            mses_for_sample.append(acc)

        mses_for_sample = np.array(mses_for_sample)
        mses_sample_mean = mses_for_sample.mean(axis=0)
        mses_mean.append(mses_sample_mean)
    return np.array(mses_mean)


def predict_soil_classes(
features: np.ndarray, gt: np.ndarray
) -> None:
    samples = [len(features)]

    samples_number_experiment(features, gt, samples)
