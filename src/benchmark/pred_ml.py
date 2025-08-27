import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.benchmark.utils import filter_background


def predict_soil_classes_background(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, fb: bool = True
) -> tuple[float]:
    if fb:
        x_train, y_train = filter_background(x_train, y_train)
    seen_classes = np.unique(y_train)
    test_mask = np.isin(y_test, seen_classes)
    y_test_filtered = y_test[test_mask]

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test_filtered)

    y_test[test_mask] = y_test_enc
    y_test[~test_mask] = -1

    model = XGBClassifier(learning_rate=1e-1, n_estimators=200)
    model.fit(x_train, y_train_enc)

    preds = model.predict(x_test)
    overall_acc = accuracy_score(y_test, preds)
    average_acc = recall_score(y_test, preds, average="macro")
    return overall_acc, average_acc


def predict_soil_classes(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> tuple[float]:

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    model = XGBClassifier(learning_rate=1e-2, n_estimators=200)
    model.fit(x_train, y_train_enc)

    preds = model.predict(x_test)
    overall_acc = accuracy_score(y_test_enc, preds)
    average_acc = recall_score(y_test_enc, preds, average="macro")
    return overall_acc, average_acc, preds
