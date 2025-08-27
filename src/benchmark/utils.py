import numpy as np
import scipy.stats as stats


def filter_background(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = y != 0
    return x[mask], y[mask]


def get_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci = stats.t.interval(confidence, df=n - 1, loc=mean, scale=std_err)
    return ci
