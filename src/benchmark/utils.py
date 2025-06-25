import numpy as np


def filter_background(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = y != 0
    return x[mask], y[mask]
