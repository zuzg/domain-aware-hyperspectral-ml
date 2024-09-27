from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def mean_to_bias(mean_path: Path, divisor: np.ndarray, device: str, img_size: int, batch_size: int) -> Tensor:
    with open(mean_path, "rb") as f:
        bias_mean = np.load(f)
        bias_mean = bias_mean / divisor
        bias_mean = torch.from_numpy(bias_mean).to(device)

    bias_model = bias_mean.repeat(batch_size, 1)
    bias_model = bias_model.unsqueeze(-1).unsqueeze(-1)
    bias_model = bias_model.repeat(1, 1, img_size, img_size)
    return bias_model
