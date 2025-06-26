import numpy as np
import resreg


def permute_spatial_pixels(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    x_flat = x.reshape(n, c, h * w)
    x_shuffled = np.empty_like(x_flat)

    for i in range(n):
        perm = np.random.permutation(h * w)
        x_shuffled[i] = x_flat[i][:, perm]

    return x_shuffled.reshape(n, c, h, w)


def mix_samples(features: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    idx1, idx2 = np.random.choice(features.shape[0], 2, replace=False)

    mixed_feature = alpha * features[idx1] + (1 - alpha) * features[idx2]
    mixed_gt = alpha * gt[idx1] + (1 - alpha) * gt[idx2]

    return mixed_feature, mixed_gt


def mix_batch(
    features: np.ndarray, gt: np.ndarray, num_samples: int = 100, alpha_range: tuple[float, float] = (0.2, 0.8)
) -> tuple[np.ndarray, np.ndarray]:
    mixed_features = []
    mixed_gts = []

    for _ in range(num_samples):
        alpha = np.random.uniform(*alpha_range)
        f, g = mix_samples(features, gt, alpha)
        mixed_features.append(f)
        mixed_gts.append(g)

    return np.array(mixed_features), np.array(mixed_gts)


def balance_dataset(x: np.ndarray, y: np.ndarray, cl: float, ch: float) -> tuple[np.ndarray, np.ndarray]:
    relevance = resreg.sigmoid_relevance(y, cl=cl, ch=ch)
    x_balanced, y_balanced = resreg.smoter(x, y, relevance=relevance)
    return x_balanced, y_balanced
