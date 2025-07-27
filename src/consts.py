from pathlib import Path

import numpy as np

from src.config import RendererConfig
from src.models.autoencoder import Autoencoder
from src.models.renderers.beta_renderer import BetaRenderer
from src.models.renderers.gaussian_asymmetric_renderer import GaussianAsymmetricRenderer
from src.models.renderers.gaussian_renderer import GaussianRenderer
from src.models.renderers.gaussian_skew_renderer import GaussianSkewRenderer
from src.models.renderers.polynomial_degree_renderer import PolynomialDegreeRenderer
from src.models.renderers.polynomial_renderer import PolynomialRenderer
from src.models.renderers.spline_renderer import SplineRenderer

# models
RENDERERS_DICT: dict[str, RendererConfig] = {
    "BetaRenderer": RendererConfig(BetaRenderer, 3),
    "GaussianRenderer": RendererConfig(GaussianRenderer, 4),
    "GaussianAsymmetricRenderer": RendererConfig(GaussianAsymmetricRenderer, 5),
    "GaussianSkewRenderer": RendererConfig(GaussianSkewRenderer, 5),
    "PolynomialDegreeRenderer": RendererConfig(PolynomialDegreeRenderer, 1),
    "PolynomialRenderer": RendererConfig(PolynomialRenderer, 2),
    "SplineRenderer": RendererConfig(SplineRenderer, 1),
    "Autoencoder": RendererConfig(Autoencoder, 3),
    "None": RendererConfig(None, 0),
}


# ------------HYPERVIEW 2--------------
# soil
GT_DIM: int = 6
GT_NAMES: list[str] = ["B", "Cu", "Zn", "Fe", "S", "Mn"]
GT_MAX: list[float] = [1.9, 4.3, 14.4, 481.7, 96.8092647002, 167.6]
MSE_BASE_B: int = 2500
MSE_BASE_CU: int = 1100
MSE_BASE_ZN: int = 2000
MSE_BASE_FE: int = 1000
MSE_BASE_S: int = 500
MSE_BASE_MN: int = 300

# paths
DATA_PATH: Path = Path("data/HYPERVIEW2")
OUTPUT_PATH: Path = Path("output")
ARCHITECTURES_PATH: Path = OUTPUT_PATH / "architectures"
MODELS_PATH: Path = OUTPUT_PATH / "models"
SUBMISSION_PATH: Path = OUTPUT_PATH / "submissions"
TRAIN_PATH: Path = DATA_PATH / "train/hsi_satellite"
TEST_PATH: Path = DATA_PATH / "test/hsi_satellite"
GT_PATH: Path = DATA_PATH / "train_gt.csv"
STATS_PATH: Path = DATA_PATH / "stats"
MAX_PATH: Path = STATS_PATH / "max_c.npy"
MEAN_PATH: Path = STATS_PATH / "mean_c.npy"
STD_PATH: Path = STATS_PATH / "std_c.npy"

MSI_TRAIN_PATH: Path = DATA_PATH / "train/msi_satellite"
MSI_TEST_PATH: Path = DATA_PATH / "test/msi_satellite"
AIRBORNE_TRAIN_PATH: Path = DATA_PATH / "train/hsi_airborne"

# data
TRAIN_IDS: list[int] = list(range(1876))
TEST_IDS: list[int] = list(range(1888))
SPLIT_RATIO: list[int] = [1000, 124, 752]

# PRISMA
water_bands: np.ndarray = np.array(
    [[97, 108], [141, 160], [224, 229]]
)  # open intervals
