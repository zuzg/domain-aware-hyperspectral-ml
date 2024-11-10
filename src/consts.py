from pathlib import Path

from src.config import RendererConfig
from src.models.autoencoder import Autoencoder
from src.models.renderers.gaussian_renderer import GaussianRenderer
from src.models.renderers.polynomial_degree_renderer import PolynomialDegreeRenderer
from src.models.renderers.polynomial_renderer import PolynomialRenderer
from src.models.renderers.spline_renderer import SplineRenderer


# paths
DATA_PATH: Path = Path("data/hyperview")
OUTPUT_PATH: Path = Path("output")
TRAIN_PATH: Path = DATA_PATH / "train_data/train_data"
GT_PATH: Path = DATA_PATH / "train_data/train_gt.csv"
STATS_PATH: Path = DATA_PATH / "stats"
MAX_PATH: Path = STATS_PATH / "max_c_masked.npy"
MEAN_PATH: Path = STATS_PATH / "mean_c_masked.npy"
STD_PATH: Path = STATS_PATH / "std_c.npy"

# data
TRAIN_IDS: list[int] = list(range(1732))
TEST_IDS: list[int] = list(range(1154))
SPLIT_RATIO: list[int] = [1000, 124, 608]
CHANNELS: int = 150

# models
RENDERERS_DICT: dict[str, RendererConfig] = {
    "GaussianRenderer": RendererConfig(GaussianRenderer, 4),
    "PolynomialDegreeRenderer": RendererConfig(PolynomialDegreeRenderer, 1),
    "PolynomialRenderer": RendererConfig(PolynomialRenderer, 2),
    "SplineRenderer": RendererConfig(SplineRenderer, 1),
    "Autoencoder": RendererConfig(Autoencoder, 3),
    "None": RendererConfig(None, 0),
}

# soil
GT_DIM: int = 4
GT_NAMES: list[str] = ["P", "K", "Mg", "pH"]
GT_MAX: list[float] = [325.0, 625.0, 400.0, 7.8]
MSE_BASE_K: int = 2500
MSE_BASE_P: int = 1100
MSE_BASE_MG: int = 2000
MSE_BASE_PH: int = 3
