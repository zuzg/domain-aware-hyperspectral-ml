from pathlib import Path

from src.config import RendererConfig
from src.models.renderers.gaussian_renderer import GaussianRenderer
from src.models.renderers.polynomial_degree_renderer import PolynomialDegreeRenderer
from src.models.renderers.polynomial_renderer import PolynomialRenderer
from src.models.renderers.spline_renderer import SplineRenderer


# paths
DATA_PATH: Path = Path("data/hyperview")
TRAIN_PATH: Path = DATA_PATH / "train_data/train_data"
STATS_PATH: Path = DATA_PATH / "stats"
MAX_PATH: Path = STATS_PATH / "max_c.npy"
MEAN_PATH: Path = STATS_PATH / "mean_c.npy"
STD_PATH: Path = STATS_PATH / "std_c.npy"

# data
SPLIT_RATIO: list[int] = [1400, 200, 132]
CHANNELS: int = 150

# models
RENDERERS_DICT: dict[str, RendererConfig] = {
    "GaussianRenderer": RendererConfig(GaussianRenderer, 3),
    "PolynomialDegreeRenderer": RendererConfig(PolynomialDegreeRenderer, 1),
    "PolynomialRenderer": RendererConfig(PolynomialRenderer, 2),
    "SplineRenderer": RendererConfig(SplineRenderer, 1),
    "None": RendererConfig(None, 0),
}
