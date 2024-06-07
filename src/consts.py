from pathlib import Path


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