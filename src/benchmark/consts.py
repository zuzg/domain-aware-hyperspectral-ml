from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    path: str
    channels: int
    folds: int


DATA_DICT: dict[str, DatasetConfig] = {
    "PU": DatasetConfig("pavia", "data/benchmark/Pavia University", 103, 5),
    "SV": DatasetConfig("salinas", "data/benchmark/Salinas", 204, 5),
    "IP": DatasetConfig("indiana", "data/benchmark/Indian Pines", 200, 4),
}
