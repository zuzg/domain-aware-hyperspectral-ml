from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    path: str
    channels: int
    folds: int


@dataclass
class SceneConfig:
    img_path: str
    gt_path: str
    img_key: str
    gt_key: str
    channels: int


DATA_DICT: dict[str, DatasetConfig] = {
    "PU": DatasetConfig("pavia", "data/benchmark/Pavia University", 103, 5),
    "SV": DatasetConfig("salinas", "data/benchmark/Salinas", 204, 5),
    "IP": DatasetConfig("indiana", "data/benchmark/Indian Pines", 200, 4),
}


SCENE_DICT: dict[str, SceneConfig] = {
    "PU": SceneConfig(
        "data/benchmark_full/PaviaU.mat", "data/benchmark_full/PaviaU_gt.mat", "paviaU", "paviaU_gt", 103
    ),
    "SV": SceneConfig(
        "data/benchmark_full/Salinas_corrected.mat",
        "data/benchmark_full/Salinas_gt.mat",
        "salinas_corrected",
        "salinas_gt",
        204,
    ),
    "IP": SceneConfig(
        "data/benchmark_full/Indian_pines_corrected.mat",
        "data/benchmark_full/Indian_pines_gt.mat",
        "indian_pines_corrected",
        "indian_pines_gt",
        200,
    ),
}

PU_CLASSES = [
    "Asphalt",
    "Meadows",
    "Gravel",
    "Trees",
    "Metal sheets",
    "Bare soil",
    "Bitumen",
    "Bricks",
    "Shadows",
]
