import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader, random_split

from src.config import ExperimentConfig
from src.consts import CHANNELS, MAX_PATH, SPLIT_RATIO, TRAIN_PATH, RENDERERS_DICT
from src.data.dataset import HyperviewDataset
from src.eval.eval_loop import evaluate
from src.models.modeller import Modeller
from src.train.train_loop import train


class Experiment:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        if self.cfg.wandb:
            wandb.init(
                project="hyperview",
                name=f"{self.cfg.renderer}_k={self.cfg.k}",
                config=vars(self.cfg)
            )

    def run(self) -> None:
        torch.manual_seed(42)
        with open(MAX_PATH, "rb") as f:
            maxx = np.load(f)
        maxx[maxx > self.cfg.max_val] = self.cfg.max_val
        dataset = HyperviewDataset(TRAIN_PATH, self.cfg.img_size, self.cfg.max_val, 0, maxx)
        train_set, val_set, test_set = random_split(dataset, SPLIT_RATIO)
        trainloader = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=True)
        valloader = DataLoader(val_set, batch_size=self.cfg.batch_size)
        testloader = DataLoader(test_set, batch_size=self.cfg.batch_size)

        renderer = RENDERERS_DICT[self.cfg.renderer]
        modeller = Modeller(self.cfg.img_size, CHANNELS, self.cfg.k, renderer.num_params).to(self.cfg.device)
        model = train(modeller, renderer.model(self.cfg.device), trainloader, valloader, self.cfg)

        if self.cfg.wandb:
            evaluate(model, renderer.model(self.cfg.device), testloader, self.cfg)
            wandb.finish()
