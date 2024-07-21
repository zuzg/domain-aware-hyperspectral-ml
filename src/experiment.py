import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader, random_split

from src.config import ExperimentConfig
from src.consts import CHANNELS, MAX_PATH, SPLIT_RATIO, TRAIN_PATH, RENDERERS_DICT
from src.data.dataset import HyperviewDataset
from src.eval.eval_loop import evaluate
from src.models.bias_variance_model import BiasModel, VarianceModel
from src.models.modeller import Modeller
from train.train_loop import pretrain, train


class Experiment:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        if self.cfg.wandb:
            wandb.init(
                project="hyperview",
                name=f"var={self.cfg.variance_renderer}_bias={self.cfg.bias_renderer}_k={self.cfg.k}",
                config=vars(self.cfg),
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
        testloader = DataLoader(test_set, batch_size=self.cfg.batch_size, drop_last=True)

        variance_renderer = RENDERERS_DICT[self.cfg.variance_renderer]
        variance_renderer_model = variance_renderer.model(self.cfg.device, CHANNELS)
        modeller = Modeller(self.cfg.img_size, CHANNELS, self.cfg.k, variance_renderer.num_params).to(self.cfg.device)
        variance_model = VarianceModel(modeller, variance_renderer_model)

        bias_renderer = RENDERERS_DICT[self.cfg.bias_renderer]
        bias_renderer_model = bias_renderer.model(self.cfg.device, CHANNELS)
        bias_shape = (self.cfg.batch_size, self.cfg.k, bias_renderer.num_params, self.cfg.img_size, self.cfg.img_size)
        bias_model = BiasModel(bias_shape, bias_renderer_model)
        bias_model = pretrain(bias_model, trainloader, self.cfg)

        model = train(variance_model, bias_model, trainloader, valloader, self.cfg)

        if self.cfg.wandb:
            evaluate(model, testloader, self.cfg)
            wandb.finish()
