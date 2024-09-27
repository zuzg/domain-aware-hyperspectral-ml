import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader, random_split

from src.config import ExperimentConfig
from src.consts import CHANNELS, MAX_DIM, MAX_PATH, MEAN_PATH, OUTPUT_PATH, SPLIT_RATIO, TRAIN_PATH, RENDERERS_DICT
from src.data.dataset import HyperviewDataset
from src.data.preprocessing import mean_to_bias
from src.eval.eval_loop import evaluate
from src.models.bias_variance_model import BiasModel, VarianceModel
from src.models.modeller import Modeller
from train.train_loop import pretrain, train


class Experiment:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.name = f"var={self.cfg.variance_renderer}_bias={self.cfg.bias_renderer}_k={self.cfg.k}"
        if self.cfg.wandb:
            wandb.init(
                project="hyperview",
                name=self.name,
                config=vars(self.cfg),
            )

    def run(self) -> None:
        torch.manual_seed(42)
        with open(MAX_PATH, "rb") as f:
            maxx = np.load(f)
        maxx[maxx > self.cfg.max_val] = self.cfg.max_val
        img_size = MAX_DIM
        dataset = HyperviewDataset(TRAIN_PATH, img_size, self.cfg.max_val, 0, maxx)
        train_set, val_set, test_set = random_split(dataset, SPLIT_RATIO)
        trainloader = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=True)
        valloader = DataLoader(val_set, batch_size=self.cfg.batch_size)
        testloader = DataLoader(test_set, batch_size=self.cfg.batch_size, drop_last=True)

        variance_renderer = RENDERERS_DICT[self.cfg.variance_renderer]
        variance_renderer_model = variance_renderer.model(self.cfg.device, CHANNELS)
        modeller = Modeller(img_size, CHANNELS, self.cfg.k, variance_renderer.num_params).to(self.cfg.device)
        variance_model = VarianceModel(modeller, variance_renderer_model)

        if self.cfg.bias_renderer == "Mean":
            bias_model = mean_to_bias(MEAN_PATH, maxx, self.cfg.device, img_size, self.cfg.batch_size)
        else:
            bias_renderer = RENDERERS_DICT[self.cfg.bias_renderer]
            bias_renderer_model = bias_renderer.model(self.cfg.device, CHANNELS)
            bias_shape = (self.cfg.k, bias_renderer.num_params)
            bias_model = BiasModel(
                bias_shape, self.cfg.batch_size, img_size, bias_renderer_model, self.cfg.device
            )
            bias_model = pretrain(bias_model, trainloader, self.cfg)

        model = train(variance_model, bias_model, trainloader, valloader, self.cfg)
        if self.cfg.save_model:
            torch.save(model.variance.modeller.state_dict(), OUTPUT_PATH / f"modeller_{self.name}.pth")

        if self.cfg.wandb:
            evaluate(model, testloader, self.cfg)
            wandb.finish()
