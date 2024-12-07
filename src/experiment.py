import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import ExperimentConfig
from src.consts import DATA_PATH, MAX_PATH, MEAN_PATH, OUTPUT_PATH, SPLIT_RATIO, TRAIN_IDS, TRAIN_PATH, RENDERERS_DICT
from src.data.dataset import HyperviewDataset
from src.data.preprocessing import mean_path_to_bias
from src.eval.eval_loop import Evaluator
from src.models.autoencoder import Autoencoder
from src.models.bias_variance_model import BiasModel, BiasVarianceModel, VarianceModel
from src.models.modeller import Modeller
from soil_params.pred_ml import predict_soil_parameters
from src.train.train_loop import pretrain, train


class Experiment:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.ae = self.cfg.variance_renderer == "Autoencoder"
        if self.cfg.wandb:
            self.name, self.tags = self._set_experiment_name_and_tags()
            self._initialize_wandb()

    def _set_experiment_name_and_tags(self) -> tuple[str, list[str]]:
        if self.ae:
            return f"Autoencoder_latent={3*self.cfg.k:.0f}", []
        return f"var={self.cfg.variance_renderer}_bias={self.cfg.bias_renderer}_k={self.cfg.k}", [
            f"μ: {self.cfg.mu_type}", "MLP"
        ]

    def _initialize_wandb(self) -> None:
        wandb.init(
            project="hyperview",
            name=self.name,
            config=vars(self.cfg),
            tags=self.tags,
        )

    def prepare_datasets(self, div: np.ndarray) -> tuple[Dataset]:
        splits = np.split(np.random.permutation(TRAIN_IDS), np.cumsum(SPLIT_RATIO))
        return (
            HyperviewDataset(TRAIN_PATH, splits[0], self.cfg.img_size, self.cfg.max_val, 0, div, mask=True),
            HyperviewDataset(TRAIN_PATH, splits[1], self.cfg.img_size, self.cfg.max_val, 0, div, mask=True),
            HyperviewDataset(TRAIN_PATH, splits[2], self.cfg.img_size, self.cfg.max_val, 0, div, mask=True),
        )

    def prepare_dataloaders(
        self, trainset: Dataset, valset: Dataset, testset: Dataset) -> tuple[DataLoader]:
        return (
            DataLoader(trainset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True),
            DataLoader(valset, batch_size=self.cfg.batch_size, drop_last=True),
            DataLoader(testset, batch_size=self.cfg.batch_size, drop_last=True),
        )

    def _load_max_values(self) -> np.ndarray:
        with open(MAX_PATH, "rb") as f:
            max_values = np.load(f)
        max_values[max_values > self.cfg.max_val] = self.cfg.max_val
        return max_values

    def _setup_autoencoder(self, div: np.ndarray) -> nn.Module:
        self.num_params = 3
        variance_model =  Autoencoder(self.cfg.channels, self.cfg.k, self.num_params).to(self.cfg.device)
        bias_model = self._prepare_bias_model(div)
        return BiasVarianceModel(bias_model, variance_model)

    def _setup_bias_variance_model(self, div: np.ndarray) -> nn.Module:
        variance_renderer = RENDERERS_DICT[self.cfg.variance_renderer]
        self.num_params = variance_renderer.num_params
        modeller = Modeller(self.cfg.img_size, self.cfg.channels, self.cfg.k, self.num_params).to(self.cfg.device)
        variance_model = VarianceModel(modeller, variance_renderer.model(self.cfg.device, self.cfg.channels, self.cfg.mu_type))
        bias_model = self._prepare_bias_model(div)
        return BiasVarianceModel(bias_model, variance_model)

    def _prepare_bias_model(self, div: np.ndarray) -> nn.Module:
        if self.cfg.bias_renderer == "Mean":
            return mean_path_to_bias(MEAN_PATH, div, self.cfg.device, self.cfg.img_size, self.cfg.batch_size)

        else:
            bias_renderer = RENDERERS_DICT[self.cfg.bias_renderer]
            bias_renderer_model = bias_renderer.model(self.cfg.device, self.cfg.channels)
            bias_shape = (self.cfg.k, bias_renderer.num_params)
            bias_model = BiasModel(bias_shape, self.cfg.batch_size, self.cfg.img_size, bias_renderer_model, self.cfg.device)

            bias_model = pretrain(bias_model, self.trainloader, self.cfg)
            for param in bias_model.parameters():
                param.requires_grad = False
        return bias_model

    def run(self) -> None:
        torch.manual_seed(42)
        max_values = self._load_max_values()
        trainset, valset, testset = self.prepare_datasets(max_values)
        if not self.cfg.save_model:
            modeller = Modeller(self.cfg.img_size, self.cfg.channels, self.cfg.k, 4)
            modeller.load_state_dict(torch.load(OUTPUT_PATH / "models" / f"modeller_{self.name}_{self.cfg.epochs}_k=5_full_soill.pth"))
            modeller = modeller.to(self.cfg.device)
            predict_soil_parameters(testset, modeller, 4, self.cfg, self.ae)

        else:
            self.trainloader, self.valloader, self.testloader = self.prepare_dataloaders(trainset, valset, testset)
            model = self._setup_autoencoder(max_values) if self.ae else self._setup_bias_variance_model(max_values)
            model = train(model, self.trainloader, self.valloader, self.cfg)

            modeller = model.variance.encoder if self.ae else model.variance.modeller
            torch.save(modeller.state_dict(), OUTPUT_PATH  / "models"/ f"modeller_{self.name}_{self.cfg.epochs}_k=5_full_soill.pth")

            if self.cfg.wandb:
                Evaluator(model, self.cfg, self.ae).evaluate(self.testloader)
            if self.cfg.predict_soil:
                predict_soil_parameters(
                    testset, modeller, self.num_params, self.cfg, self.ae
                )
        wandb.finish()
