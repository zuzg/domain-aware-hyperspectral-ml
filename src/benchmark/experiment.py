import numpy as np
import wandb
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from src.benchmark.pred_ml import predict_soil_classes
from src.config import ExperimentConfig
from src.consts import RENDERERS_DICT
from src.data.dataset import HyperspectralScene
from src.data.preprocessing import mean_to_bias
from src.eval.eval_loop import Evaluator
from src.models.autoencoder import Autoencoder
from src.models.bias_variance_model import BiasModel, BiasVarianceModel, VarianceModel
from src.models.modeller import Modeller
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
        return (
            f"[B] var={self.cfg.variance_renderer}_bias={self.cfg.bias_renderer}_k={self.cfg.k}",
            [f"μ: {self.cfg.mu_type}", "MLP"],  # TODO benchmark name
        )

    def _initialize_wandb(self) -> None:
        wandb.init(
            project="hyperview",
            name=self.name,
            config=vars(self.cfg),
            tags=self.tags,
        )

    def prepare_datasets(
        self, arr_train: np.ndarray, arr_test: np.ndarray, div: np.ndarray
    ) -> tuple[Dataset]:
        trainset = HyperspectralScene(arr_train, 0, div)
        testset = HyperspectralScene(arr_test, 0, div)
        return trainset, testset

    def prepare_dataloaders(
        self, trainset: Dataset, valset: Dataset
    ) -> tuple[DataLoader]:
        return (
            DataLoader(
                trainset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False
            ),
            DataLoader(valset, batch_size=self.cfg.batch_size, drop_last=True),
        )

    def _load_max_values(self, arr: np.ndarray) -> np.ndarray:
        arr_max_c = arr.max(axis=(0, 1)).astype(np.float32)
        return arr_max_c

    def _setup_autoencoder(self, div: np.ndarray) -> nn.Module:
        self.num_params = 3
        variance_model = Autoencoder(self.cfg.channels, self.cfg.k, self.num_params).to(
            self.cfg.device
        )
        bias_model = self._prepare_bias_model(div)
        return BiasVarianceModel(bias_model, variance_model)

    def _setup_bias_variance_model(self, div: np.ndarray) -> nn.Module:
        variance_renderer = RENDERERS_DICT[self.cfg.variance_renderer]
        self.num_params = variance_renderer.num_params
        modeller = Modeller(
            self.cfg.img_size, self.cfg.channels, self.cfg.k, self.num_params
        ).to(self.cfg.device)
        variance_model = VarianceModel(
            modeller,
            variance_renderer.model(
                self.cfg.device, self.cfg.channels, self.cfg.mu_type
            ),
        )
        bias_model = self._prepare_bias_model(div)
        return BiasVarianceModel(bias_model, variance_model)

    def _prepare_bias_model(self, div: np.ndarray) -> nn.Module:
        pavia = loadmat("data/benchmark/PaviaU.mat")["paviaU"]
        pavia_train = pavia[:170, :170]
        pavia_mean_c = pavia_train.mean(axis=(0, 1)).astype(np.float32)
        if self.cfg.bias_renderer == "Mean":
            return mean_to_bias(
                pavia_mean_c,
                div,
                self.cfg.device,
                self.cfg.img_size,
                self.cfg.batch_size,
            )

        else:
            bias_renderer = RENDERERS_DICT[self.cfg.bias_renderer]
            bias_renderer_model = bias_renderer.model(
                self.cfg.device, self.cfg.channels
            )
            bias_shape = (self.cfg.k, bias_renderer.num_params)
            bias_model = BiasModel(
                bias_shape,
                self.cfg.batch_size,
                self.cfg.img_size,
                bias_renderer_model,
                self.cfg.device,
            )

            bias_model = pretrain(bias_model, self.trainloader, self.cfg)
            for param in bias_model.parameters():
                param.requires_grad = False
        return bias_model

    def run(self) -> None:
        torch.manual_seed(42)
        # read data
        pavia = loadmat("data/benchmark/PaviaU.mat")["paviaU"]
        pavia_gt = loadmat("data/benchmark/PaviaU_gt.mat")["paviaU_gt"]
        pavia_train = pavia[:170, :170]
        pavia_test = pavia[:170, 170:]
        pavia_gt_test = pavia_gt[:170, 170:]

        max_values = self._load_max_values(pavia_train)
        trainset, valset = self.prepare_datasets(pavia_train, pavia_test, max_values)
        self.trainloader, self.valloader = self.prepare_dataloaders(trainset, valset)
        model = (
            self._setup_autoencoder(max_values)
            if self.ae
            else self._setup_bias_variance_model(max_values)
        )
        model = train(model, self.trainloader, self.valloader, self.cfg)

        modeller = model.variance.encoder if self.ae else model.variance.modeller
        if self.cfg.wandb:
            Evaluator(model, self.cfg, self.ae).evaluate(self.valloader)

        for val_img in self.valloader:
            features = modeller(val_img.to(self.cfg.device))[0]
        print(features.shape)
        features_flat = features.reshape(
            features.shape[0] * features.shape[1], features.shape[2] * features.shape[3]
        )
        features_flat = features_flat.permute(1, 0)
        print(features_flat.shape)
        pavia_gt_test = np.expand_dims(pavia_gt_test, axis=0)
        print(pavia_gt_test.shape)
        gt_flat = pavia_gt_test.reshape(
            pavia_gt_test.shape[1] * pavia_gt_test.shape[2], pavia_gt_test.shape[0]
        )
        print(gt_flat.shape)

        predict_soil_classes(features_flat.cpu().detach().numpy(), gt_flat)

        wandb.finish()
