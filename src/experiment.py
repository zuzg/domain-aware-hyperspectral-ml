import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary

import wandb
from src.config import ExperimentConfig
from src.consts import (
    ARCHITECTURES_PATH,
    MAX_PATH,
    MEAN_PATH,
    OUTPUT_PATH,
    RENDERERS_DICT,
    SPLIT_RATIO,
    SUBMISSION_PATH,
    TRAIN_IDS,
    TRAIN_PATH,
    VIZ_PATH,
)
from src.data.dataset import HyperviewDataset
from src.data.preprocessing import mean_path_to_bias
from src.eval.eval_loop import Evaluator
from src.models.autoencoder import Autoencoder
from src.models.bias_variance_model import BiasModel, BiasVarianceModel, VarianceModel
from src.models.dual import DualModeAutoencoder
from src.models.modeller import Modeller
from src.models.modeller_fixed import Modeller as ModellerF
from src.soil_params.pred_ml import predict_soil_parameters


def collate_fn_pad(batch: Tensor):
    max_h = max(img.shape[1] for img in batch)  # Find max height in batch
    max_w = max(img.shape[2] for img in batch)  # Find max width in batch

    padded_batch = []
    for img in batch:
        padded_img = nn.functional.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))  # Pad to max size
        padded_batch.append(padded_img)

    return torch.stack(padded_batch)  # Stack into a single tensor


class Experiment:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.ae = self.cfg.variance_renderer == "Autoencoder"
        if self.cfg.wandb:
            self.name = self._set_experiment_name()
            self._initialize_wandb()

    def _set_experiment_name(self) -> str:
        if self.ae:
            return f"Autoencoder_latent={4*self.cfg.k:.0f}"
        if self.cfg.dual_mode:
            pre = "[DUAL]"
        else:
            pre = ""
        return pre + f"var={self.cfg.variance_renderer}_bias={self.cfg.bias_renderer}_k={self.cfg.k}"

    def _initialize_wandb(self) -> None:
        wandb.init(
            project="hyperview",
            name=self.name,
            config=vars(self.cfg),
            tags=self.cfg.tags,
        )

    def prepare_datasets(self, div: np.ndarray) -> tuple[Dataset]:
        rng = np.random.default_rng(12345)
        splits = np.split(rng.permutation(TRAIN_IDS), np.cumsum(SPLIT_RATIO))
        return (
            HyperviewDataset(
                TRAIN_PATH, splits[0], self.cfg.img_size, self.cfg.max_val, 0, div, mask=True, bias_path=MEAN_PATH
            ),
            HyperviewDataset(
                TRAIN_PATH, splits[1], self.cfg.img_size, self.cfg.max_val, 0, div, mask=True, bias_path=MEAN_PATH
            ),
            HyperviewDataset(
                TRAIN_PATH, splits[2], self.cfg.img_size, self.cfg.max_val, 0, div, mask=True, bias_path=MEAN_PATH
            ),
        )

    def prepare_dataloaders(self, trainset: Dataset, valset: Dataset, testset: Dataset) -> tuple[DataLoader]:
        return (
            DataLoader(
                trainset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collate_fn_pad, drop_last=True
            ),
            DataLoader(valset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collate_fn_pad, drop_last=True),
            DataLoader(testset, batch_size=self.cfg.batch_size, collate_fn=collate_fn_pad, drop_last=True),
        )

    def _load_max_values(self) -> np.ndarray:
        with open(MAX_PATH, "rb") as f:
            max_values = np.load(f)
        max_values[max_values > self.cfg.max_val] = self.cfg.max_val
        return max_values

    def _setup_autoencoder(self) -> nn.Module:
        self.num_params = 4
        variance_model = Autoencoder(self.cfg.channels, self.cfg.k, self.num_params).to(self.cfg.device)
        # bias_model = self._prepare_bias_model(div)
        return variance_model

    def _setup_bias_variance_model(self) -> nn.Module:
        variance_renderer = RENDERERS_DICT[self.cfg.variance_renderer]
        self.num_params = variance_renderer.num_params
        if "scale_only" in self.cfg.tags:
            modeller = ModellerF(self.cfg.channels, self.cfg.k, self.num_params).to(self.cfg.device)
        else:
            modeller = Modeller(self.cfg.channels, self.cfg.k, self.num_params).to(self.cfg.device)
        renderer = variance_renderer.model(self.cfg.device, self.cfg.channels, self.cfg.mu_type)
        # bias_model = self._prepare_bias_model(div)
        if self.cfg.dual_mode:
            return DualModeAutoencoder(
                modeller, renderer, self.cfg.batch_size, self.cfg.img_size, self.cfg.k, self.num_params
            )
        return VarianceModel(modeller, renderer)

    def _prepare_bias_model(self, div: np.ndarray) -> nn.Module:
        if self.cfg.bias_renderer == "Mean":
            return mean_path_to_bias(MEAN_PATH, div, self.cfg.device, self.cfg.img_size, self.cfg.batch_size)

        elif self.cfg.bias_renderer == "None":
            return 0

        else:
            bias_renderer = RENDERERS_DICT[self.cfg.bias_renderer]
            bias_renderer_model = bias_renderer.model(self.cfg.device, self.cfg.channels)
            bias_shape = (self.cfg.k, bias_renderer.num_params)
            bias_model = BiasModel(
                bias_shape, self.cfg.batch_size, self.cfg.img_size, bias_renderer_model, self.cfg.device
            )

            # bias_model = pretrain(bias_model, self.trainloader, self.cfg)
            for param in bias_model.parameters():
                param.requires_grad = False
        return bias_model

    def _init_dirs(self) -> None:
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        ARCHITECTURES_PATH.mkdir(parents=True, exist_ok=True)
        SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
        VIZ_PATH.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        self._init_dirs()
        max_values = self._load_max_values()
        trainset, valset, testset = self.prepare_datasets(max_values)
        if not self.cfg.save_model:
            if self.ae:
                self.num_params = 3
            else:
                self.num_params = 5
            modeller = Modeller(self.cfg.channels, self.cfg.k, self.num_params)
            modeller.load_state_dict(torch.load(self.cfg.modeller_path))
            modeller = modeller.to(self.cfg.device)
            predict_soil_parameters(testset, modeller, self.num_params, self.cfg, self.ae)

        else:
            self.trainloader, self.valloader, self.testloader = self.prepare_dataloaders(trainset, valset, testset)
            model = self._setup_autoencoder() if self.ae else self._setup_bias_variance_model()
            if self.cfg.wandb:
                modeller_summary = summary(
                    model,
                    input_size=(self.cfg.batch_size, self.cfg.channels, self.cfg.img_size, self.cfg.img_size),
                )
                with open(ARCHITECTURES_PATH / f"{self.name}.txt", "w") as f:
                    f.write(str(modeller_summary))
                artifact = wandb.Artifact(name="Model", type="architecture")
                artifact.add_file(local_path=ARCHITECTURES_PATH / f"{self.name}.txt")
                wandb.log_artifact(artifact)

            if self.cfg.dual_mode:
                from src.train.train_loop_dual import train
            else:
                from src.train.train_loop import train

            model = train(model, self.trainloader, self.valloader, self.cfg)

            modeller = model.encoder if self.ae else model.modeller
            torch.save(modeller.state_dict(), self.cfg.modeller_path)

            if self.cfg.wandb:
                div = max_values.reshape(max_values.shape[0], 1, 1)
                Evaluator(model, self.cfg, self.ae, testset.bias / div).evaluate(self.testloader)
            if self.cfg.predict_soil:
                predict_soil_parameters(testset, modeller, self.num_params, self.cfg, self.ae)
        wandb.finish()
