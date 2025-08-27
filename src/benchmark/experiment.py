from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna
import torch
import wandb
from torch.utils.data import DataLoader, Dataset

from src.benchmark.consts import DATA_DICT, DatasetConfig
from src.benchmark.pred_ml import predict_soil_classes_background
from src.benchmark.utils import get_confidence_interval
from src.config import ExperimentConfig
from src.data.dataset import HyperpectralPatch
from src.eval.eval_loop import Evaluator
from src.experiment import Experiment
from src.models.modeller import Modeller
from src.train.train_loop import train, train_hpo


class BenchmarkExperiment(Experiment):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__(cfg)

    def run(self) -> None:
        oa_fold, aa_fold = [], []
        oa_list, aa_list = [], []

        data_cfg = DATA_DICT[self.cfg.dataset_name]
        self.cfg.channels = data_cfg.channels

        for fold in range(data_cfg.folds):
            oacc, aacc, oacc_list, aacc_list = self._run_single_fold(data_cfg, fold)
            oa_fold.append(oacc)
            aa_fold.append(aacc)
            oa_list.extend(oacc_list)
            aa_list.extend(aacc_list)

        oa_low, oa_high = get_confidence_interval(oa_list)
        aa_low, aa_high = get_confidence_interval(aa_list)

        wandb.log(
            {
                "OA": np.mean(oa_fold),
                "AA": np.mean(aa_fold),
                "OA_int_low": oa_low,
                "OA_int_high": oa_high,
                "AA_int_low": aa_low,
                "AA_int_high": aa_high,
            }
        )
        wandb.finish()

    def _run_single_fold(self, data_cfg: DatasetConfig, fold: int) -> tuple[float, float]:
        fold_dir = Path(data_cfg.path) / f"{data_cfg.name}_fold_{fold}"

        trainset, predset, testset = self._prepare_datasets(fold_dir)
        trainloader, predloader, valloader, testloader = self._prepare_dataloaders(trainset, predset, testset)

        # study = optuna.create_study(direction="minimize")
        # study.optimize(self.objective, n_trials=30)  # number of trials

        # print("Best trial:")
        # trial = study.best_trial
        # print(f"  Loss: {trial.value}")
        # print("  Params: ")
        # for k, v in trial.params.items():
        #     print(f"    {k}: {v}")

        oacc_sum = 0
        aacc_sum = 0
        run_num = 5

        oacc_list = []
        aacc_list = []

        for run_idx in range(run_num):
            model = self._setup_autoencoder() if self.ae else self._setup_bias_variance_model()
            model = train(model, trainloader, trainloader, self.cfg)

            if fold == 0 and run_idx == 0:
                print("Evaluating")
                bias = trainset.channel_mean / trainset.channel_max
                bias = bias.unsqueeze(1).unsqueeze(1).numpy()
                # Evaluator(model, self.cfg, self.ae, bias).evaluate(valloader)

            modeller = model.encoder if self.ae else model.modeller

            features_train = self._extract_features(modeller, predloader)
            features_test = self._extract_features(modeller, testloader)

            features_train = features_train.transpose(0, 3, 4, 1, 2).reshape(-1, self.cfg.k * 4)  # num params
            features_test = features_test.reshape(-1, self.cfg.k * 4)

            gt_train = np.array(predset.gt).reshape(-1)
            gt_test = np.array(testset.gt)

            oacc, aacc = predict_soil_classes_background(features_train, features_test, gt_train, gt_test)
            oacc_sum += oacc
            aacc_sum += aacc
            oacc_list.append(oacc)
            aacc_list.append(aacc)

        avg_oacc = oacc_sum / run_num
        avg_aacc = aacc_sum / run_num
        return avg_oacc, avg_aacc, oacc_list, aacc_list

    def _prepare_datasets(self, fold_dir: Path) -> tuple[Dataset]:
        trainset = HyperpectralPatch(fold_dir, extra=False)
        predset = HyperpectralPatch(fold_dir)
        testset = HyperpectralPatch(
            fold_dir,
            mean=trainset.channel_mean,
            std=trainset.channel_max,
            test=True,
            ch=trainset.ch,
            h=trainset.h,
            w=trainset.w,
        )
        return trainset, predset, testset

    def _prepare_dataloaders(self, trainset: Dataset, predset: Dataset, testset: Dataset) -> tuple[DataLoader]:
        batch_size = self.cfg.batch_size
        trainloader = DataLoader(trainset, batch_size, shuffle=True)
        predloader = DataLoader(predset, batch_size, shuffle=False)
        valloader = DataLoader(trainset, batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(testset, batch_size=1, shuffle=False)
        return trainloader, predloader, valloader, testloader

    def _extract_features(self, modeller: Modeller, dataloader: DataLoader) -> np.ndarray:
        device = self.cfg.device
        features = []
        with torch.no_grad():
            for img in dataloader:
                img = img.to(device)
                ft = modeller(img)
                features.append(ft.cpu().numpy())
        return np.concatenate(features, axis=0)

    def objective(self, trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
        model = self._setup_autoencoder()
        val_loss = train_hpo(model, self.trainloader, self.trainloader, self.cfg, optimizer_name, lr)
        return val_loss
