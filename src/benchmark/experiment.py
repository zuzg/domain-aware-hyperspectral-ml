from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from src.benchmark.pred_ml import predict_soil_classes
from src.config import ExperimentConfig
from src.data.dataset import HyperpectralPatch
from src.eval.eval_loop import Evaluator
from src.experiment import Experiment
from src.train.train_loop import train


class BenchmarkExperiment(Experiment):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__(cfg)

    def run(self) -> None:
        oa_fold, aa_fold = [], []

        for fold in range(4):
            oacc, aacc = self._run_single_fold(fold)
            oa_fold.append(oacc)
            aa_fold.append(aacc)

        wandb.log(
            {
                "soil/oacc": np.mean(oa_fold),
                "soil/aacc": np.mean(aa_fold),
            }
        )
        wandb.finish()

    def _run_single_fold(self, fold: int) -> tuple[float, float]:
        fold_dir = Path(self.cfg.fold_dir) / f"indiana_fold_{fold}"

        trainset, predset, testset = self._prepare_datasets(fold_dir)
        trainloader, predloader, valloader, testloader = self._prepare_dataloaders(trainset, predset, testset)

        oacc_sum = 0
        aacc_sum = 0

        for run_idx in range(5):
            model = self._setup_variance_model()
            model = train(model, trainloader, None, self.cfg)

            # Evaluator(model, self.cfg, self.ae, 0).evaluate(valloader)
            modeller = model.encoder if self.ae else model.modeller

            features_train = self._extract_features(modeller, predloader)
            features_test = self._extract_features(modeller, testloader)

            features_train = features_train.transpose(0, 3, 4, 1, 2).reshape(-1, self.cfg.k * 5)
            features_test = features_test.reshape(-1, self.cfg.k * 5)

            gt_train = np.array(predset.gt).reshape(-1)
            gt_test = np.array(testset.gt)

            oacc, aacc = predict_soil_classes(features_train, features_test, gt_train, gt_test)
            oacc_sum += oacc
            aacc_sum += aacc

        avg_oacc = oacc_sum / 5
        avg_aacc = aacc_sum / 5
        return avg_oacc, avg_aacc

    def _prepare_datasets(self, fold_dir: Path):
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

    def _prepare_dataloaders(self, trainset, predset, testset):
        batch_size = self.cfg.batch_size
        trainloader = DataLoader(trainset, batch_size, shuffle=True)
        predloader = DataLoader(predset, batch_size, shuffle=False)
        valloader = DataLoader(trainset, batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(testset, batch_size=1, shuffle=False)
        return trainloader, predloader, valloader, testloader

    def _extract_features(self, modeller, dataloader):
        device = self.cfg.device
        features = []

        with torch.no_grad():
            for img in dataloader:
                img = img.to(device)
                ft = modeller(img)
                features.append(ft.cpu().numpy())

        return np.concatenate(features, axis=0)
