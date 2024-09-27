import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig
from src.models.bias_variance_model import BiasModel, BiasVarianceModel, VarianceModel


def calculate_penalization(tensor: Tensor, device: str) -> Tensor:
    n, k, _, h, w = tensor.shape
    mu = tensor[:, :, 0, :, :]
    result = torch.zeros((n, h, w), device=device)

    # iterate over all possible pairs (i, j) with i != j
    for i in range(k):
        for j in range(k):
            if i != j:
                diff = torch.abs(mu[:, i, :, :] - mu[:, j, :, :])
                exp_diff = torch.exp(-diff)
                result += exp_diff

    # average over all pixels and batch size
    result = result.sum() / (n * h * w)
    return result


def pretrain(
    bias_model: BiasModel,
    trainloader: DataLoader,
    cfg: ExperimentConfig,
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(bias_model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(cfg.epochs):
        step_losses = []
        with tqdm(trainloader, unit="batch") as tepoch:
            for input in tepoch:
                tepoch.set_description(f"Pretraining epoch {epoch}")
                optimizer.zero_grad()
                input = input.to(cfg.device)
                y_pred = bias_model(0)
                loss = criterion(y_pred, input)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
                step_losses.append(loss.cpu().detach().numpy())
        scheduler.step()
        if cfg.wandb:
            wandb.log({"metrics/pretrain/MSE": np.mean(step_losses)})
        scheduler.step()
    return bias_model


def train(
    variance_model: VarianceModel,
    bias_model: BiasModel | None,
    trainloader: DataLoader,
    valloader: DataLoader,
    cfg: ExperimentConfig,
) -> tuple[nn.Module]:
    model = BiasVarianceModel(bias_model, variance_model)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(cfg.epochs):
        model.train()
        # model.bias.W.requires_grad_(False)
        step_losses = []
        with tqdm(trainloader, unit="batch") as tepoch:
            for input in tepoch:
                div = np.count_nonzero(input)
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                input = input.to(cfg.device)
                y_pred = model(input)
                loss = criterion(y_pred, input)
                # loss += calculate_penalization(y_pred, device)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item() / div)
                step_losses.append(loss.cpu().detach().numpy() / div)
        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(valloader):
                vdiv = np.count_nonzero(vdata)
                vinputs = vdata.to(cfg.device)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vinputs)
                running_vloss += vloss / vdiv

        avg_vloss = running_vloss / (i + 1)
        if cfg.wandb:
            wandb.log({"metrics/train/MSE": np.mean(step_losses), "metrics/val/MSE": avg_vloss})
        scheduler.step()
    return model
