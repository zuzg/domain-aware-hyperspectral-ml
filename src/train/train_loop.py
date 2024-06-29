import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig
from src.models.renderers.base_renderer import BaseRenderer


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


def train(
    model: nn.Module,
    renderer: BaseRenderer,
    trainloader: DataLoader,
    valloader: DataLoader,
    cfg: ExperimentConfig,
) -> tuple[nn.Module, list[float], list[float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(cfg.epochs):
        step_losses = []
        with tqdm(trainloader, unit="batch") as tepoch:
            for input in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                input = input.to(cfg.device)
                y_pred = model(input)
                y_pred_r = renderer(y_pred)
                loss = criterion(y_pred_r, input)
                # loss += calculate_penalization(y_pred, device)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
                step_losses.append(loss.cpu().detach().numpy())
        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(valloader):
                vinputs = vdata.to(cfg.device)
                voutputs = model(vinputs)
                voutputs_r = renderer(voutputs)
                vloss = torch.sqrt(criterion(voutputs_r, vinputs))
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        if cfg.wandb:
            wandb.log({"metrics/train/MSE": np.mean(step_losses), "metrics/val/MSE": avg_vloss})
        scheduler.step()
    return model
