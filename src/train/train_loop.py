import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm

from src.config import ExperimentConfig


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


def dim_zero_cat(x: Tensor) -> Tensor:
    """Concatenation along the zero dimension."""
    x = x if isinstance(x, (list, tuple)) else [x]
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)


def calculate_sam(preds: Tensor, target: Tensor) -> float:
    preds = dim_zero_cat(preds)
    target = dim_zero_cat(target)

    dot_product = (preds * target).sum(dim=1)
    preds_norm = preds.norm(dim=1)
    target_norm = target.norm(dim=1)
    sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
    return sam_score.nanmean()


def pretrain(
    bias_model: nn.Module,
    trainloader: DataLoader,
    cfg: ExperimentConfig,
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(bias_model.parameters(), lr=cfg.lr / 10)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(2 * cfg.epochs):
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


def train(model: nn.Module, trainloader: DataLoader, valloader: DataLoader, cfg) -> nn.Module:
    criterion = nn.MSELoss(reduction="sum")  # Summing loss over all elements
    psnr = PeakSignalNoiseRatio().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(cfg.epochs):
        model.train()
        step_losses = []
        with tqdm(trainloader, unit="batch") as tepoch:
            for input in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                input = input.to(cfg.device)
                mask = input != 0  # True where input is non-zero
                div = mask.sum().item()  # Count of non-masked values

                y_pred = model(input)

                # Apply mask to both input and y_pred
                masked_y_pred = y_pred[mask]
                masked_input = input[mask]

                loss = criterion(masked_y_pred, masked_input) / div
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                step_losses.append(loss.item())

        model.eval()
        running_vloss = 0.0
        running_psnr = 0.0
        running_sam = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(valloader):
                vinputs = vdata.to(cfg.device)
                mask = vinputs != 0
                vdiv = mask.sum().item()
                voutputs = model(vinputs)

                # Apply mask
                masked_voutputs = voutputs[mask]
                masked_vinputs = vinputs[mask]

                vloss = criterion(masked_voutputs, masked_vinputs) / vdiv

                voutputs[vinputs == 0] = 0
                vpsnr = psnr(voutputs, vinputs)
                vsam = calculate_sam(voutputs, vinputs)

                running_vloss += vloss.item()
                running_psnr += vpsnr
                running_sam += vsam

        if cfg.wandb:
            wandb.log(
                {
                    "metrics/train/MSE": np.mean(step_losses),
                    "metrics/val/MSE": running_vloss / (i + 1),
                    "metrics/val/PSNR": running_psnr / (i + 1),
                    "metrics/val/SAM": running_sam / (i + 1),
                }
            )
        scheduler.step()
    return model
