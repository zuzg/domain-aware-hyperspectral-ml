import numpy as np
import torch
import wandb
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm

from src.config import ExperimentConfig
from src.train.metrics import calculate_sam


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


def compute_loss(criterion: nn.Module, pred: Tensor, target: Tensor, mask: Tensor) -> nn.Module:
    """Compute masked loss normalized by the number of unmasked elements."""
    masked_pred = pred[mask]
    masked_target = target[mask]
    loss = criterion(masked_pred, masked_target) / mask.sum().item()
    return loss


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> float:
    """Run a single training step over the trainloader."""
    model.train()
    step_losses = []

    with tqdm(dataloader, unit="batch", desc=f"Epoch {epoch}") as tepoch:
        for batch in tepoch:
            optimizer.zero_grad()
            batch = batch.to(device)
            mask = batch != 0
            pred = model(batch)

            loss = compute_loss(criterion, pred, batch, mask)
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())
            step_losses.append(loss.item())

    return np.mean(step_losses)


def validate_step(
    model: nn.Module, dataloader: DataLoader | None, criterion: nn.Module, psnr: nn.Module, mae: nn.Module, device: str
) -> tuple[float]:
    """Run a validation step over the valloader and compute metrics."""
    if dataloader is None:
        return 0, 0, 0, 0

    model.eval()
    running_loss = running_psnr = running_sam = running_mae = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mask = batch != 0
            pred = model(batch)

            loss = compute_loss(criterion, pred, batch, mask)
            pred[batch == 0] = 0  # Ignore masked pixels for metrics
            mae_val = mae(pred, batch)
            psnr_val = psnr(pred, batch)
            sam_val = calculate_sam(pred, batch)

            running_loss += loss.item()
            running_mae += mae_val
            running_psnr += psnr_val
            running_sam += sam_val

    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_mae = running_mae / num_batches
    avg_psnr = running_psnr / num_batches
    avg_sam = running_sam / num_batches
    return avg_loss, avg_mae, avg_psnr, avg_sam


def train(model: nn.Module, trainloader: DataLoader, valloader: DataLoader | None, cfg: ExperimentConfig) -> nn.Module:
    criterion = nn.HuberLoss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    psnr = PeakSignalNoiseRatio().to(cfg.device)
    mae = MeanAbsoluteError().to(cfg.device)

    if cfg.wandb:
        wandb.watch(model, criterion, log="all", log_freq=10, log_graph=True)

    # Early stopping parameters
    patience = 5
    min_delta = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_model = model

    for epoch in range(cfg.epochs):
        train_loss = train_step(model, trainloader, criterion, optimizer, cfg.device, epoch)
        val_loss, val_mae, val_psnr, val_sam = validate_step(model, valloader, criterion, psnr, mae, cfg.device)
        if cfg.wandb:
            wandb.log(
                {
                    "metrics/train/Huber": train_loss,
                    "metrics/val/Huber": val_loss,
                    "metrics/val/PSNR": val_psnr,
                    "metrics/val/SAM": val_sam,
                    "metrics/val/MAE": val_mae,
                }
            )
        scheduler.step()

        # Early stopping logic
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model
        else:
            patience_counter += 1
            if val_loss > 0 and patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return best_model
