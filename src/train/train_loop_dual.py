import numpy as np
import torch
import wandb
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm

from src.config import ExperimentConfig
from src.models.dual import DualModeAutoencoder
from src.train.metrics import calculate_sam


def compute_loss(criterion: nn.Module, pred: Tensor, target: Tensor, mask: Tensor) -> nn.Module:
    """Compute masked loss normalized by the number of unmasked elements."""
    masked_pred = pred[mask]
    masked_target = target[mask]
    loss = criterion(masked_pred, masked_target) / mask.sum().item()
    return loss


def train_step(
    model: DualModeAutoencoder,
    dataloader: DataLoader,
    criterion: nn.Module,
    criterion_dual: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    alpha: float,
) -> float:
    """Run a single training step over the trainloader."""
    model.train()
    step_losses = []
    step_losses_classical = []
    step_losses_inverse = []

    with tqdm(dataloader, unit="batch", desc=f"Epoch {epoch}") as tepoch:
        for batch in tepoch:
            optimizer.zero_grad()
            batch = batch.to(device)
            mask = batch != 0

            # Classical Autoencoder Mode
            original_images, reconstructed_images = model(batch, mode="classical")
            classical_loss = compute_loss(criterion, reconstructed_images, original_images, mask)

            # Inverse Autoencoder Mode
            original_latents, reconstructed_latents = model(mode="inverse")
            inverse_loss = criterion_dual(reconstructed_latents, original_latents)

            # Combined Loss
            loss = alpha * classical_loss + (1 - alpha) * inverse_loss

            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())
            step_losses.append(loss.item())
            step_losses_classical.append(classical_loss.item())
            step_losses_inverse.append(inverse_loss.item())

    return np.mean(step_losses), np.mean(step_losses_classical), np.mean(step_losses_inverse)


def validate_step(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    criterion_dual: nn.Module,
    psnr: nn.Module,
    mae: nn.Module,
    device: str,
    alpha: float,
) -> tuple[float]:
    """Run a validation step over the valloader and compute metrics."""
    model.eval()
    running_loss = running_psnr = running_sam = running_mse = running_mae = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mask = batch != 0

            # Classical Autoencoder Mode
            original_images, reconstructed_images = model(batch, mode="classical")
            classical_loss = compute_loss(criterion, reconstructed_images, original_images, mask)

            # Inverse Autoencoder Mode
            original_latents, reconstructed_latents = model(mode="inverse")
            inverse_loss = criterion_dual(reconstructed_latents, original_latents)

            # Combined Loss
            loss = alpha * classical_loss + (1 - alpha) * inverse_loss

            reconstructed_images[batch == 0] = 0  # Ignore masked pixels for metrics
            mse_val = classical_loss
            mae_val = mae(reconstructed_images, batch)
            psnr_val = psnr(reconstructed_images, batch)
            sam_val = calculate_sam(reconstructed_images, batch)

            running_loss += loss.item()
            running_mse += mse_val.item()
            running_mae += mae_val.item()
            running_psnr += psnr_val
            running_sam += sam_val

    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_mse = running_mse / num_batches
    avg_mae = running_mae / num_batches
    avg_psnr = running_psnr / num_batches
    avg_sam = running_sam / num_batches
    return avg_loss, avg_mse, avg_mae, avg_psnr, avg_sam


def train(model: nn.Module, trainloader: DataLoader, valloader: DataLoader, cfg: ExperimentConfig) -> nn.Module:
    """Train the model with given configurations."""
    criterion = nn.HuberLoss(reduction="sum")
    criterion_dual = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    psnr = PeakSignalNoiseRatio().to(cfg.device)
    mae = MeanAbsoluteError().to(cfg.device)

    alpha = 0.5
    if cfg.wandb:
        wandb.watch(model, criterion, log="all", log_freq=10, log_graph=True)

    for epoch in range(cfg.epochs):
        train_loss, train_loss_classical, train_loss_inverse = train_step(
            model, trainloader, criterion, criterion_dual, optimizer, cfg.device, epoch, alpha=alpha
        )
        val_loss, val_mse, val_mae, val_psnr, val_sam = validate_step(
            model, valloader, criterion, criterion_dual, psnr, mae, cfg.device, alpha=alpha
        )
        if cfg.wandb:
            wandb.log(
                {
                    "metrics/train/Huber_dual": train_loss,
                    "metrics/train/Huber": train_loss_classical,
                    "metrics/train/Huber_inv": train_loss_inverse,
                    "metrics/val/Huber_dual": val_loss,
                    "metrics/val/Huber": val_mse,
                    "metrics/val/PSNR": val_psnr,
                    "metrics/val/SAM": val_sam,
                    "metrics/val/MAE": val_mae,
                }
            )
        scheduler.step()
        if alpha < 1:
            alpha += 0.03
    return model
