import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.renderers.base_renderer import BaseRenderer


def train(
    model: nn.Module,
    renderer: BaseRenderer,
    trainloader: DataLoader,
    valloader: DataLoader,
    num_epochs: int,
    device: str,
    lr: float,
) -> tuple[nn.Module, list[float], list[float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(num_epochs):
        step_losses = []
        with tqdm(trainloader, unit="batch") as tepoch:
            for input in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                input = input.to(device)
                y_pred = model(input)
                y_pred_r = renderer(y_pred)
                loss = criterion(y_pred_r, input)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
                step_losses.append(loss.cpu().detach().numpy())
        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(valloader):
                vinputs = vdata.to(device)
                voutputs = model(vinputs)
                voutputs_r = renderer(voutputs)
                vloss = torch.sqrt(criterion(voutputs_r, vinputs))
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        wandb.log({"metrics/train/MSE": np.mean(step_losses), "metrics/val/MSE": avg_vloss})
        scheduler.step()
    return model
