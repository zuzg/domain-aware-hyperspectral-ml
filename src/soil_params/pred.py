import numpy as np
import torch
import wandb
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

from src.config import ExperimentConfig
from src.consts import CHANNELS, GT_DIM, MSE_BASE_K, MSE_BASE_MG, MSE_BASE_P, MSE_BASE_PH
from src.models.modeller import Modeller
from src.soil_params.data import prepare_datasets, prepare_gt, SoilDataset


def compute_masks(img: Tensor, gt: Tensor, mask: Tensor, gt_div_tensor: Tensor) -> tuple[Tensor]:
    expanded_mask = mask.unsqueeze(1)
    crop_mask = expanded_mask.expand(-1, gt.shape[1], -1, -1)
    masked_gt = torch.where(crop_mask == 0, gt, torch.zeros_like(gt))
    masked_pred = torch.where(crop_mask == 0, img, torch.zeros_like(img))
    return masked_gt * gt_div_tensor, masked_pred * gt_div_tensor


def predict_params(
    trainloader: DataLoader, testloader: DataLoader, f_dim: int, gt_div: np.ndarray, device: str
) -> np.ndarray:
    model = train(trainloader, features=f_dim)
    model.to(device)
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    total_loss = torch.zeros(GT_DIM, device=device)
    gt_div_tensor = torch.tensor(gt_div, device=device).reshape(1, GT_DIM, 1, 1)

    with torch.no_grad():
        for img, gt in testloader:
            img, gt = img.to(device), gt.to(device)
            mask = img[:, 0] == 0
            div = (img[:, 0] != 0).sum().item()  # Count of non-zero elements in channel 0

            pred = model(img)
            masked_gt, masked_pred = compute_masks(pred, gt, mask, gt_div_tensor)

            loss = criterion(masked_pred, masked_gt)
            channel_loss = loss.sum(dim=(0, 2, 3)) / div  # Summing across height and width
            total_loss += channel_loss

    mean_loss = total_loss / len(testloader)
    return mean_loss.cpu().detach().numpy()


def samples_number_experiment(
    dataset: Dataset, sample_nums: list[int], gt_div: np.ndarray, f_dim: int, n_runs: int = 5) -> None:
    mses_mean = []
    mses_std = []
    wandb.define_metric("soil/step")
    wandb.define_metric("soil/*", step_metric="soil/step")

    for sn in sample_nums:
        mses_for_sample = []
        for run in range(n_runs):
            generator = torch.Generator().manual_seed(run)
            trainset_base, testset = random_split(dataset, [0.8, 0.2], generator=generator)
            trainset = Subset(trainset_base, indices=range(sn))
            trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
            testloader = DataLoader(testset, batch_size=8, shuffle=False)

            mse = predict_params(trainloader, testloader, f_dim, gt_div, "cuda")
            mses_for_sample.append(mse / [MSE_BASE_P, MSE_BASE_K, MSE_BASE_MG, MSE_BASE_PH])
        
        mses_for_sample = np.array(mses_for_sample)
        mses_sample_mean = mses_for_sample.mean(axis=0)
        mses_mean.append(mses_sample_mean)
        mses_std.append(mses_for_sample.std(axis=0))
        wandb.log(
            {
                "soil/step": -sn,
                "soil/P": mses_sample_mean[0],
                "soil/K": mses_sample_mean[1],
                "soil/Mg": mses_sample_mean[2],
                "soil/pH": mses_sample_mean[3],
                "soil/score": 0.25 * np.sum(mses_sample_mean),
            }
        )


class MultiRegressionCNN(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = GT_DIM):
        """
        CNN for hyperspectral image regression, outputting n continuous values per pixel.
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Sigmoid(),
        )
        self.output_layer = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        output = self.output_layer(x)
        return output


def train(dataloader: DataLoader, features: int = 150, epochs: int = 20) -> nn.Module:
    model = MultiRegressionCNN(input_channels=features)
    model = model.to("cuda")
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        with tqdm(dataloader, unit="batch", desc=f"Epoch {epoch}") as tepoch:
            for images, gt in tepoch:
                images, gt = images.to("cuda"), gt.to("cuda")
                optimizer.zero_grad()
                mask = (images != 0).any(dim=1).unsqueeze(1)
                non_zero_count = mask.sum().item()

                outputs = model(images)

                masked_outputs = outputs[mask.expand_as(outputs)]
                masked_gt = gt[mask.expand_as(gt)]
                loss = criterion(masked_outputs, masked_gt) / non_zero_count
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
    return model


def predict_soil_parameters(
    dataset: Dataset,
    model: Modeller,
    num_params: int,
    cfg: ExperimentConfig,
    ae: bool,
    baseline: bool = False,
) -> None:
    features = prepare_datasets(dataset, model, cfg.k, num_params, cfg.batch_size, cfg.device, ae, baseline)
    gt = prepare_gt(dataset.ids)
    gt_div = gt.max().values
    dataset = SoilDataset(features, cfg.img_size, gt, gt_div)
    samples = [487, 250, 200, 150, 100, 50, 25, 10]
    f_dim = CHANNELS if baseline else cfg.k * num_params
    samples_number_experiment(dataset, samples, gt_div, f_dim)
