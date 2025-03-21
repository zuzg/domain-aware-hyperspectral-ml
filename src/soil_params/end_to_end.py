import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

from src.config import ExperimentConfig
from src.consts import GT_DIM, GT_MAX, GT_NAMES, MSE_BASE_K, MSE_BASE_MG, MSE_BASE_P, MSE_BASE_PH, OUTPUT_PATH
from src.models.modeller import Modeller
from src.soil_params.data import prepare_datasets, prepare_gt, SoilDataset


class MultiRegressionCNN(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = GT_DIM):
        """
        CNN for hyperspectral image regression, outputting n continuous values per pixel.
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            # nn.Conv2d(64, 32, kernel_size=1),
            # nn.ReLU(),
        )
        self.output_layer = nn.Conv2d(32, output_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        output = self.output_layer(x)
        return output


def apply_mask(pred: Tensor, mask: Tensor, k: int, num_params: int) -> Tensor:
    mask = mask.bool()
    pred[:, :, num_params - 2] = pred[:, 0, num_params - 2 : num_params - 1]
    crop_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, k, num_params, -1, -1)
    return torch.where(crop_mask, torch.tensor(0.0, device=pred.device), pred)


class EndToEndModel(nn.Module):
    def __init__(self, modeller: Modeller, regressor: MultiRegressionCNN):
        super().__init__()
        self.modeller = modeller
        self.regressor = regressor

    def forward(self, x):
        features = self.modeller(x)
        # mask = x[:, 0] == 0
        # masked_features = apply_mask(features, mask)
        output = self.regressor(features)
        return output


def train_end_to_end(trainloader: DataLoader, modeller: Modeller, out_dim: int, epochs: int = 15) -> nn.Module:
    regressor = MultiRegressionCNN(input_channels=modeller.output_dim, output_channels=out_dim)
    model = EndToEndModel(modeller, regressor).to("cuda")

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        with tqdm(trainloader, unit="batch", desc=f"Epoch {epoch}") as tepoch:
            for images, gt in tepoch:
                images, gt = images.to("cuda"), gt.to("cuda")
                optimizer.zero_grad()

                outputs = model(images)

                mask = (images != 0).any(dim=1).unsqueeze(1)
                non_zero_count = mask.sum().item()

                masked_outputs = outputs[mask.expand_as(outputs)]
                masked_gt = gt[mask.expand_as(gt)]
                loss = criterion(masked_outputs, masked_gt) / non_zero_count

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

    return model


def compute_masks(img: Tensor, gt: Tensor, mask: Tensor, gt_div_tensor: Tensor) -> tuple[Tensor]:
    expanded_mask = mask.unsqueeze(1)
    crop_mask = expanded_mask.expand(-1, gt.shape[1], -1, -1)
    masked_gt = torch.where(crop_mask == 0, gt, torch.zeros_like(gt))
    masked_pred = torch.where(crop_mask == 0, img, torch.zeros_like(img))
    return masked_gt * gt_div_tensor, masked_pred * gt_div_tensor


def predict_params(trainloader, testloader, modeller, gt_div, device="cuda"):
    model = train_end_to_end(trainloader, modeller, out_dim=len(gt_div))
    model.to(device)
    model.eval()

    criterion = nn.MSELoss(reduction="none")
    total_loss = torch.zeros(len(gt_div), device=device)
    gt_div_tensor = torch.tensor(gt_div, device=device).reshape(1, len(gt_div), 1, 1)

    with torch.no_grad():
        for img, gt in testloader:
            img, gt = img.to(device), gt.to(device)
            mask = img[:, 0] == 0
            div = (img[:, 0] != 0).sum().item()

            pred = model(img)
            masked_gt, masked_pred = compute_masks(pred, gt, mask, gt_div_tensor)

            loss = criterion(masked_pred, masked_gt)
            channel_loss = loss.sum(dim=(0, 2, 3)) / div
            total_loss += channel_loss

    return (total_loss / len(testloader)).cpu().detach().numpy()


class ImgGtDataset(Dataset):
    def __init__(self, image_list: np.ndarray, gt: pd.DataFrame, gt_div: np.ndarray, size: int):
        """
        Dataset for training with raw images (not precomputed features).
        """
        super().__init__()
        self.images = torch.from_numpy(image_list).float()
        self.gt = gt
        self.gt_div = gt_div
        self.size = size
        self.gt_dim = len(gt_div)

        self.gt_values = self._prepare_gt_values()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.gt_values[index]

    def _prepare_gt_values(self) -> torch.Tensor:
        num_images = len(self.images)
        gt_values = torch.zeros((num_images, self.gt_dim, self.size, self.size), dtype=torch.float32)

        for i in range(num_images):
            soil = torch.from_numpy(self.gt.loc[i].values / self.gt_div).float()
            gt_unsqueezed = soil.unsqueeze(1).unsqueeze(2)
            gt_values[i] = gt_unsqueezed.repeat(1, self.size, self.size)

        return gt_values


def samples_number_experiment(
    dataset: Dataset,
    sample_nums: list[int],
    gt_div: np.ndarray,
    f_dim: int,
    mse_base: float | list[float],
    param: str | None = None,
    n_runs: int = 1,
) -> None:
    mses_mean = []
    wandb.define_metric("soil/step")
    wandb.define_metric("soil/*", step_metric="soil/step")
    save_model = True

    for sn in sample_nums:
        mses_for_sample = []
        for run in range(n_runs):
            print(run)
            generator = torch.Generator().manual_seed(run)
            trainset_base, testset = random_split(dataset, [0.8, 0.2], generator=generator)
            # if sn == sample_nums[0]:
            #     visualize_distribution(trainset_base, testset, run)

            # trainset = Subset(trainset_base, indices=range(sn))
            trainloader = DataLoader(trainset_base, batch_size=8, shuffle=True)
            testloader = DataLoader(testset, batch_size=8, shuffle=False)

            mse = predict_params(trainloader, testloader, f_dim, gt_div, "cuda", save_model)
            print(mse)
            save_model = False
            mses_for_sample.append(mse / mse_base)

        mses_for_sample = np.array(mses_for_sample)
        mses_sample_mean = mses_for_sample.mean(axis=0)
        mses_mean.append(mses_sample_mean)
        if len(gt_div) == 1:
            wandb.log( 
                {
                    "soil/step": -sn,
                    f"soil/{param}": mses_sample_mean[0],
                }
            )
        else:
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

def predict_soil_parameters(
    dataset: Dataset,
    model: Modeller,
    num_params: int,
    cfg: ExperimentConfig,
    ae: bool,
    single_model: bool = True,
    baseline: bool = False,
) -> None:
    features = prepare_datasets(dataset, model, cfg.k, cfg.channels, num_params, cfg.batch_size, cfg.device, ae, baseline)
    gt = prepare_gt(dataset.ids)
    samples = [1728]  # , 250, 200, 150, 100, 50, 25, 10]
    f_dim = cfg.channels if baseline else cfg.k * num_params
    mse_base = [MSE_BASE_P, MSE_BASE_K, MSE_BASE_MG, MSE_BASE_PH]

    if single_model:
        dataset = SoilDataset(features, 300, gt, GT_MAX, shuffle=False)  # TODO fix shuffle
        samples_number_experiment(dataset, samples, GT_MAX, f_dim, mse_base)