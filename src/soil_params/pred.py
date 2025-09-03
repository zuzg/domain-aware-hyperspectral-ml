import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

from src.config import ExperimentConfig
from src.consts import (
    GT_DIM,
    GT_MAX,
    GT_NAMES,
    MSE_BASE_K,
    MSE_BASE_MG,
    MSE_BASE_P,
    MSE_BASE_PH,
    OUTPUT_PATH,
)
from src.models.modeller import Modeller
from src.models.soil_predictor import MultiRegressionCNN
from src.soil_params.data import prepare_datasets, prepare_gt, SoilDataset
from src.soil_params.utils import compute_masks
from src.soil_params.visualizations import visualize_distribution


def predict_params(
    trainloader: DataLoader,
    testloader: DataLoader,
    f_dim: int,
    gt_div: np.ndarray,
    device: str,
    save_model: bool,
) -> np.ndarray:
    if save_model:
        param = gt_div[0]
    else:
        param = None
    model = train(trainloader, features=f_dim, out_dim=len(gt_div), param=param)
    if save_model:
        torch.save(
            model.state_dict(), OUTPUT_PATH / "models" / f"regressor_full_single.pth"
        )
    model.to(device)
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    total_loss = torch.zeros(len(gt_div), device=device)
    gt_div_tensor = torch.tensor(gt_div, device=device).reshape(1, len(gt_div), 1, 1)

    with torch.no_grad():
        for img, gt in testloader:
            img, gt = img.to(device), gt.to(device)
            mask = img[:, 0] == 0
            div = (
                (img[:, 0] != 0).sum().item()
            )  # Count of non-zero elements in channel 0

            pred = model(img)
            masked_gt, masked_pred = compute_masks(pred, gt, mask, gt_div_tensor)

            loss = criterion(masked_pred, masked_gt)
            channel_loss = (
                loss.sum(dim=(0, 2, 3)) / div
            )  # Summing across height and width
            total_loss += channel_loss

    mean_loss = total_loss / len(testloader)
    return mean_loss.cpu().detach().numpy()


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
            trainset_base, testset = random_split(
                dataset, [0.8, 0.2], generator=generator
            )
            # if sn == sample_nums[0]:
            #     visualize_distribution(trainset_base, testset, run)

            # trainset = Subset(trainset_base, indices=range(sn))
            trainloader = DataLoader(trainset_base, batch_size=8, shuffle=True)
            testloader = DataLoader(testset, batch_size=8, shuffle=False)

            mse = predict_params(
                trainloader, testloader, f_dim, gt_div, "cuda", save_model
            )
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


def train(
    dataloader: DataLoader,
    features: int = 150,
    out_dim: int = GT_DIM,
    epochs: int = 15,
    param: str = "",
) -> nn.Module:
    model = MultiRegressionCNN(input_channels=features, output_channels=out_dim)
    model = model.to("cuda")
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
                if param:
                    wandb.log({f"soil_MSE/{param}": loss.item()})
    return model


def predict_soil_parameters(
    dataset: Dataset,
    model: Modeller,
    num_params: int,
    cfg: ExperimentConfig,
    ae: bool,
    single_model: bool = True,
    baseline: bool = False,
) -> None:
    features = prepare_datasets(
        dataset,
        model,
        cfg.k,
        cfg.channels,
        num_params,
        cfg.batch_size,
        cfg.device,
        ae,
        baseline,
    )
    gt = prepare_gt(dataset.ids)
    samples = [1728]  # , 250, 200, 150, 100, 50, 25, 10]
    f_dim = cfg.channels if baseline else cfg.k * num_params
    mse_base = [MSE_BASE_P, MSE_BASE_K, MSE_BASE_MG, MSE_BASE_PH]

    if single_model:
        dataset = SoilDataset(features, cfg.img_size, gt, GT_MAX)
        samples_number_experiment(dataset, samples, GT_MAX, f_dim, mse_base)
    # train separate model for each parameter
    else:
        for i, soil_param in enumerate(GT_NAMES):
            gt_param = gt.iloc[:, i]
            dataset_param = SoilDataset(
                features, cfg.img_size, gt_param.to_frame(), [GT_MAX[i]]
            )
            samples_number_experiment(
                dataset_param,
                samples,
                np.array([GT_MAX[i]]),
                f_dim,
                mse_base[i],
                soil_param,
            )
