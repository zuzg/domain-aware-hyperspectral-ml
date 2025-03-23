import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.config import ExperimentConfig
from src.consts import (
    GT_MAX,
    MAX_PATH,
    MSE_BASE_K,
    MSE_BASE_MG,
    MSE_BASE_P,
    MSE_BASE_PH,
    OUTPUT_PATH,
    TRAIN_IDS,
    TRAIN_PATH,
    SPLIT_RATIO,
)
from src.models.modeller import Modeller
from src.models.soil_predictor import EndToEndModel, MultiRegressionCNN
from src.soil_params.data import ImgGtDataset, prepare_gt
from src.soil_params.utils import collate_fn_pad_full, compute_masks


def train_end_to_end(
    trainloader: DataLoader, modeller: Modeller, f_dim, out_dim: int, epochs: int = 20, log: bool = True
) -> nn.Module:
    regressor = MultiRegressionCNN(input_channels=f_dim, output_channels=out_dim)
    model = EndToEndModel(modeller, regressor).to("cuda")

    criterion = nn.HuberLoss(reduction="sum")
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
                if log:
                    wandb.log({f"soil_MSE/agg": loss.item()})
    return model


def predict_params(trainloader, testloader, modeller, f_dim, gt_div, device="cuda", save_model: bool = False):
    log = not save_model
    model = train_end_to_end(trainloader, modeller, f_dim, out_dim=len(gt_div), log=log)
    if save_model:
        torch.save(model.state_dict(), OUTPUT_PATH / "models" / f"e2e_model.pth")
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


def samples_number_experiment(
    dataset: Dataset,
    modeller: nn.Module,
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

    for sn in sample_nums:
        mses_for_sample = []
        for run in range(n_runs):
            print(run)
            generator = torch.Generator().manual_seed(run)
            trainset_base, testset = random_split(dataset, [0.8, 0.2], generator=generator)

            trainloader = DataLoader(
                trainset_base,
                batch_size=8,
                shuffle=True,
                collate_fn=collate_fn_pad_full,
            )
            testloader = DataLoader(
                testset,
                batch_size=8,
                shuffle=False,
                collate_fn=collate_fn_pad_full,
            )
            fullloader = DataLoader(
                dataset,
                batch_size=8,
                shuffle=True,
                collate_fn=collate_fn_pad_full,
            )

            mse = predict_params(trainloader, testloader, modeller, f_dim, gt_div, "cuda")
            print(mse)
            mses_for_sample.append(mse / mse_base)
            _ = predict_params(fullloader, fullloader, modeller, f_dim, gt_div, "cuda", save_model=True)

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
    with open(MAX_PATH, "rb") as f:
        max_values = np.load(f)
    max_values[max_values > cfg.max_val] = cfg.max_val

    rng = np.random.default_rng(12345)
    splits = np.split(rng.permutation(TRAIN_IDS), np.cumsum(SPLIT_RATIO))
    gt = prepare_gt(dataset.ids)
    samples = [1728]  # , 250, 200, 150, 100, 50, 25, 10]
    f_dim = cfg.channels if baseline else cfg.k * num_params
    mse_base = [MSE_BASE_P, MSE_BASE_K, MSE_BASE_MG, MSE_BASE_PH]

    if single_model:
        dataset = ImgGtDataset(TRAIN_PATH, splits[2], max_values, cfg.max_val, gt, GT_MAX, 300)
        samples_number_experiment(dataset, model, samples, GT_MAX, f_dim, mse_base)
