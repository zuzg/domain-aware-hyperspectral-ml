import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import wandb
from src.config import ExperimentConfig
from src.consts import (
    GT_MAX,
    GT_NAMES,
    MAX_PATH,
    MSE_BASE_B,
    MSE_BASE_CU,
    MSE_BASE_FE,
    MSE_BASE_MN,
    MSE_BASE_S,
    MSE_BASE_ZN,
    OUTPUT_PATH,
    SPLIT_RATIO,
    TRAIN_IDS,
    TRAIN_PATH,
)
from src.models.modeller import Modeller
from src.models.soil_predictor import EndToEndModel, MultiRegressionCNN
from src.soil_params.data import ImgGtDataset, prepare_gt
from src.soil_params.utils import collate_fn_pad_full, compute_masks

DEVICE = "cpu"
LR_DICT = {"B": 1e-5, "Cu": 1e-4, "Zn": 1e-3, "Fe": 1e-3, "S": 1e-3, "Mn": 1e-4}


def train_end_to_end(
    trainloader: DataLoader,
    modeller: Modeller,
    f_dim: int,
    out_dim: int,
    epochs: int = 15,
    log: bool = True,
    param: str = "",
    freeze: bool = False,
) -> nn.Module:
    regressor = MultiRegressionCNN(input_channels=f_dim, output_channels=out_dim)
    model = EndToEndModel(modeller, regressor).to(DEVICE)
    if freeze:
        model.modeller.requires_grad_(False)

    criterion = nn.HuberLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_DICT[param])

    for epoch in range(epochs):
        with tqdm(trainloader, unit="batch", desc=f"Epoch {epoch}") as tepoch:
            for images, gt in tepoch:
                images, gt = images.to(DEVICE), gt.to(DEVICE)
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
                    wandb.log({f"soil_MSE/{param}": loss.item()})
    return model


def predict_params(
    trainloader: DataLoader,
    testloader: DataLoader,
    modeller: nn.Module,
    f_dim: int,
    gt_div: np.ndarray,
    device: str = DEVICE,
    model_name: str = "",
    save_model: bool = False,
):
    log = not save_model
    out_dim = 1

    model = train_end_to_end(
        trainloader, modeller, f_dim, out_dim=out_dim, log=log, param=model_name
    )
    if save_model:
        torch.save(
            model.state_dict(), OUTPUT_PATH / "models" / f"e2e_model_{model_name}.pth"
        )
    model.to(device)
    model.eval()

    criterion = nn.MSELoss(reduction="none")
    total_loss = torch.zeros(out_dim, device=device)
    gt_div_tensor = torch.tensor(gt_div, device=device).reshape(1, out_dim, 1, 1)

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
    mses_std = []
    wandb.define_metric("soil/step")
    wandb.define_metric("soil/*", step_metric="soil/step")

    for sn in sample_nums:
        mses_for_sample = []
        for run in range(n_runs):
            print(run)
            generator = torch.Generator().manual_seed(run)
            trainset_base, testset = random_split(
                dataset, [0.8, 0.2], generator=generator
            )
            # trainset = Subset(trainset_base, indices=range(sn))

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
            # fullloader = DataLoader(
            #     dataset,
            #     batch_size=8,
            #     shuffle=True,
            #     collate_fn=collate_fn_pad_full,
            # )

            mse = predict_params(
                trainloader,
                testloader,
                modeller,
                f_dim,
                gt_div,
                DEVICE,
                model_name=param,
            )
            print(mse)
            mses_for_sample.append(mse / mse_base)
            # _ = predict_params(fullloader, fullloader, modeller, f_dim, gt_div, DEVICE, param, save_model=True)

        mses_for_sample = np.array(mses_for_sample)
        mses_sample_mean = mses_for_sample.mean(axis=0)
        mses_mean.append(mses_sample_mean)
        mses_std.append(mses_for_sample.std(axis=0))
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
    return mses_sample_mean[0], mses_std[0]


def predict_soil_parameters(
    dataset: Dataset,
    model: Modeller,
    num_params: int,
    cfg: ExperimentConfig,
    ae: bool,
    single_model: bool = False,
    baseline: bool = False,
) -> None:
    with open(MAX_PATH, "rb") as f:
        max_values = np.load(f)
    max_values[max_values > cfg.max_val] = cfg.max_val

    rng = np.random.default_rng(12345)
    splits = np.split(rng.permutation(TRAIN_IDS), np.cumsum(SPLIT_RATIO))
    gt = prepare_gt(dataset.ids)
    f_dim = cfg.channels if baseline else cfg.k * num_params
    mse_base = [
        MSE_BASE_B,
        MSE_BASE_CU,
        MSE_BASE_FE,
        MSE_BASE_MN,
        MSE_BASE_S,
        MSE_BASE_ZN,
    ]

    max_samples = 1876
    samples = [max_samples]

    if single_model:
        dataset = ImgGtDataset(
            TRAIN_PATH, splits[2], max_values, cfg.max_val, gt, GT_MAX, 300
        )
        samples_number_experiment(dataset, model, samples, GT_MAX, f_dim, mse_base)

    else:
        mses = []
        for i, soil_param in enumerate(GT_NAMES):
            gt_param = gt.iloc[:, i]
            dataset_param = ImgGtDataset(
                TRAIN_PATH,
                splits[2],
                max_values,
                cfg.max_val,
                gt_param.to_frame(),
                [GT_MAX[i]],
                300,
            )
            mse, stds = samples_number_experiment(
                dataset_param,
                model,
                samples,
                np.array([GT_MAX[i]]),
                f_dim,
                mse_base[i],
                soil_param,
            )
            mses.append(mse)
        wandb.log({"soil/score": np.mean(mses), "soil/std": np.mean(stds)})
