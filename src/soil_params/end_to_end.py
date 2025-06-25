import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import StratifiedKFold
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

from src.config import ExperimentConfig
from src.consts import (
    GT_MAX,
    GT_NAMES,
    MAX_PATH,
    MSE_BASE_K,
    MSE_BASE_MG,
    MSE_BASE_P,
    MSE_BASE_PH,
    OUTPUT_PATH,
    SPLIT_RATIO,
    TRAIN_IDS,
    TRAIN_PATH,
    VIZ_PATH,
)
from src.models.modeller import Modeller
from src.models.soil_predictor import EndToEndModel, MultiRegressionCNN
from src.soil_params.data import ImgGtDataset, prepare_gt
from src.soil_params.utils import collate_fn_pad_full, compute_masks
from src.soil_params.visualizations import plot_hyperview_score, plot_param_heatmap, plot_rgb_image


def train_end_to_end(
    trainloader: DataLoader,
    modeller: Modeller,
    f_dim: int,
    out_dim: int,
    epochs: int = 20,
    log: bool = True,
    freeze: bool = False,
) -> nn.Module:
    regressor = MultiRegressionCNN(input_channels=f_dim, output_channels=out_dim)
    model = EndToEndModel(modeller, regressor).to("cuda")
    if freeze:
        model.modeller.requires_grad_(False)

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


def compute_scores(
    masked_gt: Tensor,
    masked_pred: Tensor,
    base_values: Tensor,
    criterion: nn.Module,
) -> Tensor:
    mse = criterion(masked_gt, masked_pred)
    base_values_expanded = base_values.expand_as(masked_pred)
    score = mse / base_values_expanded
    return score.mean(dim=1)


def predict_params(
    trainloader: DataLoader,
    testloader: DataLoader,
    modeller: nn.Module,
    f_dim: int,
    gt_div: np.ndarray,
    device: str = "cuda",
    model_name: str = "",
    save_model: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    log = not save_model
    out_dim = 1

    model: nn.Module = train_end_to_end(trainloader, modeller, f_dim, out_dim=out_dim, log=log)

    if save_model:
        torch.save(model.state_dict(), OUTPUT_PATH / "models" / f"e2e_model_{model_name}.pth")

    model.to(device)
    model.eval()

    criterion = nn.MSELoss(reduction="none")
    gt_div_tensor = torch.tensor(gt_div, device=device).reshape(1, out_dim, 1, 1)
    base_values = torch.tensor(
        [MSE_BASE_K, MSE_BASE_MG, MSE_BASE_P, MSE_BASE_PH], dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)

    total_loss = torch.zeros(out_dim, device=device)
    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 16))

    with torch.no_grad():
        for i, (img, gt) in enumerate(testloader):
            img, gt = img.to(device), gt.to(device)
            mask = img[:, 0] == 0
            div = (img[:, 0] != 0).sum().item()

            pred = model(img)
            masked_gt, masked_pred = compute_masks(pred, gt, mask, gt_div_tensor)

            if 5 <= i < 10:
                vis_idx = i - 5
                plot_rgb_image(axes[vis_idx, 0], img.cpu().numpy()[0], vis_idx)
                score_mean = compute_scores(masked_gt, masked_pred, base_values, criterion)
                plot_hyperview_score(axes[vis_idx, 1], score_mean.cpu().numpy()[0], vis_idx, fig)

                for p_idx in range(4):
                    gt_np = masked_gt.cpu().numpy()[0][p_idx]
                    pred_np = masked_pred.cpu().numpy()[0][p_idx]
                    plot_param_heatmap(
                        axes[vis_idx, p_idx + 2],
                        pred_np,
                        gt_np.max(),
                        GT_NAMES[p_idx],
                        is_header=(vis_idx == 0),
                        fig=fig,
                    )

            loss = criterion(masked_pred, masked_gt)
            channel_loss = loss.sum(dim=(0, 2, 3)) / div
            total_loss += channel_loss

    fig.tight_layout()
    fig.savefig(VIZ_PATH / "soil_param_heatmaps_rgb.png")

    # === Train Evaluation ===
    total_loss_train = torch.zeros(len(gt_div), device=device)
    with torch.no_grad():
        for img, gt in trainloader:
            img, gt = img.to(device), gt.to(device)
            mask = img[:, 0] == 0
            div = (img[:, 0] != 0).sum().item()

            pred = model(img)
            masked_gt, masked_pred = compute_masks(pred, gt, mask, gt_div_tensor)

            loss_t = criterion(masked_pred, masked_gt)
            channel_loss_t = loss_t.sum(dim=(0, 2, 3)) / div
            total_loss_train += channel_loss_t

    avg_test_loss = (total_loss / len(testloader)).cpu().numpy()
    avg_train_loss = (total_loss_train / len(trainloader)).cpu().numpy()

    return avg_test_loss, avg_train_loss


def predict_params_stratified(
    dataset: ImgGtDataset, modeller: nn.Module, f_dim: int, gt_div: np.ndarray, num_bins: int = 5
) -> np.ndarray:
    gt = np.array(dataset.gt.values).ravel()
    y_binned = pd.qcut(gt, q=num_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    target_mse = []

    for train_idx, test_idx in skf.split(gt, y_binned):
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
        trainloader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_fn_pad_full,
        )
        testloader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=collate_fn_pad_full,
        )

        mse = predict_params(trainloader, testloader, modeller, f_dim, gt_div, "cuda")
        target_mse.append(mse)

    print("Target MSE")
    print(target_mse)
    return np.array([np.mean(target_mse)])


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
            trainset_base, testset = random_split(dataset, [sn, 1 - sn], generator=generator)
            # trainset = Subset(trainset_base, indices=range(sn))

            trainloader = DataLoader(
                trainset_base,
                batch_size=8,
                shuffle=True,
                collate_fn=collate_fn_pad_full,
            )
            testloader = DataLoader(
                testset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn_pad_full,
            )

            # mse = predict_params(trainloader, testloader, modeller, f_dim, gt_div, "cuda")
            mse = predict_params_stratified(dataset, modeller, f_dim, gt_div)
            print(mse)
            print(0.25 * np.sum(mse / mse_base))
            mses_for_sample.append(mse / mse_base)

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
    limited_data: bool = False,
) -> None:
    with open(MAX_PATH, "rb") as f:
        max_values = np.load(f)
    max_values[max_values > cfg.max_val] = cfg.max_val

    rng = np.random.default_rng(12345)
    splits = np.split(rng.permutation(TRAIN_IDS), np.cumsum(SPLIT_RATIO))
    gt = prepare_gt(dataset.ids)
    f_dim = cfg.channels if baseline else cfg.k * num_params
    mse_base = [MSE_BASE_P, MSE_BASE_K, MSE_BASE_MG, MSE_BASE_PH]

    samples = [0.8, 0.5, 0.25, 0.1, 0.05, 0.01] if limited_data else [0.8]

    if single_model:
        dataset = ImgGtDataset(TRAIN_PATH, splits[2], max_values, cfg.max_val, gt, GT_MAX, 300)
        samples_number_experiment(dataset, model, samples, GT_MAX, f_dim, mse_base)

    else:
        mses = []
        for i, soil_param in enumerate(GT_NAMES):
            gt_param = gt.iloc[:, i]
            dataset_param = ImgGtDataset(
                TRAIN_PATH, splits[2], max_values, cfg.max_val, gt_param.to_frame(), [GT_MAX[i]], 300
            )
            mse, stds = samples_number_experiment(
                dataset_param, model, samples, np.array([GT_MAX[i]]), f_dim, mse_base[i], soil_param
            )
            mses.append(mse)
        wandb.log({"soil/score": np.mean(mses), "soil/std": np.mean(stds)})
