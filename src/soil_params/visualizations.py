import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import scienceplots
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.subplots import make_subplots
from torch.utils.data import Dataset

import wandb
from src.consts import GT_NAMES, MAX_PATH


plt.style.use(["science", "no-latex"])


def gt_to_lists(dataset: Dataset) -> tuple[list]:
    (
        ps,
        ks,
        mgs,
        phs,
    ) = (
        [],
        [],
        [],
        [],
    )
    for img, gt in dataset:
        pixel = gt[:, 0, 0]
        p, k, mg, ph = pixel[0], pixel[1], pixel[2], pixel[3]
        ps.append(p)
        ks.append(k)
        mgs.append(mg)
        phs.append(ph)
    return ps, ks, mgs, phs


def visualize_distribution(trainset: Dataset, testset: Dataset, run: int) -> None:
    train_params = gt_to_lists(trainset)
    test_params = gt_to_lists(testset)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    kwargs = dict(alpha=0.5, bins=100)
    for i, ax in enumerate(fig.axes):
        ax.hist(train_params[i], **kwargs, label="Train set")
        ax.hist(test_params[i], **kwargs, label="Test set")
        ax.legend()
        ax.set_title(GT_NAMES[i])
    fig.suptitle("Distribution of targets")
    plt.savefig(f"output/hists_{run}.png")


def plot_soil_params(
    samples: list[int],
    mean_img: np.ndarray,
    std_img: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    gt: pd.DataFrame,
    k: int,
    num_params: int,
) -> None:
    fig = make_subplots(rows=2, cols=2, subplot_titles=gt.columns)
    x = list(range(len(samples)))
    means = [mean_img, mean_pred]
    stds = [std_img, std_pred]
    palette = px.colors.qualitative.Plotly
    names = ["Raw-150", f"ModG-{k*num_params-k+1}"]  # TODO Latent-n
    colors = [palette[i % len(palette)] for i in range(2)]

    for i in range(4):
        row = i // 2 + 1
        col = i % 2 + 1
        for j in range(2):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=means[j][:, i],
                    mode="lines",
                    line=dict(color=colors[j]),
                    name=names[j],
                    showlegend=i == 0,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    name="-std",
                    x=x,
                    y=means[j][:, i] - stds[j][:, i],
                    opacity=0.4,
                    marker=dict(color=colors[j]),
                    mode="lines",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    name="+std",
                    x=x,
                    y=means[j][:, i] + stds[j][:, i],
                    opacity=0.4,
                    marker=dict(color=colors[j]),
                    mode="lines",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Number of samples", tickvals=x, ticktext=samples, row=row, col=col)
        fig.update_yaxes(title_text="MSE", row=row, col=col)

    fig.update_layout(
        title_text="MLP: mean and std of MSE for each predicted soil parameter", showlegend=True, hovermode="x"
    )
    wandb.log({"soil_params": fig})


def normalize_band(band: np.ndarray) -> np.ndarray:
    band_min, band_max = np.nanmin(band), np.nanmax(band)
    return (band - band_min) / (band_max - band_min + 1e-8)


# Load and cap max values
with open(MAX_PATH, "rb") as f:
    max_values = np.load(f)
max_values[max_values > 6000] = 6000
MAX_VAL = max_values


def get_rgb(img: np.ndarray) -> np.ndarray:
    img[img == 0] = np.nan
    channels = [2, 28, 59]
    rgb = np.stack([normalize_band(img[c] * MAX_VAL[c]) for c in channels], axis=-1)
    return np.nan_to_num(rgb, nan=1.0)


def plot_rgb_image(ax: Axes, img: np.ndarray, instance_idx: int) -> None:
    rgb = get_rgb(img)
    im = ax.imshow(rgb)
    ax.set_title("false-color" if instance_idx == 0 else "")
    ax.set_ylabel(f"Instance {instance_idx + 1}", fontsize=20, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="5%", pad=0.05)
    cax.set_visible(False)


def plot_hyperview_score(ax: Axes, score_mean: np.ndarray, instance_idx: int, fig: Figure) -> None:
    score_mean[score_mean == 0] = np.nan
    im = ax.imshow(score_mean, cmap="coolwarm", vmin=0.0, vmax=2.0)
    ax.set_title("Hyperview Score" if instance_idx == 0 else "")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=15)


def plot_param_heatmap(
    ax: Axes,
    pred_plot: np.ndarray,
    gt_max: float,
    param_name: str,
    is_header: bool,
    fig: Figure,
) -> None:
    cmap = matplotlib.colormaps.get_cmap("Blues")
    cmap.set_bad("white")
    pred_plot[pred_plot == 0] = np.nan
    pred_plot[pred_plot < 0] = 0
    im = ax.imshow(pred_plot, cmap=cmap)
    if is_header:
        ax.set_title(param_name, fontsize=20, weight="bold")
    ax.set_xlabel(f"GT = {gt_max:.1f}", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=15)
