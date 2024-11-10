import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import wandb
from plotly.subplots import make_subplots
from torch.utils.data import Dataset

from src.consts import GT_NAMES


def gt_to_lists(dataset: Dataset) -> tuple[list]:
    ps, ks, mgs, phs, = [], [], [], []
    for (img, gt) in dataset:
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
