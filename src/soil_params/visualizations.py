import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import wandb


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
