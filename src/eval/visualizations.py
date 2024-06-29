import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import wandb
from scipy.stats import norm
from torch import Tensor

from src.consts import CHANNELS


def plot_partial_hats(pixel_hats: Tensor) -> None:
    fig = go.Figure()
    x = np.linspace(0, CHANNELS-1)
    for i, stats in enumerate(pixel_hats):
        stats = stats.cpu().detach().numpy()
        dist = norm.pdf(range(0, CHANNELS), CHANNELS * stats[0], CHANNELS * stats[1])
        fig.add_traces(go.Scatter(x=x, y=dist, mode="lines", name=f"hat {i}"))
    fig.update_layout(title="Partial hats for a random pixel", xaxis_title="Band", yaxis_title="Intensity")
    wandb.log({"partial_hats": fig})


def plot_partial_polynomials(polys: Tensor) -> None:
    fig = go.Figure()
    x = np.linspace(1, CHANNELS, num=150)
    for i, params in enumerate(polys):
        params = params.cpu().detach().numpy()
        poly = params[0] * x ** params[1]
        fig.add_traces(go.Scatter(x=x, y=poly, mode="lines", name=f"{params[0]:.2f}*x^{params[1]:.2f}"))
    fig.update_layout(title="Partial functions for a random pixel", xaxis_title="Band", yaxis_title="Intensity")
    wandb.log({"partial_polys": fig})


def plot_partial_polynomials_degree(polys: Tensor, k: int) -> None:
    fig = go.Figure()
    x = np.linspace(1 / 100, CHANNELS / 100, num=150)
    exp = np.linspace(0, k-1, num=k)
    for i, params in enumerate(polys):
        params = params.cpu().detach().numpy()
        poly = params[0] * x ** exp[i]
        fig.add_traces(go.Scatter(x=x, y=poly, mode="lines", name=f"{params[0]:.2f}*x^{exp[i]:.0f}"))
    fig.update_layout(title="Partial functions for a random pixel", xaxis_title="Band", yaxis_title="Intensity")
    wandb.log({"partial_polys_degree": fig})


def plot_images(gt_img: Tensor, pred_img: Tensor) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Band 0")
    axs[0].imshow(gt_img[0], vmin=0, vmax=1)
    axs[0].set_title("GT")
    axs[1].imshow(pred_img[0], vmin=0, vmax=1)
    axs[1].set_title("PRED")
    wandb.log({"images": fig})


def plot_average_reflectance(gt_img: Tensor, pred_img: Tensor) -> None:
    fig = plt.figure(figsize=(10, 5))
    gt_mean_spectral_reflectance = [gt_img[i].mean() for i in range(gt_img.shape[0])]
    pred_mean_spectral_reflectance = [pred_img[i].mean() for i in range(pred_img.shape[0])]
    plt.plot(pred_mean_spectral_reflectance, label="PRED")
    plt.plot(gt_mean_spectral_reflectance, label="GT")
    plt.xlabel("Band")
    fig.legend()
    plt.title("Average reflectance")
    wandb.log({"reflectance": fig})


def plot_pixelwise(gt_img: Tensor, pred_img: Tensor, size: int) -> None:
    fig, axs = plt.subplots(size, size, figsize=(20, 25))
    for i in range(size):
        for j in range(size):
            axs[i, j].plot(pred_img[:, i, j])
            axs[i, j].plot(gt_img[:, i, j])
    wandb.log({"pixelwise": wandb.Image(fig)})
