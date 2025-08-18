import cmcrameri.cm as cmc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import wandb
from scipy.stats import beta, norm
from torch import Tensor


def plot_partial_betas(betas: Tensor, channels: int) -> None:
    fig = go.Figure()
    x = np.linspace(1, channels, num=channels)
    shift = betas[0][2].numpy()
    channels_div = np.arange(0, channels) / channels
    beta_sum = np.zeros(channels)
    betas[betas == 0] = 1e-5

    for params in betas:
        params = params.numpy()
        dist = beta.pdf(channels_div, params[0], params[1])
        beta_sum += dist
        fig.add_traces(go.Scatter(x=x, y=dist, mode="lines", name=f"alpha={params[0]:.2f}, beta={params[1]:.2f}"))

    fig.add_traces(
        go.Scatter(x=x, y=beta_sum + shift, mode="lines", name=f"Sum of betas, shift={shift:.2}", line_color="black")
    )
    fig.update_layout(title="Partial betas for a random pixel", xaxis_title="Band", yaxis_title="Intensity")
    wandb.log({"partial_betas": fig})


def plot_partial_hats(pixel_hats: Tensor, mu_type: str, channels: int) -> None:
    fig = go.Figure()
    x = np.linspace(1, channels, num=channels)
    shift = pixel_hats[0][3].numpy()
    k = len(pixel_hats)
    fix_ref = channels // k
    intervals = [channels / (k + 2) * i for i in range(1, k + 1)]
    hat_sum = np.zeros(channels)

    for i, stats in enumerate(pixel_hats):
        stats = stats.numpy()
        if mu_type == "unconstrained":
            mu = channels * stats[0]
        elif mu_type == "fixed_reference":
            mu = stats[0] * fix_ref + (fix_ref * i)
        elif mu_type == "equal_interval":
            mu = intervals[i]
        sigma, scale = channels * stats[1], channels * stats[2]
        dist = scale * norm.pdf(range(0, channels), mu, sigma)
        hat_sum += dist
        fig.add_traces(go.Scatter(x=x, y=dist, mode="lines", name=f"μ={mu:.1f}, σ={sigma:.1f}, scale={scale:.1f}"))

    fig.add_traces(
        go.Scatter(x=x, y=hat_sum + shift, mode="lines", name=f"Sum of hats, shift={shift:.2}", line_color="black")
    )
    fig.update_layout(title="Partial hats for a random pixel", xaxis_title="Band", yaxis_title="Intensity")
    wandb.log({"partial_hats": fig})


def plot_partial_hats_asymmetric(pixel_hats: Tensor, mu_type: str, channels: int) -> None:
    fig = go.Figure()
    x = np.linspace(1, channels, num=channels)
    shift = pixel_hats[0][4].numpy()
    k = len(pixel_hats)
    hat_sum = np.zeros(channels)

    for i, stats in enumerate(pixel_hats):
        stats = stats.numpy()
        mu = channels * stats[0]
        sigma_left, sigma_right, scale = channels * stats[1], channels * stats[2], channels * stats[3]

        dist = np.zeros(channels)
        for idx in range(channels):
            if idx + 1 <= mu:
                dist[idx] = scale * norm.pdf(idx + 1, mu, sigma_left)
            else:
                dist[idx] = scale * norm.pdf(idx + 1, mu, sigma_right)

        hat_sum += dist
        fig.add_traces(
            go.Scatter(
                x=x,
                y=dist,
                mode="lines",
                name=f"μ={mu:.1f}, σₗ={sigma_left:.1f}, σᵣ={sigma_right:.1f}, scale={scale:.1f}",
            )
        )

    fig.add_traces(
        go.Scatter(x=x, y=hat_sum + shift, mode="lines", name=f"Sum of hats, shift={shift:.2}", line_color="black")
    )
    fig.update_layout(title="Partial hats for a random pixel", xaxis_title="Band", yaxis_title="Intensity")
    wandb.log({"partial_hats": fig})


def plot_partial_hats_skew(
    pixel_hats: Tensor, mu_type: str, channels: int, idx: int = 0, key: str = "partial_hats"
) -> None:
    fig = go.Figure()
    x = np.linspace(1, channels, num=channels)
    # shift = pixel_hats[0][3].numpy()
    k = len(pixel_hats)
    hat_sum = np.zeros(channels)  # Sum of all distributions

    for i, stats in enumerate(pixel_hats):
        stats = stats.numpy()
        mu = channels * stats[0]  # Mean
        sigma = channels * stats[1]  # Standard deviation
        scale = channels * stats[2]  # Scaling factor
        skew = channels * stats[3]  # Skewness parameter

        # Generate skew-normal PDF
        x_vals = np.linspace(0, channels, num=channels)
        pdf = 2 * scale * norm.pdf(x_vals, mu, sigma) * norm.cdf(skew * (x_vals - mu) / sigma)

        hat_sum += pdf
        fig.add_traces(
            go.Scatter(x=x, y=pdf, mode="lines", name=f"μ={mu:.1f}, σ={sigma:.1f}, scale={scale:.1f}, skew={skew:.1f}")
        )
        # np.save(f"output/spectres/{idx}_{key}_{i}.npy", pdf)

    fig.add_traces(go.Scatter(x=x, y=hat_sum, mode="lines", name=f"Sum of hats", line_color="black"))

    fig.update_layout(
        title="Partial Hats for a Random Pixel",
        xaxis_title="Band",
        yaxis_title="Intensity",
    )
    wandb.log({key: fig})


def plot_partial_polynomials(polys: Tensor, channels: int) -> None:
    fig = go.Figure()
    x = np.linspace(1, channels, num=channels)
    for i, params in enumerate(polys):
        params = params.numpy()
        poly = params[0] * x ** params[1]
        fig.add_traces(go.Scatter(x=x, y=poly, mode="lines", name=f"{params[0]:.2f}*x^{params[1]:.2f}"))
    fig.update_layout(title="Partial functions for a random pixel", xaxis_title="Band", yaxis_title="Intensity")
    wandb.log({"partial_polys": fig})


def plot_partial_polynomials_degree(polys: Tensor, k: int, channels: int) -> None:
    fig = go.Figure()
    x = np.linspace(1 / 100, channels / 100, num=channels)
    exp = np.linspace(0, k - 1, num=k)
    for i, params in enumerate(polys):
        params = params.numpy()
        poly = params[0] * x ** exp[i]
        fig.add_traces(go.Scatter(x=x, y=poly, mode="lines", name=f"{params[0]:.2f}*x^{exp[i]:.0f}"))
    fig.update_layout(title="Partial functions for a random pixel", xaxis_title="Band", yaxis_title="Intensity")
    wandb.log({"partial_polys_degree": fig})


def plot_splines(splines: Tensor) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(splines.numpy(), label="splines")
    plt.xlabel("Band")
    fig.legend()
    plt.title("Splines")
    wandb.log({"splines": fig})


def plot_images(gt_img: Tensor, pred_img: Tensor, mask_nan: bool, key: str) -> None:
    cmap = matplotlib.colormaps.get_cmap(cmc.batlow)
    if mask_nan:
        pred_img[gt_img == 0] = np.nan
        gt_img[gt_img == 0] = np.nan
        cmap.set_bad("white")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Band 0")
    axs[0].imshow(gt_img[0], cmap=cmap)
    axs[0].set_title("GT")
    axs[1].imshow(pred_img[0], cmap=cmap)
    axs[1].set_title("PRED")
    wandb.log({key: fig})


def plot_average_reflectance(gt_img: Tensor, pred_img: Tensor, key: str) -> None:
    fig = plt.figure(figsize=(10, 5))
    pred_img[gt_img == 0] = np.nan
    gt_img[gt_img == 0] = np.nan
    gt_mean_spectral_reflectance = [np.nanmean(gt_img[i]) for i in range(gt_img.shape[0])]
    pred_mean_spectral_reflectance = [np.nanmean(pred_img[i]) for i in range(pred_img.shape[0])]
    plt.plot(pred_mean_spectral_reflectance, label="PRED")
    plt.plot(gt_mean_spectral_reflectance, label="GT")
    plt.xlabel("Band")
    fig.legend()
    plt.title("Average reflectance")
    wandb.log({key: fig})


def plot_pixelwise(gt_img: Tensor, pred_img: Tensor, size: int, key: str, idx) -> None:
    fig, axs = plt.subplots(10, 10, figsize=(20, 25))
    for i in range(10):
        for j in range(10):
            axs[i, j].plot(pred_img[:, i + size, j + size])
            axs[i, j].plot(gt_img[:, i + size, j + size])
            # if key == "pixelwise" and i < 3 and j < 3:
            #     np.save(f"output/spectres/{idx}_gt_{key}_{i}_{j}.npy", gt_img[:, i + size, j + size])
            #     np.save(f"output/spectres/{idx}_pred_{key}_{i}_{j}.npy", pred_img[:, i + size, j + size])
    wandb.log({key: wandb.Image(fig)})


def plot_bias(bias: Tensor) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(bias.numpy(), label="bias")
    plt.xlabel("Band")
    fig.legend()
    plt.title("Bias")
    wandb.log({"bias": fig})


def plot_param_stats(stats: Tensor) -> None:
    fig = px.box(stats)
    ticks = ["mean", "std", "scale", "shift"]
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=[0, 1, 2, 3], ticktext=ticks),  # change 1
    )
    fig.write_html("output/stats.html")
    wandb.log({"Hat stats": fig})
