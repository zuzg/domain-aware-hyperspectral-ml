import matplotlib.pyplot as plt
import wandb
from scipy.stats import norm
from torch import Tensor


def plot_partial_hats(pixel_hats: Tensor) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for stats in pixel_hats:
        stats = stats.cpu().detach().numpy()
        dist = norm.pdf(range(0, 150), 150 * stats[0], 150 * stats[1]) + stats[2]
        ax.plot(dist)
    plt.title("Partial hats for a random pixel")
    wandb.log({"partial_hats": fig})


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
    plt.xlabel("band")
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
