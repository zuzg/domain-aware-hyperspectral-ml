import cmcrameri.cm as cmc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.consts import MAX_PATH
from src.eval.visualizations import (
    plot_average_reflectance,
    plot_bias,
    plot_images,
    plot_param_stats,
    plot_partial_betas,
    plot_partial_hats,
    plot_partial_hats_asymmetric,
    plot_partial_hats_skew,
    plot_partial_polynomials,
    plot_partial_polynomials_degree,
    plot_pixelwise,
    plot_splines,
)
from src.models.dual import generate_latent
from src.soil_params.visualizations import get_rgb


class Evaluator:
    def __init__(self, model: nn.Module, cfg: ExperimentConfig, ae: bool, bias: np.ndarray | float):
        self.model = model
        self.cfg = cfg
        self.ae = ae
        self.bias = bias
        self.modeller, self.renderer, self.bias_model = self._initialize_model_components()

    def _initialize_model_components(self) -> tuple[nn.Module | None]:
        if self.ae:
            return self.model, None, None
        return self.model.modeller, self.model.renderer, None

    def evaluate(self, testloader: DataLoader) -> None:
        imgs, renders, raw_outputs, params = self._process_testloader(testloader)
        self._plot_results(imgs, renders, raw_outputs)
        # plot_param_stats(params)

    def _process_testloader(
        self, testloader: DataLoader, visualize: bool = False
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], None]:
        imgs, renders, raw_outputs = [], [], []
        self.modeller.eval()

        if visualize:
            fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(4 * 4, 10))
            cmap = matplotlib.colormaps.get_cmap(cmc.imola)
            example_indices = [1, 15, 31, 40, 65]
            j = 0

        with torch.no_grad():
            for i, img in enumerate(testloader):
                img = img.to(self.cfg.device)
                out = self.modeller(img)
                render = self._apply_rendering(out)

                imgs.append(img)
                renders.append(render.cpu().detach())
                raw_outputs.append(out.cpu().detach())

                if visualize and i in example_indices:
                    mask = img[:, 0] == 0
                    expanded_mask = mask.unsqueeze(1).expand(-1, 150, -1, -1)
                    render[expanded_mask] = 0

                    gt_plot = img.cpu().numpy()[0]
                    pred_plot = render.cpu().numpy()[0]

                    rgb_gt = get_rgb(gt_plot)
                    rgb_pred = get_rgb(pred_plot)
                    error = np.mean(np.abs(gt_plot - pred_plot), axis=0)
                    print(np.nanmax(error))

                    axes[0, j].imshow(rgb_gt)
                    axes[1, j].imshow(rgb_pred)
                    im = axes[2, j].imshow(error, cmap=cmap, vmin=0, vmax=0.015)

                    if i == example_indices[0]:
                        axes[0, j].set_ylabel("Ground truth", fontsize=18, rotation=90)
                        axes[1, j].set_ylabel("Reconstructed", fontsize=18, rotation=90)
                        axes[2, j].set_ylabel("Error", fontsize=18, rotation=90)

                    j += 1

        if visualize:
            for row in axes:
                for ax in row:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)

            fig.tight_layout(rect=[0, 0.03, 1, 1])
            cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.01])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            cbar.set_label("Mean Absolute Error", fontsize=16)
            cbar.ax.tick_params(labelsize=14)
            fig.tight_layout()
            fig.savefig("image_viz.png")

        return imgs, renders, raw_outputs, None

    def _apply_rendering(self, out: Tensor) -> Tensor:
        if self.ae:
            return out
        rendered = self.renderer(out)
        # if self.cfg.bias_renderer == "Mean" or self.cfg.bias_renderer == "None":
        #     rendered += self.bias_model
        # elif self.bias_model is not None:
        #     rendered += self.bias_model(0)
        return rendered

    def _plot_results(self, imgs: list[Tensor], renders: list[Tensor], raw_outputs: list[Tensor]) -> None:
        if len(imgs) == 1:
            ids = [0]
            mask_nan = False
        else:
            ids = [0, 1, 5]
            mask_nan = True
        for i in ids:
            self._plot_image_comparisons(imgs[i][i].cpu(), renders[i][i].cpu(), mask_nan, i)
            self._plot_variance_renderer(raw_outputs[i][i] if not self.ae else None, i)
        # self._plot_bias()
        if self.cfg.dual_mode:
            self.dual_mode()

    def _plot_image_comparisons(self, gt_img: Tensor, pred_img: Tensor, mask_nan: bool, i) -> None:
        plot_images(gt_img.numpy(), pred_img.numpy(), mask_nan, "images")
        plot_images(gt_img.numpy() + self.bias, pred_img.numpy() + self.bias, mask_nan, "images without bias")
        plot_average_reflectance(gt_img.numpy(), pred_img.numpy(), "reflectance without bias")
        plot_average_reflectance(gt_img.numpy() + self.bias, pred_img.numpy() + self.bias, "reflectance")
        img_center = 0
        plot_pixelwise(gt_img.numpy(), pred_img.numpy(), img_center, "pixelwise without bias", i)
        plot_pixelwise(gt_img.numpy() + self.bias, pred_img.numpy() + self.bias, img_center, "pixelwise", i)

    def _plot_variance_renderer(self, raw_output: Tensor, idx: int) -> None:
        if self.ae or raw_output is None:
            return
        img_center = raw_output.shape[2] // 2
        center_slice = raw_output[..., img_center, img_center]
        renderer_type = self.cfg.variance_renderer
        if renderer_type == "BetaRenderer":
            plot_partial_betas(center_slice, self.cfg.channels)
        elif renderer_type == "GaussianRenderer":
            plot_partial_hats(center_slice, self.cfg.mu_type, self.cfg.channels)
        elif renderer_type == "GaussianAsymmetricRenderer":
            plot_partial_hats_asymmetric(center_slice, self.cfg.mu_type, self.cfg.channels)
        elif renderer_type == "GaussianSkewRenderer":
            plot_partial_hats_skew(center_slice, self.cfg.mu_type, self.cfg.channels, idx)
        elif renderer_type == "PolynomialRenderer":
            plot_partial_polynomials(center_slice, self.cfg.channels)
        elif renderer_type == "PolynomialDegreeRenderer":
            plot_partial_polynomials_degree(center_slice, self.cfg.k, self.cfg.channels)
        elif renderer_type == "SplineRenderer":
            plot_splines(self.renderer(center_slice)[idx])

    def _plot_bias(self) -> None:
        if self.bias_model is not None:
            bias = self.bias_model if self.cfg.bias_renderer == "Mean" else self.bias_model(0)
            plot_bias(bias[0, :, 0, 0])

    def dual_mode(self) -> None:
        for i in range(3):
            latent = generate_latent(self.cfg.batch_size, self.cfg.img_size, self.cfg.k, 5, "cuda")
            plot_partial_hats_skew(
                latent[0, ..., 50, 50].cpu().detach(), self.cfg.mu_type, self.cfg.channels, key="ph_dual_gt"
            )
            generated_images = self.renderer(latent)  # 16 150 100 100
            reconstructed_latents = self.modeller(generated_images)  # 16 5 5 100 100
            plot_partial_hats_skew(
                reconstructed_latents[0, ..., 50, 50].cpu().detach(),
                self.cfg.mu_type,
                self.cfg.channels,
                key="ph_dual_pred",
            )
