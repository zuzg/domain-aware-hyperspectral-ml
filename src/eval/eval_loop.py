import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.eval.visualizations import (
    plot_average_reflectance,
    plot_bias,
    plot_images,
    plot_partial_hats,
    plot_partial_polynomials,
    plot_partial_polynomials_degree,
    plot_pixelwise,
    plot_splines,
)


class Evaluator:
    def __init__(self, model: nn.Module, cfg: ExperimentConfig, ae: bool):
        self.model = model
        self.cfg = cfg
        self.ae = ae
        self.modeller, self.variance_model, self.bias_model = self._initialize_model_components()

    def _initialize_model_components(self) -> tuple[nn.Module | None]:
        if self.ae:
            return self.model, None, self.model.bias
        return self.model.variance.modeller, self.model.variance, self.model.bias

    def evaluate(self, testloader: DataLoader) -> None:
        self.modeller.eval()
        imgs, renders, raw_outputs = self._process_testloader(testloader)
        self._plot_results(imgs, renders, raw_outputs)

    def _process_testloader(self, testloader: DataLoader) -> tuple[list[Tensor]]:
        imgs, renders, raw_outputs = [], [], []
        with torch.no_grad():
            for img in testloader:
                img = img.to(self.cfg.device)
                out = self.modeller(img)
                render = self._apply_rendering(out)
                imgs.append(img)
                renders.append(render)
                raw_outputs.append(out)
        return imgs, renders, raw_outputs

    def _apply_rendering(self, out: Tensor) -> Tensor:
        if self.ae:
            return out
        rendered = self.variance_model.renderer(out)
        if self.cfg.bias_renderer == "Mean":
            rendered += self.bias_model
        elif self.bias_model is not None:
            rendered += self.bias_model(0)
        return rendered

    def _plot_results(self, imgs: list[Tensor], renders: list[Tensor], raw_outputs: list[Tensor]) -> None:
        if len(imgs) == 1:
            ids = [0]
            mask_nan = False
        else:
            ids = [1, 8, 10]
            mask_nan = True
        for i in ids:
            self._plot_image_comparisons(imgs[i][0].cpu(), renders[i][0].cpu(), mask_nan)
            self._plot_variance_renderer(raw_outputs[i][0] if not self.ae else None, i)
        self._plot_bias()

    def _plot_image_comparisons(self, gt_img: Tensor, pred_img: Tensor, mask_nan: bool) -> None:
        plot_images(gt_img.numpy(), pred_img.numpy(), mask_nan)
        plot_average_reflectance(gt_img.numpy(), pred_img.numpy())
        img_center = gt_img.shape[1] // 2
        plot_pixelwise(gt_img.numpy(), pred_img.numpy(), img_center)

    def _plot_variance_renderer(self, raw_output: Tensor, idx: int) -> None:
        if self.ae or raw_output is None:
            return
        img_center = raw_output.shape[2] // 2
        center_slice = raw_output[..., img_center, img_center]
        renderer_type = self.cfg.variance_renderer
        if renderer_type == "GaussianRenderer":
            plot_partial_hats(center_slice, self.cfg.mu_type, self.cfg.channels)
        elif renderer_type == "PolynomialRenderer":
            plot_partial_polynomials(center_slice, self.cfg.channels)
        elif renderer_type == "PolynomialDegreeRenderer":
            plot_partial_polynomials_degree(center_slice, self.cfg.k, self.cfg.channels)
        elif renderer_type == "SplineRenderer":
            plot_splines(self.variance_model.renderer(center_slice)[idx])

    def _plot_bias(self) -> None:
        if self.bias_model is not None:
            bias = self.bias_model if self.cfg.bias_renderer == "Mean" else self.bias_model(0)
            plot_bias(bias[0, :, 0, 0])
