from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.eval.visualizations import (plot_average_reflectance, plot_images,
                                     plot_partial_hats,
                                     plot_partial_polynomials,
                                     plot_partial_polynomials_degree,
                                     plot_pixelwise)
from src.models.modeller import Modeller
from src.models.renderers.base_renderer import BaseRenderer


def evaluate(model: Modeller, renderer: BaseRenderer, testloader: DataLoader, cfg: ExperimentConfig):
    imgs = []
    raw_outputs = []
    renders = []
    for img in testloader:
        out = model(img.to(cfg.device))
        rendered = renderer(out)
        imgs.append(img)
        raw_outputs.append(out)
        renders.append(rendered)

    i = 1
    gt_img = imgs[0][i]
    pred_img = renders[0][i].cpu().detach().numpy()
    plot_images(gt_img, pred_img)
    plot_average_reflectance(gt_img, pred_img)
    plot_pixelwise(gt_img, pred_img, cfg.img_size)
    if cfg.renderer == "GaussianRenderer":
        plot_partial_hats(raw_outputs[0][i, ..., 5, 5])
    elif cfg.renderer == "PolynomialRenderer":
        plot_partial_polynomials(raw_outputs[0][i, ..., 5, 5])
    elif cfg.renderer == "PolynomialDegreeRenderer":
        plot_partial_polynomials_degree(raw_outputs[0][i, ..., 5, 5], cfg.k)
