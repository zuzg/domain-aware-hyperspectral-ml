from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.eval.visualizations import (
    plot_average_reflectance,
    plot_images,
    plot_partial_hats,
    plot_partial_polynomials,
    plot_partial_polynomials_degree,
    plot_pixelwise,
)
from src.models.bias_variance_model import BiasVarianceModel


def evaluate(
    model: BiasVarianceModel,  testloader: DataLoader, cfg: ExperimentConfig
):
    variance_model = model.variance
    bias_model = model.bias
    variance_model.eval()
    if bias_model:
        bias_model.eval()
    imgs = []
    raw_outputs = []
    renders = []
    for img in testloader:
        out = variance_model.modeller(img.to(cfg.device))
        rendered = variance_model.renderer(out)
        if bias_model:
            rendered += bias_model(0)
        imgs.append(img)
        raw_outputs.append(out)
        renders.append(rendered)

    i = 1
    gt_img = imgs[0][i]
    pred_img = renders[0][i].cpu().detach().numpy()
    plot_images(gt_img, pred_img)
    plot_average_reflectance(gt_img, pred_img)
    plot_pixelwise(gt_img, pred_img, cfg.img_size)
    if cfg.variance_renderer == "GaussianRenderer":
        plot_partial_hats(raw_outputs[0][i, ..., 5, 5])
    elif cfg.variance_renderer == "PolynomialRenderer":
        plot_partial_polynomials(raw_outputs[0][i, ..., 5, 5])
    elif cfg.variance_renderer == "PolynomialDegreeRenderer":
        plot_partial_polynomials_degree(raw_outputs[0][i, ..., 5, 5], cfg.k)
