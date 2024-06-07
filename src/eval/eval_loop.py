from torch.utils.data import DataLoader

from src.eval.visualizations import plot_average_reflectance, plot_images, plot_partial_hats, plot_pixelwise
from src.models.modeller import Modeller
from src.models.renderers.base_renderer import BaseRenderer


def evaluate(model: Modeller, renderer: BaseRenderer, testloader: DataLoader, device: str, size: int):
    imgs = []
    raw_outputs = []
    renders = []
    for img in testloader:
        out = model(img.to(device))
        rendered = renderer(out)
        imgs.append(img)
        raw_outputs.append(out)
        renders.append(rendered)

    i = 1
    gt_img = imgs[0][i]
    pred_img = renders[0][i].cpu().detach().numpy()
    plot_images(gt_img, pred_img)
    plot_average_reflectance(gt_img, pred_img)
    plot_partial_hats(raw_outputs[0][i, ..., 5, 5])
    plot_pixelwise(gt_img, pred_img, size)
