import torch
from torch import Tensor
from src.models.renderers.base_renderer import BaseRenderer


class SplineRenderer(BaseRenderer):
    def __init__(self, device: str) -> None:
        super().__init__(device)

    def __call__(self, batch: Tensor) -> Tensor:
        batch_size, k, params, h, w = batch.shape
        self.bands = torch.arange(150).float().repeat(k, h, w, 1).to(self.device)
        rendered = torch.zeros((batch_size, 150, h, w))

        for idx in range(batch_size):
            splines = self.generate_spline(batch[idx, :, 0, ...], batch[idx, :, 1, ...])
            pixel_dist = torch.sum(splines, dim=0)
            rendered[idx] = pixel_dist.permute(2, 0, 1)# + batch[idx, 0, 2, ...]

        return rendered.to(self.device)

    def generate_spline(self, a: Tensor, b: Tensor) -> Tensor:
        knots = torch.linspace(0, 1, steps=a.shape[-1], device=self.device)
        t = self.bands.unsqueeze(-1)
        knots = knots.unsqueeze(0).expand_as(t)
        a = a.unsqueeze(-1).expand_as(t)
        b = b.unsqueeze(-1).expand_as(t)
        mask = (t >= knots) & (t < knots + 1)
        t = (t - knots) * mask.float()
        return a * (1 - t**3) + b * t**3
