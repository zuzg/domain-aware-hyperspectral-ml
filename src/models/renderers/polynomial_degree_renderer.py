import torch
from torch import Tensor

from src.models.renderers.base_renderer import BaseRenderer


class PolynomialDegreeRenderer(BaseRenderer):
    def __init__(self, device: str, channels: int) -> None:
        super().__init__(device, channels)

    def __call__(self, batch: Tensor) -> Tensor:
        batch_size, k, params, h, w = batch.shape
        self.bands = (
            torch.arange(1 / 100, (self.channels + 1) / 100, step=0.01, device=self.device).float().repeat(k, h, w, 1)
        )
        rendered = torch.zeros((batch_size, self.channels, h, w))

        for idx in range(batch_size):
            polynoms = self.generate_polynomial(batch[idx, :, 0, ...], k)
            pixel_dist = torch.sum(polynoms, dim=0)
            rendered[idx] = pixel_dist.permute(2, 0, 1)

        return rendered.to(self.device)

    def generate_polynomial(self, coefficient: Tensor, degree: int) -> Tensor:
        exponent = torch.arange(degree, device=self.device).repeat(10, 10, self.channels, 1).permute(3, 0, 1, 2)
        return coefficient.unsqueeze(-1) * torch.pow(self.bands, exponent)
