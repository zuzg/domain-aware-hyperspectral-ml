import torch
from torch import Tensor

from src.models.renderers.base_renderer import BaseRenderer


class PolynomialDegreeRenderer(BaseRenderer):
    def __init__(self, device: str, channels: int, mod: str) -> None:
        super().__init__(device, channels)

    def __call__(self, batch: Tensor) -> Tensor:
        batch_size, k, params, self.h, self.w = batch.shape
        self.bands = (
            torch.arange(1 / 100, (self.channels + 1) / 100, step=0.01, device=self.device).float().repeat(k, self.h, self.w, 1)
        )
        rendered_list = []

        for idx in range(batch_size):
            polynoms = self.generate_polynomial(batch[idx, :, 0, ...], k)
            pixel_dist = torch.sum(polynoms, dim=0)
            rendered_list.append(pixel_dist.permute(2, 0, 1))

        rendered = torch.stack(rendered_list)
        return rendered.to(self.device)

    def generate_polynomial(self, coefficient: Tensor, degree: int) -> Tensor:
        exponent = torch.arange(degree, device=self.device).repeat(self.h, self.w, self.channels, 1).permute(3, 0, 1, 2)
        return coefficient.unsqueeze(-1) * torch.pow(self.bands, exponent)
