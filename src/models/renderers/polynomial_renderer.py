import torch
from torch import Tensor

from src.models.renderers.base_renderer import BaseRenderer


class PolynomialRenderer(BaseRenderer):
    def __init__(self, device: str) -> None:
        super().__init__(device)

    def __call__(self, batch: Tensor) -> Tensor:
        batch_size, k, params, h, w = batch.shape
        self.bands = torch.arange(1, 151).float().repeat(k, h, w, 1).to(self.device)
        rendered = torch.zeros((batch_size, 150, h, w))

        for idx in range(batch_size):
            polynoms = self.generate_polynomial(batch[idx, :, 0, ...], batch[idx, :, 1, ...])
            pixel_dist = torch.sum(polynoms, dim=0)
            rendered[idx] = pixel_dist.permute(2, 0, 1)

        return rendered.to(self.device)

    def generate_polynomial(self, coefficient: Tensor, exponent: Tensor) -> Tensor:
        # two params: coeff * x^exponent
        return coefficient.unsqueeze(-1) * torch.pow(self.bands, exponent.unsqueeze(-1))
