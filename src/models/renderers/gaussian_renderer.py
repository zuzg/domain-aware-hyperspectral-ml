import torch
from torch import Tensor
from torch.distributions.normal import Normal

from src.models.renderers.base_renderer import BaseRenderer


class GaussianRenderer(BaseRenderer):
    def __init__(self, device: str) -> None:
        super().__init__(device)

    def __call__(self, batch: Tensor) -> Tensor:
        batch_size, k, params, h, w = batch.shape
        self.bands = torch.arange(1, 151).float().repeat(k, h, w, 1).to(self.device)
        rendered = torch.zeros((batch_size, 150, h, w))

        for idx in range(batch_size):
            dists = self.generate_distribution(batch[idx, :, 0, ...], batch[idx, :, 1, ...])
            pixel_dist = torch.sum(dists, dim=0)
            rendered[idx] = pixel_dist.permute(2, 0, 1) + batch[idx, 0, 2, ...]

        return rendered.to(self.device)

    def generate_distribution(self, mu: Tensor, sigma: Tensor) -> Tensor:
        eps = 1e-4
        sigma = torch.add(sigma, eps)
        normal_dist = Normal(150 * mu.unsqueeze(-1), 150 * sigma.unsqueeze(-1))
        return torch.exp(normal_dist.log_prob(self.bands))
