import torch
from torch import Tensor
from torch.distributions.beta import Beta

from src.models.renderers.base_renderer import BaseRenderer


class BetaRenderer(BaseRenderer):
    def __init__(self, device: str, channels: int, mod: str) -> None:
        super().__init__(device, channels)

    def __call__(self, batch: Tensor) -> Tensor:
        batch_size, k, params, h, w = batch.shape
        self.bands = torch.arange(1, self.channels + 1).float().repeat(k, h, w, 1).to(
            self.device
        ) / (self.channels + 1)
        rendered_list = []

        for idx in range(batch_size):
            dists = self.generate_distribution(
                batch[idx, :, 0, ...], batch[idx, :, 1, ...]
            )
            pixel_dist = torch.sum(dists, dim=0)
            rendered_list.append(pixel_dist.permute(2, 0, 1) + batch[idx, 0, 2, ...])

        rendered = torch.stack(rendered_list)
        return rendered.to(self.device)

    def generate_distribution(self, alpha: Tensor, beta: Tensor) -> Tensor:
        eps = 1e-5
        alpha = torch.add(alpha, eps)
        beta = torch.add(beta, eps)
        beta_dist = Beta(alpha.unsqueeze(-1), beta.unsqueeze(-1))
        return torch.exp(beta_dist.log_prob(self.bands))
