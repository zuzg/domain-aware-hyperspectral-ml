import torch
from torch import Tensor
from torch.distributions import Normal

from src.models.renderers.gaussian_renderer import GaussianRenderer


class GaussianSkewRenderer(GaussianRenderer):
    def __init__(self, device: str, channels: int, mu_type: str) -> None:
        super().__init__(device, channels, mu_type)

    def __call__(self, batch: Tensor) -> Tensor:
        batch_size, k, params, h, w = batch.shape
        self.bands = torch.arange(1, self.channels + 1, device=self.device).float().repeat(k, h, w, 1)
        rendered_list = []

        for idx in range(batch_size):
            dists = self.generate_distribution(
                batch[idx, :, 0, ...],  # mu
                batch[idx, :, 1, ...],  # sigma
                batch[idx, :, 2, ...],  # scale
                batch[idx, :, 3, ...],  # skew
            )
            pixel_dist = torch.sum(dists, dim=0)
            rendered_frame = pixel_dist.permute(2, 0, 1)  # + batch[idx, 0, 3, ...]  # Add bias
            rendered_list.append(rendered_frame)

        rendered = torch.stack(rendered_list)
        return rendered.to(self.device)

    def generate_distribution(self, mu: Tensor, sigma: Tensor, scale: Tensor, skew: Tensor) -> Tensor:
        if self.mu_type == "unconstrained":
            mu_transformed = self.channels * mu.unsqueeze(-1)
        elif self.mu_type == "fixed_reference":
            mu_transformed = self.adjust_tensor_intervals(mu.unsqueeze(-1), self.channels)
        elif self.mu_type == "equal_interval":
            mu_transformed = self.fix_intervals(mu.unsqueeze(-1), self.channels).to(self.device)

        skew = self.channels * skew.unsqueeze(-1)

        # Generate standard normal CDF and PDF
        normal_dist = Normal(mu_transformed, self.channels * sigma.unsqueeze(-1))  # sigma constrained - *15
        pdf = torch.exp(normal_dist.log_prob(self.bands))
        cdf = normal_dist.cdf(skew * (self.bands - mu_transformed))

        # Combine PDF and skew factor
        skew_normal_pdf = 2 * pdf * cdf
        return self.channels * scale.unsqueeze(-1) * skew_normal_pdf
