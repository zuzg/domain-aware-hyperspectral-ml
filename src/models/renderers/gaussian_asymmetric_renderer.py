import torch
from torch import Tensor
from torch.distributions.normal import Normal

from src.models.renderers.base_renderer import BaseRenderer


class GaussianAsymmetricRenderer(BaseRenderer):
    def __init__(self, device: str, channels: int, mu_type: str) -> None:
        super().__init__(device, channels)
        self.mu_type = mu_type

    def __call__(self, batch: Tensor) -> Tensor:
        batch_size, k, params, h, w = batch.shape
        self.bands = (
            torch.arange(1, self.channels + 1, device=self.device)
            .float()
            .repeat(k, h, w, 1)
        )
        rendered_list = []

        for idx in range(batch_size):
            dists = self.generate_distribution(
                batch[idx, :, 0, ...],  # mu
                batch[idx, :, 1, ...],  # sigma_left
                batch[idx, :, 2, ...],  # sigma_right
                batch[idx, :, 3, ...],  # scale
            )
            pixel_dist = torch.sum(dists, dim=0)
            rendered_frame = (
                pixel_dist.permute(2, 0, 1) + batch[idx, 0, 4, ...]
            )  # shift
            rendered_list.append(rendered_frame)

        rendered = torch.stack(rendered_list)
        return rendered.to(self.device)

    def generate_distribution(
        self, mu: Tensor, sigma_left: Tensor, sigma_right: Tensor, scale: Tensor
    ) -> Tensor:
        eps = 1e-4
        sigma_left = torch.add(sigma_left, eps)
        sigma_right = torch.add(sigma_right, eps)

        mu_transformed = self.channels * mu.unsqueeze(-1)

        left_mask = self.bands <= mu_transformed
        right_mask = self.bands > mu_transformed

        # Create separate distributions for the left and right sides
        normal_dist_left = Normal(
            mu_transformed, self.channels * sigma_left.unsqueeze(-1)
        )
        normal_dist_right = Normal(
            mu_transformed, self.channels * sigma_right.unsqueeze(-1)
        )

        # Compute probabilities for each side
        prob_left = torch.exp(normal_dist_left.log_prob(self.bands)) * left_mask
        prob_right = torch.exp(normal_dist_right.log_prob(self.bands)) * right_mask

        # Combine probabilities and scale them
        asym_dist = self.channels * scale.unsqueeze(-1) * (prob_left + prob_right)
        return asym_dist
