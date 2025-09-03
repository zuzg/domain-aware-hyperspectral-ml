import torch
from torch import Tensor
from torch.distributions.normal import Normal

from src.models.renderers.base_renderer import BaseRenderer


class GaussianRenderer(BaseRenderer):
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
                batch[idx, :, 0, ...], batch[idx, :, 1, ...], batch[idx, :, 2, ...]
            )
            pixel_dist = torch.sum(dists, dim=0)
            rendered_frame = pixel_dist.permute(2, 0, 1) + batch[idx, 0, 3, ...]
            rendered_list.append(rendered_frame)

        rendered = torch.stack(rendered_list)
        return rendered.to(self.device)

    def generate_distribution(self, mu: Tensor, sigma: Tensor, scale: Tensor) -> Tensor:
        eps = 1e-4
        sigma = torch.add(sigma, eps)
        if self.mu_type == "unconstrained":
            mu_transformed = self.channels * mu.unsqueeze(-1)
        elif self.mu_type == "fixed_reference":
            mu_transformed = self.adjust_tensor_intervals(
                mu.unsqueeze(-1), self.channels
            )
        elif self.mu_type == "equal_interval":
            mu_transformed = self.fix_intervals(mu.unsqueeze(-1), self.channels).to(
                self.device
            )

        normal_dist = Normal(mu_transformed, self.channels * sigma.unsqueeze(-1))
        return (
            self.channels
            * scale.unsqueeze(-1)
            * torch.exp(normal_dist.log_prob(self.bands))
        )

    @staticmethod
    def adjust_tensor_intervals(tensor: Tensor, channels: int) -> Tensor:
        batch, k, h, w = tensor.shape
        interval = channels // k
        adjusted_tensor = torch.zeros_like(tensor)
        for i in range(k):
            adjusted_tensor[:, i, :, :] = tensor[:, i, :, :] * interval + (interval * i)
        return adjusted_tensor

    @staticmethod
    def fix_intervals(tensor: Tensor, channels: int) -> Tensor:
        batch, k, h, w = tensor.shape
        scale_factors = torch.tensor([channels / (k + 2) * i for i in range(1, k + 1)])
        scaled_tensor = scale_factors.view(1, k, 1, 1).expand(batch, k, h, w)
        return scaled_tensor
