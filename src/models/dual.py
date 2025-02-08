import torch
from torch import Tensor, nn

from src.models.modeller import Modeller
from src.models.renderers.base_renderer import BaseRenderer


def generate_latent(batch_size: int, img_size: int, k: int, num_params: int, device: str) -> Tensor:
    # Generate mu and std from a normal distribution, then normalize to [0, 1]
    first_two_dims = torch.randn(batch_size, k, 2, img_size, img_size, device=device)
    first_two_dims = (first_two_dims - first_two_dims.min()) / (first_two_dims.max() - first_two_dims.min())

    # Generate scale, shift, and skew from a normal distribution, then scale to [-1, 1]
    remaining_dims = torch.randn(batch_size, k, num_params - 2, img_size, img_size, device=device)
    remaining_dims = 2 * (remaining_dims - remaining_dims.min()) / (remaining_dims.max() - remaining_dims.min()) - 1

    latent_vectors = torch.cat([first_two_dims, remaining_dims], dim=2)
    return latent_vectors


class DualModeAutoencoder(nn.Module):
    def __init__(
        self,
        modeller: Modeller,
        renderer: BaseRenderer,
        batch_size: int,
        img_size: int,
        k: int,
        num_params: int,
    ):
        super().__init__()
        self.modeller = modeller  # Encoder
        self.renderer = renderer  # Decoder
        self.batch_size = batch_size
        self.img_size = img_size
        self.k = k
        self.num_params = num_params

    def forward(self, x: Tensor | None = None, mode: str = "classical"):
        """
        Forward pass for dual-mode autoencoder.

        Args:
            x (Tensor): Input images for classical mode.
            batch_size (int): Batch size for inverse mode.
            mode (str): "classical" or "inverse".

        Returns:
            - Classical Mode: (input, reconstructed images)
            - Inverse Mode: (latent vectors, reconstructed latents)
        """
        if mode == "classical":
            assert x is not None, "Classical mode requires an image input"
            latent = self.modeller(x)
            reconstructed_x = self.renderer(latent)
            return x, reconstructed_x  # Compare input images vs. reconstructed images

        elif mode == "inverse":
            latent = generate_latent(
                self.batch_size, self.img_size, self.k, self.num_params, next(self.parameters()).device
            )
            generated_images = self.renderer(latent)  # 16 150 100 100
            reconstructed_latents = self.modeller(generated_images)  # 16 5 5 100 100
            return latent, reconstructed_latents  # Compare input latent vs. reconstructed latent

        else:
            raise ValueError("Mode must be 'classical' or 'inverse'")
