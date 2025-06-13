from torch import nn, Tensor

from src.models.modeller import Modeller
from src.models.renderers.mlp_renderer import MLPBasedRenderer


class Decoder(nn.Module):
    def __init__(self, channels: int, param_dim: int):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(param_dim, param_dim // 2, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(param_dim // 2)
        self.conv2 = nn.Conv2d(param_dim // 2, param_dim // 4, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(param_dim // 4)
        self.conv3 = nn.Conv2d(param_dim // 4, param_dim // 4, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(param_dim // 4)
        self.conv4 = nn.Conv2d(param_dim // 4, 128, kernel_size=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, channels, kernel_size=1)
        self.output_activation = nn.Tanh()
        self.float()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(self.relu(self.conv1(x)))
        x = self.norm2(self.relu(self.conv2(x)))
        x = self.norm3(self.relu(self.conv3(x)))
        x = self.norm4(self.relu(self.conv4(x)))
        x = self.output_activation(self.conv5(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, channels: int, k: int, num_params: int, mlp: bool = True):
        super().__init__()
        self.k = k
        self.num_params = num_params
        self.encoder = Modeller(channels, k, num_params)
        self.decoder = MLPBasedRenderer(channels, k * num_params) if mlp else Decoder(channels, k * num_params)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = x.view(x.shape[0], self.k * self.num_params, x.shape[-2], x.shape[-1])
        x = self.decoder(x)
        return x
