import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, channels: int, k: int, num_params: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, num_params * k, kernel_size=1)
        self.float()

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, channels: int, k: int, num_params: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.deconv1 = nn.Conv2d(num_params * k, 64, kernel_size=1)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.deconv3 = nn.Conv2d(64, channels, kernel_size=1)
        self.output_activation = nn.Tanh()
        self.float()

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.output_activation(self.deconv3(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, channels: int, k: int, num_params: int):
        super().__init__()
        self.num_params = num_params
        self.encoder = Encoder(channels, k, num_params)
        self.decoder = Decoder(channels, k, num_params)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
