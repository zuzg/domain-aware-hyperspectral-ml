import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, channels: int, k: int, num_params: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, 128, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1)
        self.norm4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, num_params * k, kernel_size=1)
        self.float()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(self.relu(self.conv1(x)))
        x = self.norm2(self.relu(self.conv2(x)))
        x = self.norm3(self.relu(self.conv3(x)))
        x = self.norm4(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        return x


class Decoder(nn.Module):
    def __init__(self, channels: int, k: int, num_params: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(num_params * k, 32, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1)
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
    def __init__(self, channels: int, k: int, num_params: int):
        super().__init__()
        self.num_params = num_params
        self.encoder = Encoder(channels, k, num_params)
        self.decoder = Decoder(channels, k, num_params)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
