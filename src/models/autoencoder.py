from torch import nn, Tensor

from src.models.modeller import Modeller


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
        super(Decoder, self).__init__()
        self.in_channels = num_params * k  # Should match encoder output

        self.deconv1 = nn.Conv2d(self.in_channels, 256, kernel_size=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.deconv2 = nn.Conv2d(256, 512, kernel_size=1)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.deconv3 = nn.Conv2d(512, 1024, kernel_size=1)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.deconv4 = nn.Conv2d(1024, 512, kernel_size=1)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.deconv5 = nn.Conv2d(512, channels, kernel_size=1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.lrelu1(self.deconv1(x))
        x = self.lrelu2(self.deconv2(x))
        x = self.lrelu3(self.deconv3(x))
        x = self.lrelu4(self.deconv4(x))
        x = self.deconv5(x)
        x = self.output_activation(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, channels: int, k: int, num_params: int):
        super().__init__()
        self.k = k
        self.num_params = num_params
        self.encoder = Modeller(channels, k, num_params)
        self.decoder = Decoder(channels, k, num_params)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = x.view(x.shape[0], self.k * self.num_params, x.shape[-2], x.shape[-1])
        x = self.decoder(x)
        return x
