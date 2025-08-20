import torch
from torch import Tensor, nn


class Modeller(nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        num_params: int,
    ):
        super().__init__()
        self.k = k
        self.num_params = num_params

        self.conv1 = nn.Conv2d(channels, 512, kernel_size=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=1)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=1)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_third = nn.Conv2d(256, k, kernel_size=1)
        # Fixed learnable constants for other params
        self.fixed_params = nn.Parameter(torch.zeros(1, k, self.num_params, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # Init weights
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv_third]:
            nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain("leaky_relu"))

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape

        # Extract features
        feat = self.lrelu1(self.conv1(x))
        feat = self.lrelu2(self.conv2(feat))
        feat = self.lrelu3(self.conv3(feat))
        feat = self.lrelu4(self.conv4(feat))

        # Start with fixed parameters broadcasted
        out = self.fixed_params.expand(b, self.k, self.num_params, h, w).clone()

        # Replace only the third parameter with dynamic prediction
        third_param = self.conv_third(feat)  # (b, k, h, w)
        out[:, :, 2, :, :] = self.tanh(third_param)

        out[:, :, :2] = self.sigmoid(out[:, :, :2])  # mu, std
        out[:, :, 3:] = self.tanh(out[:, :, 3:])  # shift, skew

        return out
