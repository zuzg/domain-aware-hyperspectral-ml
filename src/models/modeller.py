import numpy as np
import torch
from torch import Tensor, nn


def calculate_mu(mu_group: Tensor) -> Tensor:
    channels_equal = torch.from_numpy(np.array([1, 50, 100, 150])).to("cuda")
    channels_equal_reshaped = channels_equal.view(1, 1, 4, 1, 1)

    # Perform element-wise multiplication and sum along the third dimension
    mu = (mu_group * channels_equal_reshaped).sum(dim=2, keepdim=True)
    return mu / 150


class Modeller(nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        num_params: int,
        multi_mu: bool = False,
    ):
        super().__init__()
        self.k = k
        self.num_params = num_params
        self.multi_mu = multi_mu
        if multi_mu:
            self.num_params += 3
            self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(channels, 512, kernel_size=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.norm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=1)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.norm2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=1)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv5 = nn.Conv2d(256, self.num_params * self.k, kernel_size=1)
        # self.norm = nn.BatchNorm2d(self.num_params * self.k, momentum=0.1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain("leaky_relu"))

    def forward(self, x: Tensor) -> Tensor:
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))
        x = self.conv5(x)
        _, _, h, w = x.shape
        x = x.view(x.shape[0], self.k, self.num_params, h, w)  # Use h, w dynamically
        if self.num_params > 3:
            if self.multi_mu:
                x[:, :, 3:4] = calculate_mu(self.softmax(x[:, :, :4]))
                x[:, :, 4:5] = self.sigmoid(x[:, :, 4:5])  # std
                x[:, :, 5:] = self.tanh(x[:, :, 5:])  # scale, shift, skew
                x = x[:, :, 3:]  # delete first two dimensions
            else:
                x[:, :, :2] = self.sigmoid(x[:, :, :2])  # mu, std
                x[:, :, 2:] = self.tanh(x[:, :, 2:])  # scale, shift, skew
        else:
            x = self.tanh(x)  # TODO relu for beta dist
        return x
