import numpy as np
import torch
from torch import Tensor, nn


def calculate_mu(mu_group: Tensor) -> Tensor:
    print(mu_group.shape)
    channels_equal = torch.from_numpy(np.array([0, 50, 100, 150]))
    channels_equal_reshaped = channels_equal.view(1, 1, 4, 1, 1)

    # Perform element-wise multiplication and sum along the third dimension
    mu = (mu_group * channels_equal_reshaped).sum(dim=2, keepdim=True)
    print(mu.shape)
    return mu


class Modeller(nn.Module):
    def __init__(
        self,
        img_size: int,
        channels: int,
        k: int,
        num_params: int,
        multi_mu: bool = True,
    ):
        super().__init__()
        self.k = k
        self.size = img_size
        self.num_params = num_params
        self.multi_mu = multi_mu
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain("relu"))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain("relu"))

        if multi_mu:
            self.num_params += 3

        self.conv3 = nn.Conv2d(64, self.num_params * self.k, kernel_size=1)
        nn.init.xavier_uniform_(
            self.conv3.weight, gain=nn.init.calculate_gain("sigmoid")
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(x.shape[0], self.k, self.num_params, self.size, self.size)
        if self.num_params >= 3:
            if self.multi_mu:
                print(x.shape)
                x[:, :, 3:4] = calculate_mu(self.softmax(x[:, :, :4]))
                x[:, :, 4:5] = self.sigmoid(x[:, :, 4:5])  # std
                x[:, :, 5:] = self.tanh(x[:, :, 5:])  # scale, shift
                # delete first two dimensions
                x = x[:, :, 3:]
            else:
                x[:, :, :2] = self.sigmoid(x[:, :, :2])
                x[:, :, 2:] = self.tanh(x[:, :, 2:])
        else:
            x = self.tanh(x)
        return x
