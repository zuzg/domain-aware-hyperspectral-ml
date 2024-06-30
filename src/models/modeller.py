import torch.nn as nn
from torch import Tensor


class Modeller(nn.Module):
    def __init__(self, img_size: int, channels: int, k: int, num_params: int):
        super(Modeller, self).__init__()
        self.k = k
        self.size = img_size
        self.num_params = num_params
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain("relu"))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain("relu"))
        self.conv4 = nn.Conv2d(64, self.num_params * self.k, kernel_size=1)
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain("sigmoid"))

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv4(x)
        x = x.view(x.shape[0], self.k, self.num_params, self.size, self.size)
        if self.num_params == 3:
            x[:, :, :2] = self.sigmoid(x[:, :, :2])
            x[:, :, 2] = self.tanh(x[:, :, 2]) 
        else:
            x = self.tanh(x)
        return x
