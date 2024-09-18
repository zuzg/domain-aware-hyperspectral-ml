import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.models.modeller import Modeller
from src.models.renderers.base_renderer import BaseRenderer


class BiasModel(nn.Module):
    def __init__(self, shape: tuple[int], batch_size: int, img_size: int, renderer: BaseRenderer, device: str):
        super().__init__()
        rand_init = torch.FloatTensor(shape).uniform_(-1, 1)
        rand_init = rand_init.to(device)
        self.W = nn.Parameter(rand_init, requires_grad=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.renderer = renderer

    def forward(self, x: int) -> Tensor:
        x = self.W.unsqueeze(0)
        x = x.repeat(self.batch_size, 1, 1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.repeat(1, 1, 1, self.img_size, self.img_size)
        x = self.renderer(x)
        return x


class VarianceModel(nn.Module):
    def __init__(self, modeller: Modeller, renderer: BaseRenderer):
        super().__init__()
        self.modeller = modeller
        self.renderer = renderer

    def forward(self, x: Tensor):
        x = self.modeller(x)
        x = self.renderer(x)
        return x


class BiasVarianceModel(nn.Module):
    def __init__(self, bias: BiasModel | np.ndarray | None, variance: VarianceModel):
        super().__init__()
        self.bias = bias
        self.variance = variance

    def forward(self, x: Tensor):
        x = self.variance(x)
        if self.bias is not None and isinstance(self.bias, BiasModel):
            x += self.bias(0)
        elif self.bias is not None:
            x += self.bias
        return x
