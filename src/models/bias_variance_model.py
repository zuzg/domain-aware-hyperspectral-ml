import torch
import torch.nn as nn
from torch import Tensor

from src.models.modeller import Modeller
from src.models.renderers.base_renderer import BaseRenderer


class BiasModel(nn.Module):
    def __init__(self, shape: tuple[int], renderer: BaseRenderer):
        super().__init__()
        self.W = nn.Parameter(torch.randn(shape), requires_grad=True)
        self.renderer = renderer

    def forward(self, x: Tensor) -> Tensor:
        x = self.renderer(self.W)
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
    def __init__(self, bias: BiasModel | None, variance: VarianceModel):
        super().__init__()
        self.bias = bias
        self.variance = variance

    def forward(self, x: Tensor):
        x = self.variance(x)
        if self.bias:
            x += self.bias(0)
        return x
