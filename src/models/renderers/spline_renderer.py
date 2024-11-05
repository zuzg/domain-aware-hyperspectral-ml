from torch import nn, Tensor

from src.models.renderers.base_renderer import BaseRenderer


class SplineRenderer(BaseRenderer):
    def __init__(self, device: str, channels: int) -> None:
        super().__init__(device, channels)

    def __call__(self, batch: Tensor) -> Tensor:
        splines = self.generate_spline(batch[:, :, 0, ...])
        rendered = splines.permute(0, 2, 1, 3)
        return rendered.to(self.device)

    def generate_spline(self, points: Tensor) -> Tensor:
        points = points.permute(0, 2, 1, 3)
        return nn.functional.interpolate(input=points, size=[self.channels, points.shape[-1]], mode="bicubic")
