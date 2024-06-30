from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor


class BaseRenderer(ABC):
    """
    Abstract base class for renderers
    """
    def __init__(self, device: str, channels: int) -> None:
        self.device = device
        self.channels = channels

    @abstractmethod
    def __call__(self, batch: Tensor) -> Any:
        pass
