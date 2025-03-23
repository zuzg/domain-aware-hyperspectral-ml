from torch import Tensor, nn
from src.consts import GT_DIM
from src.models.modeller import Modeller


class MultiRegressionCNN(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = GT_DIM):
        """
        CNN for hyperspectral image regression, outputting n continuous values per pixel.
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.Sigmoid(),
            # nn.BatchNorm2d(64),
            # nn.Conv2d(64, 32, kernel_size=1),
            # nn.ReLU(),
        )
        self.output_layer = nn.Conv2d(32, output_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        output = self.output_layer(x)
        return output


class EndToEndModel(nn.Module):
    def __init__(self, modeller: Modeller, regressor: MultiRegressionCNN):
        super().__init__()
        self.modeller = modeller
        self.regressor = regressor

    def forward(self, x):
        features = self.modeller(x)
        fs = features.shape
        features = features.reshape(fs[0], fs[1] * fs[2], fs[3], fs[4])
        output = self.regressor(features)
        return output
