import torch
from torch import nn, Tensor


class MLPBasedRenderer(nn.Module):
    def __init__(self, channels: int, param_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.param_dim = param_dim
        self.channels = channels

        # f(x, params) -> scalar output
        self.mlp = nn.Sequential(
            nn.Linear(param_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Tensor) -> Tensor:
        """
        batch: (batch_size, param_dim, h, w)
        Output: (batch_size, c, h, w)
        """
        batch_size, param_dim, h, w = batch.shape
        device = batch.device

        x = torch.linspace(1, self.channels, self.channels, device=device)  # (c,)
        x = x.view(1, self.channels, 1, 1, 1)  # (1, c, 1, 1, 1)
        x = x.expand(batch_size, -1, -1, h, w)  # (batch_size, c, 1, h, w)

        params = batch.unsqueeze(1).expand(-1, self.channels, -1, -1, -1)  # (batch_size, c, num_params, h, w)

        inputs = torch.cat([params, x], dim=2)  # (batch_size, c, num_params+1, h, w)
        inputs = inputs.permute(0, 3, 4, 1, 2).reshape(-1, self.param_dim + 1)  # (batch_size * c * h * w, num_params+1)

        output = self.mlp(inputs)  # (batch_size * c * h * w, 1)
        output = output.view(batch_size, h, w, self.channels).permute(0, 3, 1, 2)  # (batch_size, c, h, w)

        return output
