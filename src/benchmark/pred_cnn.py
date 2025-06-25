import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.benchmark.utils import filter_background


class Spectral1DNet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(Spectral1DNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SoilMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(SoilMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # raw logits


def predict_soil_classes_cnn(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    device: str,
    epochs: int = 70,
    batch_size: int = 16,
) -> tuple[float]:
    x_train, y_train = filter_background(x_train, y_train)
    seen_classes = np.unique(y_train)
    test_mask = np.isin(y_test, seen_classes)
    y_test_filtered = y_test[test_mask]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test_filtered)
    num_classes = len(le.classes_)

    y_test[test_mask] = y_test_enc
    y_test[~test_mask] = -1

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size)

    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_enc), y=y_train_enc)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model = Spectral1DNet(x_train_tensor.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())

    acc = accuracy_score(all_targets, all_preds)
    avg_acc = recall_score(all_targets, all_preds, average="macro")
    return acc, avg_acc
