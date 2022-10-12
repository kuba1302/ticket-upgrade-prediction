import pandas as pd
import torch
import torch.nn as nn
from data_loader import UpgradeDataset

from ticket_upgrade_prediction.models import BaseModel


class LinearReluModule(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size), nn.ReLU()
        )

    def formward(self, x):
        return self.block(x)


class Network(nn.Module, BaseModel):
    def __init__(
        self, input_size: int, hidden_layers_sizes: list[int]
    ) -> None:
        super().__init__()
        self.hidden_layers_sizes = hidden_layers_sizes
        self.layers = nn.Sequential(
            nn.Linear(input_size, self.hidden_layers_sizes[0]),
            self._get_hidden_layers(),
            nn.Linear(self.hidden_layers_sizes[-1], 1),
        )
        print(self.layers)

    def _get_hidden_layers(self):
        modules = []

        for idx in range(len(self.hidden_layers_sizes) - 1):
            module = LinearReluModule(
                input_size=self.hidden_layers_sizes[idx],
                output_size=self.hidden_layers_sizes[idx + 1],
            )
            modules.append(module)

        return nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)

    def predict_proba(self, X: pd.DataFrame) -> torch.Tensor:
        X_tensor = torch.from_numpy(X.values)
        network = nn.Sequential(self.layers, nn.Sigmoid())
        return network(X_tensor.float())

    def predict(self, X):
        threshold = 0.5
        proba = self.predict_proba(X=X)
        return torch.where(proba >= threshold, 1, 0).numpy()
