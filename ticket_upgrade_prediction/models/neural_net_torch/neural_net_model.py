import torch.nn as nn


class LinearReluModule(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size), nn.ReLU()
        )

    def formward(self, x):
        return self.block(x)


class Network(nn.Module):
    def __init__(self, layers_sizes: list[int]) -> None:
        super().__init__()
        self.layers_sizes = layers_sizes
        self.layers = nn.Sequential(self._get_hidden_layers(), nn.Sigmoid())

    def _get_hidden_layers(self):
        modules = []

        for idx in range(len(self.layers_sizes) - 1):
            module = LinearReluModule(
                input_size=self.layers_sizes[idx],
                output_size=self.layers_sizes[idx + 1],
            )
            modules.append(module)

        return nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


# class NetworkWrapper:
#     def __init__(self, model) -> None:
#         self.model = model

#     def predict_proba(self, X):
#         return self.model(X)
