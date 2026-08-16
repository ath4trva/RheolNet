import torch.nn as nn


class PINN(nn.Module):
    """
    Basic Physics-Informed Neural Network.

    Input:
        x, y

    Output:
        u, v, p
    """

    def __init__(
        self,
        in_dim=2,
        out_dim=3,
        hidden_layers=4,
        hidden_width=64,
    ):
        super().__init__()

        layers = [
            nn.Linear(in_dim, hidden_width),
            nn.Tanh(),
        ]

        for _ in range(hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_width, hidden_width),
                nn.Tanh(),
            ])

        layers.append(
            nn.Linear(hidden_width, out_dim)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)