"""Configurable BloodFlowPINN architecture used by Task 1.3."""

from __future__ import annotations

import torch
import torch.nn as nn

from .fourier_features import FourierFeatureEncoding


class BloodFlowPINN(nn.Module):
    """Five-layer tanh PINN with optional fixed Fourier features.

    ``out_dim`` intentionally remains configurable. The documented four-output
    architecture is supported directly, while the validated 2D mixed formulation
    uses six outputs: ``u, v, p, tau_xx, tau_xy, tau_yy``.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 4,
        hidden_layers: int = 5,
        hidden_width: int = 128,
        fourier_enabled: bool = True,
        num_frequencies: int = 20,
        frequency_scale: float = 1.0,
        fourier_seed: int = 20260918,
        include_input: bool = True,
        zero_output: bool = True,
    ) -> None:
        super().__init__()
        if out_dim < 1 or hidden_layers < 1 or hidden_width < 1:
            raise ValueError("out_dim, hidden_layers and hidden_width must be positive")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_layers = int(hidden_layers)
        self.hidden_width = int(hidden_width)
        self.encoder = FourierFeatureEncoding(
            in_dim=self.in_dim,
            num_frequencies=num_frequencies,
            frequency_scale=frequency_scale,
            seed=fourier_seed,
            enabled=fourier_enabled,
            include_input=include_input,
        )

        layers: list[nn.Module] = []
        features = self.encoder.output_dim
        for _ in range(self.hidden_layers):
            layer = nn.Linear(features, self.hidden_width)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.Tanh()))
            features = self.hidden_width

        output = nn.Linear(features, self.out_dim)
        if zero_output:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        else:
            nn.init.xavier_normal_(output.weight)
            nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    @property
    def fourier_enabled(self) -> bool:
        return self.encoder.enabled

    def forward(self, normalized_coordinates: torch.Tensor) -> torch.Tensor:
        return self.network(self.encoder(normalized_coordinates))
