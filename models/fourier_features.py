"""Deterministic Fourier-feature encoding for RheolNet PINNs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatureEncoding(nn.Module):
    """Encode normalized coordinates with a fixed Gaussian projection.

    The projection matrix is generated once from ``seed`` and registered as a
    non-trainable buffer, so it moves with the module and is saved in its state
    dictionary without receiving optimizer updates.
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int = 20,
        frequency_scale: float = 1.0,
        seed: int = 20260918,
        enabled: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError("in_dim must be positive")
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        if frequency_scale <= 0:
            raise ValueError("frequency_scale must be positive")

        self.in_dim = int(in_dim)
        self.num_frequencies = int(num_frequencies)
        self.frequency_scale = float(frequency_scale)
        self.seed = int(seed)
        self.enabled = bool(enabled)
        self.include_input = bool(include_input)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        projection = self.frequency_scale * torch.randn(
            self.num_frequencies,
            self.in_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.register_buffer("projection", projection, persistent=True)

    @property
    def output_dim(self) -> int:
        if not self.enabled:
            return self.in_dim
        return 2 * self.num_frequencies + (self.in_dim if self.include_input else 0)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected final coordinate dimension {self.in_dim}, "
                f"received {coordinates.shape[-1]}"
            )
        if not self.enabled:
            return coordinates

        phase = 2.0 * math.pi * coordinates @ self.projection.T
        encoded = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        if self.include_input:
            encoded = torch.cat((coordinates, encoded), dim=-1)
        return encoded

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, num_frequencies={self.num_frequencies}, "
            f"frequency_scale={self.frequency_scale}, seed={self.seed}, "
            f"enabled={self.enabled}, include_input={self.include_input}"
        )
