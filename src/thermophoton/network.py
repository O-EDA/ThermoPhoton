"""ThermoPhoton heater-map and inference-network components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_heat_source_from_config(
    config_path: Path, grid_size: int = 120
) -> np.ndarray:
    """Create a flattened heater map from a JSON configuration file."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    axis = np.linspace(0, 1, grid_size)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    pattern = np.zeros_like(x)
    for heater in config["heaters"]:
        shape = heater["shape"]
        center_x, center_y = heater["center"]
        if shape == "circle":
            mask = (
                (x - center_x) ** 2 + (y - center_y) ** 2
                <= heater["radius"] ** 2
            )
        elif shape == "rectangle":
            width, height = heater["size"]
            mask = (
                (np.abs(x - center_x) <= width / 2)
                & (np.abs(y - center_y) <= height / 2)
            )
        else:
            raise ValueError(f"Unsupported heater shape: {shape}")
        pattern[mask] = 1
    return pattern.ravel()


class _SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = inputs.shape
        weights = self.fc(self.pool(inputs).view(batch, channels))
        return inputs * weights.view(batch, channels, 1, 1)


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = _SqueezeExcitation(out_channels)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        outputs = self.se(self.bn2(self.conv2(outputs)))
        return F.relu(outputs + residual, inplace=True)


class _SelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4, head_dimension: int = 32):
        super().__init__()
        self.heads = heads
        self.head_dimension = head_dimension
        hidden_dimension = heads * head_dimension
        self.to_qkv = nn.Conv2d(channels, hidden_dimension * 3, 1, bias=False)
        self.proj_out = nn.Conv2d(hidden_dimension, channels, 1)
        self.scale = head_dimension**-0.5

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = inputs.shape
        qkv = self.to_qkv(inputs).view(
            batch, self.heads, 3, self.head_dimension, height * width
        )
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        query = query.transpose(-2, -1) * self.scale
        attention = F.softmax(torch.matmul(query, key), dim=-1)
        outputs = torch.matmul(attention, value.transpose(-2, -1))
        outputs = outputs.transpose(-2, -1).reshape(
            batch, self.heads * self.head_dimension, height, width
        )
        return self.proj_out(outputs)


class TransformerBranch(nn.Module):
    """ResNet-18 branch with four-head attention after stage two."""

    def __init__(self, output_dimension: int, grid_size: int) -> None:
        super().__init__()
        self.channels = 64
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_stage(64, blocks=2)
        self.layer2 = self._make_stage(128, blocks=2, stride=2)
        self.attn = _SelfAttention(128)
        self.layer3 = self._make_stage(256, blocks=2, stride=2)
        self.layer4 = self._make_stage(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, output_dimension)

    def _make_stage(self, channels: int, blocks: int, stride: int = 1) -> nn.Module:
        downsample = None
        if stride != 1 or self.channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers = [_ResidualBlock(self.channels, channels, stride, downsample)]
        self.channels = channels
        layers.extend(_ResidualBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.view(-1, 1, self.grid_size, self.grid_size)
        outputs = self.maxpool(self.relu(self.bn1(self.conv1(outputs))))
        outputs = self.layer2(self.layer1(outputs))
        outputs = self.layer4(self.layer3(self.attn(outputs)))
        return self.fc(torch.flatten(self.avgpool(outputs), 1))


class _ResidualFullyConnected(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dimension, dimension)
        self.fc2 = nn.Linear(dimension, dimension)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc2(torch.tanh(self.fc1(inputs))) + inputs)


class _ResidualTrunk(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(input_dimension, 128)
        self.act = nn.Tanh()
        self.net = nn.Sequential(*[_ResidualFullyConnected(128) for _ in range(4)])
        self.final_fc = nn.Linear(128, output_dimension)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.final_fc(self.net(self.act(self.fc_in(inputs))))


class FourierTrunk(nn.Module):
    """Fourier-feature trunk with four 128-wide residual blocks."""

    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        super().__init__()
        self.base_model = _ResidualTrunk(input_dimension * 7, output_dimension)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (
                inputs,
                torch.sin(2 * torch.pi * inputs),
                torch.cos(2 * torch.pi * inputs),
                torch.sin(4 * torch.pi * inputs),
                torch.cos(4 * torch.pi * inputs),
                torch.sin(6 * torch.pi * inputs),
                torch.cos(6 * torch.pi * inputs),
            ),
            dim=1,
        )
        return self.base_model(features)


def get_branch_model(output_dim: int, grid_size: int) -> TransformerBranch:
    return TransformerBranch(output_dim, grid_size)


def get_track_model(
    model_type: str, output_dim: int, input_dim: int
) -> FourierTrunk:
    if model_type != "ResFC":
        raise ValueError("ThermoPhoton only uses the ResFC trunk")
    return FourierTrunk(input_dim, output_dim)


class ThermoPhotonNet(nn.Module):
    """Inference-only Transformer DeepONet matching the released checkpoint."""

    def __init__(self, grid_size: int = 120, latent_dimension: int = 256) -> None:
        super().__init__()
        self.branch = get_branch_model(latent_dimension, grid_size)
        self.trunk = get_track_model(
            "ResFC", output_dim=latent_dimension, input_dim=3
        )
        self.b = nn.ParameterList([nn.Parameter(torch.tensor(0.0))])

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        heat_source, spatial_points = inputs
        branch_features = self.branch(heat_source)
        trunk_features = F.gelu(self.trunk(spatial_points))
        return torch.einsum("bi,ni->bn", branch_features, trunk_features) + self.b[0]
