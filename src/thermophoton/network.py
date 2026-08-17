"""ThermoPhoton input sampler and Transformer DeepONet components."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate
from sklearn import gaussian_process as gp


def create_example_heat_source(grid_size: int = 120) -> np.ndarray:
    """Create the 3 x 3 MZI heater map used by the inference example."""
    axis = np.linspace(0, 1, grid_size)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    pattern = np.zeros_like(x)
    for center_x, center_y in (
        (0.10, 0.60),
        (0.35, 0.60),
        (0.375, 0.20),
        (0.625, 0.20),
        (0.65, 0.60),
        (0.90, 0.60),
    ):
        mask = (np.abs(x - center_x) <= 0.05) & (np.abs(y - center_y) <= 0.025)
        pattern[mask] = 1
    return pattern.ravel()


class GRF2D:
    """Mixture of Gaussian random fields and non-overlapping heater blocks."""

    def __init__(
        self,
        kernel: str = "RBF",
        length_scale: float = 0.2,
        N: int = 120,
        interp: str = "splinef2d",
        mean_power: float = 1.0,
        power_std: float = 0.5,
        sample_ratio: float = 0.75,
    ) -> None:
        self.N = N
        self.interp = interp
        self.mean_power = mean_power
        self.power_std = power_std
        self.sample_ratio = sample_ratio
        self.x = np.linspace(0, 1, num=N)
        self.y = np.linspace(0, 1, num=N)
        xv, yv = np.meshgrid(self.x, self.y, indexing="ij")
        self.points = np.column_stack((xv.ravel(), yv.ravel()))

        if kernel == "RBF":
            covariance = gp.kernels.RBF(length_scale=length_scale)(self.points)
        elif kernel == "AE":
            covariance = gp.kernels.Matern(length_scale=length_scale, nu=0.5)(
                self.points
            )
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        self.cholesky = np.linalg.cholesky(
            covariance + 1e-12 * np.eye(self.N**2)
        )

    def _rectangles(self) -> list[tuple[float, float, float, float]]:
        grid_resolution = 20
        occupied = np.zeros((grid_resolution, grid_resolution), dtype=bool)
        cell_size = 1.0 / grid_resolution
        width, height = 0.1, 0.05
        rectangles = []

        for _ in range(np.random.randint(5, 20)):
            for _ in range(50):
                center_x = np.random.uniform(width / 2, 1 - width / 2)
                center_y = np.random.uniform(height / 2, 1 - height / 2)
                x0 = max(0, int((center_x - width / 2) / cell_size))
                x1 = min(
                    grid_resolution, int((center_x + width / 2) / cell_size) + 1
                )
                y0 = max(0, int((center_y - height / 2) / cell_size))
                y1 = min(
                    grid_resolution, int((center_y + height / 2) / cell_size) + 1
                )
                if not np.any(occupied[x0:x1, y0:y1]):
                    occupied[x0:x1, y0:y1] = True
                    rectangles.append((center_x, center_y, width, height))
                    break
        return rectangles

    def random(self, size: int) -> np.ndarray:
        """Sample flattened heat-source fields."""
        gaussian_count = int(size * self.sample_ratio)
        rectangle_count = size - gaussian_count

        gaussian = np.random.randn(self.N**2, gaussian_count)
        gaussian = (self.cholesky @ gaussian).T
        gaussian = (gaussian - gaussian.mean(axis=1, keepdims=True)) / gaussian.std(
            axis=1, keepdims=True
        )
        gaussian = np.exp(self.mean_power + self.power_std * gaussian)
        gaussian = (gaussian - gaussian.min(axis=1, keepdims=True)) / (
            gaussian.max(axis=1, keepdims=True)
            - gaussian.min(axis=1, keepdims=True)
        )
        gaussian *= 2 * self.mean_power

        rectangles = np.zeros((rectangle_count, self.N**2))
        for index in range(rectangle_count):
            for center_x, center_y, width, height in self._rectangles():
                mask = (
                    (np.abs(self.points[:, 0] - center_x) <= width / 2)
                    & (np.abs(self.points[:, 1] - center_y) <= height / 2)
                )
                rectangles[index, mask] = 1

        samples = np.vstack((gaussian, rectangles))
        np.random.shuffle(samples)
        return samples

    def eval_one(self, feature: np.ndarray, point: np.ndarray) -> float:
        return interpolate.interpn(
            (self.x, self.y),
            feature.reshape(self.N, self.N),
            point,
            method=self.interp,
        )[0]

    def eval_batch(self, features: np.ndarray, points: np.ndarray) -> np.ndarray:
        fields = np.reshape(features, (-1, self.N, self.N))
        return np.asarray(
            [
                interpolate.interpn(
                    (self.x, self.y), field, points, method=self.interp
                )
                for field in fields
            ],
            dtype=np.float32,
        )


def _initialize_parameters(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


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
        self.apply(_initialize_parameters)

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
        self.apply(_initialize_parameters)

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
