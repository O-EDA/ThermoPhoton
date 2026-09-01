"""Run full-field ThermoPhoton inference for a heater configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from thermophoton.network import (
    ThermoPhotonNet,
    create_heat_source_from_config,
)


class ThermoPhotonSolver:
    """Inference wrapper for the released ThermoPhoton checkpoint."""

    def __init__(
        self, checkpoint: Path, grid_size: int = 120, device: str | None = None
    ) -> None:
        self.grid_size = grid_size
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.net = ThermoPhotonNet(grid_size=grid_size).to(self.device)

    def load(self) -> None:
        try:
            state = torch.load(
                self.checkpoint, map_location=self.device, weights_only=True
            )
        except TypeError:
            state = torch.load(self.checkpoint, map_location=self.device)
        self.net.load_state_dict(state, strict=True)
        self.net.eval()
        print(f"Model loaded from {self.checkpoint}")

    def predict(
        self, heat_source_pattern: np.ndarray, spatial_points: np.ndarray
    ) -> np.ndarray:
        branch_input = heat_source_pattern.reshape(1, -1)
        with torch.inference_mode():
            values = self.net(
                (
                    torch.as_tensor(
                        branch_input, dtype=torch.float32, device=self.device
                    ),
                    torch.as_tensor(
                        spatial_points, dtype=torch.float32, device=self.device
                    ),
                )
            )
        return values.detach().cpu().numpy()


def visualize_results(
    solver: ThermoPhotonSolver,
    heat_source: np.ndarray,
    case: str,
    output_dir: Path,
) -> None:
    """Save the heat-source map and three temperature slices."""
    grid_size = solver.grid_size
    heat_source_2d = heat_source.reshape(grid_size, grid_size)
    x = y = z = np.linspace(0, 1, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    temperatures = solver.predict(heat_source, points).reshape(
        grid_size, grid_size, grid_size
    )

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    image = axes[0].imshow(
        heat_source_2d.T, origin="lower", extent=[0, 1, 0, 1], cmap="Reds"
    )
    figure.colorbar(image, ax=axes[0], label="Heat-source intensity")
    axes[0].set_title(f"{case.replace('_', ' ')} heaters")

    for axis, z_level in zip(axes[1:], (0.25, 0.5, 0.75)):
        z_idx = int(z_level * (grid_size - 1))
        temperature_slice = temperatures[:, :, z_idx].T
        image = axis.imshow(
            temperature_slice,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="viridis",
        )
        figure.colorbar(image, ax=axis, label="Temperature (°C)")
        axis.set_title(f"z = {z_level:.2f} mm")

    for axis in axes:
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_dir / "temperature_slices.png", dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ThermoPhoton inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/thermophoton.pt"),
    )
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/3x3_mzi.json"),
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not args.config.is_file():
        parser.error(f"config not found: {args.config}")

    solver = ThermoPhotonSolver(args.checkpoint, args.grid_size)
    solver.load()
    heat_source = create_heat_source_from_config(args.config, args.grid_size)
    visualize_results(solver, heat_source, args.config.stem, args.output_dir)


if __name__ == "__main__":
    main()
