"""Train or evaluate the ThermoPhoton Transformer on a 3 x 3 MZI case."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import deepheat as dde
import torch

from thermophoton.network import (
    GRF2D,
    create_case_pattern,
    get_branch_model,
    get_track_model,
)


class ThermalProblem3D:
    """3D Thermal Conduction Problem with configurable boundary conditions"""

    def __init__(self, config=None):
        self.default_config = {
            "geometry": {"size": [1, 1, 1]},  # Unit: mm
            "boundary_conditions": {
                "front": {"type": "convection", "h": 0.5},
                "back": {"type": "convection", "h": 0.5},
                "left": {"type": "convection", "h": 0.5},
                "right": {"type": "convection", "h": 0.5},
                "top": {"type": "convection", "h": 0.5},  # h in mW/(mm²·K)
                "bottom": {"type": "convection", "h": 0.5},
            },
            "material": {
                "conductivity": 1.4,  # k in mW/(mm·K)
                "source_scale": 4e3  # Heat source scaling factor
            },
            "source": {
                "height": 0.5,  # z-position (mm)
                "thickness": 0.05  # mm
            },
            "ambient_temp": 25  # °C
        }

        self.config = self.default_config.copy()
        if config:
            self.config.update(config)

        self.geometry = dde.geometry.Cuboid(
            [0, 0, 0],
            self.config["geometry"]["size"]
        )
        self._setup_pde()
        self._setup_boundary_conditions()

    def _setup_pde(self):
        """Define the governing PDE"""
        k = self.config["material"]["conductivity"]
        source_scale = self.config["material"]["source_scale"]
        z0 = self.config["source"]["height"]
        W = self.config["source"]["thickness"]
        alpha = 4 / W ** 2  # Heat source decay factor

        def pde(_, T, heat_source, input):
            grad_u = dde.zcs.LazyGrad(_, T)
            T_xx = grad_u.compute((2, 0, 0))  # ∂²u/∂x²
            T_yy = grad_u.compute((0, 2, 0))  # ∂²u/∂y²
            T_zz = grad_u.compute((0, 0, 2))  # ∂²u/∂z²
            laplacian = T_xx + T_yy + T_zz
            z = input[:, 2:3]
            weight = 1 / (1 + alpha * (z - z0) ** 2)/1.2122
            batch_size = laplacian.size(0)
            weight = weight.T.repeat(batch_size, 1)
            source_term = source_scale * heat_source * weight

            return k * laplacian + source_term

        self.pde = pde

    def _setup_boundary_conditions(self):
        """Configure boundary conditions"""
        self.bcs = []
        size = self.config["geometry"]["size"]
        tol = 1e-5

        for boundary, config in self.config["boundary_conditions"].items():
            if boundary == "front":
                on_boundary = lambda x, _: np.abs(x[1]) < tol
            elif boundary == "back":
                on_boundary = lambda x, _: np.abs(x[1] - size[1]) < tol
            elif boundary == "left":
                on_boundary = lambda x, _: np.abs(x[0]) < tol
            elif boundary == "right":
                on_boundary = lambda x, _: np.abs(x[0] - size[0]) < tol
            elif boundary == "bottom":
                on_boundary = lambda x, _: np.abs(x[2]) < tol
            elif boundary == "top":
                on_boundary = lambda x, _: np.abs(x[2] - size[2]) < tol
            else:
                continue

            # Create BC based on type
            if config["type"] == "adiabatic":
                self.bcs.append(dde.NeumannBC(self.geometry, lambda _: 0, on_boundary))
            elif config["type"] == "convection":
                h = config["h"] / self.config["material"]["conductivity"]
                robin_bc = lambda x, T: -h * (T - self.config["ambient_temp"])
                self.bcs.append(dde.RobinBC(self.geometry, robin_bc, on_boundary))


class DeepONetSolver:
    """DeepONet-based solver for parametric PDE problems"""

    def __init__(self, problem, path, grid_size=20):
        self.problem = problem
        self.grid_size = grid_size
        self.model_path = path
        self.net = None
        self.model = None
        self._init_function_space()
        self._init_data()

    def _init_function_space(self):
        """Initialize input function space"""
        self.func_space = GRF2D(
            kernel="RBF",
            length_scale=0.2,
            N=self.grid_size,
            sample_ratio=0.75,
        )

        # Create evaluation grid
        x = y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y, indexing="ij")
        self.eval_points = np.vstack([X.ravel(), Y.ravel()]).T

    def _init_data(self):
        """Initialize PDE operator dataset"""

        pde_data = dde.data.PDE(
            self.problem.geometry,
            self.problem.pde,
            self.problem.bcs,
            num_domain=30000,
            num_boundary=5000,
        )

        self.data = dde.zcs.PDEOperatorCartesianProd(
            pde_data,
            self.func_space,
            self.eval_points,
            2000, function_variables=[0, 1],
            num_test=8,
            batch_size=20
        )

    def build_network(self):
        # Select branch network using factory function
        branch_net = get_branch_model(256, self.grid_size)
        truck_net = get_track_model("ResFC", output_dim=256, input_dim=3)
        self.net = dde.nn.DeepONetCartesianProd(
            (self.grid_size ** 2, branch_net),
            #(21,truck_net),
            (3, truck_net),
            {"branch": "tanh", "trunk": "gelu"},
            kernel_initializer="Glorot normal"
        )
        self.model = dde.zcs.Model(self.data, self.net)

    def train(self, epochs=1000):
        """Train the DeepONet model"""
        self.model.compile(
            optimizer="adam",
            lr=1e-3,
        )
        self.model.train(iterations=20000, display_every=1000)
        self.model.compile(
            optimizer="adamw",
            lr=1e-4,
            loss_weights=[1, 10, 10, 10, 10, 10, 10]
        )
        callbacks = [
            dde.callbacks.PDEPointResampler(
                period=4000,
                pde_points=True,
                bc_points=False,
            ),
        ]
        return   self.model.train(iterations=epochs, display_every=1000, callbacks=callbacks)


    def predict(self, heat_source_pattern, spatial_points):
        """
        Predict temperature distribution for given heat source pattern

        Args:
            heat_source_pattern: 2D array of shape (grid_size, grid_size)
            spatial_points: Array of (x,y,z) coordinates for prediction

        Returns:
            Predicted temperatures at spatial_points
        """
        branch_input = heat_source_pattern.reshape(1, -1)
        return self.model.predict((branch_input, spatial_points))

    def save(self, path=None):
        """Save trained model weights"""
        path = path or self.model_path

        if os.path.exists(path):
            base, ext = os.path.splitext(path)
            new_path = f"{base}_1{ext}"
        else:
            new_path = path

        Path(new_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), new_path)
        print(f"Model saved to {new_path}")

    def load(self, path=None):
        """Load pretrained model weights"""
        self.model.compile(
            optimizer="adam",
            lr=1,
        )
        path = path or self.model_path
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        self.net.load_state_dict(state, strict=True)
        print(f"Model loaded from {path}")


def visualize_results(solver, heat_source, output_dir):
    """Save the heat-source map and three temperature slices."""
    grid_size = solver.grid_size
    heat_source_2d = heat_source.reshape(grid_size, grid_size)
    x = y = z = np.linspace(0, 1, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    temps = solver.predict(heat_source, points).reshape(grid_size, grid_size, grid_size)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    image = axes[0].imshow(
        heat_source_2d.T, origin="lower", extent=[0, 1, 0, 1], cmap="Reds"
    )
    figure.colorbar(image, ax=axes[0], label="Heat-source intensity")
    axes[0].set_title("3 x 3 MZI heaters")

    for axis, z_level in zip(axes[1:], (0.25, 0.5, 0.75)):
        z_idx = int(z_level * (grid_size - 1))
        temp_slice = temps[:, :, z_idx].T
        image = axis.imshow(
            temp_slice, origin="lower", extent=[0, 1, 0, 1], cmap="viridis"
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


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate ThermoPhoton")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/thermophoton.pt"),
    )
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=80000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--train-new-model", action="store_true")
    args = parser.parse_args()

    TRAIN_NEW_MODEL = args.train_new_model
    GRID_SIZE = args.grid_size
    EPOCHS = args.epochs
    path = str(args.checkpoint)
    # dde.optimizers.config.set_LBFGS_options(maxcor=20, ftol=1e-5, gtol=1e-06, maxiter=26000, maxfun=None, maxls=25)
    problem = ThermalProblem3D()
    solver = DeepONetSolver(problem, path, GRID_SIZE)
    solver.build_network()

    # Model loading/training
    if os.path.exists(solver.model_path) and not TRAIN_NEW_MODEL:
        solver.load()
    else:
        print("Training new model...")
        losshistory, train_state = solver.train(epochs=EPOCHS)
        solver.save()
        dde.utils.plot_loss_history(losshistory)

    # Generate sample patterns and visualize
    visualize_results(solver, create_case_pattern("3x3_mzi", GRID_SIZE), args.output_dir)


if __name__ == "__main__":
    main()
