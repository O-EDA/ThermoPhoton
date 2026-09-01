import os
import numpy as np
import matplotlib.pyplot as plt
import deepheat as dde

from deepheat.data.function_spaces import FunctionSpace
from deepheat.data import PDEOperatorCartesianProd
from scipy import interpolate
from sklearn import gaussian_process as gp

import torch
import torch.nn as nn

from branch_networks import get_branch_model,GRF2D,get_track_model


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
                "source_scale": 4e3/1.1071  # Heat source scaling factor
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
            weight = 1 / (1 + alpha * (z - z0) ** 2)/1.21
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
            sample_ratio=0.75
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

    def input_transform(self, X):
        sin = torch.sin(2 * torch.pi * X)
        cos = torch.cos(2 * torch.pi * X)
        sin2 = torch.sin(2 * 2 * torch.pi * X)
        cos2 = torch.cos(2 * 2 * torch.pi * X)
        sin3 = torch.sin(3 * 2 * torch.pi * X)
        cos3 = torch.cos(3 * 2 * torch.pi * X)
        final_input = torch.cat((sin, cos, sin2, cos2, sin3, cos3), dim=1)
        final_input = torch.cat((X, final_input), dim=1)
        return final_input

    def build_network(self):
        # Select branch network using factory function
        branch_net = get_branch_model('Transformer', 256, self.grid_size )
        truck_net = get_track_model("ResFC", output_dim=256, input_dim=3)
        self.net = dde.nn.DeepONetCartesianProd(
            (self.grid_size ** 2, branch_net),
            #(21,truck_net),
            (3, truck_net),
            {"branch": "tanh", "trunk": "gelu"},
            kernel_initializer="Glorot normal"
        )
        #self.net.apply_feature_transform(self.input_transform)
        self.model = dde.zcs.Model(self.data, self.net)

    def train(self, epochs=1000, lr=1e-6):
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

        torch.save(self.net.state_dict(), new_path)
        print(f"Model saved to {new_path}")

    def load(self, path=None):
        """Load pretrained model weights"""
        self.model.compile(
            optimizer="adam",
            lr=1,
        )
        path = path or self.model_path
        self.net.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


def create_heat_source_pattern(grid_size, pattern_type):
    """
    Generate 2D heat source patterns

    Args:
        grid_size: Resolution of the pattern
        pattern_type: One of ['diagonal', 'center', 'random', 'stripes']

    Returns:
        2D numpy array of shape (grid_size, grid_size)
    """
    # 统一使用相同尺寸的网格
    x = y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y, indexing="ij")

    if pattern_type == "random":
        grf = GRF2D(
            kernel="RBF",
            length_scale=0.2,
            N=grid_size
        )
        pattern = grf.random(1)[0]
    elif pattern_type == "center":
        # 高斯型中心热源
        x0, y0 = 0.5, 0.5  # 中心坐标
        sigma = 0.1  # 扩散系数
        dist = (X - x0) ** 2 + (Y - y0) ** 2
        pattern = np.ravel(np.exp(-dist / (2 * sigma)))
    elif pattern_type == "diagonal":
        # 对角线模式
        pattern = np.ravel(1 - np.abs(X - Y) ** 0.5)
    elif pattern_type == "stripes":
        # 条纹模式（水平方向）带平滑处理
        stripe_spacing = 0.5
        stripe_width = 0.05
        # 使用高斯函数进行平滑过渡
        x_phase = X / stripe_spacing
        distance = np.abs((x_phase % 1) - 0.5) * 2  # 转换为[0,1]区间
        sigma = 0.1  # 控制平滑程度
        pattern = np.exp(-(distance ** 2) / (2 * sigma ** 2))
    elif pattern_type == "blocks":
        # 六个矩形热源模式
        block_params = [
            (0.15, 0.25, 0.1, 0.05),
            (0.50, 0.25, 0.1, 0.05),
            (0.85, 0.25, 0.1, 0.05),
            (0.15, 0.75, 0.1, 0.05),
            (0.50, 0.75, 0.1, 0.05),
            (0.85, 0.75, 0.1, 0.05)
        ]
        pattern = np.zeros_like(X)

        for x0, y0, w, h in block_params:
            # 直接设置矩形区域为1
            mask = (np.abs(X - x0) <= w / 2) & (np.abs(Y - y0) <= h / 2)
            pattern[mask] = 1
    elif pattern_type == "MZI":
        # 六个矩形热源模式
        block_params = [
            (0.10, 0.6, 0.1, 0.05),
            (0.35, 0.60, 0.1, 0.05),
            (0.375, 0.2, 0.1, 0.05),
            (0.625, 0.2, 0.1, 0.05),
            (0.65, 0.60, 0.1, 0.05),
            (0.9, 0.6, 0.1, 0.05),
        ]
        # block_params = [
        #     (0.10, 0.35, 0.1, 0.05),
        #     (0.10, 0.50, 0.1, 0.05),
        #     (0.10, 0.65, 0.1, 0.05),
        #     (0.35, 0.35, 0.1, 0.05),
        #     (0.35, 0.50, 0.1, 0.05),
        #     (0.35, 0.65, 0.1, 0.05),
        #     (0.65, 0.20, 0.1, 0.05),
        #     (0.65, 0.40, 0.1, 0.05),
        #     (0.65, 0.60, 0.1, 0.05),
        #     (0.65, 0.80, 0.1, 0.05),
        #     (0.90, 0.20, 0.1, 0.05),
        #     (0.90, 0.40, 0.1, 0.05),
        #     (0.90, 0.60, 0.1, 0.05),
        #     (0.90, 0.80, 0.1, 0.05)
        # ]
        pattern = np.zeros_like(X)

        for x0, y0, w, h in block_params:
            # 直接设置矩形区域为1
            mask = (np.abs(X - x0) <= w / 2) & (np.abs(Y - y0) <= h / 2)
            pattern[mask] = 1
    else:
        raise ValueError(
            f"Unknown pattern type: {pattern_type}. Available: ['diagonal', 'center', 'random', 'stripes']")

    # 统一归一化到 [0, 2] 范围
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min()) * 1
    return pattern


def visualize_results(solver, heat_source, pattern_name, z_slices=[0.25, 0.5, 0.75]):
    """
    Visualize heat source pattern and predicted temperature distribution

    Args:
        solver: Trained DeepONetSolver instance
        heat_source: 2D heat source pattern
        pattern_name: Name for plot titles
        z_slices: Z-levels to visualize
    """
    grid_size = solver.grid_size
    heat_source_2d = heat_source.reshape(grid_size, grid_size)

    plt.figure(figsize=(15, 5))

    # Plot heat source pattern
    plt.subplot(2, 2, 1)
    plt.imshow(heat_source_2d.T, origin="lower", extent=[0, 1, 0, 1], cmap="Reds")
    plt.colorbar(label="Heat Source Intensity")
    plt.title(f"{pattern_name}\nHeat Source Distribution")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    

    # Generate 3D grid for prediction
    grid_size = solver.grid_size
    x = y = z = np.linspace(0, 1, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Predict temperatures
    temps = solver.predict(heat_source, points).reshape(grid_size, grid_size, grid_size)
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    # Plot temperature slices
    for i, z_level in enumerate(z_slices, 2):
        plt.subplot(2, 2, i)
        z_idx = int(z_level * (grid_size - 1))
        temp_slice = temps[:, :, z_idx].T

        plt.imshow(temp_slice, origin="lower", extent=[0, 1, 0, 1],
                   cmap="viridis")
        plt.colorbar(label="Temperature (°C)")
        plt.title(f"Z = {z_level:.2f} mm\nTemp Range: {temp_slice.min():.1f}-{temp_slice.max():.1f}°C")
        plt.xlabel("X (mm)")
        x_coords = np.linspace(0, 1, grid_size)
        y_coords = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        data_to_save = np.column_stack((X.ravel(), Y.ravel(), temp_slice.ravel()))
        
        np.savetxt(
            os.path.join(save_dir, f"{pattern_name}_z{z_level:.2f}_data.txt"),
            data_to_save,
            header="X Y Temperature",
            fmt="%.6f"
        )

    plt.tight_layout()
    
    plt.savefig(
        os.path.join(save_dir, f"{pattern_name}_2d.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()
    fig_scatter = plt.figure(figsize=(12, 8))
    ax_scatter = fig_scatter.add_subplot(111, projection='3d')

    # 提取坐标和温度值
    X_pts = points[:, 0]
    Y_pts = points[:, 1]
    Z_pts = points[:, 2]
    T_vals = temps.ravel()

    scatter = ax_scatter.scatter3D(
        X_pts[::2], Y_pts[::2], Z_pts[::2],
        c=T_vals[::2],
        cmap='viridis',
        marker='.',
        alpha=0.6,
        s=30
    )

    # 设置坐标轴
    ax_scatter.set_xlabel('X (mm)')
    ax_scatter.set_ylabel('Y (mm)')
    ax_scatter.set_zlabel('Z (mm)')
    ax_scatter.set_title(f'3D Temperature Scatter - {pattern_name}')
     # 保存为.txt文件 (X,Y,Z,T格式)
    x = y = z = np.linspace(0, 1, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    temperature_data = np.column_stack((
        X.ravel(), 
        Y.ravel(), 
        Z.ravel(), 
        temps.ravel()
    ))
    np.savetxt(
        os.path.join(save_dir, f"{pattern_name}_3d_temperature.txt"),
        temperature_data,
        header="X Y Z Temperature",
        comments=""
    )
    cbar = fig_scatter.colorbar(scatter, ax=ax_scatter, pad=0.1)
    cbar.set_label('Temperature (°C)')
    plt.savefig(
        os.path.join(save_dir, f"{pattern_name}_3d.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()


def main():
    # Configuration
    import datetime
    print(f"程序开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    TRAIN_NEW_MODEL = True
    GRID_SIZE =120
    EPOCHS = 80000
    path = "thermophoton.pt"
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

    return

    # Generate sample patterns and visualize
    patterns = {
        "Random3": create_heat_source_pattern(GRID_SIZE, "blocks"),
        "3x3_MZI": create_heat_source_pattern(GRID_SIZE, "MZI")
    }

    for name, pattern in patterns.items():
        visualize_results(solver, pattern, name)


if __name__ == "__main__":
    main()
