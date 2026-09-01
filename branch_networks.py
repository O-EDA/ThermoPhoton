import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate
from sklearn import gaussian_process as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# ------------------------------------------------------------------------------
# 统一权重初始化函数
# ------------------------------------------------------------------------------
def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# ------------------------------------------------------------------------------
# 1. GRF2D: 高斯随机场类
# ------------------------------------------------------------------------------
class GRF2D:
    """Gaussian Random Field (GRF) with positive values in [0,1]x[0,1]."""

    def __init__(self, kernel="RBF", length_scale=1.0, N=120, interp="splinef2d",
                 mean_power=1.0, power_std=0.5, sample_ratio=0.75, rect_config=None):
        self.N = N
        self.interp = interp
        self.mean_power = mean_power
        self.power_std = power_std
        self.sample_ratio = sample_ratio

        # 定义网格
        self.x = np.linspace(0, 1, num=N)
        self.y = np.linspace(0, 1, num=N)
        xv, yv = np.meshgrid(self.x, self.y, indexing="ij")
        self.X = np.vstack((np.ravel(xv), np.ravel(yv))).T

        self._init_kernel(kernel, length_scale)

        # 默认矩形配置
        self.rect_config = rect_config or {
            "grid_res": 20,
            "size": (0.1, 0.05),
            "num_range": (5, 20),
            "max_try": 50
        }

    def _init_kernel(self, kernel, length_scale):
        if kernel == "RBF":
            self.K = gp.kernels.RBF(length_scale=length_scale)(self.X)
        elif kernel == "AE":
            self.K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)(self.X)
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        # 加上微小扰动保证数值稳定性
        self.L = np.linalg.cholesky(self.K + 1e-12 * np.eye(self.N ** 2))

    def _generate_rectangles(self):
        """生成不重叠的矩形"""
        rects = []
        grid_res = self.rect_config['grid_res']
        grid_map = np.zeros((grid_res, grid_res), dtype=bool)
        cell_size = 1.0 / grid_res
        w, h = self.rect_config['size']
        num_rects = np.random.randint(*self.rect_config['num_range'])

        for _ in range(num_rects):
            for _ in range(self.rect_config['max_try']):
                cx = np.random.uniform(w / 2, 1 - w / 2)
                cy = np.random.uniform(h / 2, 1 - h / 2)
                xi = max(0, int((cx - w / 2) / cell_size))
                xj = min(grid_res, int((cx + w / 2) / cell_size) + 1)
                yi = max(0, int((cy - h / 2) / cell_size))
                yj = min(grid_res, int((cy + h / 2) / cell_size) + 1)
                if not np.any(grid_map[xi:xj, yi:yj]):
                    grid_map[xi:xj, yi:yj] = True
                    rects.append([cx, cy, w, h])
                    break
        return np.array(rects)

    def random(self, size):
        """
        生成包含高斯和矩形成分的随机样本
        参数:
            size: 生成样本总数
        返回:
            normalized_field: (size, N**2) 的归一化场数据
        """
        num_gaussian = int(size * self.sample_ratio)
        num_rect = size - num_gaussian

        # 生成高斯场
        u = np.random.randn(self.N ** 2, num_gaussian)
        base_field = np.dot(self.L, u).T
        base_field = (base_field - np.mean(base_field, axis=1, keepdims=True)) / \
                     np.std(base_field, axis=1, keepdims=True)
        positive_field = self.mean_power + self.power_std * base_field
        positive_field = np.exp(positive_field)
        positive_field = (positive_field - positive_field.min(axis=1, keepdims=True)) / \
                         (positive_field.max(axis=1, keepdims=True) - positive_field.min(axis=1, keepdims=True))
        gauss_part = positive_field * 2 * self.mean_power

        # 生成矩形部分
        rect_part = np.zeros((num_rect, self.N ** 2))
        # 对每个样本生成矩形
        for i, rects in enumerate([self._generate_rectangles() for _ in range(num_rect)]):
            for cx, cy, w, h in rects:
                mask = (np.abs(self.X[:, 0] - cx) <= w / 2) & (np.abs(self.X[:, 1] - cy) <= h / 2)
                rect_part[i, mask] = 1

        combined = np.vstack([gauss_part, rect_part])
        # 随机打乱行顺序
        np.random.shuffle(combined)
        return combined

    def _visualize_fields(self, gauss_field, rect_field):
        """显示高斯随机场和矩形热源的2D热图"""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        gauss_field_reshaped = gauss_field.reshape(self.N, self.N).T
        rect_field_reshaped = rect_field.reshape(self.N, self.N).T

        im1 = axes[0].imshow(gauss_field_reshaped, extent=[0, 1, 0, 1], origin="lower", cmap="viridis", aspect="auto")
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_title("Gaussian Random Field")

        im2 = axes[1].imshow(rect_field_reshaped, extent=[0, 1, 0, 1], origin="lower", cmap="gray_r", aspect="auto")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title("Rectangular Heat Sources")

        plt.tight_layout()
        plt.show()

    def eval_one(self, feature, x):
        """单点插值计算"""
        return interpolate.interpn(
            (self.x, self.y), feature.reshape(self.N, self.N), x, method=self.interp
        )[0]

    def eval_batch(self, features, xs):
        """批量插值计算"""
        ys = np.reshape(features, (-1, self.N, self.N))
        return np.array([
            interpolate.interpn((self.x, self.y), y, xs, method=self.interp)
            for y in ys
        ], dtype=np.float32)

# ------------------------------------------------------------------------------
# 2. SE_ResNetBranch 分支 (基于 SE 模块的 ResNet)
# ------------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SE_ResNetBranch(nn.Module):
    def __init__(self, output_dim, grid_size, input_channels=1):
        """
        一个基于 SE-ResNet 的分支模型。
        参数:
            output_dim: 输出维度
            grid_size: 网格尺寸（输入视为 grid_size x grid_size）
            input_channels: 输入通道数
        """
        super(SE_ResNetBranch, self).__init__()
        self.inplanes = 64
        self.grid_size = grid_size

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(SEBasicBlock, 64, blocks=2)
        self.layer2 = self._make_layer(SEBasicBlock, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(SEBasicBlock, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(SEBasicBlock, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, output_dim)

        # 统一初始化
        self.apply(initialize_parameters)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 假设输入形状为 (B, grid_size**2) -> reshape为 (B, 1, grid_size, grid_size)
        x = x.view(-1, 1, self.grid_size, self.grid_size)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# 3. ConvNeXtBranch 分支
# ------------------------------------------------------------------------------
class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(ConvNeXtBlock, self).__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                                 padding=kernel_size // 2, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)  # 在通道维度上归一化
        self.pw_conv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(torch.zeros(channels), requires_grad=True)

    def forward(self, x):
        residual = x
        x = self.dw_conv(x)  # (B, channels, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, channels)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, channels, H, W)
        x = residual + self.gamma.view(1, -1, 1, 1) * x
        return x

class ConvNeXtBranch(nn.Module):
    def __init__(self, output_dim, grid_size, input_channels=1):
        """
        基于 ConvNeXt 模块的分支模型。
        参数:
            output_dim: 输出维度
            grid_size: 输入图像边长（输入形状将 reshape 为 (B, 1, grid_size, grid_size)）
            input_channels: 输入通道数
        """
        super(ConvNeXtBranch, self).__init__()
        self.inplanes = 64
        self.grid_size = grid_size

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = nn.Sequential(
            ConvNeXtBlock(64),
            ConvNeXtBlock(64)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ConvNeXtBlock(128),
            ConvNeXtBlock(128)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ConvNeXtBlock(256),
            ConvNeXtBlock(256)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(256, output_dim)

        self.apply(initialize_parameters)

    def forward(self, x):
        # 将输入 reshape 为 (B, 1, grid_size, grid_size)
        x = x.view(-1, 1, self.grid_size, self.grid_size)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# 4. TransformerBranch 分支
# ------------------------------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    """
    通过卷积实现自注意力模块
    """

    def __init__(self, in_channels, heads=4, dim_head=32):
        super(SelfAttentionBlock, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(in_channels, hidden_dim * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.scale = dim_head ** -0.5

    def forward(self, x):
        B, _, H, W = x.shape
        qkv = self.to_qkv(x)  # (B, 3*hidden_dim, H, W)
        qkv = qkv.view(B, self.heads, 3, self.dim_head, H * W)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(-2, -1) * self.scale  # (B, heads, H*W, dim_head)
        attn = torch.matmul(q, k)  # (B, heads, H*W, H*W)
        attn = F.softmax(attn, dim=-1)
        v = v.transpose(-2, -1)
        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).reshape(B, self.heads * self.dim_head, H, W)
        return self.proj_out(out)

class TransformerBranch(nn.Module):
    def __init__(self, output_dim, grid_size, input_channels=1):
        """
        带有自注意力模块的 Transformer 分支
        参数:
            output_dim: 输出维度
            grid_size: 输入边长（输入 reshape 成 (B, 1, grid_size, grid_size)）
            input_channels: 输入通道
        """
        super(TransformerBranch, self).__init__()
        self.inplanes = 64
        self.grid_size = grid_size

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(SEBasicBlock, 64, blocks=2)
        self.layer2 = self._make_layer(SEBasicBlock, 128, blocks=2, stride=2)
        self.attn   = SelfAttentionBlock(128, heads=4, dim_head=32)
        self.layer3 = self._make_layer(SEBasicBlock, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(SEBasicBlock, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, output_dim)

        self.apply(initialize_parameters)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 将输入 reshape 为 (B, 1, grid_size, grid_size)
        x = x.view(-1, 1, self.grid_size, self.grid_size)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.attn(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# 5. CNNTransformerBranch 分支
# ------------------------------------------------------------------------------
class CNNTransformerBranch(nn.Module):
    def __init__(self, output_dim, grid_size, input_channels=1,
                 num_transformer_layers=2, nhead=4, d_model=256):
        """
        先用 CNN 提取局部特征，后用 Transformer 编码器进行全局交互
        参数:
            output_dim: 输出维度
            grid_size: 输入边长
            input_channels: 输入通道
            num_transformer_layers: Transformer层数
            nhead: 注意力头数
            d_model: Transformer 嵌入维度
        """
        super(CNNTransformerBranch, self).__init__()
        self.grid_size = grid_size

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(SEBasicBlock, 64, blocks=2)
        self.layer2 = self._make_layer(SEBasicBlock, 128, blocks=2, stride=2)
        # 将 CNN 输出映射到 Transformer 的维度
        self.conv_project = nn.Conv2d(128, d_model, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dropout=0.1, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(d_model, output_dim)

        self.apply(initialize_parameters)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        inplanes = 64 if planes == 64 else planes // 2
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = [block(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 将输入 reshape 为 (B, 1, grid_size, grid_size)
        x = x.view(-1, 1, self.grid_size, self.grid_size)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv_project(x)  # (B, d_model, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2)            # (B, d_model, H*W)
        x = x.transpose(1, 2)         # (B, H*W, d_model)
        x = x.transpose(0, 1)         # (H*W, B, d_model) 适用于 Transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)            # 序列上全局平均池化
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# 6. ResFullyConnectedBranch 分支
# ------------------------------------------------------------------------------
class ResFCLayer(nn.Module):
    """带残差连接的全连接层模块"""
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return torch.tanh(x + identity)

class ResFullyConnectedBranch(nn.Module):
    def __init__(self, output_dim, grid_size, input_channels=1, hidden_dim=64, num_blocks=2):
        """
        带残差连接的全连接网络分支。
        假设输入为一维 (B, grid_size**2)
        """
        super().__init__()
        self.grid_size = grid_size
        self.fc_in = nn.Linear(grid_size , hidden_dim)
        self.act = nn.Tanh()
        layers = []
        for _ in range(num_blocks):
            layers.append(ResFCLayer(hidden_dim, hidden_dim))
        self.net = nn.Sequential(*layers)
        self.final_fc = nn.Linear(hidden_dim, output_dim)

        self.apply(initialize_parameters)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act(self.fc_in(x))
        x = self.net(x)
        x = self.final_fc(x)
        return x

# ------------------------------------------------------------------------------
# 7. ResNetBranch 分支 (标准 ResNet，不使用 SE 模块)
# ------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetBranch(nn.Module):
    def __init__(self, output_dim, grid_size, input_channels=1):
        """
        标准 ResNet 分支，不使用 SE 模块。
        参数:
            output_dim: 输出维度
            grid_size: 输入视作 (grid_size x grid_size)
            input_channels: 输入通道数
        """
        super(ResNetBranch, self).__init__()
        self.inplanes = 64
        self.grid_size = grid_size

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, blocks=2)
        self.layer2 = self._make_layer(BasicBlock, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, output_dim)

        self.apply(initialize_parameters)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 1, self.grid_size, self.grid_size)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# 8. TrackModel 与工厂函数
# ------------------------------------------------------------------------------
class TrackModel(nn.Module):
    def __init__(self, base_model, input_dim, output_dim, input_transform=False):
        """
        跟踪模型包装器。
        参数:
            base_model: 分支或前馈基模型
            input_dim: 输入维度（原始未扩展维度）
            output_dim: 输出维度
            input_transform: 是否对输入进行正弦余弦扩展
        """
        super().__init__()
        self.base_model = base_model
        self._input_transform = input_transform
        self.input_dim = input_dim
        self.output_dim = output_dim

    def input_transform(self, X):
        sin = torch.sin(2 * torch.pi * X)
        cos = torch.cos(2 * torch.pi * X)
        sin2 = torch.sin(4 * torch.pi * X)
        cos2 = torch.cos(4 * torch.pi * X)
        sin3 = torch.sin(6 * torch.pi * X)
        cos3 = torch.cos(6 * torch.pi * X)
        # 拼接原始输入与三组正余弦特征（扩展 7 倍）
        final_input = torch.cat((X, sin, cos, sin2, cos2, sin3, cos3), dim=1)
        return final_input

    def forward(self, x):
        if self._input_transform:
            x = self.input_transform(x)
        return self.base_model(x)

def get_track_model(model_type, output_dim, input_dim, **kwargs):
    """
    跟踪模型工厂函数
    model_type:
        'FNN'    : 简单前馈网络
        'ResFC'  : 带残差连接的全连接网络
    """
    if model_type == 'FNN':
        base = nn.Sequential(
            nn.Linear(input_dim * 7, 64),  # 输入维度扩充7倍
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )
    elif model_type == 'ResFC':
        base = ResFullyConnectedBranch(
            output_dim=output_dim,
            grid_size=(input_dim * 7),
            input_channels=1,
            hidden_dim=128,
            num_blocks=4
        )
    else:
        raise ValueError(f"Unsupported track model type: {model_type}")

    return TrackModel(
        base_model=base,
        input_dim=input_dim,
        output_dim=output_dim,
        input_transform=True
    )

def get_branch_model(model_type, output_dim, grid_size, input_channels=1, **kwargs):
    """
    分支模型工厂函数，可选项:
        'SE_ResNet'      : 基于 SE-ResNet 的分支
        'ConvNeXt'       : 基于 ConvNeXt 的分支
        'Transformer'    : 带自注意力模块的 Transformer 分支
        'CNNTransformer' : CNN + Transformer 混合分支
        'Swin'           : 基于 Swin Transformer 的分支（未在此代码中完整实现）
        'ResFC'          : 带残差的全连接网络分支
        'ResNet'         : 标准 ResNet 分支
    """
    if model_type == 'SE_ResNet':
        return SE_ResNetBranch(output_dim, grid_size, input_channels)
    elif model_type == 'ConvNeXt':
        return ConvNeXtBranch(output_dim, grid_size, input_channels)
    elif model_type == 'Transformer':
        return TransformerBranch(output_dim, grid_size, input_channels)
    elif model_type == 'CNNTransformer':
        return CNNTransformerBranch(output_dim, grid_size, input_channels, **kwargs)
    elif model_type == 'ResFC':
        return ResFullyConnectedBranch(output_dim, grid_size**2, input_channels,num_blocks=6,hidden_dim=1024, **kwargs)
    elif model_type == 'ResNet':
        return ResNetBranch(output_dim, grid_size, input_channels)
    else:
        raise ValueError(f"Unknown model_type: {model_type}.")