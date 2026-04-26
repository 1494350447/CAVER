import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, act=True):
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, act=True):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.depthwise = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            act=True,
        )
        self.pointwise = ConvNormAct(in_channels, out_channels, kernel_size=1, padding=0, act=act)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualNCGNBlock(nn.Module):
    """Norm-Conv-GELU-Norm 残差块。

    对应图 3.2 中 SPD-DRE 的预处理残差块。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.norm2 = LayerNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(self.norm1(x))
        x = self.act(x)
        x = self.conv2(self.norm2(x))
        return x + residual


class SpatialAttention2d(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        max_pool = x.amax(dim=1, keepdim=True)
        avg_pool = x.mean(dim=1, keepdim=True)
        attn = torch.sigmoid(self.proj(torch.cat([max_pool, avg_pool], dim=1)))
        return x * attn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class ConvResidualStack(nn.Module):
    def __init__(self, channels, num_blocks=2):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(ConvNormAct(channels, channels), ConvNormAct(channels, channels, act=False)))
        self.blocks = nn.ModuleList(blocks)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        for block in self.blocks:
            x = self.act(x + block(x))
        return x


def resize_like(x, ref, mode="bilinear"):
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)

