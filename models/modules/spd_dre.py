import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNormAct, DepthwiseSeparableConv, MLP, ResidualNCGNBlock, SpatialAttention2d


class PriorAffineRouter(nn.Module):
    """SAM2 先验引导的仿射 MoE 路由器。

    对应图 3.2 与公式 (1.5) - (1.6)。
    """

    def __init__(self, prior_channels, feat_channels, num_experts, hidden_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.feat_channels = feat_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.router = MLP(prior_channels, hidden_dim, num_experts)
        self.gamma = MLP(prior_channels, hidden_dim, num_experts * feat_channels)
        self.beta = MLP(prior_channels, hidden_dim, num_experts * feat_channels)

    def forward(self, prior):
        pooled = self.pool(prior).flatten(1)
        weights = self.router(pooled).softmax(dim=1)
        gamma = torch.sigmoid(self.gamma(pooled)).view(-1, self.num_experts, self.feat_channels, 1, 1)
        beta = self.beta(pooled).view(-1, self.num_experts, self.feat_channels, 1, 1)
        return weights, gamma, beta

    def aggregate(self, expert_outputs, prior):
        weights, gamma, beta = self.forward(prior)
        stacked = torch.stack(expert_outputs, dim=1)
        modulated = stacked * gamma + beta
        return (modulated * weights[:, :, None, None, None]).sum(dim=1)


class GridSpatialExpert(nn.Module):
    def __init__(self, channels, patch_size, kernel_size):
        super().__init__()
        self.patch_size = patch_size
        self.expert = DepthwiseSeparableConv(channels, channels, kernel_size=kernel_size)

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, hp, wp = x.shape

        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        fold = nn.Fold(output_size=(hp, wp), kernel_size=self.patch_size, stride=self.patch_size)

        patches = unfold(x).transpose(1, 2).reshape(-1, c, self.patch_size, self.patch_size)
        patches = self.expert(patches)
        patches = patches.reshape(b, -1, c * self.patch_size * self.patch_size).transpose(1, 2)
        x = fold(patches)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]
        return x


class ChannelFrequencyExpert(nn.Module):
    def __init__(self, channels, channel_groups):
        super().__init__()
        self.channel_groups = channel_groups
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        groups = max(1, min(self.channel_groups, c))
        if c % groups != 0:
            groups = math.gcd(c, groups)
        x_grouped = x.view(b, groups, c // groups, h, w)
        fft = torch.fft.fft2(x_grouped.float(), dim=(-2, -1))
        amplitude = fft.abs().flatten(1, 2)
        gate = self.gate(amplitude)
        gate = gate.view(b, groups, c // groups, 1, 1)
        fft = fft * gate
        out = torch.fft.ifft2(fft, dim=(-2, -1)).real
        return out.reshape(b, c, h, w).type_as(x)


class PriorFusionMapper(nn.Module):
    """将 RGB-IR 融合映射为教师网络的统一先验输入。"""

    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        if in_channels != 4 or out_channels != 3:
            raise ValueError("PriorFusionMapper 当前固定服务于 RGB(3) + IR(1) -> pseudo-RGB(3) 映射。")
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, image, depth):
        rgb = (image * self.rgb_std) + self.rgb_mean
        rgb = rgb.clamp_(0.0, 1.0)
        depth = depth[:, :1].clamp_(0.0, 1.0)
        luminance = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        blended = 0.5 * (luminance + depth)
        return torch.cat([luminance, depth, blended], dim=1)


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            DepthwiseSeparableConv(channels, channels, act=False),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class SAM2ProxyGenerator(nn.Module):
    """学生侧 SAM2 代理先验生成器。

    对应公式 (1.3) - (1.4)。
    """

    def __init__(self, in_channels=4, prior_channels=128, num_blocks=3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, 64, stride=2),
            ConvNormAct(64, prior_channels, stride=2),
        )
        self.blocks = nn.Sequential(*[DepthwiseResidualBlock(prior_channels) for _ in range(num_blocks)])
        self.spatial_attention = SpatialAttention2d(kernel_size=7)

    def forward(self, image, depth):
        x = torch.cat([image, depth], dim=1)
        x = self.blocks(self.stem(x))
        return self.spatial_attention(x)


class SPDDREStage(nn.Module):
    """SPD-DRE 单尺度阶段。

    对应图 3.2 与公式 (1.1) - (1.6)。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        prior_channels,
        patch_size=4,
        channel_groups=4,
        num_frequency_experts=3,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pre_block = ResidualNCGNBlock(in_channels, out_channels)
        self.spatial_experts = nn.ModuleList(
            [GridSpatialExpert(out_channels, patch_size=patch_size, kernel_size=k) for k in (3, 5, 7)]
        )
        self.frequency_experts = nn.ModuleList(
            [ChannelFrequencyExpert(out_channels, channel_groups=channel_groups) for _ in range(num_frequency_experts)]
        )
        self.spatial_router = PriorAffineRouter(prior_channels, out_channels, num_experts=len(self.spatial_experts))
        self.frequency_router = PriorAffineRouter(prior_channels, out_channels, num_experts=len(self.frequency_experts))
        self.fuse = nn.Sequential(
            ConvNormAct(out_channels * 3, out_channels, kernel_size=1, padding=0),
            ConvNormAct(out_channels, out_channels),
        )

    def forward(self, x, student_prior):
        x = self.pre_block(x)
        prior = F.interpolate(student_prior, size=x.shape[-2:], mode="bilinear", align_corners=False)

        spatial_outputs = [expert(x) for expert in self.spatial_experts]
        frequency_outputs = [expert(x) for expert in self.frequency_experts]

        spatial_out = self.spatial_router.aggregate(spatial_outputs, prior)
        frequency_out = self.frequency_router.aggregate(frequency_outputs, prior)

        fused = self.fuse(torch.cat([x, spatial_out, frequency_out], dim=1))
        aux = dict(spatial=spatial_out, frequency=frequency_out)
        return fused, aux
