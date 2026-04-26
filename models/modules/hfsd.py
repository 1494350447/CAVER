import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNormAct, ConvResidualStack, LayerNorm2d, MLP


class HeuristicFeatureExtractor(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm2d(channels),
            ConvNormAct(channels, channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = MLP(channels, hidden_dim, hidden_dim)

    def forward(self, x):
        return self.mlp(self.net(x).flatten(1))


class FrequencyCoordinateEncoder(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.u_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.v_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, h, w, device):
        u = torch.linspace(-1.0, 1.0, steps=h, device=device)[:, None]
        v = torch.linspace(-1.0, 1.0, steps=w, device=device)[:, None]
        u_feat = self.u_mlp(u)[:, None, :]
        v_feat = self.v_mlp(v)[None, :, :]
        coord = torch.cat([u_feat.expand(h, w, -1), v_feat.expand(h, w, -1)], dim=-1)
        # 频率半径图需要保持二维 [H, W]，后续才能和专家生成的二维掩码逐点相乘。
        radius = torch.sqrt(u.pow(2) + v.transpose(0, 1).pow(2))
        return coord, radius


class ParametricFrequencyExpert(nn.Module):
    def __init__(self, coord_dim, hidden_dim=64):
        super().__init__()
        self.coord_proj = nn.Sequential(nn.Linear(coord_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.low_proj = nn.Linear(hidden_dim, 1)
        self.high_proj = nn.Linear(hidden_dim, 1)
        self.low_mu = nn.Parameter(torch.tensor(0.15))
        self.low_sigma = nn.Parameter(torch.tensor(0.25))
        self.high_mu = nn.Parameter(torch.tensor(0.35))
        self.high_sigma = nn.Parameter(torch.tensor(0.30))

    @staticmethod
    def gaussian(radius, mu, sigma):
        sigma = sigma.abs() + 1e-4
        return torch.exp(-((radius - mu) ** 2) / (2 * sigma ** 2))

    def forward(self, coord, radius):
        encoded = self.coord_proj(coord)
        low_base = torch.sigmoid(self.low_proj(encoded)).squeeze(-1)
        high_base = torch.sigmoid(self.high_proj(encoded)).squeeze(-1)
        low_mask = low_base * self.gaussian(radius, self.low_mu, self.low_sigma)
        high_mask = high_base * (1.0 - self.gaussian(radius, self.high_mu, self.high_sigma))
        return low_mask, high_mask


class HFSDStage(nn.Module):
    """启发式频率选择解码器单尺度模块。

    对应图 3.4 与公式 (1.14) - (1.25)。
    """

    def __init__(self, channels, heuristic_dim=128, num_experts=3, coord_hidden_dim=16):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.heuristic = HeuristicFeatureExtractor(channels, heuristic_dim)
        self.coord_encoder = FrequencyCoordinateEncoder(hidden_dim=coord_hidden_dim)
        self.experts = nn.ModuleList(
            [ParametricFrequencyExpert(coord_dim=coord_hidden_dim * 2, hidden_dim=heuristic_dim) for _ in range(num_experts)]
        )
        self.low_gate = nn.Linear(heuristic_dim, num_experts)
        self.high_gate = nn.Linear(heuristic_dim, num_experts)
        self.cls_refine = ConvResidualStack(channels, num_blocks=2)
        self.reg_refine = ConvResidualStack(channels, num_blocks=2)

    def _apply_masks(self, x_fft, masks):
        return torch.fft.ifft2(x_fft * masks[:, None, :, :], dim=(-2, -1)).real

    def forward(self, x):
        x_norm = self.norm(x)
        heuristic = self.heuristic(x_norm)
        x_fft = torch.fft.fft2(x_norm.float(), dim=(-2, -1))

        h, w = x.shape[-2:]
        coord, radius = self.coord_encoder(h=h, w=w, device=x.device)

        low_masks = []
        high_masks = []
        for expert in self.experts:
            low_mask, high_mask = expert(coord, radius)
            low_masks.append(low_mask)
            high_masks.append(high_mask)
        low_masks = torch.stack(low_masks, dim=0)
        high_masks = torch.stack(high_masks, dim=0)

        low_gate = self.low_gate(heuristic).softmax(dim=1)
        high_gate = self.high_gate(heuristic).softmax(dim=1)

        low_mask = (low_gate[:, :, None, None] * low_masks[None, :, :, :]).sum(dim=1)
        high_mask = (high_gate[:, :, None, None] * high_masks[None, :, :, :]).sum(dim=1)

        cls_feat = self._apply_masks(x_fft, low_mask).type_as(x)
        reg_feat = self._apply_masks(x_fft, high_mask).type_as(x)

        cls_feat = self.cls_refine(cls_feat + x)
        reg_feat = self.reg_refine(reg_feat + x)
        aux = dict(low_mask=low_mask, high_mask=high_mask, low_gate=low_gate, high_gate=high_gate)
        return cls_feat, reg_feat, aux


class HFSDDecoder(nn.Module):
    def __init__(self, channels, num_levels=3, heuristic_dim=128, num_experts=3, coord_hidden_dim=16):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                HFSDStage(
                    channels=channels,
                    heuristic_dim=heuristic_dim,
                    num_experts=num_experts,
                    coord_hidden_dim=coord_hidden_dim,
                )
                for _ in range(num_levels)
            ]
        )

    def forward(self, feats):
        cls_feats = []
        reg_feats = []
        aux = []
        for feat, stage in zip(feats, self.stages):
            cls_feat, reg_feat, stage_aux = stage(feat)
            cls_feats.append(cls_feat)
            reg_feats.append(reg_feat)
            aux.append(stage_aux)
        return cls_feats, reg_feats, aux
