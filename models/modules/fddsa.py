import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNormAct, resize_like


class DynamicGroupResampler(nn.Module):
    """聚焦-发散动态重采样器。

    对应公式 (1.8) - (1.11)。
    用局部聚焦分支与空洞发散分支完成动态加权聚合。
    """

    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        distance = torch.tensor([2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0], dtype=torch.float32)
        inverse_distance = 1.0 / (1.0 + distance)
        self.register_buffer("inverse_distance", inverse_distance.view(1, 1, 1, 9, 1))

    def _aggregate(self, x, kernel_size, dilation, padding):
        b, c, h, w = x.shape
        groups = min(self.groups, c)
        if c % groups != 0:
            groups = 1
        cg = c // groups
        x = x.reshape(b * groups, cg, h, w)
        patches = F.unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding)
        patches = patches.view(b, groups, cg, kernel_size * kernel_size, h * w)
        center = x.reshape(b, groups, cg, 1, h * w)
        similarity = (patches * center).mean(dim=2, keepdim=True)
        return patches, similarity

    def forward(self, x, focus_map, divergence_map):
        b, c, h, w = x.shape
        local_patches, local_sim = self._aggregate(x, kernel_size=3, dilation=1, padding=1)
        global_patches, global_sim = self._aggregate(x, kernel_size=3, dilation=2, padding=2)

        alpha = torch.softmax(torch.cat([focus_map, divergence_map], dim=1), dim=1)[:, :1]
        beta = 1.0 - alpha
        alpha = alpha.view(b, 1, 1, 1, h * w)
        beta = beta.view(b, 1, 1, 1, h * w)

        local_weights = torch.softmax(local_sim * self.inverse_distance * alpha, dim=3)
        global_weights = torch.softmax(global_sim * self.inverse_distance * beta, dim=3)

        local_agg = (local_weights * local_patches).sum(dim=3)
        global_agg = (global_weights * global_patches).sum(dim=3)
        out = alpha.squeeze(3) * local_agg + beta.squeeze(3) * global_agg
        return out.view(b, c, h, w), alpha.view(b, 1, h, w), beta.view(b, 1, h, w)


class ChannelAttentionFusion(nn.Module):
    """通道注意力调制融合。

    对应公式 (1.12) - (1.13)。
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels * 2),
            nn.Tanh(),
        )
        self.smooth = ConvNormAct(channels, channels)

    def forward(self, rgb, ir):
        pooled = torch.cat([self.pool(rgb).flatten(1), self.pool(ir).flatten(1)], dim=1)
        v_rgb, v_ir = self.mlp(pooled).chunk(2, dim=1)
        v_rgb = v_rgb[:, :, None, None]
        v_ir = v_ir[:, :, None, None]
        fused = (1.0 + v_rgb) * rgb + (1.0 + v_ir) * ir
        return self.smooth(fused), v_rgb, v_ir


class FDDSABlock(nn.Module):
    """聚焦-发散动态空间对齐模块。

    对应图 3.3 与公式 (1.7) - (1.13)。
    """

    def __init__(self, channels, groups=4):
        super().__init__()
        self.focus_branch = ConvNormAct(channels * 2, channels, kernel_size=3)
        self.divergence_branch = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.focus_map = nn.Sequential(ConvNormAct(channels, channels, kernel_size=1, padding=0), nn.Conv2d(channels, 1, 1))
        self.div_map = nn.Sequential(ConvNormAct(channels, channels, kernel_size=1, padding=0), nn.Conv2d(channels, 1, 1))
        self.rgb_resampler = DynamicGroupResampler(channels, groups=groups)
        self.ir_resampler = DynamicGroupResampler(channels, groups=groups)
        self.fusion = ChannelAttentionFusion(channels)

    def forward(self, rgb, ir):
        concat = torch.cat([rgb, ir], dim=1)
        focus_feat = self.focus_branch(concat)
        div_feat = self.divergence_branch(concat)
        focus_map = self.focus_map(focus_feat)
        div_map = self.div_map(div_feat)

        rgb_aligned, alpha_rgb, beta_rgb = self.rgb_resampler(rgb, focus_map, div_map)
        ir_aligned, alpha_ir, beta_ir = self.ir_resampler(ir, focus_map, div_map)
        fused, v_rgb, v_ir = self.fusion(rgb_aligned, ir_aligned)
        aux = dict(
            focus_map=focus_map,
            divergence_map=div_map,
            alpha_rgb=alpha_rgb,
            beta_rgb=beta_rgb,
            alpha_ir=alpha_ir,
            beta_ir=beta_ir,
            v_rgb=v_rgb,
            v_ir=v_ir,
        )
        return fused, aux


class FDDASNeck(nn.Module):
    """FPN/PAN 内嵌 F-DDSA 的双向检测颈部。"""

    def __init__(self, channels=256, num_levels=3, groups=4):
        super().__init__()
        self.num_levels = num_levels
        self.rgb_topdown = nn.ModuleList([ConvNormAct(channels, channels) for _ in range(num_levels - 1)])
        self.ir_topdown = nn.ModuleList([ConvNormAct(channels, channels) for _ in range(num_levels - 1)])
        self.rgb_bottomup = nn.ModuleList([ConvNormAct(channels, channels, stride=2) for _ in range(num_levels - 1)])
        self.ir_bottomup = nn.ModuleList([ConvNormAct(channels, channels, stride=2) for _ in range(num_levels - 1)])
        self.fpn_fusions = nn.ModuleList([FDDSABlock(channels, groups=groups) for _ in range(num_levels)])
        self.pan_fusions = nn.ModuleList([FDDSABlock(channels, groups=groups) for _ in range(num_levels)])

    def forward(self, rgb_feats, ir_feats):
        rgb_td = list(rgb_feats)
        ir_td = list(ir_feats)
        fpn_aux = []
        pan_aux = []

        for idx in range(self.num_levels - 2, -1, -1):
            rgb_td[idx] = rgb_td[idx] + resize_like(self.rgb_topdown[idx](rgb_td[idx + 1]), rgb_td[idx])
            ir_td[idx] = ir_td[idx] + resize_like(self.ir_topdown[idx](ir_td[idx + 1]), ir_td[idx])

        fpn_feats = []
        for rgb_feat, ir_feat, fusion in zip(rgb_td, ir_td, self.fpn_fusions):
            fused, aux = fusion(rgb_feat, ir_feat)
            fpn_feats.append(fused)
            fpn_aux.append(aux)

        rgb_bu = [rgb_td[0]]
        ir_bu = [ir_td[0]]
        fused_out = []
        fused, aux = self.pan_fusions[0](rgb_bu[0], ir_bu[0])
        fused_out.append(fused)
        pan_aux.append(aux)

        for idx in range(1, self.num_levels):
            rgb_curr = rgb_td[idx] + self.rgb_bottomup[idx - 1](rgb_bu[idx - 1])
            ir_curr = ir_td[idx] + self.ir_bottomup[idx - 1](ir_bu[idx - 1])
            rgb_bu.append(rgb_curr)
            ir_bu.append(ir_curr)
            fused, aux = self.pan_fusions[idx](rgb_curr, ir_curr)
            fused_out.append(fused)
            pan_aux.append(aux)

        return fused_out, dict(fpn=fpn_aux, pan=pan_aux, fpn_feats=fpn_feats)
