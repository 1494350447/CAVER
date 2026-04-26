import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_norm_groups(channels, preferred_groups=32):
    groups = min(int(preferred_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def _build_norm(norm_type, channels, norm_groups=32):
    norm_type = str(norm_type).lower()
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    if norm_type == "gn":
        return nn.GroupNorm(_resolve_group_norm_groups(channels, preferred_groups=norm_groups), channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}. Expected one of: bn, gn.")


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        act=True,
        norm_type="gn",
        norm_groups=32,
    ):
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
            _build_norm(norm_type=norm_type, channels=out_channels, norm_groups=norm_groups),
        ]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)


class DetectionTower(nn.Module):
    def __init__(self, channels, num_layers=2, norm_type="gn", norm_groups=32):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ConvNormAct(channels, channels, norm_type=norm_type, norm_groups=norm_groups))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LearnableScale(nn.Module):
    """FCOS 风格逐层标度参数。

    不改变检测头拓扑，只负责把回归分支的输出标度拉回到更合适的范围。
    """

    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x):
        return x * self.scale


class DecoupledYOLODetectionHead(nn.Module):
    """YOLO 式 anchor-free 解耦检测头。"""

    def __init__(
        self,
        channels,
        num_classes,
        num_levels=3,
        tower_layers=2,
        norm_type="gn",
        norm_groups=32,
        cls_prior_prob=0.01,
        reg_init_bias=1.5,
        reg_scale_init=1.0,
        use_reg_scales=True,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.use_reg_scales = bool(use_reg_scales)
        self.cls_towers = nn.ModuleList(
            [DetectionTower(channels, tower_layers, norm_type=norm_type, norm_groups=norm_groups) for _ in range(num_levels)]
        )
        self.reg_towers = nn.ModuleList(
            [DetectionTower(channels, tower_layers, norm_type=norm_type, norm_groups=norm_groups) for _ in range(num_levels)]
        )
        self.cls_preds = nn.ModuleList([nn.Conv2d(channels, num_classes, kernel_size=1) for _ in range(num_levels)])
        self.reg_preds = nn.ModuleList([nn.Conv2d(channels, 4, kernel_size=1) for _ in range(num_levels)])
        if self.use_reg_scales:
            self.reg_scales = nn.ModuleList([LearnableScale(init_value=reg_scale_init) for _ in range(num_levels)])
        else:
            self.reg_scales = None
        self._init_weights(cls_prior_prob=cls_prior_prob, reg_init_bias=reg_init_bias)

    def _init_weights(self, cls_prior_prob, reg_init_bias):
        prior_bias = -math.log((1 - cls_prior_prob) / cls_prior_prob)
        for tower in list(self.cls_towers) + list(self.reg_towers):
            for module in tower.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

        for layer in self.cls_preds:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, prior_bias)
        for layer in self.reg_preds:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, reg_init_bias)

    def forward(self, cls_feats, reg_feats):
        cls_logits = []
        bbox_preds = []
        for level in range(self.num_levels):
            cls_feature = self.cls_towers[level](cls_feats[level])
            reg_feature = self.reg_towers[level](reg_feats[level])
            cls_logits.append(self.cls_preds[level](cls_feature))
            reg_logits = self.reg_preds[level](reg_feature)
            if self.use_reg_scales:
                reg_logits = self.reg_scales[level](reg_logits)
            bbox_preds.append(F.softplus(reg_logits))
        return cls_logits, bbox_preds
