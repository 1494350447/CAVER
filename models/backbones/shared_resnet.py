import os

import timm
import torch
import torch.nn as nn


class SharedDualStreamResNetBackbone(nn.Module):
    """共享参数双流 ResNet 骨干网络。

    对应第三章图 3.1 中的双流 ResNet 编码阶段。
    RGB 与 IR 分别前向，但共享同一套 backbone 参数。
    """

    def __init__(self, backbone_name="resnet50d", out_indices=(2, 3, 4), pretrained=None):
        super().__init__()
        self.backbone_name = backbone_name
        self.out_indices = out_indices
        timm_model_name = backbone_name
        use_timm_pretrained = False
        if isinstance(pretrained, str):
            normalized = pretrained.strip().lower()
            if normalized in {"timm", "imagenet", "imagenet1k", "default"}:
                use_timm_pretrained = True
            elif normalized.startswith("timm:"):
                timm_model_name = pretrained.split(":", 1)[1].strip()
                use_timm_pretrained = True

        self.backbone = timm.create_model(
            model_name=timm_model_name,
            pretrained=use_timm_pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.feature_channels = tuple(self.backbone.feature_info.channels())

        if isinstance(pretrained, str) and os.path.isfile(pretrained):
            state_dict = torch.load(pretrained, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.backbone.load_state_dict(state_dict, strict=False)

    def forward_single(self, x):
        return list(self.backbone(x))

    def forward(self, image, depth):
        if depth.shape[1] == 1:
            depth = depth.repeat(1, 3, 1, 1)
        rgb_feats = self.forward_single(image)
        ir_feats = self.forward_single(depth)
        return rgb_feats, ir_feats
