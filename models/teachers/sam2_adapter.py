import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.spd_dre import PriorFusionMapper


class SAM2TeacherPriorProjector(nn.Module):
    """将官方 SAM2 的高分辨率 FPN 特征映射到论文中的先验通道数。"""

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, feat):
        channels = feat.shape[1]
        if channels < self.out_channels:
            repeat_factor = math.ceil(self.out_channels / channels)
            feat = feat.repeat(1, repeat_factor, 1, 1)
        if feat.shape[1] == self.out_channels:
            return feat
        return F.adaptive_avg_pool3d(feat.unsqueeze(1), (self.out_channels, feat.shape[-2], feat.shape[-1])).squeeze(1)


class SAM2TeacherAdapter(nn.Module):
    """可插拔的 SAM2 教师适配器。

    对应第三章中的训练期教师分支：
    PriorFusionMapper -> Official SAM2 Large Image Encoder -> 高分辨率 FPN 先验
    """

    def __init__(self, teacher_cfg, prior_channels):
        super().__init__()
        self.enabled = teacher_cfg.get("enable", False)
        self.variant = teacher_cfg.get("variant", None)
        self.repo_root = teacher_cfg.get("repo_root", None)
        self.model_cfg = teacher_cfg.get("model_cfg", "configs/sam2/sam2_hiera_l.yaml")
        self.checkpoint_path = teacher_cfg.get("checkpoint_path", None)
        self.freeze = teacher_cfg.get("freeze", True)
        self.compile_image_encoder = bool(teacher_cfg.get("compile_image_encoder", False))
        self.prior_channels = prior_channels
        self.prior_mapper = PriorFusionMapper(in_channels=4, out_channels=3)
        self.prior_projector = SAM2TeacherPriorProjector(out_channels=prior_channels)
        self.teacher = None
        self.teacher_device = None

        if self.enabled and self.variant not in {None, "sam2_large"}:
            raise ValueError(
                f"未知的 SAM2 teacher 版本: {self.variant}. "
                "当前严格复现实现只支持 variant='sam2_large'。"
            )

    def _ensure_teacher_cfg(self):
        if not self.repo_root:
            raise ValueError("teacher.enable=True 时必须提供 teacher_adapter_cfg.repo_root。")
        if not self.checkpoint_path:
            raise ValueError("teacher.enable=True 时必须提供 teacher_adapter_cfg.checkpoint_path。")
        if not os.path.isdir(self.repo_root):
            raise FileNotFoundError(f"SAM2 repo_root 不存在: {self.repo_root}")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"SAM2 checkpoint 不存在: {self.checkpoint_path}")

        config_abspath = os.path.join(self.repo_root, "sam2", self.model_cfg)
        if not os.path.isfile(config_abspath):
            raise FileNotFoundError(
                f"SAM2 配置文件不存在: {config_abspath}. "
                f"请确认 model_cfg 是否为仓库内相对 sam2/ 的路径。"
            )

    def _build_teacher(self, device):
        if self.teacher is not None:
            if self.teacher_device != str(device):
                self.teacher = self.teacher.to(device)
                self.teacher_device = str(device)
            return self.teacher

        self._ensure_teacher_cfg()
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)

        try:
            from sam2.build_sam import build_sam2
        except ImportError as exc:
            raise ImportError(
                "未能导入官方 SAM2。请先执行 tools/install_base_env.sh 或手动安装 SAM2。"
            ) from exc

        hydra_overrides = [f"++model.compile_image_encoder={str(self.compile_image_encoder).lower()}"]
        teacher = build_sam2(
            config_file=self.model_cfg,
            ckpt_path=self.checkpoint_path,
            device=device,
            mode="eval",
            hydra_overrides_extra=hydra_overrides,
            apply_postprocessing=False,
        )
        teacher.eval()
        if self.freeze:
            for param in teacher.parameters():
                param.requires_grad = False
        self.teacher = teacher
        self.teacher_device = str(device)
        return self.teacher

    def release_teacher(self):
        if self.teacher is None:
            return
        teacher = self.teacher
        self.teacher = None
        self.teacher_device = None
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _pick_highest_resolution_feature(backbone_out):
        backbone_fpn = backbone_out.get("backbone_fpn", None)
        if not backbone_fpn:
            raise KeyError("SAM2 forward_image 输出中缺少 backbone_fpn。")
        return max(backbone_fpn, key=lambda item: item.shape[-2] * item.shape[-1])

    def forward(self, image, depth, target_size=None):
        if not self.enabled:
            return None

        teacher = self._build_teacher(device=image.device)
        prior_input = self.prior_mapper(image, depth)
        with torch.no_grad():
            backbone_out = teacher.forward_image(prior_input)
            high_res_feature = self._pick_highest_resolution_feature(backbone_out)
            teacher_prior = self.prior_projector(high_res_feature).type_as(prior_input)

        if target_size is not None and teacher_prior.shape[-2:] != tuple(target_size):
            teacher_prior = F.interpolate(teacher_prior, size=target_size, mode="bilinear", align_corners=False)
        return teacher_prior.detach()
