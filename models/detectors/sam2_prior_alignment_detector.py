import torch
import torch.nn as nn

from models.backbones import SharedDualStreamResNetBackbone
from models.heads import DecoupledYOLODetectionHead
from models.modules import FDDASNeck, HFSDDecoder, SAM2ProxyGenerator, SPDDREStage
from models.teachers import SAM2TeacherAdapter


class SAM2PriorAlignmentYOLODetector(nn.Module):
    """第三章模型总装类。

    结构对应图 3.1:
    共享参数双流 ResNet -> SPD-DRE -> FPN/PAN 内嵌 F-DDSA -> HFSD -> 解耦检测头
    """

    def __init__(
        self,
        pretrained=None,
        num_classes=6,
        backbone_name="resnet50d",
        backbone_out_indices=None,
        neck_channels=256,
        patch_size=4,
        channel_groups=4,
        prior_channels=128,
        num_frequency_experts=3,
        frequency_coord_dim=16,
        cls_prior_prob=0.01,
        det_head=None,
        strides=(8, 16, 32),
        teacher_adapter_cfg=None,
    ):
        super().__init__()
        if teacher_adapter_cfg is None:
            teacher_adapter_cfg = dict(enable=False)
        det_head_cfg = dict(det_head or {})
        det_head_cfg.setdefault("cls_prior_prob", cls_prior_prob)

        self.num_classes = num_classes
        self.strides = tuple(strides)
        if backbone_out_indices is None:
            num_levels = len(self.strides)
            if num_levels < 1 or num_levels > 5:
                raise ValueError(f"Unsupported number of detection levels: {num_levels}. Expected 1 to 5.")
            backbone_out_indices = tuple(range(5 - num_levels, 5))
        self.backbone_out_indices = tuple(backbone_out_indices)
        self.backbone = SharedDualStreamResNetBackbone(
            backbone_name=backbone_name,
            out_indices=self.backbone_out_indices,
            pretrained=pretrained,
        )
        self.student_prior_generator = SAM2ProxyGenerator(in_channels=4, prior_channels=prior_channels)
        self.teacher_adapter = SAM2TeacherAdapter(teacher_cfg=teacher_adapter_cfg, prior_channels=prior_channels)

        self.spd_dre_stages = nn.ModuleList(
            [
                SPDDREStage(
                    in_channels=in_channels,
                    out_channels=neck_channels,
                    prior_channels=prior_channels,
                    patch_size=patch_size,
                    channel_groups=channel_groups,
                    num_frequency_experts=num_frequency_experts,
                )
                for in_channels in self.backbone.feature_channels
            ]
        )
        self.fddsa_neck = FDDASNeck(channels=neck_channels, num_levels=len(self.spd_dre_stages), groups=channel_groups)
        self.hfsd_decoder = HFSDDecoder(
            channels=neck_channels,
            num_levels=len(self.spd_dre_stages),
            heuristic_dim=prior_channels,
            num_experts=num_frequency_experts,
            coord_hidden_dim=frequency_coord_dim,
        )
        self.det_head = DecoupledYOLODetectionHead(
            channels=neck_channels,
            num_classes=num_classes,
            num_levels=len(self.spd_dre_stages),
            **det_head_cfg,
        )

    def get_grouped_params(self):
        pretrained = list(self.backbone.parameters())
        retrained = []
        for module in [self.student_prior_generator, self.spd_dre_stages, self.fddsa_neck, self.hfsd_decoder, self.det_head]:
            retrained.extend(list(module.parameters()))
        if self.teacher_adapter.enabled and self.teacher_adapter.teacher is not None:
            no_training = list(self.teacher_adapter.parameters())
        else:
            no_training = []
        return dict(pretrained=pretrained, retrained=retrained, no_training=no_training)

    def _encode_multiscale(self, feats, student_prior):
        outputs = []
        aux = []
        for feat, stage in zip(feats, self.spd_dre_stages):
            out, stage_aux = stage(feat, student_prior)
            outputs.append(out)
            aux.append(stage_aux)
        return outputs, aux

    def forward(self, data):
        image = data["image"]
        depth = data["depth"]

        if not self.training and self.teacher_adapter.enabled:
            # 论文设定中 teacher 仅服务训练蒸馏；评估/推理阶段主动释放以回收显存。
            self.teacher_adapter.release_teacher()

        rgb_feats, ir_feats = self.backbone(image, depth)
        student_prior = self.student_prior_generator(image, depth)
        teacher_prior = None
        if self.training and self.teacher_adapter.enabled:
            teacher_prior = self.teacher_adapter(image, depth, target_size=student_prior.shape[-2:])

        rgb_encoded, rgb_aux = self._encode_multiscale(rgb_feats, student_prior)
        ir_encoded, ir_aux = self._encode_multiscale(ir_feats, student_prior)
        fused_feats, neck_aux = self.fddsa_neck(rgb_encoded, ir_encoded)
        cls_feats, reg_feats, hfsd_aux = self.hfsd_decoder(fused_feats)
        cls_logits, bbox_preds = self.det_head(cls_feats, reg_feats)

        return dict(
            cls_logits=cls_logits,
            bbox_preds=bbox_preds,
            cls_features=cls_feats,
            reg_features=reg_feats,
            student_prior=student_prior,
            teacher_prior=teacher_prior,
            aux=dict(
                rgb_spd=rgb_aux,
                ir_spd=ir_aux,
                neck=neck_aux,
                hfsd=hfsd_aux,
                fused_feats=fused_feats,
                strides=self.strides,
            ),
        )
