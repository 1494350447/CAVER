_base_ = ["chapter3_vedai_fold0_strong_base.py"]

info = "vedai_fold0_p2"

task = dict(
    num_classes=8,
    strides=(4, 8, 16, 32),
    assigner=dict(
        mode="fcos",
        center_radius=2.5,
        use_stride_normalized_reg=True,
        regress_ranges=((0, 32), (32, 64), (64, 128), (128, 1e8)),
    ),
)

model = dict(
    num_classes=8,
    backbone_out_indices=(1, 2, 3, 4),
    strides=(4, 8, 16, 32),
    det_head=dict(
        norm_type="gn",
        use_reg_scales=True,
        cls_prior_prob=0.01,
    ),
    teacher_adapter_cfg=dict(enable=False),
)

args = dict(
    batch_size=1,
    epoch_num=50,
    use_amp=True,
)

optimizers = dict(
    lr=2e-4,
    strategy="finetune",
    optimizer="adamw",
    optimizer_candidates=dict(
        adamw=dict(
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=5e-4,
            amsgrad=False,
        ),
    ),
)

schedulers = dict(
    sche_usebatch=True,
    strategy="cos",
    scheduler_candidates=dict(
        cos=dict(
            warmup_length=1014,
            min_coef=0.05,
            max_coef=1,
        ),
    ),
)
