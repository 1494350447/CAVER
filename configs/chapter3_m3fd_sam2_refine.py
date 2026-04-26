_base_ = ["base.py"]

pretrained = "timm:resnet50d.ra4_e3600_r224_in1k"
info = "m3fd_sam2refine"
freeze_backbone_bn = True
val_sweep_start_epoch = 1
metric_precision = 6
ema = dict(enable=True, decay=0.9998)

task = dict(
    name="bimodal_detection",
    num_classes=6,
    strides=(8, 16, 32),
    assigner=dict(
        regress_ranges=((0, 64), (64, 128), (128, 1e8)),
        center_radius=2.5,
        use_stride_normalized_reg=True,
    ),
    loss=dict(
        cls_weight=2.0,
        reg_weight=1.0,
        distill_weight=0.25,
        focal_alpha=0.25,
        focal_gamma=2.0,
        pos_ce_weight=1.0,
        reg_l1_weight=0.25,
        class_weights=[0.353, 0.298, 1.567, 1.852, 0.749, 1.181],
    ),
    inference=dict(
        score_thr=0.005,
        nms_iou_thr=0.55,
        pre_nms_topk=3000,
        max_per_img=300,
        sweep_score_thrs=[0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015],
        sweep_nms_iou_thrs=[0.50, 0.55, 0.60],
    ),
)

model = dict(
    num_classes=6,
    backbone_name="resnet50d",
    neck_channels=256,
    patch_size=4,
    channel_groups=4,
    prior_channels=128,
    num_frequency_experts=3,
    frequency_coord_dim=16,
    cls_prior_prob=0.05,
    strides=(8, 16, 32),
    teacher_adapter_cfg=dict(
        enable=True,
        variant="sam2_large",
        repo_root="/root/autodl-fs/third_party/sam2_official",
        model_cfg="configs/sam2/sam2_hiera_l.yaml",
        checkpoint_path="/root/autodl-fs/checkpoints/sam2/sam2_hiera_large.pt",
        freeze=True,
        compile_image_encoder=False,
    ),
)

args = dict(
    base_seed=42,
    batch_size=3,
    num_workers=4,
    print_freq=20,
    val_freq=1,
    epoch_num=85,
    use_amp=True,
    iter_num=10000,
    epoch_based=True,
)

optimizers = dict(
    lr=1e-4,
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
            warmup_length=560,
            min_coef=0.01,
            max_coef=1,
        ),
    ),
)

data = dict(
    train=dict(
        name=["M3FD_TRAIN"],
        shape=dict(h=768, w=1024),
        dataset_infos=dict(
            M3FD_TRAIN=dict(
                image_root="/root/CAVER/M3FD/Vis",
                depth_root="/root/CAVER/M3FD/Ir",
                ann_file="/root/CAVER/data/m3fd_detection/train_coco.json",
                depth_file_key="depth_file_name",
            ),
        ),
    ),
    test=dict(
        name=["M3FD_VAL"],
        shape=dict(h=768, w=1024),
        dataset_infos=dict(
            M3FD_VAL=dict(
                image_root="/root/CAVER/M3FD/Vis",
                depth_root="/root/CAVER/M3FD/Ir",
                ann_file="/root/CAVER/data/m3fd_detection/val_coco.json",
                depth_file_key="depth_file_name",
            ),
        ),
    ),
)
