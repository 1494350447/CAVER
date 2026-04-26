_base_ = ["base.py"]

pretrained = "/root/CAVER/pretrained/resnet50d_ra2-464e36ba.pth"
load_from = None
info = "vedai_fold0_strongbase"
freeze_backbone_bn = False
metric_precision = 6
val_sweep_start_epoch = 1

task = dict(
    name="bimodal_detection",
    num_classes=8,
    strides=(8, 16, 32),
    assigner=dict(
        mode="fcos",
        center_radius=2.5,
        use_stride_normalized_reg=True,
        regress_ranges=((0, 48), (48, 96), (96, 1e8)),
    ),
    area_metric_ranges=dict(
        small=(0, 32 * 32),
        medium=(32 * 32, 96 * 96),
    ),
    train_aug=dict(
        mode="aerial",
        random_rotate90_p=0.5,
        horizontal_flip_p=0.5,
        vertical_flip_p=0.5,
        affine=dict(
            scale=(0.9, 1.1),
            translate_percent=(-0.04, 0.04),
            rotate=0.0,
            shear=0.0,
            p=0.6,
        ),
        brightness=0.15,
        contrast=0.15,
        saturation=0.08,
        hue=0.02,
        color_jitter_p=0.5,
        blur_limit=3,
        gaussian_blur_p=0.08,
    ),
    loss=dict(
        cls_weight=1.0,
        reg_weight=1.0,
        distill_weight=0.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        pos_ce_weight=0.0,
        reg_l1_weight=0.25,
        class_weights=[0.276, 0.332, 0.514, 0.585, 0.717, 0.743, 0.784, 1.0],
    ),
    inference=dict(
        score_thr=0.03,
        nms_iou_thr=0.55,
        pre_nms_topk=1000,
        max_per_img=300,
        sweep_score_thrs=[0.02, 0.03, 0.05, 0.07],
        sweep_nms_iou_thrs=[0.50, 0.55, 0.60],
    ),
)

model = dict(
    num_classes=8,
    backbone_name="resnet50d",
    backbone_out_indices=(2, 3, 4),
    neck_channels=256,
    patch_size=4,
    channel_groups=4,
    prior_channels=128,
    num_frequency_experts=3,
    frequency_coord_dim=16,
    strides=(8, 16, 32),
    det_head=dict(
        norm_type="gn",
        use_reg_scales=True,
        cls_prior_prob=0.01,
    ),
    teacher_adapter_cfg=dict(
        enable=False,
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
    batch_size=2,
    num_workers=4,
    print_freq=20,
    val_freq=1,
    epoch_num=50,
    use_amp=True,
    iter_num=10000,
    epoch_based=True,
)

ema = dict(enable=True, decay=0.9998)

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
            warmup_length=507,
            min_coef=0.05,
            max_coef=1,
        ),
    ),
)

data = dict(
    train=dict(
        name=["VEDAI_TRAIN_FOLD0"],
        shape=dict(h=1024, w=1024),
        dataset_infos=dict(
            VEDAI_TRAIN_FOLD0=dict(
                image_root="/root/CAVER/VEDAI/visible/all2017",
                depth_root="/root/CAVER/VEDAI/infrared/all2017",
                ann_file="/root/CAVER/data/vedai_detection/train_fold0.json",
                depth_file_key="depth_file_name",
            ),
        ),
    ),
    test=dict(
        name=["VEDAI_VAL_FOLD0"],
        shape=dict(h=1024, w=1024),
        dataset_infos=dict(
            VEDAI_VAL_FOLD0=dict(
                image_root="/root/CAVER/VEDAI/visible/all2017",
                depth_root="/root/CAVER/VEDAI/infrared/all2017",
                ann_file="/root/CAVER/data/vedai_detection/val_fold0.json",
                depth_file_key="depth_file_name",
            ),
        ),
    ),
)
