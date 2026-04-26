_base_ = ["base.py"]

task = dict(
    name="bimodal_detection",
    num_classes=6,
    strides=(8, 16, 32),
    center_radius=2.5,
    loss=dict(
        cls_weight=1.0,
        reg_weight=1.0,
        distill_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ),
    inference=dict(
        score_thr=0.05,
        nms_iou_thr=0.6,
        pre_nms_topk=1000,
        max_per_img=300,
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
    print_freq=20,
    val_freq=1,
    epoch_num=100,
    use_amp=False,
    iter_num=10000,
    epoch_based=True,
)

optimizers = dict(
    lr=1e-4,
    strategy="all",
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
            warmup_length=5,
            min_coef=0.001,
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
