_base_ = ["base.py"]

pretrained = None
load_from = "/root/CAVER/output_chapter3_m3fd_full/SAM2PriorAlignmentYOLODetector_768x1024_BS3_E100_AMPn_LR0.0001_OTall_OPadamw_LTcos/exp_2/pth/state.pth"
info = "m3fd_recover_resume"
freeze_backbone_bn = False
metric_precision = 6
val_sweep_start_epoch = 1

task = dict(
    name="bimodal_detection",
    num_classes=6,
    strides=(8, 16, 32),
    assigner=dict(
        mode="legacy",
        center_radius=2.5,
    ),
    loss=dict(
        cls_weight=1.0,
        reg_weight=1.0,
        distill_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        pos_ce_weight=0.0,
        reg_l1_weight=0.0,
        class_weights=None,
    ),
    inference=dict(
        score_thr=0.05,
        nms_iou_thr=0.6,
        pre_nms_topk=1000,
        max_per_img=300,
        sweep_score_thrs=[0.03, 0.05, 0.07, 0.10],
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
    strides=(8, 16, 32),
    det_head=dict(
        norm_type="bn",
        use_reg_scales=False,
    ),
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
    epoch_num=30,
    use_amp=False,
    iter_num=10000,
    epoch_based=True,
)

optimizers = dict(
    lr=6e-5,
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
            warmup_length=0,
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
