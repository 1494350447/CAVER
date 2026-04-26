_base_ = ["chapter3_m3fd_sam2_large.py"]

args = dict(
    base_seed=42,
    batch_size=1,
    num_workers=0,
    print_freq=1,
    val_freq=1,
    epoch_num=1,
    use_amp=False,
    iter_num=20,
    epoch_based=True,
)

data = dict(
    train=dict(
        name=["M3FD_DEBUG20"],
        shape=dict(h=768, w=1024),
        dataset_infos=dict(
            M3FD_DEBUG20=dict(
                image_root="/root/CAVER/M3FD/Vis",
                depth_root="/root/CAVER/M3FD/Ir",
                ann_file="/root/CAVER/data/m3fd_detection/debug_20.json",
                depth_file_key="depth_file_name",
            ),
        ),
    ),
    test=dict(
        name=["M3FD_DEBUG20"],
        shape=dict(h=768, w=1024),
        dataset_infos=dict(
            M3FD_DEBUG20=dict(
                image_root="/root/CAVER/M3FD/Vis",
                depth_root="/root/CAVER/M3FD/Ir",
                ann_file="/root/CAVER/data/m3fd_detection/debug_20.json",
                depth_file_key="depth_file_name",
            ),
        ),
    ),
)
