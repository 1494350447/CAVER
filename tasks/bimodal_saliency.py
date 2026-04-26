import os

import albumentations as A
import cv2
import numpy as np
import torch
from tqdm import tqdm

import datasets as dataset_lib
from method.mssim import ssim
from utils import ops
from utils.data import get_data_from_txt, get_datasets_info_with_keys, read_binary_array, read_color_array
from utils.recorder import CalTotalMetric

from .base import BaseTask


def iou(prob, gt):
    inter = torch.sum(gt * prob, dim=(1, 2, 3))
    union = gt.sum(dim=(1, 2, 3)) + prob.sum(dim=(1, 2, 3)) - inter
    iou_score = inter / union
    return iou_score.mean()


class BimodalSaliencyTrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, shape, extra_scales=None):
        super().__init__()
        if extra_scales is not None:
            self.scales = (1,) + tuple(extra_scales)

        self.total_paths = []
        for dataset_name, dataset_info in root.items():
            image_root = dataset_info["image"]["path"]
            image_suffix = dataset_info["image"]["suffix"]
            mask_root = dataset_info["mask"]["path"]
            mask_suffix = dataset_info["mask"]["suffix"]
            depth_root = dataset_info["depth"]["path"]
            depth_suffix = dataset_info["depth"]["suffix"]
            if "index_file" in dataset_info:
                valid_names = get_data_from_txt(dataset_info["index_file"])
            else:
                image_names = [x[: -len(image_suffix)] for x in os.listdir(image_root)]
                mask_names = [x[: -len(mask_suffix)] for x in os.listdir(mask_root)]
                depth_names = [x[: -len(depth_suffix)] for x in os.listdir(depth_root)]
                valid_names = list(set(image_names).intersection(mask_names).intersection(depth_names))

            for valid_name in sorted(valid_names):
                sample_paths = (
                    os.path.join(image_root, valid_name + image_suffix),
                    os.path.join(mask_root, valid_name + mask_suffix),
                    os.path.join(depth_root, valid_name + depth_suffix),
                )
                self.total_paths.append(sample_paths)
            print(f"Loading data from {dataset_name} with {len(valid_names)} samples.")

        self.joint_trans = A.Compose(
            [
                A.Resize(height=shape["h"], width=shape["w"]),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=90),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.75),
                A.Normalize(),
            ],
            additional_targets=dict(depth="mask"),
        )

    def __len__(self):
        return len(self.total_paths)

    def __getitem__(self, index):
        image_path, mask_path, depth_path = self.total_paths[index]

        image = read_color_array(image_path)
        mask = read_binary_array(mask_path, to_normalize=True, thr=0.5)
        depth = read_binary_array(depth_path, to_normalize=True, thr=-1)

        transformed = self.joint_trans(image=image, mask=mask, depth=depth)
        image = transformed["image"]
        mask = transformed["mask"]
        depth = transformed["depth"]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        return dict(image=image_tensor, mask=mask_tensor, depth=depth_tensor)


class BimodalSaliencyTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, shape):
        super().__init__()
        self.datasets = get_datasets_info_with_keys(dataset_infos=root, extra_keys=["mask", "depth"])
        self.image_paths = self.datasets["image"]
        self.mask_paths = self.datasets["mask"]
        self.depth_paths = self.datasets["depth"]

        self.joint_trans = A.Compose([A.Resize(height=shape["h"], width=shape["w"]), A.Normalize()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        depth_path = self.depth_paths[index]

        image = read_color_array(image_path)
        depth = read_binary_array(depth_path, to_normalize=True, thr=-1)

        transformed = self.joint_trans(image=image, mask=depth)
        image = transformed["image"]
        depth = transformed["mask"]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        return dict(
            image=image_tensor,
            depth=depth_tensor,
            image_info=dict(mask_path=mask_path, mask_name=os.path.basename(mask_path)),
        )


class BimodalSaliencyTask(BaseTask):
    name = "bimodal_saliency"
    metric_names = (
        "Smeasure",
        "wFmeasure",
        "MAE",
        "adpEm",
        "meanEm",
        "maxEm",
        "adpFm",
        "meanFm",
        "maxFm",
    )

    def build_train_dataset(self, cfg):
        train_data_paths = {name: dataset_lib.__dict__[name] for name in cfg.data.train.name}
        return BimodalSaliencyTrainDataset(root=train_data_paths, shape=cfg.data.train.shape)

    def build_test_dataset(self, dataset_name, cfg):
        dataset_info = dataset_lib.__dict__[dataset_name]
        dataset = BimodalSaliencyTestDataset(root=(dataset_name, dataset_info), shape=cfg.data.test.shape)
        return dataset, dataset_info

    def get_model_inputs(self, batch):
        return dict(image=batch["image"], depth=batch["depth"])

    def compute_loss(self, model_outputs, batch):
        seg_gts = batch["mask"]
        losses = []
        loss_parts = []

        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=model_outputs, target=seg_gts, reduction="mean"
        )
        losses.append(bce_loss)
        loss_parts.append(f"bce:{bce_loss.item():.5f}")

        prob = model_outputs.sigmoid()
        ssim_loss = 1 - ssim(prob, seg_gts)
        losses.append(ssim_loss)
        loss_parts.append(f"ssim:{ssim_loss.item():.5f}")

        iou_loss = 1 - iou(prob, seg_gts)
        losses.append(iou_loss)
        loss_parts.append(f"iou:{iou_loss.item():.5f}")
        return sum(losses), " ".join(loss_parts)

    def get_visualization_data(self, model_outputs, batch):
        return dict(smap=model_outputs.sigmoid().detach().cpu(), img=batch["image"], dep=batch["depth"], msk=batch["mask"])

    @torch.no_grad()
    def evaluate_once(self, model, data_loader, save_path="", show_bar=True):
        model.eval()
        cal_total_seg_metrics = CalTotalMetric()

        bar_iter = enumerate(data_loader)
        if show_bar:
            bar_iter = tqdm(bar_iter, total=len(data_loader), leave=False, ncols=79)

        for _, batch in bar_iter:
            device_batch = self.move_batch_to_device(batch)
            logits = model(data=self.get_model_inputs(device_batch))
            probs = logits.sigmoid().squeeze(1).cpu().detach().numpy()

            for i, pred in enumerate(probs):
                mask_path = batch["image_info"]["mask_path"][i]
                mask_array = read_binary_array(mask_path, dtype=np.uint8)
                mask_h, mask_w = mask_array.shape

                pred = cv2.resize(pred, dsize=(mask_w, mask_h), interpolation=cv2.INTER_LINEAR)

                if save_path:
                    pred_name = os.path.splitext(batch["image_info"]["mask_name"][i])[0] + ".png"
                    ops.save_array_as_image(data_array=pred, save_name=pred_name, save_dir=save_path)

                pred = (pred * 255).astype(np.uint8)
                cal_total_seg_metrics.step(pred, mask_array, mask_path)
        return cal_total_seg_metrics.get_results()
