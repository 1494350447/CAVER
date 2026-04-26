import json
import math
import os
from collections import Counter, defaultdict

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.data import read_binary_array, read_color_array

from .base import BaseTask


class BalancedConcatSampler(torch.utils.data.Sampler):
    """在多个子数据集之间做定比采样。

    默认每个 epoch 从每个子数据集采同样数量的样本，用于 full/tile 这类 1:1 训练配比。
    """

    def __init__(self, concat_dataset, base_seed=42, samples_per_dataset="min", dataset_weights=None):
        if not hasattr(concat_dataset, "datasets"):
            raise TypeError("BalancedConcatSampler expects a ConcatDataset-like object with .datasets")
        self.concat_dataset = concat_dataset
        self.dataset_lengths = [len(dataset) for dataset in concat_dataset.datasets]
        if not self.dataset_lengths or any(length <= 0 for length in self.dataset_lengths):
            raise ValueError("All child datasets must be non-empty for BalancedConcatSampler.")
        if dataset_weights is None:
            dataset_weights = [1] * len(self.dataset_lengths)
        if len(dataset_weights) != len(self.dataset_lengths):
            raise ValueError("dataset_weights length must match the number of child datasets.")

        self.base_seed = int(base_seed)
        self.dataset_weights = [int(weight) for weight in dataset_weights]
        if any(weight <= 0 for weight in self.dataset_weights):
            raise ValueError("dataset_weights must be positive integers.")

        if isinstance(samples_per_dataset, str):
            normalized = samples_per_dataset.lower()
            if normalized != "min":
                raise ValueError(f"Unsupported samples_per_dataset mode: {samples_per_dataset}")
            base_count = min(length // weight for length, weight in zip(self.dataset_lengths, self.dataset_weights))
        else:
            base_count = int(samples_per_dataset)
        if base_count <= 0:
            raise ValueError("samples_per_dataset must be positive.")

        self.base_count = base_count
        self.num_samples_per_dataset = [self.base_count * weight for weight in self.dataset_weights]
        self.offsets = []
        curr_offset = 0
        for length in self.dataset_lengths:
            self.offsets.append(curr_offset)
            curr_offset += length
        self.total_num_samples = sum(self.num_samples_per_dataset)
        self.epoch = 0

    def __len__(self):
        return self.total_num_samples

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + self.epoch)
        sampled_indices = []
        for offset, length, count in zip(self.offsets, self.dataset_lengths, self.num_samples_per_dataset):
            if count <= length:
                local_indices = torch.randperm(length, generator=generator)[:count]
            else:
                local_indices = torch.randint(low=0, high=length, size=(count,), generator=generator)
            sampled_indices.extend((offset + local_indices).tolist())

        shuffle_order = torch.randperm(len(sampled_indices), generator=generator).tolist()
        self.epoch += 1
        return iter(sampled_indices[idx] for idx in shuffle_order)


def xywh_to_xyxy(boxes):
    boxes = boxes.copy()
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    return boxes


def bbox_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def ciou_loss(pred_boxes, target_boxes):
    iou = bbox_iou(pred_boxes, target_boxes).diag()

    pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
    target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
    center_dist = ((pred_center - target_center) ** 2).sum(dim=1)

    enc_lt = torch.minimum(pred_boxes[:, :2], target_boxes[:, :2])
    enc_rb = torch.maximum(pred_boxes[:, 2:], target_boxes[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=1e-6)
    enc_diag = (enc_wh ** 2).sum(dim=1) + 1e-6

    pred_wh = (pred_boxes[:, 2:] - pred_boxes[:, :2]).clamp(min=1e-6)
    target_wh = (target_boxes[:, 2:] - target_boxes[:, :2]).clamp(min=1e-6)
    v = (4 / math.pi ** 2) * (torch.atan(target_wh[:, 0] / target_wh[:, 1]) - torch.atan(pred_wh[:, 0] / pred_wh[:, 1])) ** 2
    alpha = v / (1 - iou + v + 1e-6)
    ciou = iou - center_dist / enc_diag - alpha * v
    return 1 - ciou


def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0, class_weights=None):
    prob = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    modulating = (1 - p_t) ** gamma
    loss = alpha_factor * modulating * ce_loss
    if class_weights is not None:
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype).view(1, -1)
        loss = loss * (targets * class_weights + (1 - targets))
    return loss


def batched_nms(boxes, scores, labels, iou_threshold):
    keep = []
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    for cls in labels.unique():
        cls_mask = labels == cls
        cls_indices = torch.nonzero(cls_mask, as_tuple=False).flatten()
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        order = cls_scores.argsort(descending=True)
        while order.numel() > 0:
            i = order[0]
            keep.append(cls_indices[i])
            if order.numel() == 1:
                break
            ious = bbox_iou(cls_boxes[i : i + 1], cls_boxes[order[1:]]).squeeze(0)
            order = order[1:][ious <= iou_threshold]
    if not keep:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return torch.stack(keep)


def compute_average_precision(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])


class DetectionMetricAccumulator:
    def __init__(self, num_classes, iou_thresholds=None, area_ranges=None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5 + i * 0.05 for i in range(10)]
        self.area_ranges = area_ranges or dict(
            small=(0.0, 32.0 * 32.0),
            medium=(32.0 * 32.0, 96.0 * 96.0),
        )
        self.predictions = []
        self.targets = []

    def step(self, preds, targets):
        self.predictions.extend(preds)
        self.targets.extend(targets)

    @staticmethod
    def _in_area_range(areas, area_range):
        if area_range is None:
            return torch.ones_like(areas, dtype=torch.bool)
        lower, upper = area_range
        return (areas >= float(lower)) & (areas < float(upper))

    def _evaluate_class(self, class_id, iou_threshold, area_range=None):
        gt_by_image = defaultdict(list)
        total_gt = 0
        for sample in self.targets:
            boxes = sample["boxes"][sample["labels"] == class_id]
            if area_range is not None and boxes.numel() > 0:
                areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
                boxes = boxes[self._in_area_range(areas, area_range)]
            gt_by_image[sample["image_id"]].append(
                dict(boxes=boxes, matched=torch.zeros(boxes.shape[0], dtype=torch.bool))
            )
            total_gt += boxes.shape[0]

        if total_gt == 0:
            return None, 0.0, 0.0

        class_preds = []
        for sample in self.predictions:
            mask = sample["labels"] == class_id
            if mask.any():
                for box, score in zip(sample["boxes"][mask], sample["scores"][mask]):
                    if area_range is not None:
                        box_area = float(((box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)).item())
                        if not (float(area_range[0]) <= box_area < float(area_range[1])):
                            continue
                    class_preds.append(dict(image_id=sample["image_id"], box=box, score=score.item()))
        class_preds.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(class_preds), dtype=np.float32)
        fp = np.zeros(len(class_preds), dtype=np.float32)

        for idx, pred in enumerate(class_preds):
            image_gts = gt_by_image.get(pred["image_id"], [])
            best_iou = 0.0
            best_group = None
            best_gt_idx = -1
            pred_box = pred["box"][None, :]

            for group in image_gts:
                if group["boxes"].numel() == 0:
                    continue
                ious = bbox_iou(pred_box, group["boxes"]).squeeze(0)
                iou_val, gt_idx = ious.max(dim=0)
                if iou_val.item() > best_iou:
                    best_iou = iou_val.item()
                    best_group = group
                    best_gt_idx = gt_idx.item()

            if best_group is not None and best_iou >= iou_threshold and not best_group["matched"][best_gt_idx]:
                tp[idx] = 1.0
                best_group["matched"][best_gt_idx] = True
            else:
                fp[idx] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(total_gt, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
        ap = compute_average_precision(recalls, precisions) if len(recalls) else 0.0
        precision = float(precisions[-1]) if len(precisions) else 0.0
        recall = float(recalls[-1]) if len(recalls) else 0.0
        return ap, precision, recall

    def summarize(self, bit_width=3, class_names=None):
        aps_50 = []
        aps_5095 = []
        precisions = []
        recalls = []
        classwise = []
        for class_id in range(self.num_classes):
            ap50_result = self._evaluate_class(class_id, 0.5)
            class_name = class_names[class_id] if class_names is not None and class_id < len(class_names) else str(class_id)
            if ap50_result[0] is None:
                classwise.append(
                    dict(
                        class_id=class_id,
                        class_name=class_name,
                        AP50=round(0.0, bit_width),
                        AP50_95=round(0.0, bit_width),
                        Precision=round(0.0, bit_width),
                        Recall=round(0.0, bit_width),
                    )
                )
                continue

            aps_50.append(ap50_result[0])
            precisions.append(ap50_result[1])
            recalls.append(ap50_result[2])

            class_aps = []
            for thr in self.iou_thresholds:
                ap_result = self._evaluate_class(class_id, thr)
                if ap_result[0] is not None:
                    class_aps.append(ap_result[0])
            ap5095 = sum(class_aps) / len(class_aps) if class_aps else 0.0
            aps_5095.append(ap5095)
            classwise.append(
                dict(
                    class_id=class_id,
                    class_name=class_name,
                    AP50=round(float(ap50_result[0]), bit_width),
                    AP50_95=round(float(ap5095), bit_width),
                    Precision=round(float(ap50_result[1]), bit_width),
                    Recall=round(float(ap50_result[2]), bit_width),
                )
            )

        summary = dict(
            mAP50=round(float(sum(aps_50) / max(len(aps_50), 1)), bit_width),
            mAP50_95=round(float(sum(aps_5095) / max(len(aps_5095), 1)), bit_width),
            Precision=round(float(sum(precisions) / max(len(precisions), 1)), bit_width),
            Recall=round(float(sum(recalls) / max(len(recalls), 1)), bit_width),
        )
        area_summary = {}
        for area_name, area_range in self.area_ranges.items():
            area_aps = []
            for class_id in range(self.num_classes):
                ap50_result = self._evaluate_class(class_id, 0.5, area_range=area_range)
                if ap50_result[0] is not None:
                    area_aps.append(ap50_result[0])
            metric_name = f"AP50_{area_name}"
            area_summary[metric_name] = round(float(sum(area_aps) / max(len(area_aps), 1)), bit_width)
        summary.update(area_summary)
        return summary, classwise

    def get_results(self, bit_width=3):
        results, _ = self.summarize(bit_width=bit_width)
        return results

    def get_detailed_results(self, bit_width=3, class_names=None):
        summary, classwise = self.summarize(bit_width=bit_width, class_names=class_names)
        return dict(summary=summary, classwise=classwise)


class COCODetectionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_info, shape, is_train, train_aug_cfg=None):
        super().__init__()
        self.dataset_info = dataset_info
        self.image_root = dataset_info["image_root"]
        self.depth_root = dataset_info["depth_root"]
        self.ann_file = dataset_info["ann_file"]
        self.depth_file_key = dataset_info.get("depth_file_key", "depth_file_name")
        self.depth_suffix = dataset_info.get("depth_suffix", None)
        self.is_train = is_train
        self.shape = shape
        self.train_aug_cfg = train_aug_cfg or {}

        with open(self.ann_file, encoding="utf-8", mode="r") as f:
            coco = json.load(f)

        self.images = {item["id"]: item for item in coco["images"]}
        self.cat_id_to_label = {cat["id"]: idx for idx, cat in enumerate(sorted(coco["categories"], key=lambda x: x["id"]))}
        self.class_names = tuple(cat["name"] for cat in sorted(coco["categories"], key=lambda x: x["id"]))

        ann_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            ann_by_image[ann["image_id"]].append(ann)
        self.samples = []
        for image_id, image_info in self.images.items():
            anns = ann_by_image.get(image_id, [])
            self.samples.append(dict(image_info=image_info, annotations=anns))

        joint_transforms = [A.Resize(height=shape["h"], width=shape["w"])]
        if is_train:
            aug_mode = str(self.train_aug_cfg.get("mode", "ground")).lower()
            if aug_mode == "aerial":
                affine_cfg = self.train_aug_cfg.get("affine", {})
                joint_transforms = [
                    A.RandomRotate90(p=float(self.train_aug_cfg.get("random_rotate90_p", 0.5))),
                    A.HorizontalFlip(p=float(self.train_aug_cfg.get("horizontal_flip_p", 0.5))),
                    A.VerticalFlip(p=float(self.train_aug_cfg.get("vertical_flip_p", 0.5))),
                    A.Affine(
                        scale=tuple(affine_cfg.get("scale", (0.9, 1.1))),
                        translate_percent=tuple(affine_cfg.get("translate_percent", (-0.04, 0.04))),
                        rotate=float(affine_cfg.get("rotate", 0.0)),
                        shear=float(affine_cfg.get("shear", 0.0)),
                        p=float(affine_cfg.get("p", 0.6)),
                    ),
                    A.Resize(height=shape["h"], width=shape["w"]),
                ]
            else:
                joint_transforms = [
                    A.Affine(
                        scale=tuple(self.train_aug_cfg.get("scale", (0.85, 1.15))),
                        translate_percent=tuple(self.train_aug_cfg.get("translate_percent", (-0.05, 0.05))),
                        rotate=float(self.train_aug_cfg.get("rotate", 0.0)),
                        shear=float(self.train_aug_cfg.get("shear", 0.0)),
                        p=float(self.train_aug_cfg.get("affine_p", 0.7)),
                    ),
                    A.HorizontalFlip(p=float(self.train_aug_cfg.get("horizontal_flip_p", 0.5))),
                    A.Resize(height=shape["h"], width=shape["w"]),
                ]

        self.joint_trans = A.Compose(
            joint_transforms,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            additional_targets=dict(depth="image"),
        )
        self.image_only_trans = (
            A.Compose(
                [
                    A.ColorJitter(
                        brightness=float(self.train_aug_cfg.get("brightness", 0.2)),
                        contrast=float(self.train_aug_cfg.get("contrast", 0.2)),
                        saturation=float(self.train_aug_cfg.get("saturation", 0.1)),
                        hue=float(self.train_aug_cfg.get("hue", 0.02)),
                        p=float(self.train_aug_cfg.get("color_jitter_p", 0.6)),
                    ),
                    A.GaussianBlur(
                        blur_limit=int(self.train_aug_cfg.get("blur_limit", 3)),
                        p=float(self.train_aug_cfg.get("gaussian_blur_p", 0.1)),
                    ),
                ]
            )
            if is_train
            else None
        )
        self.normalize = A.Normalize()

    def __len__(self):
        return len(self.samples)

    def _resolve_depth_name(self, image_info):
        if self.depth_file_key in image_info:
            return image_info[self.depth_file_key]
        image_name = image_info["file_name"]
        if self.depth_suffix is None:
            return image_name
        image_stem = os.path.splitext(image_name)[0]
        return image_stem + self.depth_suffix

    def __getitem__(self, index):
        sample = self.samples[index]
        image_info = sample["image_info"]
        source_file_name = image_info.get("source_file_name", image_info["file_name"])
        image_path = os.path.join(self.image_root, source_file_name)
        depth_path = os.path.join(self.depth_root, self._resolve_depth_name(image_info))

        image = read_color_array(image_path)
        depth = read_binary_array(depth_path, to_normalize=True, thr=-1)
        tile_box = image_info.get("tile_box", None)
        if tile_box is not None:
            tile_x, tile_y, tile_w, tile_h = [int(round(value)) for value in tile_box]
            image = image[tile_y: tile_y + tile_h, tile_x: tile_x + tile_w].copy()
            depth = depth[tile_y: tile_y + tile_h, tile_x: tile_x + tile_w].copy()
        crop_h, crop_w = image.shape[:2]

        bboxes = []
        labels = []
        for ann in sample["annotations"]:
            bbox = np.array(ann["bbox"], dtype=np.float32)[None, :]
            bbox = xywh_to_xyxy(bbox)[0].tolist()
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            bboxes.append(bbox)
            labels.append(self.cat_id_to_label[ann["category_id"]])

        transformed = self.joint_trans(image=image, depth=depth, bboxes=bboxes, class_labels=labels)
        image = transformed["image"]
        depth = transformed["depth"]
        bboxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4) if transformed["bboxes"] else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(transformed["class_labels"], dtype=np.int64) if transformed["class_labels"] else np.zeros((0,), dtype=np.int64)

        if self.image_only_trans is not None:
            image = self.image_only_trans(image=image)["image"]

        image = self.normalize(image=image)["image"]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        depth_tensor = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)

        target = dict(
            boxes=torch.from_numpy(bboxes).float(),
            labels=torch.from_numpy(labels).long(),
            image_id=torch.tensor(int(image_info["id"]), dtype=torch.long),
            orig_size=torch.tensor([crop_h, crop_w], dtype=torch.long),
            size=torch.tensor([self.shape["h"], self.shape["w"]], dtype=torch.long),
        )
        packed_image_info = dict(file_name=image_info["file_name"], image_id=int(image_info["id"]))
        for optional_key in ("source_file_name", "tile_box", "parent_image_id", "tile_id", "tile_name"):
            if optional_key in image_info:
                packed_image_info[optional_key] = image_info[optional_key]
        return dict(
            image=image_tensor,
            depth=depth_tensor,
            targets=target,
            image_info=packed_image_info,
        )


class BimodalDetectionTask(BaseTask):
    name = "bimodal_detection"
    metric_names = ("mAP50", "mAP50_95", "AP50_small", "AP50_medium", "Precision", "Recall", "score_thr", "nms_iou_thr")
    primary_metric_name = "mAP50_95"

    def __init__(self, cfg):
        super().__init__(cfg)
        task_cfg = cfg.get("task", {})
        self.metric_precision = int(cfg.get("metric_precision", task_cfg.get("metric_precision", 3)))
        self.pred_hist_num_images = int(task_cfg.get("pred_hist_num_images", cfg.get("pred_hist_num_images", 100)))
        self.train_sampler_cfg = cfg.get("train_sampler", task_cfg.get("train_sampler", dict(enable=False)))
        self.train_aug_cfg = task_cfg.get("train_aug", {})
        self.num_classes = int(task_cfg.get("num_classes", cfg.get("model", {}).get("num_classes", 1)))
        self.strides = tuple(task_cfg.get("strides", cfg.get("model", {}).get("strides", (8, 16, 32))))
        area_metric_ranges = task_cfg.get(
            "area_metric_ranges",
            dict(small=(0, 32 * 32), medium=(32 * 32, 96 * 96)),
        )
        self.area_metric_ranges = {name: tuple(value) for name, value in area_metric_ranges.items()}
        assigner_cfg = task_cfg.get("assigner", {})
        default_assigner_mode = "fcos" if assigner_cfg else "legacy"
        self.assigner_mode = str(assigner_cfg.get("mode", default_assigner_mode)).lower()
        if self.assigner_mode not in {"legacy", "fcos"}:
            raise ValueError(f"Unsupported assigner mode: {self.assigner_mode}. Expected one of: legacy, fcos.")
        self.center_radius = float(assigner_cfg.get("center_radius", task_cfg.get("center_radius", 2.5)))
        self.regress_ranges = tuple(
            tuple(item)
            for item in assigner_cfg.get("regress_ranges", ((0, 64), (64, 128), (128, 1e8)))
        )
        if self.assigner_mode == "legacy":
            self.use_stride_normalized_reg = False
        else:
            self.use_stride_normalized_reg = bool(assigner_cfg.get("use_stride_normalized_reg", True))
        self.loss_cfg = task_cfg.get(
            "loss",
            dict(
                cls_weight=1.0,
                reg_weight=1.0,
                distill_weight=1.0,
                focal_alpha=0.25,
                focal_gamma=2.0,
                pos_ce_weight=1.0,
                reg_l1_weight=0.25,
            ),
        )
        class_weights = self.loss_cfg.get("class_weights", None)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        if self.class_weights is not None and len(self.class_weights) != self.num_classes:
            raise ValueError(
                f"class_weights length ({len(self.class_weights)}) must match num_classes ({self.num_classes})."
            )
        self.infer_cfg = task_cfg.get(
            "inference",
            dict(score_thr=0.05, nms_iou_thr=0.6, pre_nms_topk=1000, max_per_img=300),
        )
        self.sweep_score_thrs = tuple(float(x) for x in self.infer_cfg.get("sweep_score_thrs", ()))
        self.sweep_nms_iou_thrs = tuple(float(x) for x in self.infer_cfg.get("sweep_nms_iou_thrs", ()))
        if len(self.regress_ranges) != len(self.strides):
            raise ValueError(
                f"regress_ranges length ({len(self.regress_ranges)}) must match strides length ({len(self.strides)})."
            )
        self.class_names = None
        self._vis_palette = (
            (255, 99, 71),
            (30, 144, 255),
            (34, 139, 34),
            (255, 215, 0),
            (186, 85, 211),
            (255, 140, 0),
        )

    def get_collate_fn(self, split):
        def collate_fn(batch):
            return dict(
                image=torch.stack([item["image"] for item in batch], dim=0),
                depth=torch.stack([item["depth"] for item in batch], dim=0),
                targets=[item["targets"] for item in batch],
                image_info=[item["image_info"] for item in batch],
            )

        return collate_fn

    def _build_dataset_sampling_weights(self, dataset):
        if hasattr(dataset, "datasets"):
            weights = []
            for sub_dataset in dataset.datasets:
                weights.extend(self._build_dataset_sampling_weights(sub_dataset))
            return weights
        if not hasattr(dataset, "samples"):
            return None

        sampler_cfg = self.train_sampler_cfg or {}
        min_weight = float(sampler_cfg.get("min_weight", 1.0))
        max_weight = float(sampler_cfg.get("max_weight", 2.0))
        class_weights = self.class_weights.tolist() if self.class_weights is not None else [1.0] * self.num_classes
        weights = []
        for sample in dataset.samples:
            sample_labels = []
            for ann in sample["annotations"]:
                label = dataset.cat_id_to_label.get(ann["category_id"], None)
                if label is not None:
                    sample_labels.append(label)
            if sample_labels:
                sample_weight = max(class_weights[label] for label in sample_labels)
            else:
                sample_weight = 1.0
            sample_weight = max(min_weight, min(max_weight, float(sample_weight)))
            weights.append(sample_weight)
        return weights

    def get_train_sampler(self, train_dataset, cfg=None):
        sampler_cfg = self.train_sampler_cfg or {}
        if not bool(sampler_cfg.get("enable", False)):
            return None
        sampler_mode = str(sampler_cfg.get("mode", "class_weighted")).lower()
        if sampler_mode == "balanced_concat":
            dataset_weights = sampler_cfg.get("dataset_weights", None)
            samples_per_dataset = sampler_cfg.get("samples_per_dataset", "min")
            base_seed = cfg.args.base_seed if cfg is not None else self.cfg.args.base_seed
            return BalancedConcatSampler(
                concat_dataset=train_dataset,
                base_seed=base_seed,
                samples_per_dataset=samples_per_dataset,
                dataset_weights=dataset_weights,
            )
        if sampler_mode != "class_weighted":
            raise ValueError(f"Unsupported train_sampler mode: {sampler_mode}")
        weights = self._build_dataset_sampling_weights(train_dataset)
        if not weights:
            return None
        weights = torch.tensor(weights, dtype=torch.double)
        return torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def _get_dataset_info(self, split, dataset_name=None):
        split_cfg = self.cfg.data.train if split == "train" else self.cfg.data.test
        dataset_infos = split_cfg.get("dataset_infos", {})
        if dataset_name is not None:
            return dataset_infos[dataset_name]
        return {name: dataset_infos[name] for name in split_cfg.name}

    def build_train_dataset(self, cfg):
        train_infos = self._get_dataset_info(split="train")
        datasets = [
            COCODetectionDataset(info, shape=cfg.data.train.shape, is_train=True, train_aug_cfg=self.train_aug_cfg)
            for info in train_infos.values()
        ]
        if datasets:
            self.class_names = tuple(datasets[0].class_names)
        if len(datasets) == 1:
            return datasets[0]
        return torch.utils.data.ConcatDataset(datasets)

    def build_test_dataset(self, dataset_name, cfg):
        dataset_info = self._get_dataset_info(split="test", dataset_name=dataset_name)
        dataset = COCODetectionDataset(dataset_info, shape=cfg.data.test.shape, is_train=False, train_aug_cfg=self.train_aug_cfg)
        self.class_names = tuple(dataset.class_names)
        return dataset, dataset_info

    def get_model_inputs(self, batch):
        return dict(image=batch["image"], depth=batch["depth"], targets=batch.get("targets"))

    def _generate_points(self, cls_logits, device):
        all_points = []
        all_strides = []
        all_ranges = []
        for level, level_logits in enumerate(cls_logits):
            _, _, h, w = level_logits.shape
            stride = self.strides[level]
            lower, upper = self.regress_ranges[level]
            ys = (torch.arange(h, device=device) + 0.5) * stride
            xs = (torch.arange(w, device=device) + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
            all_points.append(points)
            all_strides.append(points.new_full((points.shape[0],), stride))
            all_ranges.append(points.new_tensor([lower, upper]).view(1, 2).repeat(points.shape[0], 1))
        return torch.cat(all_points, dim=0), torch.cat(all_strides, dim=0), torch.cat(all_ranges, dim=0)

    def _flatten_outputs(self, model_outputs):
        cls_logits = []
        bbox_preds = []
        for level_cls, level_box in zip(model_outputs["cls_logits"], model_outputs["bbox_preds"]):
            b, c, h, w = level_cls.shape
            cls_logits.append(level_cls.permute(0, 2, 3, 1).reshape(b, h * w, c))
            bbox_preds.append(level_box.permute(0, 2, 3, 1).reshape(b, h * w, 4))
        return torch.cat(cls_logits, dim=1), torch.cat(bbox_preds, dim=1)

    def _assign_targets(self, points, point_strides, point_ranges, targets, device):
        num_points = points.shape[0]
        cls_targets = torch.zeros(num_points, self.num_classes, device=device)
        reg_targets = torch.zeros(num_points, 4, device=device)
        box_targets = torch.zeros(num_points, 4, device=device)
        positive_mask = torch.zeros(num_points, dtype=torch.bool, device=device)

        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]
        if gt_boxes.numel() == 0:
            return cls_targets, reg_targets, box_targets, positive_mask

        xs = points[:, 0][:, None]
        ys = points[:, 1][:, None]
        left = xs - gt_boxes[None, :, 0]
        top = ys - gt_boxes[None, :, 1]
        right = gt_boxes[None, :, 2] - xs
        bottom = gt_boxes[None, :, 3] - ys
        reg_deltas = torch.stack([left, top, right, bottom], dim=-1)
        inside_boxes = reg_deltas.min(dim=-1).values > 0

        centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        radius = point_strides[:, None] * self.center_radius
        center_boxes = torch.stack(
            [
                torch.maximum(centers[None, :, 0] - radius, gt_boxes[None, :, 0]),
                torch.maximum(centers[None, :, 1] - radius, gt_boxes[None, :, 1]),
                torch.minimum(centers[None, :, 0] + radius, gt_boxes[None, :, 2]),
                torch.minimum(centers[None, :, 1] + radius, gt_boxes[None, :, 3]),
            ],
            dim=-1,
        )
        center_left = xs - center_boxes[..., 0]
        center_top = ys - center_boxes[..., 1]
        center_right = center_boxes[..., 2] - xs
        center_bottom = center_boxes[..., 3] - ys
        inside_centers = torch.stack([center_left, center_top, center_right, center_bottom], dim=-1).min(dim=-1).values > 0

        if self.assigner_mode == "legacy":
            candidate_mask = inside_boxes & inside_centers
        else:
            max_reg_deltas = reg_deltas.max(dim=-1).values
            lower_bound = point_ranges[:, 0][:, None]
            upper_bound = point_ranges[:, 1][:, None]
            in_range = (max_reg_deltas >= lower_bound) & (max_reg_deltas <= upper_bound)
            positive_lower = lower_bound > 0
            in_range = torch.where(positive_lower, max_reg_deltas > lower_bound, in_range) & (max_reg_deltas <= upper_bound)
            candidate_mask = inside_boxes & inside_centers & in_range
        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        areas = areas[None, :].expand(num_points, -1).clone()
        areas[~candidate_mask] = float("inf")
        min_areas, min_indices = areas.min(dim=1)
        positive_mask = torch.isfinite(min_areas)

        if positive_mask.any():
            chosen_gt = min_indices[positive_mask]
            chosen_labels = gt_labels[chosen_gt]
            cls_targets[positive_mask, chosen_labels] = 1.0
            reg_targets[positive_mask] = reg_deltas[positive_mask, chosen_gt]
            if self.use_stride_normalized_reg:
                reg_targets[positive_mask] = reg_targets[positive_mask] / point_strides[positive_mask, None]
            box_targets[positive_mask] = gt_boxes[chosen_gt]
        return cls_targets, reg_targets, box_targets, positive_mask

    @staticmethod
    def _distances_to_boxes(points, distances, point_strides=None):
        if point_strides is not None:
            distances = distances * point_strides[:, None]
        return torch.stack(
            [
                points[:, 0] - distances[:, 0],
                points[:, 1] - distances[:, 1],
                points[:, 0] + distances[:, 2],
                points[:, 1] + distances[:, 3],
            ],
            dim=1,
        )

    def compute_loss(self, model_outputs, batch):
        device = batch["image"].device
        cls_logits, bbox_preds = self._flatten_outputs(model_outputs)
        points, point_strides, point_ranges = self._generate_points(model_outputs["cls_logits"], device=device)

        total_focal_loss = cls_logits.new_tensor(0.0)
        total_pos_ce_loss = cls_logits.new_tensor(0.0)
        total_ciou_loss = cls_logits.new_tensor(0.0)
        total_l1_loss = cls_logits.new_tensor(0.0)
        total_pos = 0
        class_weights = self.class_weights.to(device=device) if self.class_weights is not None else None
        pos_ce_weight = float(self.loss_cfg.get("pos_ce_weight", 1.0))
        reg_l1_weight = float(self.loss_cfg.get("reg_l1_weight", 0.25))

        for batch_idx, targets in enumerate(batch["targets"]):
            cls_target, reg_target, box_target, positive_mask = self._assign_targets(
                points,
                point_strides,
                point_ranges,
                targets,
                device,
            )
            cls_loss = focal_loss_with_logits(
                cls_logits[batch_idx],
                cls_target,
                alpha=self.loss_cfg.get("focal_alpha", 0.25),
                gamma=self.loss_cfg.get("focal_gamma", 2.0),
                class_weights=class_weights,
            ).sum()
            total_focal_loss = total_focal_loss + cls_loss

            num_pos = int(positive_mask.sum().item())
            total_pos += num_pos
            if num_pos > 0:
                if pos_ce_weight > 0:
                    positive_labels = cls_target[positive_mask].argmax(dim=1)
                    ce_class_weights = class_weights.to(dtype=cls_logits[batch_idx].dtype) if class_weights is not None else None
                    pos_ce_loss = F.cross_entropy(
                        cls_logits[batch_idx][positive_mask],
                        positive_labels,
                        weight=ce_class_weights,
                        reduction="sum",
                    )
                    total_pos_ce_loss = total_pos_ce_loss + pos_ce_loss
                positive_points = points[positive_mask]
                positive_strides = point_strides[positive_mask]
                pred_distances = bbox_preds[batch_idx][positive_mask]
                pred_boxes = self._distances_to_boxes(
                    positive_points,
                    pred_distances,
                    point_strides=positive_strides if self.use_stride_normalized_reg else None,
                )
                ciou_reg_loss = ciou_loss(pred_boxes, box_target[positive_mask]).sum()
                total_ciou_loss = total_ciou_loss + ciou_reg_loss
                if reg_l1_weight > 0:
                    l1_reg_loss = F.smooth_l1_loss(pred_distances, reg_target[positive_mask], reduction="sum")
                    total_l1_loss = total_l1_loss + l1_reg_loss

        normalizer = max(total_pos, 1)
        focal_cls_loss = total_focal_loss / normalizer
        pos_ce_loss = total_pos_ce_loss / normalizer if total_pos > 0 and pos_ce_weight > 0 else total_focal_loss.new_tensor(0.0)
        cls_loss = focal_cls_loss + pos_ce_weight * pos_ce_loss
        ciou_reg_loss = total_ciou_loss / normalizer if total_pos > 0 else total_focal_loss.new_tensor(0.0)
        l1_reg_loss = total_l1_loss / normalizer if total_pos > 0 else total_focal_loss.new_tensor(0.0)
        reg_loss = ciou_reg_loss + reg_l1_weight * l1_reg_loss

        student_prior = model_outputs.get("student_prior", None)
        teacher_prior = model_outputs.get("teacher_prior", None)
        prior_mse = focal_cls_loss.new_tensor(0.0)
        prior_cos = focal_cls_loss.new_tensor(0.0)
        if teacher_prior is not None and student_prior is not None:
            student_resized = F.interpolate(student_prior, size=teacher_prior.shape[-2:], mode="bilinear", align_corners=False)
            teacher_detached = teacher_prior.detach()
            prior_mse = F.mse_loss(student_resized, teacher_detached)
            prior_cos = F.cosine_similarity(student_resized.flatten(1), teacher_detached.flatten(1), dim=1).mean()
            distill_loss = prior_mse
        else:
            distill_loss = focal_cls_loss.new_tensor(0.0)

        weighted_cls = self.loss_cfg.get("cls_weight", 1.0) * cls_loss
        weighted_reg = self.loss_cfg.get("reg_weight", 1.0) * reg_loss
        weighted_distill = self.loss_cfg.get("distill_weight", 1.0) * distill_loss
        total_loss = weighted_cls + weighted_reg + weighted_distill
        loss_str = (
            f"focal:{focal_cls_loss.item():.5f} posce:{pos_ce_loss.item():.5f} cls:{cls_loss.item():.5f} "
            f"ciou:{ciou_reg_loss.item():.5f} l1:{l1_reg_loss.item():.5f} reg:{reg_loss.item():.5f} "
            f"distill:{distill_loss.item():.5f} prior_mse:{prior_mse.item():.5f} prior_cos:{prior_cos.item():.5f} pos:{total_pos}"
        )
        return total_loss, loss_str

    @staticmethod
    def _empty_prediction(device):
        return dict(
            boxes=torch.zeros((0, 4), device=device),
            scores=torch.zeros((0,), device=device),
            labels=torch.zeros((0,), dtype=torch.long, device=device),
        )

    def _collect_candidates_single(self, cls_logits, bbox_preds, points, point_strides, image_size, score_thr, pre_nms_topk):
        scores = cls_logits.sigmoid()
        point_scores, labels = scores.max(dim=1)
        keep = point_scores > score_thr
        if not keep.any():
            return self._empty_prediction(device=scores.device)

        keep_indices = torch.nonzero(keep, as_tuple=False).flatten()
        if keep_indices.numel() > pre_nms_topk:
            _, topk_idx = point_scores[keep_indices].topk(pre_nms_topk)
            keep_indices = keep_indices[topk_idx]

        point_indices = keep_indices
        selected_labels = labels[keep_indices]
        selected_scores = point_scores[keep_indices]
        selected_boxes = self._distances_to_boxes(
            points[point_indices],
            bbox_preds[point_indices],
            point_strides=point_strides[point_indices] if self.use_stride_normalized_reg else None,
        )
        h, w = image_size
        selected_boxes[:, 0::2] = selected_boxes[:, 0::2].clamp(min=0, max=float(w))
        selected_boxes[:, 1::2] = selected_boxes[:, 1::2].clamp(min=0, max=float(h))
        return dict(boxes=selected_boxes, scores=selected_scores, labels=selected_labels)

    def _finalize_candidates_single(self, candidates, score_thr, nms_iou_thr, max_per_img):
        if candidates["scores"].numel() == 0:
            return self._empty_prediction(device=candidates["scores"].device)
        score_mask = candidates["scores"] > score_thr
        if not score_mask.any():
            return self._empty_prediction(device=candidates["scores"].device)

        selected_boxes = candidates["boxes"][score_mask]
        selected_scores = candidates["scores"][score_mask]
        selected_labels = candidates["labels"][score_mask]
        keep_after_nms = batched_nms(
            selected_boxes,
            selected_scores,
            selected_labels,
            iou_threshold=nms_iou_thr,
        )
        if keep_after_nms.numel() > max_per_img:
            keep_after_nms = keep_after_nms[:max_per_img]

        return dict(
            boxes=selected_boxes[keep_after_nms],
            scores=selected_scores[keep_after_nms],
            labels=selected_labels[keep_after_nms],
        )

    def _collect_predictions(self, model_outputs, batch, score_thr, pre_nms_topk):
        device = model_outputs["cls_logits"][0].device
        cls_logits, bbox_preds = self._flatten_outputs(model_outputs)
        points, point_strides, _ = self._generate_points(model_outputs["cls_logits"], device=device)

        predictions = []
        for batch_idx in range(cls_logits.shape[0]):
            size = batch["targets"][batch_idx]["size"].tolist()
            pred = self._collect_candidates_single(
                cls_logits[batch_idx],
                bbox_preds[batch_idx],
                points,
                point_strides,
                image_size=size,
                score_thr=score_thr,
                pre_nms_topk=pre_nms_topk,
            )
            pred["image_id"] = int(batch["targets"][batch_idx]["image_id"].item())
            predictions.append(pred)
        return predictions

    def _finalize_predictions(self, candidate_predictions, score_thr, nms_iou_thr):
        finalized_predictions = []
        for pred in candidate_predictions:
            finalized = self._finalize_candidates_single(
                pred,
                score_thr=score_thr,
                nms_iou_thr=nms_iou_thr,
                max_per_img=self.infer_cfg.get("max_per_img", 300),
            )
            finalized["image_id"] = pred["image_id"]
            finalized_predictions.append(finalized)
        return finalized_predictions

    def predict(self, model_outputs, batch, infer_cfg=None):
        infer_cfg = infer_cfg or self.infer_cfg
        candidate_predictions = self._collect_predictions(
            model_outputs=model_outputs,
            batch=batch,
            score_thr=infer_cfg.get("score_thr", 0.05),
            pre_nms_topk=infer_cfg.get("pre_nms_topk", 1000),
        )
        return self._finalize_predictions(
            candidate_predictions=candidate_predictions,
            score_thr=infer_cfg.get("score_thr", 0.05),
            nms_iou_thr=infer_cfg.get("nms_iou_thr", 0.6),
        )

    @staticmethod
    def _denormalize_image(image):
        mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return (image * std + mean).clamp(0, 1)

    @staticmethod
    def _depth_to_rgb(depth):
        return depth.float().clamp(0, 1).repeat(3, 1, 1)

    @staticmethod
    def _prior_to_heatmap(prior, target_size):
        prior_map = prior.detach().float().mean(dim=1, keepdim=True)
        prior_map = F.interpolate(prior_map, size=target_size, mode="bilinear", align_corners=False)
        prior_map = prior_map - prior_map.amin(dim=(-2, -1), keepdim=True)
        prior_map = prior_map / (prior_map.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return prior_map.repeat(1, 3, 1, 1).cpu()

    def _draw_boxes(self, image, boxes, labels, scores=None, max_boxes=20):
        canvas = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        class_names = self.class_names or tuple(str(idx) for idx in range(self.num_classes))
        total = min(len(boxes), max_boxes)
        for idx in range(total):
            box = boxes[idx].tolist()
            label = int(labels[idx])
            color = self._vis_palette[label % len(self._vis_palette)]
            x1, y1, x2, y2 = [int(round(value)) for value in box]
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, canvas.shape[1] - 1)
            y2 = min(y2, canvas.shape[0] - 1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=color, thickness=2)
            text = class_names[label] if label < len(class_names) else str(label)
            if scores is not None:
                text += f":{float(scores[idx]):.2f}"
            cv2.putText(
                canvas,
                text,
                (x1, max(y1 - 6, 18)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.55,
                color=color,
                thickness=2,
            )
        return torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1)

    def get_visualization_data(self, model_outputs, batch):
        if "image" not in batch or batch["image"].numel() == 0:
            return None

        max_items = min(int(batch["image"].shape[0]), 2)
        predictions = self.predict(model_outputs=model_outputs, batch=batch)

        images = torch.stack([self._denormalize_image(item) for item in batch["image"][:max_items]], dim=0).cpu()
        depths = torch.stack([self._depth_to_rgb(item) for item in batch["depth"][:max_items]], dim=0).cpu()

        gt_views = []
        pred_views = []
        for idx in range(max_items):
            target = batch["targets"][idx]
            pred = predictions[idx]
            gt_views.append(self._draw_boxes(images[idx], target["boxes"], target["labels"]))
            pred_views.append(
                self._draw_boxes(
                    images[idx],
                    pred["boxes"].detach().cpu(),
                    pred["labels"].detach().cpu(),
                    scores=pred["scores"].detach().cpu(),
                )
            )

        vis_data = dict(
            rgb=images,
            depth=depths,
            gt_boxes=torch.stack(gt_views, dim=0),
            pred_boxes=torch.stack(pred_views, dim=0),
            student_prior=self._prior_to_heatmap(
                model_outputs["student_prior"][:max_items],
                target_size=batch["image"].shape[-2:],
            ),
        )
        teacher_prior = model_outputs.get("teacher_prior", None)
        if teacher_prior is not None:
            vis_data["teacher_prior"] = self._prior_to_heatmap(
                teacher_prior[:max_items],
                target_size=batch["image"].shape[-2:],
            )
        return vis_data

    def _attach_inference_cfg(self, results, score_thr, nms_iou_thr):
        merged = dict(results)
        merged["score_thr"] = round(float(score_thr), max(self.metric_precision, 4))
        merged["nms_iou_thr"] = round(float(nms_iou_thr), max(self.metric_precision, 4))
        return merged

    def _build_pred_class_hist(self, predictions):
        hist = Counter()
        max_images = min(len(predictions), self.pred_hist_num_images)
        for pred in predictions[:max_images]:
            hist.update(int(label) for label in pred["labels"].tolist())
        class_names = self.class_names or tuple(str(idx) for idx in range(self.num_classes))
        return {class_names[idx] if idx < len(class_names) else str(idx): int(hist.get(idx, 0)) for idx in range(self.num_classes)}

    def format_eval_details(self, eval_results):
        details = eval_results.get("details", {})
        lines = []
        area_summary = details.get("area_summary", None)
        if area_summary:
            lines.append(
                "scale_metrics "
                + " ".join(f"{name}:{value}" for name, value in area_summary.items())
            )
        pred_class_hist = details.get("pred_class_hist", None)
        if pred_class_hist:
            images = int(details.get("pred_hist_images", 0))
            hist_text = ", ".join(f"{name}:{count}" for name, count in pred_class_hist.items())
            lines.append(
                f"pred_class_hist@{images}imgs {hist_text} dominant:{details.get('dominant_pred_class', 'NA')} "
                f"ratio:{details.get('dominant_pred_ratio', 0.0)}"
            )
        for item in details.get("classwise", []):
            lines.append(
                f"class:{item['class_name']} AP50:{item['AP50']} AP50_95:{item['AP50_95']} "
                f"Precision:{item['Precision']} Recall:{item['Recall']}"
            )
        return lines

    @staticmethod
    def _is_better_eval_result(candidate, best):
        if best is None:
            return True
        candidate_key = (
            float(candidate["mAP50_95"]),
            float(candidate["mAP50"]),
            float(candidate["Precision"]),
        )
        best_key = (
            float(best["mAP50_95"]),
            float(best["mAP50"]),
            float(best["Precision"]),
        )
        return candidate_key > best_key

    @staticmethod
    def _dump_predictions(predictions, image_infos, save_path):
        for pred, info in zip(predictions, image_infos):
            pred_path = os.path.join(save_path, f"{os.path.splitext(info['file_name'])[0]}.json")
            with open(pred_path, encoding="utf-8", mode="w") as f:
                json.dump(
                    dict(
                        image_id=pred["image_id"],
                        boxes=pred["boxes"].cpu().tolist(),
                        scores=pred["scores"].cpu().tolist(),
                        labels=pred["labels"].cpu().tolist(),
                    ),
                    f,
                    ensure_ascii=False,
                )

    @torch.no_grad()
    def evaluate_once(self, model, data_loader, save_path="", show_bar=True, calibrate_inference=False):
        model.eval()

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        min_score_thr = self.infer_cfg.get("score_thr", 0.05)
        if calibrate_inference and self.sweep_score_thrs:
            min_score_thr = min(self.sweep_score_thrs)
        pre_nms_topk = self.infer_cfg.get("pre_nms_topk", 1000)

        candidate_predictions = []
        all_targets = []
        all_image_infos = []
        bar_iter = enumerate(data_loader)
        if show_bar:
            bar_iter = tqdm(bar_iter, total=len(data_loader), leave=False, ncols=79)

        for _, batch in bar_iter:
            device_batch = self.move_batch_to_device(batch)
            model_outputs = model(data=self.get_model_inputs(device_batch))
            batch_candidates = self._collect_predictions(
                model_outputs=model_outputs,
                batch=device_batch,
                score_thr=min_score_thr,
                pre_nms_topk=pre_nms_topk,
            )
            cpu_predictions = [
                dict(
                    boxes=pred["boxes"].detach().cpu(),
                    scores=pred["scores"].detach().cpu(),
                    labels=pred["labels"].detach().cpu(),
                    image_id=pred["image_id"],
                )
                for pred in batch_candidates
            ]
            targets = []
            for target in batch["targets"]:
                targets.append(
                    dict(
                        boxes=target["boxes"].float(),
                        labels=target["labels"].long(),
                        image_id=int(target["image_id"].item()),
                    )
                )
            candidate_predictions.extend(cpu_predictions)
            all_targets.extend(targets)
            all_image_infos.extend(batch["image_info"])

        search_score_thrs = (self.infer_cfg.get("score_thr", 0.05),)
        search_nms_iou_thrs = (self.infer_cfg.get("nms_iou_thr", 0.6),)
        if calibrate_inference and self.sweep_score_thrs and self.sweep_nms_iou_thrs:
            search_score_thrs = self.sweep_score_thrs
            search_nms_iou_thrs = self.sweep_nms_iou_thrs

        best_results = None
        best_predictions = None
        best_metric = None
        for score_thr in search_score_thrs:
            for nms_iou_thr in search_nms_iou_thrs:
                finalized_predictions = self._finalize_predictions(
                    candidate_predictions=candidate_predictions,
                    score_thr=score_thr,
                    nms_iou_thr=nms_iou_thr,
                )
                metric = DetectionMetricAccumulator(num_classes=self.num_classes, area_ranges=self.area_metric_ranges)
                metric.step(finalized_predictions, all_targets)
                results = self._attach_inference_cfg(
                    metric.get_results(bit_width=self.metric_precision),
                    score_thr=score_thr,
                    nms_iou_thr=nms_iou_thr,
                )
                if self._is_better_eval_result(results, best_results):
                    best_results = results
                    best_predictions = finalized_predictions
                    best_metric = metric

        if best_results is None:
            best_metric = DetectionMetricAccumulator(num_classes=self.num_classes, area_ranges=self.area_metric_ranges)
            best_results = self._attach_inference_cfg(
                best_metric.get_results(bit_width=self.metric_precision),
                score_thr=self.infer_cfg.get("score_thr", 0.05),
                nms_iou_thr=self.infer_cfg.get("nms_iou_thr", 0.6),
            )
            best_predictions = []

        detail_results = best_metric.get_detailed_results(bit_width=self.metric_precision, class_names=self.class_names)
        pred_class_hist = self._build_pred_class_hist(best_predictions)
        dominant_pred_class = "NA"
        dominant_pred_ratio = round(0.0, self.metric_precision)
        total_hist_count = sum(pred_class_hist.values())
        if total_hist_count > 0:
            dominant_pred_class, dominant_pred_count = max(pred_class_hist.items(), key=lambda item: item[1])
            dominant_pred_ratio = round(float(dominant_pred_count / total_hist_count), self.metric_precision)
        best_results["details"] = dict(
            classwise=detail_results["classwise"],
            area_summary={name: best_results[name] for name in ("AP50_small", "AP50_medium") if name in best_results},
            pred_class_hist=pred_class_hist,
            pred_hist_images=min(len(best_predictions), self.pred_hist_num_images),
            dominant_pred_class=dominant_pred_class,
            dominant_pred_ratio=dominant_pred_ratio,
        )

        self.infer_cfg["score_thr"] = best_results["score_thr"]
        self.infer_cfg["nms_iou_thr"] = best_results["nms_iou_thr"]

        if save_path and best_predictions is not None:
            self._dump_predictions(best_predictions, all_image_infos, save_path)
        return best_results
