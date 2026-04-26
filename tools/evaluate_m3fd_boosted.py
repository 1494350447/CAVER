#!/usr/bin/env python3
"""M3FD 冻结核心代码的增强推理脚本。

流程：
1. 评估单 checkpoint 的 raw 结果
2. 收集 full / hflip / tile 的候选预测
3. 按类阈值校准
4. 用 WBF 生成 boosted 结果
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmengine import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models as models_lib
import tasks as task_lib
from tasks.bimodal_detection import COCODetectionDataset, DetectionMetricAccumulator, bbox_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/root/CAVER/configs/chapter3_m3fd_frozen_core_boost.py")
    parser.add_argument("--model-name", type=str, default="SAM2PriorAlignmentYOLODetector")
    parser.add_argument("--checkpoint", action="append", default=None, help="可重复传入多个 checkpoint")
    parser.add_argument("--output-dir", type=str, default="/root/CAVER/output_chapter3_m3fd_boosted_eval")
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--calibration-num-images", type=int, default=0, help="仅用部分验证图像做阈值校准，0 表示使用配置默认值")
    parser.add_argument("--show-bar", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, mode="r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def dump_json(obj, path):
    with open(path, mode="w", encoding="utf-8") as file_obj:
        json.dump(obj, file_obj, ensure_ascii=False, indent=2)


def metric_key(result_like):
    return (
        float(result_like.get("mAP50_95", 0.0)),
        float(result_like.get("mAP50", 0.0)),
        float(result_like.get("Precision", 0.0)),
    )


def class_metric_key(class_result):
    return (
        float(class_result.get("AP50_95", 0.0)),
        float(class_result.get("AP50", 0.0)),
        float(class_result.get("Precision", 0.0)),
    )


def normalize_checkpoint_weights(raw_results):
    weights = []
    for item in raw_results:
        weights.append(max(float(item["results"]["mAP50_95"]), 1e-6))
    max_weight = max(weights) if weights else 1.0
    return [weight / max_weight for weight in weights]


def load_model(cfg, model_name, checkpoint_path):
    model_kwargs = dict(cfg.get("model", {}))
    model_kwargs.pop("name", None)
    model_kwargs.pop("pretrained", None)
    if "teacher_adapter_cfg" in model_kwargs:
        teacher_cfg = dict(model_kwargs["teacher_adapter_cfg"])
        teacher_cfg["enable"] = False
        model_kwargs["teacher_adapter_cfg"] = teacher_cfg
    module_class = getattr(models_lib, model_name)
    model = module_class(pretrained=cfg.pretrained, **model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys:
        print(f"[WARN] Missing keys when loading {checkpoint_path}: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[WARN] Unexpected keys when loading {checkpoint_path}: {len(incompatible.unexpected_keys)}")
    model.cuda().eval()
    return model


def to_cpu_pred(pred, topk=None):
    boxes = pred["boxes"].detach().cpu().float()
    scores = pred["scores"].detach().cpu().float()
    labels = pred["labels"].detach().cpu().long()
    if topk is not None and int(topk) > 0 and scores.numel() > int(topk):
        keep = scores.argsort(descending=True)[: int(topk)]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
    return dict(
        boxes=boxes,
        scores=scores,
        labels=labels,
        image_id=int(pred["image_id"]),
    )


def maybe_limit_full_dataset(dataset, limit_images):
    if limit_images <= 0 or limit_images >= len(dataset):
        return dataset, None
    dataset.samples = dataset.samples[:limit_images]
    dataset.images = {sample["image_info"]["id"]: sample["image_info"] for sample in dataset.samples}
    allowed_image_ids = {int(sample["image_info"]["id"]) for sample in dataset.samples}
    return dataset, allowed_image_ids


def maybe_limit_tile_dataset(dataset, allowed_parent_ids):
    if not allowed_parent_ids:
        return dataset
    dataset.samples = [
        sample for sample in dataset.samples if int(sample["image_info"].get("parent_image_id", -1)) in allowed_parent_ids
    ]
    dataset.images = {sample["image_info"]["id"]: sample["image_info"] for sample in dataset.samples}
    return dataset


def extract_targets_from_full_dataset(dataset):
    targets_by_image = {}
    for sample in dataset.samples:
        image_info = sample["image_info"]
        anns = sample["annotations"]
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(dataset.cat_id_to_label[ann["category_id"]]))
        targets_by_image[int(image_info["id"])] = dict(
            boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            labels=torch.tensor(labels, dtype=torch.long),
            image_id=int(image_info["id"]),
        )
    return targets_by_image


def build_loader(dataset, task, batch_size, show_bar):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=task.get_collate_fn(split="test"),
    )


def flip_boxes_horizontally(boxes, image_width):
    if boxes.numel() == 0:
        return boxes
    flipped = boxes.clone()
    flipped[:, 0] = float(image_width) - boxes[:, 2]
    flipped[:, 2] = float(image_width) - boxes[:, 0]
    return flipped


def collect_full_view_sources(model, task, data_loader, min_score_thr, pre_nms_topk, use_hflip, source_topk=None):
    per_image_sources = defaultdict(list)
    with torch.no_grad():
        for batch in data_loader:
            device_batch = task.move_batch_to_device(batch)
            model_outputs = model(data=task.get_model_inputs(device_batch))
            predictions = task._collect_predictions(
                model_outputs=model_outputs,
                batch=device_batch,
                score_thr=min_score_thr,
                pre_nms_topk=pre_nms_topk,
            )
            for pred in predictions:
                image_id = int(pred["image_id"])
                per_image_sources[image_id].append(to_cpu_pred(pred, topk=source_topk))

            if not use_hflip:
                continue

            flipped_batch = dict(
                image=torch.flip(device_batch["image"], dims=[3]),
                depth=torch.flip(device_batch["depth"], dims=[3]),
                targets=device_batch["targets"],
                image_info=device_batch["image_info"],
            )
            flipped_outputs = model(data=task.get_model_inputs(flipped_batch))
            flipped_predictions = task._collect_predictions(
                model_outputs=flipped_outputs,
                batch=flipped_batch,
                score_thr=min_score_thr,
                pre_nms_topk=pre_nms_topk,
            )
            for pred, target in zip(flipped_predictions, batch["targets"]):
                restored = to_cpu_pred(pred, topk=source_topk)
                restored["boxes"] = flip_boxes_horizontally(restored["boxes"], image_width=int(target["size"][1].item()))
                per_image_sources[restored["image_id"]].append(restored)
    return per_image_sources


def map_tile_boxes_to_parent(pred_boxes, tile_box, resized_size):
    if pred_boxes.numel() == 0:
        return pred_boxes
    tile_x, tile_y, tile_w, tile_h = [float(v) for v in tile_box]
    input_h, input_w = resized_size
    mapped = pred_boxes.clone()
    mapped[:, 0::2] = mapped[:, 0::2] * (tile_w / float(input_w)) + tile_x
    mapped[:, 1::2] = mapped[:, 1::2] * (tile_h / float(input_h)) + tile_y
    return mapped


def collect_tile_view_sources(model, task, data_loader, min_score_thr, pre_nms_topk, source_topk=None):
    per_image_sources = defaultdict(list)
    with torch.no_grad():
        for batch in data_loader:
            device_batch = task.move_batch_to_device(batch)
            model_outputs = model(data=task.get_model_inputs(device_batch))
            predictions = task._collect_predictions(
                model_outputs=model_outputs,
                batch=device_batch,
                score_thr=min_score_thr,
                pre_nms_topk=pre_nms_topk,
            )
            for pred, info, target in zip(predictions, batch["image_info"], batch["targets"]):
                parent_image_id = int(info["parent_image_id"])
                tile_box = info["tile_box"]
                restored = to_cpu_pred(pred, topk=source_topk)
                restored["image_id"] = parent_image_id
                restored["boxes"] = map_tile_boxes_to_parent(
                    restored["boxes"],
                    tile_box=tile_box,
                    resized_size=tuple(int(x) for x in target["size"].tolist()),
                )
                per_image_sources[parent_image_id].append(restored)
    return per_image_sources


def select_sources_for_class(image_sources, class_id, score_thr, checkpoint_weight_map, max_candidates=None):
    boxes = []
    scores = []
    weights = []
    for source in image_sources:
        if source["scores"].numel() == 0:
            continue
        mask = (source["labels"] == class_id) & (source["scores"] >= score_thr)
        if not mask.any():
            continue
        boxes.append(source["boxes"][mask].numpy())
        scores.append(source["scores"][mask].numpy())
        weight = checkpoint_weight_map[source["checkpoint_key"]]
        weights.append(np.full((int(mask.sum().item()),), weight, dtype=np.float32))
    if not boxes:
        return None, None, None
    boxes = np.concatenate(boxes, axis=0)
    scores = np.concatenate(scores, axis=0)
    weights = np.concatenate(weights, axis=0)
    if max_candidates is not None and int(max_candidates) > 0 and len(scores) > int(max_candidates):
        keep = np.argsort(scores)[::-1][: int(max_candidates)]
        boxes = boxes[keep]
        scores = scores[keep]
        weights = weights[keep]
    return boxes, scores, weights


def bbox_iou_numpy(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[:, 0] * wh[:, 1]
    area1 = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    area2 = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), a_min=0.0, a_max=None)
    union = area1 + area2 - inter + 1e-6
    return inter / union


def weighted_box_fusion_single_class(boxes, scores, weights, iou_thr):
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    order = np.argsort(scores)[::-1]
    clusters = []
    for idx in order:
        curr_box = boxes[idx]
        curr_score = float(scores[idx])
        curr_weight = float(weights[idx])
        best_cluster_id = -1
        best_iou = iou_thr
        for cluster_id, cluster in enumerate(clusters):
            cluster_iou = float(bbox_iou_numpy(curr_box, np.asarray([cluster["box"]]))[0])
            if cluster_iou > best_iou:
                best_iou = cluster_iou
                best_cluster_id = cluster_id

        if best_cluster_id < 0:
            clusters.append(
                dict(
                    boxes=[curr_box],
                    scores=[curr_score],
                    weights=[curr_weight],
                    box=curr_box.copy(),
                )
            )
            continue

        cluster = clusters[best_cluster_id]
        cluster["boxes"].append(curr_box)
        cluster["scores"].append(curr_score)
        cluster["weights"].append(curr_weight)
        box_array = np.asarray(cluster["boxes"], dtype=np.float32)
        score_array = np.asarray(cluster["scores"], dtype=np.float32)
        weight_array = np.asarray(cluster["weights"], dtype=np.float32)
        blend = np.clip(score_array * weight_array, a_min=1e-6, a_max=None)
        cluster["box"] = (box_array * blend[:, None]).sum(axis=0) / blend.sum()

    fused_boxes = []
    fused_scores = []
    for cluster in clusters:
        score_array = np.asarray(cluster["scores"], dtype=np.float32)
        weight_array = np.asarray(cluster["weights"], dtype=np.float32)
        fused_boxes.append(cluster["box"])
        fused_scores.append(float((score_array * weight_array).sum() / np.clip(weight_array.sum(), 1e-6, None)))

    fused_boxes = np.asarray(fused_boxes, dtype=np.float32)
    fused_scores = np.asarray(fused_scores, dtype=np.float32)
    order = np.argsort(fused_scores)[::-1]
    return fused_boxes[order], fused_scores[order]


def fuse_predictions_for_image(
    image_sources,
    num_classes,
    per_class_thresholds,
    checkpoint_weight_map,
    wbf_iou,
    per_class_topk=None,
):
    fused_boxes = []
    fused_scores = []
    fused_labels = []
    for class_id in range(num_classes):
        boxes, scores, weights = select_sources_for_class(
            image_sources=image_sources,
            class_id=class_id,
            score_thr=per_class_thresholds[class_id],
            checkpoint_weight_map=checkpoint_weight_map,
            max_candidates=per_class_topk,
        )
        class_boxes, class_scores = weighted_box_fusion_single_class(boxes=boxes, scores=scores, weights=weights, iou_thr=wbf_iou)
        if len(class_scores) == 0:
            continue
        fused_boxes.append(class_boxes)
        fused_scores.append(class_scores)
        fused_labels.append(np.full((len(class_scores),), class_id, dtype=np.int64))

    if not fused_boxes:
        return dict(
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.zeros((0,), dtype=torch.float32),
            labels=torch.zeros((0,), dtype=torch.long),
        )
    return dict(
        boxes=torch.from_numpy(np.concatenate(fused_boxes, axis=0)).float(),
        scores=torch.from_numpy(np.concatenate(fused_scores, axis=0)).float(),
        labels=torch.from_numpy(np.concatenate(fused_labels, axis=0)).long(),
    )


def fuse_single_class_for_image(image_sources, class_id, score_thr, checkpoint_weight_map, wbf_iou, per_class_topk=None):
    boxes, scores, weights = select_sources_for_class(
        image_sources=image_sources,
        class_id=class_id,
        score_thr=score_thr,
        checkpoint_weight_map=checkpoint_weight_map,
        max_candidates=per_class_topk,
    )
    class_boxes, class_scores = weighted_box_fusion_single_class(boxes=boxes, scores=scores, weights=weights, iou_thr=wbf_iou)
    if len(class_scores) == 0:
        return dict(
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.zeros((0,), dtype=torch.float32),
            labels=torch.zeros((0,), dtype=torch.long),
        )
    return dict(
        boxes=torch.from_numpy(class_boxes).float(),
        scores=torch.from_numpy(class_scores).float(),
        labels=torch.full((len(class_scores),), class_id, dtype=torch.long),
    )


def build_metric_predictions(predictions_by_image, targets_by_image):
    metric_predictions = []
    metric_targets = []
    for image_id in sorted(targets_by_image):
        pred = predictions_by_image.get(
            image_id,
            dict(
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
                labels=torch.zeros((0,), dtype=torch.long),
            ),
        )
        metric_predictions.append(
            dict(
                boxes=pred["boxes"],
                scores=pred["scores"],
                labels=pred["labels"],
                image_id=image_id,
            )
        )
        metric_targets.append(targets_by_image[image_id])
    return metric_predictions, metric_targets


def evaluate_single_class(predictions_by_image, targets_by_image, class_id, num_classes):
    metric_predictions, metric_targets = build_metric_predictions(predictions_by_image, targets_by_image)
    metric = DetectionMetricAccumulator(num_classes=num_classes)
    metric.step(metric_predictions, metric_targets)
    ap50_result = metric._evaluate_class(class_id, 0.5)
    if ap50_result[0] is None:
        return dict(AP50=0.0, AP50_95=0.0, Precision=0.0, Recall=0.0)
    class_aps = []
    for thr in metric.iou_thresholds:
        ap_thr_result = metric._evaluate_class(class_id, thr)
        if ap_thr_result[0] is not None:
            class_aps.append(ap_thr_result[0])
    return dict(
        AP50=float(ap50_result[0]),
        AP50_95=float(sum(class_aps) / max(len(class_aps), 1)),
        Precision=float(ap50_result[1]),
        Recall=float(ap50_result[2]),
    )


def evaluate_full_predictions(predictions_by_image, targets_by_image, num_classes, class_names, metric_precision):
    metric_predictions, metric_targets = build_metric_predictions(predictions_by_image, targets_by_image)
    metric = DetectionMetricAccumulator(num_classes=num_classes)
    metric.step(metric_predictions, metric_targets)
    detailed = metric.get_detailed_results(bit_width=metric_precision, class_names=class_names)
    return dict(summary=detailed["summary"], classwise=detailed["classwise"])


def pick_calibration_image_ids(image_ids, calibration_num_images):
    image_ids = list(sorted(image_ids))
    if calibration_num_images <= 0 or calibration_num_images >= len(image_ids):
        return image_ids
    if calibration_num_images == 1:
        return [image_ids[0]]
    positions = np.linspace(0, len(image_ids) - 1, num=calibration_num_images)
    selected = []
    seen = set()
    for pos in positions:
        idx = int(round(float(pos)))
        idx = max(0, min(idx, len(image_ids) - 1))
        image_id = image_ids[idx]
        if image_id not in seen:
            selected.append(image_id)
            seen.add(image_id)
    if len(selected) < calibration_num_images:
        for image_id in image_ids:
            if image_id not in seen:
                selected.append(image_id)
                seen.add(image_id)
            if len(selected) >= calibration_num_images:
                break
    return selected


def subset_by_image_ids(mapping, image_ids):
    return {image_id: mapping[image_id] for image_id in image_ids if image_id in mapping}


def calibrate_thresholds(
    candidates_by_image,
    targets_by_image,
    num_classes,
    checkpoint_weight_map,
    threshold_grid,
    iou_candidates,
    class_names,
    per_class_topk=None,
):
    best_global = None
    for wbf_iou in iou_candidates:
        per_class_thresholds = [0.05 for _ in range(num_classes)]
        per_class_results = []
        for class_id in range(num_classes):
            best_local = None
            best_threshold = per_class_thresholds[class_id]
            for thr in threshold_grid:
                candidate_predictions = {}
                for image_id, image_sources in candidates_by_image.items():
                    candidate_predictions[image_id] = fuse_single_class_for_image(
                        image_sources=image_sources,
                        class_id=class_id,
                        score_thr=thr,
                        checkpoint_weight_map=checkpoint_weight_map,
                        wbf_iou=wbf_iou,
                        per_class_topk=per_class_topk,
                    )
                class_result = evaluate_single_class(
                    predictions_by_image=candidate_predictions,
                    targets_by_image=targets_by_image,
                    class_id=class_id,
                    num_classes=num_classes,
                )
                if best_local is None or class_metric_key(class_result) > class_metric_key(best_local):
                    best_local = class_result
                    best_threshold = float(thr)
            per_class_thresholds[class_id] = best_threshold
            per_class_results.append(dict(class_id=class_id, threshold=best_threshold, metrics=best_local))

        fused_predictions = {}
        for image_id, image_sources in candidates_by_image.items():
            fused_predictions[image_id] = fuse_predictions_for_image(
                image_sources=image_sources,
                num_classes=num_classes,
                per_class_thresholds=per_class_thresholds,
                checkpoint_weight_map=checkpoint_weight_map,
                wbf_iou=wbf_iou,
                per_class_topk=per_class_topk,
            )
        eval_results = evaluate_full_predictions(
            predictions_by_image=fused_predictions,
            targets_by_image=targets_by_image,
            num_classes=num_classes,
            class_names=class_names,
            metric_precision=6,
        )
        candidate = dict(
            wbf_iou=float(wbf_iou),
            per_class_thresholds=per_class_thresholds,
            per_class_results=per_class_results,
            results=eval_results,
            fused_predictions=fused_predictions,
        )
        if best_global is None or metric_key(candidate["results"]["summary"]) > metric_key(best_global["results"]["summary"]):
            best_global = candidate
    return best_global


def fuse_all_predictions(candidates_by_image, num_classes, per_class_thresholds, checkpoint_weight_map, wbf_iou, per_class_topk=None):
    fused_predictions = {}
    for image_id, image_sources in candidates_by_image.items():
        fused_predictions[image_id] = fuse_predictions_for_image(
            image_sources=image_sources,
            num_classes=num_classes,
            per_class_thresholds=per_class_thresholds,
            checkpoint_weight_map=checkpoint_weight_map,
            wbf_iou=wbf_iou,
            per_class_topk=per_class_topk,
        )
    return fused_predictions


def serialize_predictions(predictions_by_image):
    serializable = []
    for image_id, pred in sorted(predictions_by_image.items()):
        serializable.append(
            dict(
                image_id=int(image_id),
                boxes=pred["boxes"].tolist(),
                scores=pred["scores"].tolist(),
                labels=pred["labels"].tolist(),
            )
        )
    return serializable


def main():
    args = parse_args()
    os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
    cfg = Config.fromfile(args.config, use_predefined_variables=False)
    boosted_cfg = cfg.get("boosted_eval", {})

    checkpoint_paths = list(args.checkpoint or [])
    if not checkpoint_paths:
        default_checkpoint = boosted_cfg.get("raw_checkpoint", None)
        if default_checkpoint is None:
            raise ValueError("No checkpoint provided and boosted_eval.raw_checkpoint is missing in config.")
        checkpoint_paths = [default_checkpoint]

    task = task_lib.build_task(cfg)
    full_dataset = COCODetectionDataset(boosted_cfg["full_val_dataset"], shape=cfg.data.test.shape, is_train=False)
    full_dataset, allowed_image_ids = maybe_limit_full_dataset(full_dataset, limit_images=int(args.limit_images))
    tile_dataset = COCODetectionDataset(boosted_cfg["tile_val_dataset"], shape=cfg.data.test.shape, is_train=False)
    tile_dataset = maybe_limit_tile_dataset(tile_dataset, allowed_parent_ids=allowed_image_ids)
    task.class_names = tuple(full_dataset.class_names)

    full_loader = build_loader(full_dataset, task=task, batch_size=cfg.args.batch_size, show_bar=args.show_bar)
    tile_loader = build_loader(tile_dataset, task=task, batch_size=cfg.args.batch_size, show_bar=args.show_bar)
    targets_by_image = extract_targets_from_full_dataset(full_dataset)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_results = []
    for checkpoint_path in checkpoint_paths:
        model = load_model(cfg=cfg, model_name=args.model_name, checkpoint_path=checkpoint_path)
        result = task.evaluate_once(
            model=model,
            data_loader=full_loader,
            save_path="",
            show_bar=args.show_bar,
            calibrate_inference=True,
        )
        raw_results.append(dict(checkpoint=checkpoint_path, results=result))
        del model
        torch.cuda.empty_cache()

    checkpoint_weights = normalize_checkpoint_weights(raw_results)
    checkpoint_weight_map = {
        checkpoint_path: weight for checkpoint_path, weight in zip(checkpoint_paths, checkpoint_weights)
    }

    all_candidates = defaultdict(list)
    candidate_pre_nms_topk = int(cfg.task.inference.get("pre_nms_topk", 1000))
    min_score_thr = float(boosted_cfg.get("fusion", {}).get("skip_box_thr", 0.001))
    use_hflip = bool(boosted_cfg.get("tta", {}).get("use_hflip", True))
    use_tiles = bool(boosted_cfg.get("tta", {}).get("use_tiles", True))
    source_topk = int(boosted_cfg.get("fusion", {}).get("source_topk", 0)) or None

    for checkpoint_path in checkpoint_paths:
        model = load_model(cfg=cfg, model_name=args.model_name, checkpoint_path=checkpoint_path)
        full_sources = collect_full_view_sources(
            model=model,
            task=task,
            data_loader=full_loader,
            min_score_thr=min_score_thr,
            pre_nms_topk=candidate_pre_nms_topk,
            use_hflip=use_hflip,
            source_topk=source_topk,
        )
        tile_sources = (
            collect_tile_view_sources(
                model=model,
                task=task,
                data_loader=tile_loader,
                min_score_thr=min_score_thr,
                pre_nms_topk=candidate_pre_nms_topk,
                source_topk=source_topk,
            )
            if use_tiles
            else defaultdict(list)
        )

        for image_id in sorted(targets_by_image):
            merged_sources = []
            for source in full_sources.get(image_id, []):
                source["checkpoint_key"] = checkpoint_path
                merged_sources.append(source)
            for source in tile_sources.get(image_id, []):
                source["checkpoint_key"] = checkpoint_path
                merged_sources.append(source)
            all_candidates[image_id].extend(merged_sources)
        del model
        torch.cuda.empty_cache()

    best_raw = max(raw_results, key=lambda item: metric_key(item["results"]))
    threshold_grid = boosted_cfg["threshold_search"]["per_class_grid"]
    iou_candidates = boosted_cfg["fusion"]["iou_candidates"]
    calibration_num_images = int(args.calibration_num_images or boosted_cfg.get("threshold_search", {}).get("calibration_num_images", 0))
    calibration_image_ids = pick_calibration_image_ids(sorted(targets_by_image), calibration_num_images)
    calibration_candidates = subset_by_image_ids(all_candidates, calibration_image_ids)
    calibration_targets = subset_by_image_ids(targets_by_image, calibration_image_ids)
    per_class_topk = int(boosted_cfg.get("fusion", {}).get("per_class_topk", 0)) or None

    best_boosted = calibrate_thresholds(
        candidates_by_image=calibration_candidates,
        targets_by_image=calibration_targets,
        num_classes=task.num_classes,
        checkpoint_weight_map=checkpoint_weight_map,
        threshold_grid=threshold_grid,
        iou_candidates=iou_candidates,
        class_names=task.class_names,
        per_class_topk=per_class_topk,
    )
    final_fused_predictions = fuse_all_predictions(
        candidates_by_image=all_candidates,
        num_classes=task.num_classes,
        per_class_thresholds=best_boosted["per_class_thresholds"],
        checkpoint_weight_map=checkpoint_weight_map,
        wbf_iou=best_boosted["wbf_iou"],
        per_class_topk=per_class_topk,
    )
    final_eval_results = evaluate_full_predictions(
        predictions_by_image=final_fused_predictions,
        targets_by_image=targets_by_image,
        num_classes=task.num_classes,
        class_names=task.class_names,
        metric_precision=6,
    )

    final_summary = dict(
        checkpoints=checkpoint_paths,
        checkpoint_weights=checkpoint_weight_map,
        raw_results=raw_results,
        best_raw=best_raw,
        calibration_num_images=len(calibration_image_ids),
        calibration_image_ids=calibration_image_ids,
        boosted_calibration_summary=best_boosted["results"]["summary"],
        boosted_calibration_classwise=best_boosted["results"]["classwise"],
        boosted_summary=final_eval_results["summary"],
        boosted_classwise=final_eval_results["classwise"],
        boosted_per_class_thresholds=best_boosted["per_class_thresholds"],
        boosted_per_class_search=best_boosted["per_class_results"],
        boosted_wbf_iou=best_boosted["wbf_iou"],
        boosted_source_topk=source_topk,
        boosted_per_class_topk=per_class_topk,
    )

    dump_json(final_summary, output_dir / "boosted_eval_summary.json")
    dump_json(serialize_predictions(final_fused_predictions), output_dir / "boosted_predictions.json")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
