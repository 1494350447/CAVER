#!/usr/bin/env python3
"""准备 VEDAI fold0 检测数据。

输入：
- /root/CAVER/VEDAI/visible/annotations/instances_all2017.json
- /root/CAVER/VEDAI/infrared/annotations/instances_all2017.json

输出：
- data/vedai_detection/train_fold0.json
- data/vedai_detection/val_fold0.json
- data/vedai_detection/fold0_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vedai-root", type=str, default="/root/CAVER/VEDAI")
    parser.add_argument("--output-dir", type=str, default="/root/CAVER/data/vedai_detection")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def load_coco(path: Path):
    with open(path, encoding="utf-8", mode="r") as file_obj:
        return json.load(file_obj)


def validate_modal_pairing(visible_coco, infrared_coco):
    vis_images = {int(item["id"]): item for item in visible_coco["images"]}
    ir_images = {int(item["id"]): item for item in infrared_coco["images"]}
    if set(vis_images) != set(ir_images):
        raise ValueError("VEDAI visible/infrared image ids do not match.")

    for image_id, vis_info in vis_images.items():
        ir_info = ir_images[image_id]
        if vis_info["file_name"] != ir_info["file_name"]:
            raise ValueError(f"File name mismatch for image_id={image_id}: {vis_info['file_name']} vs {ir_info['file_name']}")
        if vis_info["width"] != ir_info["width"] or vis_info["height"] != ir_info["height"]:
            raise ValueError(f"Size mismatch for image_id={image_id}.")

    vis_cats = [(cat["id"], cat["name"]) for cat in visible_coco["categories"]]
    ir_cats = [(cat["id"], cat["name"]) for cat in infrared_coco["categories"]]
    if vis_cats != ir_cats:
        raise ValueError("VEDAI visible/infrared categories do not match.")


def build_image_records(coco):
    categories = sorted(coco["categories"], key=lambda item: item["id"])
    category_ids = [int(cat["id"]) for cat in categories]
    category_to_index = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[int(ann["image_id"])].append(ann)

    image_records = []
    for image_info in sorted(coco["images"], key=lambda item: int(item["id"])):
        image_id = int(image_info["id"])
        anns = ann_by_image.get(image_id, [])
        class_hist = [0] * len(category_ids)
        for ann in anns:
            class_hist[category_to_index[int(ann["category_id"])]] += 1
        image_records.append(
            dict(
                id=image_id,
                file_name=image_info["file_name"],
                width=int(image_info["width"]),
                height=int(image_info["height"]),
                annotations=anns,
                class_hist=class_hist,
                num_boxes=len(anns),
                empty=int(len(anns) == 0),
            )
        )
    return image_records, categories


def make_targets(records, val_ratio):
    num_images = len(records)
    target_images = int(round(num_images * val_ratio))
    total_empty = sum(record["empty"] for record in records)
    target_empty = int(round(total_empty * val_ratio))
    class_totals = [sum(record["class_hist"][idx] for record in records) for idx in range(len(records[0]["class_hist"]))]
    target_class_totals = [int(round(total * val_ratio)) for total in class_totals]
    return dict(images=target_images, empty=target_empty, class_totals=target_class_totals)


def build_rarity_weights(class_totals):
    raw = [1.0 / math.sqrt(max(total, 1)) for total in class_totals]
    max_value = max(raw) if raw else 1.0
    return [item / max_value for item in raw]


def build_state(record_by_id, selected_ids, num_classes):
    state = dict(images=len(selected_ids), empty=0, class_totals=[0] * num_classes)
    for image_id in selected_ids:
        record = record_by_id[image_id]
        state["empty"] += record["empty"]
        for idx, count in enumerate(record["class_hist"]):
            state["class_totals"][idx] += count
    return state


def state_cost(state, targets, class_weights):
    image_term = ((state["images"] - targets["images"]) / max(targets["images"], 1)) ** 2
    empty_term = ((state["empty"] - targets["empty"]) / max(targets["empty"], 1)) ** 2
    class_term = 0.0
    for value, target, weight in zip(state["class_totals"], targets["class_totals"], class_weights):
        norm = max(target, 1)
        class_term += 2.0 * weight * ((value - target) / norm) ** 2
    return image_term + 0.8 * empty_term + class_term


def stable_hash(image_id):
    return (int(image_id) * 2654435761) & 0xFFFFFFFF


def initialize_val_ids(records, target_images):
    ranked = sorted((stable_hash(record["id"]), record["id"]) for record in records)
    return set(image_id for _, image_id in ranked[:target_images])


def candidate_score(record, residuals, class_weights, empty_gap, role):
    if role not in {"remove", "add"}:
        raise ValueError(f"Unsupported role: {role}")

    if role == "remove":
        class_signal = sum(residual * weight * count for residual, weight, count in zip(residuals, class_weights, record["class_hist"]))
        empty_signal = float(empty_gap) * float(record["empty"]) * 4.0
    else:
        class_signal = sum((-residual) * weight * count for residual, weight, count in zip(residuals, class_weights, record["class_hist"]))
        empty_signal = float(-empty_gap) * float(record["empty"]) * 4.0

    return (
        empty_signal + class_signal,
        record["num_boxes"],
        -record["id"],
    )


def swapped_state(state, remove_record, add_record):
    class_totals = []
    for value, remove_count, add_count in zip(state["class_totals"], remove_record["class_hist"], add_record["class_hist"]):
        class_totals.append(value - remove_count + add_count)
    return dict(
        images=state["images"],
        empty=state["empty"] - remove_record["empty"] + add_record["empty"],
        class_totals=class_totals,
    )


def optimize_split(records, targets, class_weights, max_iterations=64, val_candidate_limit=128, train_candidate_limit=256):
    record_by_id = {record["id"]: record for record in records}
    val_ids = initialize_val_ids(records, target_images=targets["images"])
    all_ids = {record["id"] for record in records}
    train_ids = all_ids - val_ids
    state = build_state(record_by_id, val_ids, num_classes=len(targets["class_totals"]))
    best_cost = state_cost(state, targets, class_weights)

    for _ in range(max_iterations):
        residuals = [value - target for value, target in zip(state["class_totals"], targets["class_totals"])]
        empty_gap = state["empty"] - targets["empty"]

        val_candidates = sorted(
            (record_by_id[image_id] for image_id in val_ids),
            key=lambda record: candidate_score(record, residuals, class_weights, empty_gap, role="remove"),
            reverse=True,
        )[:val_candidate_limit]
        train_candidates = sorted(
            (record_by_id[image_id] for image_id in train_ids),
            key=lambda record: candidate_score(record, residuals, class_weights, empty_gap, role="add"),
            reverse=True,
        )[:train_candidate_limit]

        best_swap = None
        best_swap_cost = best_cost
        for remove_record in val_candidates:
            for add_record in train_candidates:
                candidate_state = swapped_state(state, remove_record=remove_record, add_record=add_record)
                candidate_cost = state_cost(candidate_state, targets, class_weights)
                if candidate_cost + 1e-12 < best_swap_cost:
                    best_swap_cost = candidate_cost
                    best_swap = (remove_record, add_record, candidate_state)

        if best_swap is None:
            break

        remove_record, add_record, next_state = best_swap
        val_ids.remove(remove_record["id"])
        val_ids.add(add_record["id"])
        train_ids.remove(add_record["id"])
        train_ids.add(remove_record["id"])
        state = next_state
        best_cost = best_swap_cost

    return sorted(train_ids), sorted(val_ids), state, best_cost


def split_fold0(records, targets):
    class_weights = build_rarity_weights(targets["class_totals"])
    train_ids, val_ids, state, score = optimize_split(records, targets, class_weights)
    return train_ids, val_ids, state, score


def filter_coco(coco, selected_ids):
    selected_ids = set(int(image_id) for image_id in selected_ids)
    output = dict(
        images=[],
        annotations=[],
        categories=deepcopy(coco["categories"]),
    )
    image_lookup = {int(item["id"]): item for item in coco["images"]}
    for image_id in sorted(selected_ids):
        image_info = deepcopy(image_lookup[image_id])
        image_info["depth_file_name"] = image_info["file_name"]
        output["images"].append(image_info)

    ann_id = 1
    for ann in coco["annotations"]:
        if int(ann["image_id"]) not in selected_ids:
            continue
        ann_copy = deepcopy(ann)
        ann_copy["id"] = ann_id
        output["annotations"].append(ann_copy)
        ann_id += 1
    return output


def split_stats(coco):
    category_ids = [int(cat["id"]) for cat in sorted(coco["categories"], key=lambda item: item["id"])]
    class_counts = {cat_id: 0 for cat_id in category_ids}
    image_ids_with_ann = set()
    for ann in coco["annotations"]:
        class_counts[int(ann["category_id"])] += 1
        image_ids_with_ann.add(int(ann["image_id"]))
    empty_images = len(coco["images"]) - len(image_ids_with_ann)
    return dict(
        images=len(coco["images"]),
        annotations=len(coco["annotations"]),
        empty_images=empty_images,
        class_counts=class_counts,
    )


def main():
    args = parse_args()
    vedai_root = Path(args.vedai_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visible_json_path = vedai_root / "visible" / "annotations" / "instances_all2017.json"
    infrared_json_path = vedai_root / "infrared" / "annotations" / "instances_all2017.json"
    if not visible_json_path.is_file():
        raise FileNotFoundError(f"Missing visible COCO annotation: {visible_json_path}")
    if not infrared_json_path.is_file():
        raise FileNotFoundError(f"Missing infrared COCO annotation: {infrared_json_path}")

    visible_coco = load_coco(visible_json_path)
    infrared_coco = load_coco(infrared_json_path)
    validate_modal_pairing(visible_coco, infrared_coco)

    records, categories = build_image_records(visible_coco)
    targets = make_targets(records, val_ratio=float(args.val_ratio))
    train_ids, val_ids, val_state, split_cost_value = split_fold0(records, targets)

    train_coco = filter_coco(visible_coco, train_ids)
    val_coco = filter_coco(visible_coco, val_ids)

    train_stats = split_stats(train_coco)
    val_stats = split_stats(val_coco)
    manifest = dict(
        dataset="VEDAI",
        source_root=str(vedai_root),
        visible_root=str(vedai_root / "visible" / "all2017"),
        infrared_root=str(vedai_root / "infrared" / "all2017"),
        split_name="fold0",
        split_strategy="deterministic stable-hash initialization + local swap multilabel balancing",
        val_ratio=float(args.val_ratio),
        categories=deepcopy(categories),
        targets=targets,
        optimized_val_state=val_state,
        optimization_cost=split_cost_value,
        train=train_stats,
        val=val_stats,
        train_image_ids=train_ids,
        val_image_ids=val_ids,
    )

    with open(output_dir / "train_fold0.json", encoding="utf-8", mode="w") as file_obj:
        json.dump(train_coco, file_obj, ensure_ascii=False)
    with open(output_dir / "val_fold0.json", encoding="utf-8", mode="w") as file_obj:
        json.dump(val_coco, file_obj, ensure_ascii=False)
    with open(output_dir / "fold0_manifest.json", encoding="utf-8", mode="w") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
