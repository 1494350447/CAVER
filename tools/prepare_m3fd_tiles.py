#!/usr/bin/env python3
"""为 M3FD 生成切片版 COCO 与回映射 manifest。"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/root/CAVER/data/m3fd_detection")
    parser.add_argument("--output-dir", type=str, default="/root/CAVER/data/m3fd_detection")
    parser.add_argument("--tile-width", type=int, default=768)
    parser.add_argument("--tile-height", type=int, default=576)
    parser.add_argument("--stride-x", type=int, default=576)
    parser.add_argument("--stride-y", type=int, default=432)
    parser.add_argument("--keep-ratio", type=float, default=0.7)
    parser.add_argument("--min-side", type=float, default=6.0)
    return parser.parse_args()


def load_json(path):
    with open(path, mode="r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def dump_json(obj, path):
    with open(path, mode="w", encoding="utf-8") as file_obj:
        json.dump(obj, file_obj, ensure_ascii=False, indent=2)


def generate_starts(full_size, tile_size, nominal_stride):
    if tile_size > full_size:
        raise ValueError(f"tile_size ({tile_size}) cannot exceed full_size ({full_size})")
    starts = []
    max_start = full_size - tile_size
    cursor = 0
    while True:
        starts.append(min(cursor, max_start))
        if cursor >= max_start:
            break
        cursor += nominal_stride
    if starts[-1] != max_start:
        starts.append(max_start)
    deduped = []
    for item in starts:
        if item not in deduped:
            deduped.append(item)
    return deduped


def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def intersect_box(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def build_tile_windows(image_width, image_height, tile_width, tile_height, stride_x, stride_y):
    effective_tile_width = min(int(tile_width), int(image_width))
    effective_tile_height = min(int(tile_height), int(image_height))
    xs = generate_starts(full_size=image_width, tile_size=effective_tile_width, nominal_stride=stride_x)
    ys = generate_starts(full_size=image_height, tile_size=effective_tile_height, nominal_stride=stride_y)
    windows = []
    tile_idx = 0
    for tile_y in ys:
        for tile_x in xs:
            windows.append(
                dict(
                    tile_id=tile_idx,
                    tile_name=f"tile_{tile_idx}",
                    tile_box=[int(tile_x), int(tile_y), int(effective_tile_width), int(effective_tile_height)],
                )
            )
            tile_idx += 1
    return windows


def convert_split(coco, split_name, tile_width, tile_height, stride_x, stride_y, keep_ratio, min_side, drop_empty_tiles):
    ann_by_image = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    output = dict(images=[], annotations=[], categories=deepcopy(coco["categories"]))
    manifest_tiles = []
    next_image_id = 0
    next_ann_id = 1
    empty_tiles = 0

    for image_info in coco["images"]:
        image_id = int(image_info["id"])
        image_width = int(image_info["width"])
        image_height = int(image_info["height"])
        windows = build_tile_windows(
            image_width=image_width,
            image_height=image_height,
            tile_width=tile_width,
            tile_height=tile_height,
            stride_x=stride_x,
            stride_y=stride_y,
        )
        image_stem = Path(image_info["file_name"]).stem
        anns = ann_by_image.get(image_id, [])

        for window in windows:
            tile_x, tile_y, tile_w, tile_h = window["tile_box"]
            tile_rect = [tile_x, tile_y, tile_x + tile_w, tile_y + tile_h]
            kept_annotations = []
            for ann in anns:
                original_box = xywh_to_xyxy(ann["bbox"])
                intersected = intersect_box(original_box, tile_rect)
                if intersected is None:
                    continue
                intersect_area = (intersected[2] - intersected[0]) * (intersected[3] - intersected[1])
                original_area = max(float(ann["area"]), 1e-6)
                if intersect_area / original_area < keep_ratio:
                    continue

                clipped_box = [
                    intersected[0] - tile_x,
                    intersected[1] - tile_y,
                    intersected[2] - tile_x,
                    intersected[3] - tile_y,
                ]
                clipped_w = clipped_box[2] - clipped_box[0]
                clipped_h = clipped_box[3] - clipped_box[1]
                if min(clipped_w, clipped_h) < min_side:
                    continue

                ann_copy = deepcopy(ann)
                ann_copy["id"] = next_ann_id
                ann_copy["image_id"] = next_image_id
                ann_copy["bbox"] = xyxy_to_xywh(clipped_box)
                ann_copy["area"] = float(clipped_w * clipped_h)
                kept_annotations.append(ann_copy)
                next_ann_id += 1

            if drop_empty_tiles and not kept_annotations:
                empty_tiles += 1
                continue

            tile_file_name = f"{image_stem}_{window['tile_name']}.png"
            tile_image_info = dict(
                id=next_image_id,
                file_name=tile_file_name,
                source_file_name=image_info["file_name"],
                depth_file_name=image_info["depth_file_name"],
                width=tile_w,
                height=tile_h,
                tile_box=window["tile_box"],
                parent_image_id=image_id,
                tile_id=window["tile_id"],
                tile_name=window["tile_name"],
                split=split_name,
            )
            output["images"].append(tile_image_info)
            output["annotations"].extend(kept_annotations)
            manifest_tiles.append(tile_image_info)
            if not kept_annotations:
                empty_tiles += 1
            next_image_id += 1

    summary = dict(
        split=split_name,
        num_images=len(output["images"]),
        num_annotations=len(output["annotations"]),
        empty_tiles=empty_tiles,
        tile_width=tile_width,
        tile_height=tile_height,
        stride_x=stride_x,
        stride_y=stride_y,
        keep_ratio=keep_ratio,
        min_side=min_side,
        drop_empty_tiles=drop_empty_tiles,
    )
    return output, manifest_tiles, summary


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_coco = load_json(input_dir / "train_coco.json")
    val_coco = load_json(input_dir / "val_coco.json")

    train_tiles, train_manifest, train_summary = convert_split(
        coco=train_coco,
        split_name="train",
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        stride_x=args.stride_x,
        stride_y=args.stride_y,
        keep_ratio=args.keep_ratio,
        min_side=args.min_side,
        drop_empty_tiles=True,
    )
    val_tiles, val_manifest, val_summary = convert_split(
        coco=val_coco,
        split_name="val",
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        stride_x=args.stride_x,
        stride_y=args.stride_y,
        keep_ratio=args.keep_ratio,
        min_side=args.min_side,
        drop_empty_tiles=False,
    )

    dump_json(train_tiles, output_dir / "train_tile_coco.json")
    dump_json(val_tiles, output_dir / "val_tile_coco.json")
    dump_json(
        dict(
            dataset="M3FD",
            train_summary=train_summary,
            val_summary=val_summary,
            train_tiles=train_manifest,
            val_tiles=val_manifest,
        ),
        output_dir / "tile_manifest.json",
    )

    print(json.dumps(dict(train_summary=train_summary, val_summary=val_summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
