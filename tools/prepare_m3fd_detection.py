#!/usr/bin/env python3
"""将 M3FD 原始 XML 标注转换为当前检测任务使用的 COCO 标注。"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


CATEGORIES = [
    ("People", 1),
    ("Car", 2),
    ("Bus", 3),
    ("Motorcycle", 4),
    ("Lamp", 5),
    ("Truck", 6),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3fd-root", type=str, default="/root/CAVER/M3FD")
    parser.add_argument("--output-dir", type=str, default="/root/CAVER/data/m3fd_detection")
    parser.add_argument("--train-end", type=int, default=3359, help="按文件名顺序划分时，训练集最后一个样本编号。")
    return parser.parse_args()


def load_annotation(xml_path, category_to_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = f"{xml_path.stem}.png"
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))

    boxes = []
    for obj in root.findall("object"):
        category = obj.findtext("name")
        if category not in category_to_id:
            continue
        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        width_box = max(0.0, xmax - xmin)
        height_box = max(0.0, ymax - ymin)
        if width_box <= 0 or height_box <= 0:
            continue
        boxes.append(
            dict(
                category_id=category_to_id[category],
                bbox=[xmin, ymin, width_box, height_box],
                area=width_box * height_box,
                iscrowd=0,
            )
        )

    return dict(file_name=filename, width=width, height=height, annotations=boxes)


def build_coco(sample_ids, ann_root, category_to_id):
    images = []
    annotations = []
    ann_id = 1

    total = len(sample_ids)
    for index, sample_id in enumerate(sample_ids, start=1):
        xml_path = ann_root / f"{sample_id}.xml"
        parsed = load_annotation(xml_path, category_to_id)
        image_id = int(sample_id)
        images.append(
            dict(
                id=image_id,
                file_name=parsed["file_name"],
                depth_file_name=parsed["file_name"],
                width=parsed["width"],
                height=parsed["height"],
            )
        )
        for ann in parsed["annotations"]:
            annotations.append(
                dict(
                    id=ann_id,
                    image_id=image_id,
                    category_id=ann["category_id"],
                    bbox=ann["bbox"],
                    area=ann["area"],
                    iscrowd=ann["iscrowd"],
                )
            )
            ann_id += 1
        if index % 500 == 0 or index == total:
            print(f"[build_coco] {sample_id}: {index}/{total}", flush=True)

    categories = [dict(id=cat_id, name=name) for name, cat_id in CATEGORIES]
    return dict(images=images, annotations=annotations, categories=categories)


def save_text_list(path, sample_ids):
    with open(path, encoding="utf-8", mode="w") as file_obj:
        for sample_id in sample_ids:
            file_obj.write(sample_id + "\n")


def main():
    args = parse_args()
    m3fd_root = Path(args.m3fd_root)
    ann_root = m3fd_root / "Annotation"
    vis_root = m3fd_root / "Vis"
    ir_root = m3fd_root / "Ir"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for required_dir in [ann_root, vis_root, ir_root]:
        if not required_dir.is_dir():
            raise FileNotFoundError(f"M3FD 目录不存在: {required_dir}")

    category_to_id = {name: cat_id for name, cat_id in CATEGORIES}
    sample_ids = sorted(file_name[:-4] for file_name in os.listdir(ann_root) if file_name.endswith(".xml"))
    if len(sample_ids) != 4200:
        print(f"[Warning] 当前检测到 {len(sample_ids)} 个 XML，不等于预期的 4200。仍按现有文件继续生成。")

    train_ids = sample_ids[: args.train_end + 1]
    val_ids = sample_ids[args.train_end + 1 :]

    train_coco = build_coco(train_ids, ann_root, category_to_id)
    val_coco = build_coco(val_ids, ann_root, category_to_id)

    with open(output_dir / "train_coco.json", encoding="utf-8", mode="w") as file_obj:
        json.dump(train_coco, file_obj, ensure_ascii=False)
    with open(output_dir / "val_coco.json", encoding="utf-8", mode="w") as file_obj:
        json.dump(val_coco, file_obj, ensure_ascii=False)

    save_text_list(output_dir / "train.txt", train_ids)
    save_text_list(output_dir / "val.txt", val_ids)

    summary = dict(
        m3fd_root=str(m3fd_root),
        split_rule=f"sorted stems, train <= {train_ids[-1] if train_ids else 'N/A'}, val >= {val_ids[0] if val_ids else 'N/A'}",
        train_samples=len(train_ids),
        val_samples=len(val_ids),
        categories=[dict(id=cat_id, name=name) for name, cat_id in CATEGORIES],
        vis_root=str(vis_root),
        ir_root=str(ir_root),
    )
    with open(output_dir / "summary.json", encoding="utf-8", mode="w") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
