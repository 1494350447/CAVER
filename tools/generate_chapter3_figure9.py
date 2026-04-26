#!/usr/bin/env python3
"""生成第三章消融可视化主图（建议图 3.9）。

本工具围绕第三章的三段式叙事链条构建：
SPD-DRE -> F-DDSA -> HFSD

它提供两个子命令：
1. render：按照 JSON spec 渲染最终的 3x7 论文复合图。
2. mine：给定 full model 与对比模型，自动筛选差异更明显的候选样例。

说明：
- render 阶段默认复用当前仓库已有模型输出，不新增模型接口。
- teacher_prior 在 eval() 下不会自动返回，因此脚本会单独调用 teacher_adapter
  获取教师先验，以保留论文图中 teacher/student 对照这一列。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from mmengine import Config
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models as models_lib
import tasks as task_lib


MODULE_VIEW_SPECS = {
    "spd_dre": dict(
        mid_a_title="空间解耦响应",
        mid_b_title="频率解耦响应",
        compare_label="w/o SPD-DRE",
        evidence_label="GT + Zoom",
    ),
    "fddsa": dict(
        mid_a_title="Focus Map",
        mid_b_title="Divergence Map",
        compare_label="w/o F-DDSA",
        evidence_label="GT + Zoom",
    ),
    "hfsd": dict(
        mid_a_title="Cls Features",
        mid_b_title="Reg Features",
        compare_label="w/o HFSD",
        evidence_label="GT + Zoom",
    ),
}

DEFAULT_FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/"
    "NotoSansCJKsc-Regular.otf"
)
FONT_CACHE_DIR = Path.home() / ".cache" / "caver_fonts"
FONT_CACHE_PATH = FONT_CACHE_DIR / "NotoSansCJKsc-Regular.otf"
FALLBACK_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SYSTEM_CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
)
ACTIVE_FONT_PATH: Optional[str] = None
FONT_CACHE: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def parse_args():
    parser = argparse.ArgumentParser(description="第三章图 3.9 复合可视化生成工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="根据 spec 渲染最终论文图。")
    render_parser.add_argument("--spec", type=str, required=True, help="图像渲染 spec 的 JSON 路径。")
    render_parser.add_argument("--output", type=str, required=True, help="输出 png/jpg 路径。")
    render_parser.add_argument("--device", type=str, default="cuda", help="推理设备，默认 cuda。")
    render_parser.add_argument("--cell-width", type=int, default=420, help="单个子图单元宽度。")
    render_parser.add_argument("--cell-height", type=int, default=280, help="单个子图单元高度。")
    render_parser.add_argument("--title-height", type=int, default=34, help="每个子图标题栏高度。")
    render_parser.add_argument("--row-label-width", type=int, default=150, help="左侧行标签栏宽度。")
    render_parser.add_argument("--margin", type=int, default=18, help="整体边距。")
    render_parser.add_argument("--gap", type=int, default=10, help="子图间距。")
    render_parser.add_argument("--font-scale", type=float, default=0.62, help="普通标题字体大小。")
    render_parser.add_argument("--header-font-scale", type=float, default=0.95, help="主标题字体大小。")
    render_parser.add_argument("--line-thickness", type=int, default=2, help="框线粗细。")
    render_parser.add_argument("--font-path", type=str, default=None, help="可选，显式指定支持中文的字体文件路径。")

    mine_parser = subparsers.add_parser("mine", help="为某个模块筛选差异更明显的候选样例。")
    mine_parser.add_argument("--module", type=str, required=True, choices=sorted(MODULE_VIEW_SPECS))
    mine_parser.add_argument("--full-config", type=str, required=True)
    mine_parser.add_argument("--full-checkpoint", type=str, required=True)
    mine_parser.add_argument("--compare-config", type=str, default=None)
    mine_parser.add_argument("--compare-checkpoint", type=str, default=None)
    mine_parser.add_argument("--dataset-name", type=str, required=True)
    mine_parser.add_argument("--split", type=str, default="test", choices=("train", "test"))
    mine_parser.add_argument("--output", type=str, required=True, help="输出候选样例 JSON。")
    mine_parser.add_argument("--device", type=str, default="cuda")
    mine_parser.add_argument("--limit", type=int, default=120, help="最多扫描多少个样本。")
    mine_parser.add_argument("--topk", type=int, default=20, help="输出前多少个候选。")
    mine_parser.add_argument("--score-thr", type=float, default=0.05)
    mine_parser.add_argument("--nms-iou-thr", type=float, default=0.60)
    return parser.parse_args()


@dataclass(frozen=True)
class RunSpec:
    config_path: str
    checkpoint_path: str
    model_name: Optional[str] = None


class LoadedRun:
    def __init__(self, run_spec: RunSpec, device: torch.device):
        self.run_spec = run_spec
        self.device = device
        self.cfg = Config.fromfile(run_spec.config_path, use_predefined_variables=False)
        self.model_name = run_spec.model_name or self.cfg.get("model_name", "SAM2PriorAlignmentYOLODetector")
        self.task = task_lib.build_task(self.cfg)
        self.datasets = {}
        self.dataset_indices = {}
        self.model = self._build_model()

    def _build_model(self):
        model_kwargs = dict(self.cfg.get("model", {}))
        model_kwargs.pop("name", None)
        model_kwargs.pop("pretrained", None)
        model_cls = getattr(models_lib, self.model_name)
        model = model_cls(pretrained=self.cfg.get("pretrained", None), **model_kwargs)
        checkpoint = torch.load(self.run_spec.checkpoint_path, map_location="cpu")
        incompatible = model.load_state_dict(checkpoint, strict=False)
        if incompatible.missing_keys:
            print(
                f"[LoadedRun] missing keys when loading {self.run_spec.checkpoint_path}: "
                f"{len(incompatible.missing_keys)}"
            )
        if incompatible.unexpected_keys:
            print(
                f"[LoadedRun] unexpected keys when loading {self.run_spec.checkpoint_path}: "
                f"{len(incompatible.unexpected_keys)}"
            )
        model.to(self.device)
        model.eval()
        return model

    def get_dataset(self, split: str, dataset_name: str):
        key = (split, dataset_name)
        if key in self.datasets:
            return self.datasets[key]
        if split == "test":
            dataset, _ = self.task.build_test_dataset(dataset_name=dataset_name, cfg=self.cfg)
        else:
            dataset = self.task.build_train_dataset(cfg=self.cfg)
        self.datasets[key] = dataset
        self.dataset_indices[key] = self._build_dataset_index(dataset)
        return dataset

    @staticmethod
    def _build_dataset_index(dataset):
        image_id_to_index = {}
        file_name_to_index = {}
        if hasattr(dataset, "samples"):
            for idx, sample in enumerate(dataset.samples):
                image_info = sample["image_info"]
                image_id_to_index[int(image_info["id"])] = idx
                file_name_to_index[str(image_info["file_name"])] = idx
        return dict(image_id=image_id_to_index, file_name=file_name_to_index)

    def resolve_index(self, split: str, dataset_name: str, sample_spec: Dict[str, Any]) -> int:
        self.get_dataset(split=split, dataset_name=dataset_name)
        index_dict = self.dataset_indices[(split, dataset_name)]
        if "image_id" in sample_spec:
            image_id = int(sample_spec["image_id"])
            if image_id not in index_dict["image_id"]:
                raise KeyError(f"image_id={image_id} 不存在于 {dataset_name} ({split}) 中。")
            return index_dict["image_id"][image_id]
        if "file_name" in sample_spec:
            file_name = str(sample_spec["file_name"])
            if file_name not in index_dict["file_name"]:
                raise KeyError(f"file_name={file_name} 不存在于 {dataset_name} ({split}) 中。")
            return index_dict["file_name"][file_name]
        if "index" in sample_spec:
            return int(sample_spec["index"])
        raise KeyError("sample 字段必须至少提供 image_id / file_name / index 之一。")


def ensure_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境未检测到 GPU。")
    return torch.device(device_str)


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_font_path(explicit_path: Optional[str] = None) -> Optional[str]:
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f"font_path 不存在: {explicit_path}")
        return explicit_path
    env_path = os.environ.get("CAVER_CHAPTER3_FONT", "")
    if env_path:
        if not os.path.isfile(env_path):
            raise FileNotFoundError(f"CAVER_CHAPTER3_FONT 指向的字体不存在: {env_path}")
        return env_path
    for candidate in SYSTEM_CJK_FONT_CANDIDATES:
        if os.path.isfile(candidate) and is_valid_font_file(candidate):
            return candidate
    if FONT_CACHE_PATH.is_file() and is_valid_font_file(FONT_CACHE_PATH):
        return str(FONT_CACHE_PATH)
    FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(DEFAULT_FONT_URL, FONT_CACHE_PATH)
        if is_valid_font_file(FONT_CACHE_PATH):
            return str(FONT_CACHE_PATH)
        print(f"[font] 下载得到的字体文件无效: {FONT_CACHE_PATH}")
    except Exception as exc:
        print(f"[font] 下载中文字体失败，将退回默认字体: {exc}")
    if os.path.isfile(FALLBACK_FONT_PATH):
        return FALLBACK_FONT_PATH
    return None


def is_valid_font_file(font_path: Path | str) -> bool:
    try:
        ImageFont.truetype(str(font_path), 16)
        return True
    except Exception:
        return False


def get_font(font_scale: float, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
    font_size = max(12, int(round(font_scale * 30)))
    actual_font_path = font_path or ACTIVE_FONT_PATH
    if actual_font_path is None:
        actual_font_path = resolve_font_path(None)
    if actual_font_path is None:
        return ImageFont.load_default()
    key = (actual_font_path, font_size)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(actual_font_path, font_size)
    return FONT_CACHE[key]


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    mean = image_tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = image_tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    image = (image_tensor.float() * std + mean).clamp(0, 1)
    return tensor_rgb_to_numpy(image)


def depth_to_rgb(depth_tensor: torch.Tensor) -> np.ndarray:
    depth = depth_tensor.float().clamp(0, 1).repeat(3, 1, 1)
    return tensor_rgb_to_numpy(depth)


def tensor_rgb_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def clamp_zoom_box(zoom_box: Optional[Sequence[float]], image_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    if zoom_box is None:
        return None
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in zoom_box]
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, x1 + 1), width)
    y2 = min(max(y2, y1 + 1), height)
    return x1, y1, x2, y2


def put_text(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.55,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
):
    x, y = org
    font = get_font(font_scale)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    bbox = draw.textbbox((x, y), text, font=font, anchor="ls")
    if bg_color is not None:
        draw.rectangle(
            (bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 2),
            fill=bg_color,
        )
    draw.text((x, y), text=text, font=font, fill=color, anchor="ls")
    image[...] = np.asarray(pil_image)


def draw_zoom_hint(image: np.ndarray, zoom_box: Optional[Tuple[int, int, int, int]], color=(255, 0, 255), thickness=2):
    if zoom_box is None:
        return image
    x1, y1, x2, y2 = zoom_box
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)
    return image


def add_zoom_inset(
    image: np.ndarray,
    zoom_box: Optional[Tuple[int, int, int, int]],
    inset_ratio: float = 0.34,
    border_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    canvas = image.copy()
    if zoom_box is None:
        return canvas
    x1, y1, x2, y2 = zoom_box
    if x2 <= x1 or y2 <= y1:
        return canvas
    crop = canvas[y1:y2, x1:x2]
    if crop.size == 0:
        return canvas

    inset_w = max(64, int(round(canvas.shape[1] * inset_ratio)))
    inset_h = max(64, int(round(canvas.shape[0] * inset_ratio)))
    crop = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)

    pad = 10
    dest_x2 = canvas.shape[1] - pad
    dest_y2 = canvas.shape[0] - pad
    dest_x1 = max(dest_x2 - inset_w, pad)
    dest_y1 = max(dest_y2 - inset_h, pad)
    crop = crop[: dest_y2 - dest_y1, : dest_x2 - dest_x1]

    cv2.rectangle(canvas, (dest_x1 - 2, dest_y1 - 2), (dest_x2 + 2, dest_y2 + 2), border_color, thickness=-1)
    canvas[dest_y1:dest_y2, dest_x1:dest_x2] = crop
    cv2.rectangle(canvas, (dest_x1, dest_y1), (dest_x2, dest_y2), (0, 0, 0), thickness=2)
    return canvas


def fit_cell(image: np.ndarray, width: int, height: int, pad_color=(245, 245, 245)) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    ratio = min(width / image.shape[1], height / image.shape[0])
    resized_w = max(1, int(round(image.shape[1] * ratio)))
    resized_h = max(1, int(round(image.shape[0] * ratio)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), pad_color, dtype=np.uint8)
    x0 = (width - resized_w) // 2
    y0 = (height - resized_h) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas


def add_cell_title(
    image: np.ndarray,
    title: str,
    title_height: int,
    font_scale: float,
    bg_color=(250, 250, 250),
    text_color=(20, 20, 20),
) -> np.ndarray:
    title_bar = np.full((title_height, image.shape[1], 3), bg_color, dtype=np.uint8)
    put_text(
        title_bar,
        text=title,
        org=(10, int(title_height * 0.72)),
        font_scale=font_scale,
        color=text_color,
        thickness=1,
        bg_color=None,
    )
    return np.concatenate([title_bar, image], axis=0)


def feature_tensor_to_map(feature_tensor: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
    fmap = feature_tensor_to_norm_array(feature_tensor, target_size)
    heat = cv2.applyColorMap((fmap * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def feature_tensor_to_norm_array(feature_tensor: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
    if feature_tensor.ndim == 4:
        feature_tensor = feature_tensor[0]
    if feature_tensor.ndim == 3:
        fmap = feature_tensor.detach().float().abs().mean(dim=0, keepdim=True).unsqueeze(0)
    elif feature_tensor.ndim == 2:
        fmap = feature_tensor.detach().float().unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported feature tensor shape: {tuple(feature_tensor.shape)}")
    fmap = torch.nn.functional.interpolate(fmap, size=target_size, mode="bilinear", align_corners=False)
    fmap = fmap[0, 0].cpu().numpy()
    return normalize_array(fmap)


def prior_pair_panel(
    student_prior: torch.Tensor,
    teacher_prior: Optional[torch.Tensor],
    target_size: Tuple[int, int],
    zoom_box: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    student_map = prior_to_numpy_map(student_prior, target_size)
    teacher_map = prior_to_numpy_map(teacher_prior, target_size) if teacher_prior is not None else None
    if teacher_map is None:
        student_norm = normalize_array(student_map)
        left = np.full((student_map.shape[0], student_map.shape[1], 3), 235, dtype=np.uint8)
        put_text(left, "Teacher Prior N/A", (16, 28), font_scale=0.62, color=(30, 30, 30), bg_color=(255, 255, 255))
    else:
        teacher_norm, student_norm = normalize_joint_pair(teacher_map, student_map)
        left = cv2.cvtColor(cv2.applyColorMap((teacher_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.applyColorMap((student_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    left = add_zoom_inset(draw_zoom_hint(left, zoom_box=zoom_box), zoom_box=zoom_box)
    right = add_zoom_inset(draw_zoom_hint(right, zoom_box=zoom_box), zoom_box=zoom_box)
    put_text(left, "Teacher", (14, 28), font_scale=0.62, bg_color=(0, 0, 0))
    put_text(right, "Student", (14, 28), font_scale=0.62, bg_color=(0, 0, 0))
    divider = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def prior_to_numpy_map(prior: Optional[torch.Tensor], target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if prior is None:
        return None
    if prior.ndim == 4:
        prior = prior[0]
    fmap = prior.detach().float().mean(dim=0, keepdim=True).unsqueeze(0)
    fmap = torch.nn.functional.interpolate(fmap, size=target_size, mode="bilinear", align_corners=False)
    return fmap[0, 0].cpu().numpy()


def normalize_joint_pair(left_map: np.ndarray, right_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    joint_min = min(float(left_map.min()), float(right_map.min()))
    joint_max = max(float(left_map.max()), float(right_map.max()))
    denom = joint_max - joint_min + 1e-6
    left = np.clip((left_map - joint_min) / denom, 0, 1)
    right = np.clip((right_map - joint_min) / denom, 0, 1)
    return left, right


def normalize_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = array - float(array.min())
    return array / (float(array.max()) + 1e-6)


def build_region_masks(gt_boxes: torch.Tensor, image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    height, width = image_shape
    boxes_np = gt_boxes.detach().cpu().numpy() if torch.is_tensor(gt_boxes) else np.asarray(gt_boxes)
    inner = np.zeros((height, width), dtype=np.float32)
    expanded = np.zeros((height, width), dtype=np.float32)
    ring = np.zeros((height, width), dtype=np.float32)
    for box in boxes_np:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        inner[y1:y2, x1:x2] = 1.0
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad = max(6, int(round(0.12 * max(bw, bh))))
        ex1 = max(0, x1 - pad)
        ey1 = max(0, y1 - pad)
        ex2 = min(width, x2 + pad)
        ey2 = min(height, y2 + pad)
        expanded[ey1:ey2, ex1:ex2] = 1.0
        ring[ey1:ey2, ex1:ex2] = 1.0
        ring[y1:y2, x1:x2] = 0.0
    background = 1.0 - expanded
    return dict(inner=inner, expanded=expanded, ring=ring, background=background)


def mean_on_mask(feature_map: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=np.float32)
    denom = float(mask.sum())
    if denom <= 1e-6:
        return 0.0
    return float((feature_map * mask).sum() / denom)


def cosine_similarity_map(left: np.ndarray, right: np.ndarray) -> float:
    left_vec = left.reshape(-1).astype(np.float32)
    right_vec = right.reshape(-1).astype(np.float32)
    denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec) + 1e-6)
    if denom <= 1e-6:
        return 0.0
    return float(np.dot(left_vec, right_vec) / denom)


def draw_gt_and_pred(
    base_rgb: np.ndarray,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    class_names: Sequence[str],
    palette: Sequence[Tuple[int, int, int]],
    zoom_box: Optional[Tuple[int, int, int, int]],
    line_thickness: int,
) -> np.ndarray:
    canvas = base_rgb.copy()
    gt_boxes_np = gt_boxes.detach().cpu().numpy() if torch.is_tensor(gt_boxes) else np.asarray(gt_boxes)
    gt_labels_np = gt_labels.detach().cpu().numpy() if torch.is_tensor(gt_labels) else np.asarray(gt_labels)
    pred_boxes_np = pred_boxes.detach().cpu().numpy() if torch.is_tensor(pred_boxes) else np.asarray(pred_boxes)
    pred_labels_np = pred_labels.detach().cpu().numpy() if torch.is_tensor(pred_labels) else np.asarray(pred_labels)
    pred_scores_np = pred_scores.detach().cpu().numpy() if torch.is_tensor(pred_scores) else np.asarray(pred_scores)

    for box, label in zip(gt_boxes_np, gt_labels_np):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), thickness=max(2, line_thickness + 1))
        name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
        put_text(canvas, f"GT:{name}", (max(6, x1), max(20, y1 - 4)), font_scale=0.52, bg_color=(255, 255, 255), color=(0, 0, 0))

    for box, label, score in zip(pred_boxes_np, pred_labels_np, pred_scores_np):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = palette[int(label) % len(palette)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=line_thickness)
        name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
        put_text(canvas, f"{name}:{float(score):.2f}", (max(6, x1), min(canvas.shape[0] - 10, max(20, y1 + 18))), font_scale=0.5, bg_color=(0, 0, 0))

    put_text(canvas, "GT:white  Pred:class-color", (10, canvas.shape[0] - 12), font_scale=0.5, bg_color=(0, 0, 0))
    canvas = draw_zoom_hint(canvas, zoom_box=zoom_box)
    canvas = add_zoom_inset(canvas, zoom_box=zoom_box)
    return canvas


def draw_gt_only(
    base_rgb: np.ndarray,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    class_names: Sequence[str],
    zoom_box: Optional[Tuple[int, int, int, int]],
    line_thickness: int,
) -> np.ndarray:
    canvas = base_rgb.copy()
    gt_boxes_np = gt_boxes.detach().cpu().numpy() if torch.is_tensor(gt_boxes) else np.asarray(gt_boxes)
    gt_labels_np = gt_labels.detach().cpu().numpy() if torch.is_tensor(gt_labels) else np.asarray(gt_labels)
    for box, label in zip(gt_boxes_np, gt_labels_np):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), thickness=max(2, line_thickness + 1))
        name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
        put_text(canvas, f"GT:{name}", (max(6, x1), max(20, y1 - 4)), font_scale=0.52, bg_color=(255, 255, 255), color=(0, 0, 0))
    put_text(canvas, "GT:white", (10, canvas.shape[0] - 12), font_scale=0.5, bg_color=(0, 0, 0))
    canvas = draw_zoom_hint(canvas, zoom_box=zoom_box)
    canvas = add_zoom_inset(canvas, zoom_box=zoom_box)
    return canvas


def extract_module_views(module_name: str, outputs: Dict[str, Any], target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    module_name = module_name.lower()
    if module_name == "spd_dre":
        rgb_aux = outputs["aux"]["rgb_spd"][0]
        ir_aux = outputs["aux"]["ir_spd"][0]
        spatial = 0.5 * (rgb_aux["spatial"] + ir_aux["spatial"])
        frequency = 0.5 * (rgb_aux["frequency"] + ir_aux["frequency"])
        return feature_tensor_to_map(spatial, target_size), feature_tensor_to_map(frequency, target_size)
    if module_name == "fddsa":
        pan_aux = outputs["aux"]["neck"]["pan"][0]
        return feature_tensor_to_map(pan_aux["focus_map"], target_size), feature_tensor_to_map(pan_aux["divergence_map"], target_size)
    if module_name == "hfsd":
        return feature_tensor_to_map(outputs["cls_features"][0], target_size), feature_tensor_to_map(outputs["reg_features"][0], target_size)
    raise KeyError(f"Unsupported module name: {module_name}")


def resolve_column_titles(row_cfg: Dict[str, Any], figure_mode: str) -> Dict[str, str]:
    module_name = row_cfg["module"].lower()
    defaults = MODULE_VIEW_SPECS[module_name]
    compare_cfg = row_cfg.get("compare_run", {})
    compare_title = compare_cfg.get("label", defaults["compare_label"])
    if figure_mode == "evidence":
        compare_title = row_cfg.get("gt_title", defaults["evidence_label"])
    return dict(
        rgb=row_cfg.get("rgb_title", "RGB"),
        ir=row_cfg.get("ir_title", "IR"),
        prior=row_cfg.get("prior_title", "Teacher / Student Prior"),
        mid_a=row_cfg.get("mid_a_title", defaults["mid_a_title"]),
        mid_b=row_cfg.get("mid_b_title", defaults["mid_b_title"]),
        compare=row_cfg.get("compare_title", compare_title),
        full=row_cfg.get("full_title", "Full Model Prediction" if figure_mode == "evidence" else "Full Model"),
    )


def resolve_figure_mode(spec: Dict[str, Any], row_cfg: Dict[str, Any]) -> str:
    mode = row_cfg.get("figure_mode", spec.get("figure_mode", None))
    if mode is None:
        mode = "comparison" if row_cfg.get("compare_run", None) else "evidence"
    mode = str(mode).lower()
    if mode not in ("comparison", "evidence"):
        raise ValueError(f"Unsupported figure_mode={mode}. Expected 'comparison' or 'evidence'.")
    if mode == "comparison" and not row_cfg.get("compare_run", None):
        raise KeyError("figure_mode='comparison' 时必须提供 compare_run。")
    return mode


def move_to_device(data: Any, device: torch.device) -> Any:
    if torch.is_tensor(data):
        return data.to(device, non_blocking=device.type == "cuda")
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    return data


def clone_batch_cpu(batch: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(batch)


def run_single_sample(
    run: LoadedRun,
    split: str,
    dataset_name: str,
    sample_spec: Dict[str, Any],
    infer_cfg: Optional[Dict[str, float]],
    need_teacher_prior: bool = False,
) -> Dict[str, Any]:
    dataset = run.get_dataset(split=split, dataset_name=dataset_name)
    sample_index = run.resolve_index(split=split, dataset_name=dataset_name, sample_spec=sample_spec)
    batch_cpu = run.task.get_collate_fn(split)([dataset[sample_index]])
    batch_device = move_to_device(batch_cpu, run.device)

    with torch.no_grad():
        outputs = run.model(data=run.task.get_model_inputs(batch_device))
        if need_teacher_prior and outputs.get("teacher_prior", None) is None and getattr(run.model, "teacher_adapter", None) is not None:
            teacher_adapter = run.model.teacher_adapter
            if teacher_adapter.enabled and outputs.get("student_prior", None) is not None:
                try:
                    outputs["teacher_prior"] = teacher_adapter(
                        batch_device["image"],
                        batch_device["depth"],
                        target_size=outputs["student_prior"].shape[-2:],
                    )
                except Exception as exc:
                    outputs["teacher_prior"] = None
                    print(f"[warn] failed to fetch teacher_prior for visualization: {exc}")
                    if run.device.type == "cuda":
                        torch.cuda.empty_cache()
        predictions = run.task.predict(model_outputs=outputs, batch=batch_device, infer_cfg=infer_cfg)
    return dict(batch_cpu=clone_batch_cpu(batch_cpu), outputs=outputs, predictions=predictions, sample_index=sample_index)


def render_row(
    row_cfg: Dict[str, Any],
    full_run: LoadedRun,
    compare_run: Optional[LoadedRun],
    layout_cfg: Dict[str, Any],
    figure_mode: str,
) -> np.ndarray:
    split = str(row_cfg.get("split", "test"))
    dataset_name = row_cfg["dataset_name"]
    infer_cfg = row_cfg.get("inference", None)
    if infer_cfg is None:
        infer_cfg = dict(score_thr=full_run.task.infer_cfg.get("score_thr", 0.05), nms_iou_thr=full_run.task.infer_cfg.get("nms_iou_thr", 0.6))
    else:
        infer_cfg = {
            "score_thr": float(infer_cfg.get("score_thr", full_run.task.infer_cfg.get("score_thr", 0.05))),
            "nms_iou_thr": float(infer_cfg.get("nms_iou_thr", full_run.task.infer_cfg.get("nms_iou_thr", 0.6))),
            "pre_nms_topk": int(infer_cfg.get("pre_nms_topk", full_run.task.infer_cfg.get("pre_nms_topk", 1000))),
            "max_per_img": int(infer_cfg.get("max_per_img", full_run.task.infer_cfg.get("max_per_img", 300))),
        }

    full_result = run_single_sample(
        run=full_run,
        split=split,
        dataset_name=dataset_name,
        sample_spec=row_cfg["sample"],
        infer_cfg=infer_cfg,
        need_teacher_prior=True,
    )
    compare_result = None
    if figure_mode == "comparison":
        if compare_run is None:
            raise RuntimeError("comparison mode requires compare_run.")
        compare_result = run_single_sample(
            run=compare_run,
            split=split,
            dataset_name=dataset_name,
            sample_spec=row_cfg["sample"],
            infer_cfg=infer_cfg,
            need_teacher_prior=False,
        )

    batch_cpu = full_result["batch_cpu"]
    target = batch_cpu["targets"][0]
    image_rgb = denormalize_image(batch_cpu["image"][0])
    depth_rgb = depth_to_rgb(batch_cpu["depth"][0])
    zoom_box = clamp_zoom_box(row_cfg.get("zoom_box", None), image_rgb.shape)

    target_size = tuple(batch_cpu["image"].shape[-2:])
    teacher_student = prior_pair_panel(
        student_prior=full_result["outputs"]["student_prior"],
        teacher_prior=full_result["outputs"].get("teacher_prior", None),
        target_size=target_size,
        zoom_box=zoom_box,
    )
    mid_a_map, mid_b_map = extract_module_views(row_cfg["module"], full_result["outputs"], target_size=target_size)
    mid_a_map = add_zoom_inset(draw_zoom_hint(mid_a_map, zoom_box=zoom_box), zoom_box=zoom_box)
    mid_b_map = add_zoom_inset(draw_zoom_hint(mid_b_map, zoom_box=zoom_box), zoom_box=zoom_box)

    class_names = full_run.task.class_names or ()
    palette = full_run.task._vis_palette
    full_pred = full_result["predictions"][0]
    if figure_mode == "comparison":
        assert compare_result is not None
        compare_pred = compare_result["predictions"][0]
        compare_panel = draw_gt_and_pred(
            base_rgb=image_rgb,
            gt_boxes=target["boxes"],
            gt_labels=target["labels"],
            pred_boxes=compare_pred["boxes"].detach().cpu(),
            pred_labels=compare_pred["labels"].detach().cpu(),
            pred_scores=compare_pred["scores"].detach().cpu(),
            class_names=class_names,
            palette=palette,
            zoom_box=zoom_box,
            line_thickness=int(layout_cfg["line_thickness"]),
        )
    else:
        compare_panel = draw_gt_only(
            base_rgb=image_rgb,
            gt_boxes=target["boxes"],
            gt_labels=target["labels"],
            class_names=class_names,
            zoom_box=zoom_box,
            line_thickness=int(layout_cfg["line_thickness"]),
        )
    full_panel = draw_gt_and_pred(
        base_rgb=image_rgb,
        gt_boxes=target["boxes"],
        gt_labels=target["labels"],
        pred_boxes=full_pred["boxes"].detach().cpu(),
        pred_labels=full_pred["labels"].detach().cpu(),
        pred_scores=full_pred["scores"].detach().cpu(),
        class_names=class_names,
        palette=palette,
        zoom_box=zoom_box,
        line_thickness=int(layout_cfg["line_thickness"]),
    )

    column_titles = resolve_column_titles(row_cfg, figure_mode=figure_mode)
    cells = [
        (column_titles["rgb"], add_zoom_inset(draw_zoom_hint(image_rgb.copy(), zoom_box=zoom_box), zoom_box=zoom_box)),
        (column_titles["ir"], add_zoom_inset(draw_zoom_hint(depth_rgb.copy(), zoom_box=zoom_box), zoom_box=zoom_box)),
        (column_titles["prior"], add_zoom_inset(draw_zoom_hint(teacher_student, zoom_box=zoom_box), zoom_box=zoom_box)),
        (column_titles["mid_a"], mid_a_map),
        (column_titles["mid_b"], mid_b_map),
        (column_titles["compare"], compare_panel),
        (column_titles["full"], full_panel),
    ]

    titled_cells = []
    for title, image in cells:
        image = fit_cell(image, width=layout_cfg["cell_width"], height=layout_cfg["cell_height"])
        image = add_cell_title(
            image=image,
            title=title,
            title_height=layout_cfg["title_height"],
            font_scale=layout_cfg["font_scale"],
        )
        titled_cells.append(image)
    row_strip = stack_images_h(titled_cells, gap=layout_cfg["gap"], bg_color=(255, 255, 255))
    row_strip = attach_row_label(
        row_strip=row_strip,
        row_title=row_cfg.get("row_title", row_cfg["module"].upper()),
        row_label_width=layout_cfg["row_label_width"],
        font_scale=max(0.72, layout_cfg["font_scale"]),
    )
    return row_strip


def attach_row_label(row_strip: np.ndarray, row_title: str, row_label_width: int, font_scale: float) -> np.ndarray:
    label_canvas = np.full((row_strip.shape[0], row_label_width, 3), 245, dtype=np.uint8)
    put_multiline_center(
        label_canvas,
        text=row_title,
        font_scale=font_scale,
        color=(20, 20, 20),
    )
    return np.concatenate([label_canvas, row_strip], axis=1)


def put_multiline_center(image: np.ndarray, text: str, font_scale: float, color: Tuple[int, int, int]):
    lines = split_text_for_cell(text, max_chars=16)
    font = get_font(font_scale)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    total_text_h = sum(line_heights) + max(0, len(lines) - 1) * 10
    start_y = max(30, image.shape[0] // 2 - total_text_h // 2)
    current_y = start_y
    for idx, line in enumerate(lines):
        x = max(8, image.shape[1] // 2 - line_widths[idx] // 2)
        current_y += line_heights[idx]
        draw.text((x, current_y), line, font=font, fill=color, anchor="ls")
        current_y += 10
    image[...] = np.asarray(pil_image)


def split_text_for_cell(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    parts = []
    current = ""
    for char in text:
        current += char
        if len(current) >= max_chars:
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return parts


def stack_images_h(images: Sequence[np.ndarray], gap: int, bg_color=(255, 255, 255)) -> np.ndarray:
    total_width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    max_height = max(image.shape[0] for image in images)
    canvas = np.full((max_height, total_width, 3), bg_color, dtype=np.uint8)
    cursor = 0
    for image in images:
        canvas[: image.shape[0], cursor : cursor + image.shape[1]] = image
        cursor += image.shape[1] + gap
    return canvas


def stack_images_v(images: Sequence[np.ndarray], gap: int, bg_color=(255, 255, 255)) -> np.ndarray:
    total_height = sum(image.shape[0] for image in images) + gap * (len(images) - 1)
    max_width = max(image.shape[1] for image in images)
    canvas = np.full((total_height, max_width, 3), bg_color, dtype=np.uint8)
    cursor = 0
    for image in images:
        canvas[cursor : cursor + image.shape[0], : image.shape[1]] = image
        cursor += image.shape[0] + gap
    return canvas


def render_figure(spec: Dict[str, Any], args):
    global ACTIVE_FONT_PATH
    device = ensure_device(args.device)
    ACTIVE_FONT_PATH = resolve_font_path(spec.get("font_path", args.font_path))
    print(f"[font] use font: {ACTIVE_FONT_PATH or 'PIL default'}")
    layout_cfg = dict(
        cell_width=int(spec.get("cell_width", args.cell_width)),
        cell_height=int(spec.get("cell_height", args.cell_height)),
        title_height=int(spec.get("title_height", args.title_height)),
        row_label_width=int(spec.get("row_label_width", args.row_label_width)),
        margin=int(spec.get("margin", args.margin)),
        gap=int(spec.get("gap", args.gap)),
        font_scale=float(spec.get("font_scale", args.font_scale)),
        header_font_scale=float(spec.get("header_font_scale", args.header_font_scale)),
        line_thickness=int(spec.get("line_thickness", args.line_thickness)),
    )

    run_cache: Dict[Tuple[str, str, str], LoadedRun] = {}

    def get_run(run_cfg: Dict[str, Any]) -> LoadedRun:
        run_spec = RunSpec(
            config_path=run_cfg["config"],
            checkpoint_path=run_cfg["checkpoint"],
            model_name=run_cfg.get("model_name", None),
        )
        key = (run_spec.config_path, run_spec.checkpoint_path, run_spec.model_name or "")
        if key not in run_cache:
            run_cache[key] = LoadedRun(run_spec=run_spec, device=device)
        return run_cache[key]

    row_images = []
    for row_cfg in spec["rows"]:
        module_name = row_cfg["module"].lower()
        if module_name not in MODULE_VIEW_SPECS:
            raise KeyError(f"Unsupported module={module_name}. Expected one of: {sorted(MODULE_VIEW_SPECS)}")
        figure_mode = resolve_figure_mode(spec=spec, row_cfg=row_cfg)
        full_run = get_run(row_cfg["full_run"])
        compare_run = get_run(row_cfg["compare_run"]) if row_cfg.get("compare_run", None) else None
        row_images.append(
            render_row(
                row_cfg=row_cfg,
                full_run=full_run,
                compare_run=compare_run,
                layout_cfg=layout_cfg,
                figure_mode=figure_mode,
            )
        )

    body = stack_images_v(row_images, gap=layout_cfg["gap"], bg_color=(255, 255, 255))
    title = spec.get("title", "图3.9 M3FD数据集上第三章核心模块的可解释性与定性证据图")
    subtitle = spec.get("subtitle", "")
    header = build_header(
        width=body.shape[1],
        title=title,
        subtitle=subtitle,
        margin=layout_cfg["margin"],
        font_scale=layout_cfg["header_font_scale"],
    )
    figure = stack_images_v([header, body], gap=layout_cfg["gap"], bg_color=(255, 255, 255))
    figure = cv2.copyMakeBorder(
        figure,
        layout_cfg["margin"],
        layout_cfg["margin"],
        layout_cfg["margin"],
        layout_cfg["margin"],
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    ensure_parent(args.output)
    cv2.imwrite(args.output, cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))
    print(f"[render] saved figure to: {args.output}")


def build_header(width: int, title: str, subtitle: str, margin: int, font_scale: float) -> np.ndarray:
    header_h = 90 if subtitle else 62
    header = np.full((header_h, width, 3), 255, dtype=np.uint8)
    put_text(header, title, (margin, 32), font_scale=font_scale, color=(10, 10, 10), thickness=2, bg_color=None)
    if subtitle:
        put_text(header, subtitle, (margin, 66), font_scale=0.62, color=(70, 70, 70), thickness=1, bg_color=None)
    cv2.line(header, (0, header_h - 2), (width, header_h - 2), (215, 215, 215), thickness=2)
    return header


def safe_detach_cpu(prediction: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        boxes=prediction["boxes"].detach().cpu(),
        scores=prediction["scores"].detach().cpu(),
        labels=prediction["labels"].detach().cpu(),
        image_id=int(prediction["image_id"]),
    )


def summarize_prediction(prediction: Dict[str, Any], target: Dict[str, torch.Tensor], iou_thr: float = 0.5) -> Dict[str, float]:
    pred = safe_detach_cpu(prediction)
    gt_boxes = target["boxes"].detach().cpu()
    gt_labels = target["labels"].detach().cpu()

    pred_boxes = pred["boxes"]
    pred_labels = pred["labels"]
    pred_scores = pred["scores"]
    num_gt = int(gt_boxes.shape[0])
    num_pred = int(pred_boxes.shape[0])
    if num_gt == 0:
        return dict(tp=0, fp=num_pred, fn=0, best_iou=0.0, num_gt=0, num_pred=num_pred, matched_gt_ratio=0.0)
    if num_pred == 0:
        return dict(tp=0, fp=0, fn=num_gt, best_iou=0.0, num_gt=num_gt, num_pred=0, matched_gt_ratio=0.0)

    used_gt = torch.zeros(num_gt, dtype=torch.bool)
    order = pred_scores.argsort(descending=True)
    tp = 0
    fp = 0
    best_iou = 0.0
    for pred_idx in order.tolist():
        pred_box = pred_boxes[pred_idx : pred_idx + 1]
        pred_label = int(pred_labels[pred_idx])
        ious = bbox_iou_cpu(pred_box, gt_boxes).squeeze(0)
        if ious.numel() > 0:
            best_iou = max(best_iou, float(ious.max().item()))
        match_idx = -1
        match_iou = 0.0
        for gt_idx in range(num_gt):
            if used_gt[gt_idx]:
                continue
            if int(gt_labels[gt_idx]) != pred_label:
                continue
            iou_val = float(ious[gt_idx].item())
            if iou_val >= iou_thr and iou_val > match_iou:
                match_iou = iou_val
                match_idx = gt_idx
        if match_idx >= 0:
            used_gt[match_idx] = True
            tp += 1
        else:
            fp += 1
    fn = num_gt - tp
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        best_iou=best_iou,
        num_gt=num_gt,
        num_pred=num_pred,
        matched_gt_ratio=float(tp / max(num_gt, 1)),
    )


def bbox_iou_cpu(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def compute_module_gain(module: str, full_stats: Dict[str, float], compare_stats: Dict[str, float], target: Dict[str, torch.Tensor]) -> float:
    image_h = int(target["size"][0].item()) if "size" in target else 768
    image_w = int(target["size"][1].item()) if "size" in target else 1024
    image_area = float(image_h * image_w)
    gt_boxes = target["boxes"].detach().cpu()
    if gt_boxes.numel() > 0:
        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        mean_area_ratio = float((areas / max(image_area, 1.0)).mean().item())
    else:
        mean_area_ratio = 0.0
    small_object_bonus = 1.0 / (mean_area_ratio + 1e-4)

    if module == "spd_dre":
        score = (
            4.0 * (full_stats["tp"] - compare_stats["tp"])
            + 1.8 * (compare_stats["fn"] - full_stats["fn"])
            + 2.0 * (full_stats["best_iou"] - compare_stats["best_iou"])
            + 0.001 * small_object_bonus
        )
        return score
    if module == "fddsa":
        score = (
            2.5 * (compare_stats["fp"] - full_stats["fp"])
            + 2.0 * (full_stats["best_iou"] - compare_stats["best_iou"])
            + 1.5 * (full_stats["tp"] - compare_stats["tp"])
        )
        return score
    if module == "hfsd":
        score = (
            3.0 * (full_stats["best_iou"] - compare_stats["best_iou"])
            + 2.0 * (full_stats["tp"] - compare_stats["tp"])
            + 1.0 * (compare_stats["fp"] - full_stats["fp"])
        )
        return score
    raise KeyError(f"Unsupported module: {module}")


def summarize_module_evidence(
    module: str,
    outputs: Dict[str, Any],
    target: Dict[str, torch.Tensor],
    target_size: Tuple[int, int],
    prediction_stats: Dict[str, float],
) -> Dict[str, float]:
    masks = build_region_masks(target["boxes"], image_shape=target_size)
    quality = 2.0 * float(prediction_stats["best_iou"]) + 1.5 * float(prediction_stats["matched_gt_ratio"])

    if module == "spd_dre":
        rgb_aux = outputs["aux"]["rgb_spd"][0]
        ir_aux = outputs["aux"]["ir_spd"][0]
        spatial = feature_tensor_to_norm_array(0.5 * (rgb_aux["spatial"] + ir_aux["spatial"]), target_size)
        frequency = feature_tensor_to_norm_array(0.5 * (rgb_aux["frequency"] + ir_aux["frequency"]), target_size)
        spatial_contrast = mean_on_mask(spatial, masks["inner"]) - mean_on_mask(spatial, masks["background"])
        frequency_contrast = mean_on_mask(frequency, masks["inner"]) - mean_on_mask(frequency, masks["background"])
        student_prior = normalize_array(prior_to_numpy_map(outputs["student_prior"], target_size))
        teacher_prior = outputs.get("teacher_prior", None)
        prior_cos = 0.0
        if teacher_prior is not None:
            teacher_prior_map = normalize_array(prior_to_numpy_map(teacher_prior, target_size))
            prior_cos = cosine_similarity_map(teacher_prior_map, student_prior)
        score = quality + 1.2 * prior_cos + 0.9 * spatial_contrast + 0.8 * frequency_contrast
        return dict(
            score=float(score),
            best_iou=float(prediction_stats["best_iou"]),
            matched_gt_ratio=float(prediction_stats["matched_gt_ratio"]),
            prior_cos=float(prior_cos),
            spatial_contrast=float(spatial_contrast),
            frequency_contrast=float(frequency_contrast),
        )

    if module == "fddsa":
        pan_aux = outputs["aux"]["neck"]["pan"][0]
        focus_map = feature_tensor_to_norm_array(pan_aux["focus_map"], target_size)
        divergence_map = feature_tensor_to_norm_array(pan_aux["divergence_map"], target_size)
        focus_contrast = mean_on_mask(focus_map, masks["inner"]) - mean_on_mask(focus_map, masks["background"])
        divergence_ring = mean_on_mask(divergence_map, masks["ring"]) - mean_on_mask(divergence_map, masks["inner"])
        score = quality + 1.0 * focus_contrast + 0.9 * divergence_ring
        return dict(
            score=float(score),
            best_iou=float(prediction_stats["best_iou"]),
            matched_gt_ratio=float(prediction_stats["matched_gt_ratio"]),
            focus_contrast=float(focus_contrast),
            divergence_ring=float(divergence_ring),
        )

    if module == "hfsd":
        cls_map = feature_tensor_to_norm_array(outputs["cls_features"][0], target_size)
        reg_map = feature_tensor_to_norm_array(outputs["reg_features"][0], target_size)
        cls_contrast = mean_on_mask(cls_map, masks["inner"]) - mean_on_mask(cls_map, masks["background"])
        reg_edge = mean_on_mask(reg_map, masks["ring"]) - mean_on_mask(reg_map, masks["inner"])
        score = quality + 1.0 * cls_contrast + 1.0 * reg_edge
        return dict(
            score=float(score),
            best_iou=float(prediction_stats["best_iou"]),
            matched_gt_ratio=float(prediction_stats["matched_gt_ratio"]),
            cls_contrast=float(cls_contrast),
            reg_edge=float(reg_edge),
        )

    raise KeyError(f"Unsupported module: {module}")


def mine_candidates(args):
    device = ensure_device(args.device)
    full_run = LoadedRun(
        RunSpec(config_path=args.full_config, checkpoint_path=args.full_checkpoint),
        device=device,
    )
    if bool(args.compare_config) != bool(args.compare_checkpoint):
        raise ValueError("compare-config 与 compare-checkpoint 必须同时提供，或同时省略。")
    use_compare = bool(args.compare_config and args.compare_checkpoint)
    compare_run = None
    if use_compare:
        compare_run = LoadedRun(
            RunSpec(config_path=args.compare_config, checkpoint_path=args.compare_checkpoint),
            device=device,
        )
    infer_cfg = dict(
        score_thr=float(args.score_thr),
        nms_iou_thr=float(args.nms_iou_thr),
        pre_nms_topk=full_run.task.infer_cfg.get("pre_nms_topk", 1000),
        max_per_img=full_run.task.infer_cfg.get("max_per_img", 300),
    )

    dataset = full_run.get_dataset(split=args.split, dataset_name=args.dataset_name)
    limit = min(int(args.limit), len(dataset))
    results = []
    for index in range(limit):
        sample_spec = dict(index=index)
        full_result = run_single_sample(
            run=full_run,
            split=args.split,
            dataset_name=args.dataset_name,
            sample_spec=sample_spec,
            infer_cfg=infer_cfg,
            need_teacher_prior=args.module == "spd_dre" and not use_compare,
        )
        target = full_result["batch_cpu"]["targets"][0]
        full_stats = summarize_prediction(full_result["predictions"][0], target)
        image_info = full_result["batch_cpu"]["image_info"][0]
        if use_compare:
            assert compare_run is not None
            compare_result = run_single_sample(
                run=compare_run,
                split=args.split,
                dataset_name=args.dataset_name,
                sample_spec=sample_spec,
                infer_cfg=infer_cfg,
                need_teacher_prior=False,
            )
            compare_stats = summarize_prediction(compare_result["predictions"][0], target)
            gain = compute_module_gain(args.module, full_stats, compare_stats, target)
            results.append(
                dict(
                    index=index,
                    image_id=int(target["image_id"].item()),
                    file_name=str(image_info["file_name"]),
                    gain=round(float(gain), 6),
                    full=full_stats,
                    compare=compare_stats,
                )
            )
        else:
            evidence = summarize_module_evidence(
                module=args.module,
                outputs=full_result["outputs"],
                target=target,
                target_size=tuple(full_result["batch_cpu"]["image"].shape[-2:]),
                prediction_stats=full_stats,
            )
            score = evidence.pop("score")
            results.append(
                dict(
                    index=index,
                    image_id=int(target["image_id"].item()),
                    file_name=str(image_info["file_name"]),
                    score=round(float(score), 6),
                    prediction=full_stats,
                    evidence=evidence,
                )
            )
        if (index + 1) % 10 == 0 or index + 1 == limit:
            print(f"[mine] processed {index + 1}/{limit}")

    key_name = "gain" if use_compare else "score"
    results.sort(key=lambda item: item[key_name], reverse=True)
    payload = dict(
        mode="comparison" if use_compare else "evidence",
        module=args.module,
        dataset_name=args.dataset_name,
        split=args.split,
        infer_cfg=infer_cfg,
        topk=results[: args.topk],
        scanned=limit,
    )
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
    print(f"[mine] saved candidates to: {args.output}")


def main():
    args = parse_args()
    if args.command == "render":
        with open(args.spec, "r", encoding="utf-8") as file_obj:
            spec = json.load(file_obj)
        render_figure(spec=spec, args=args)
        return
    if args.command == "mine":
        mine_candidates(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
