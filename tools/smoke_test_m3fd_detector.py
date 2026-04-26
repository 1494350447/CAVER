#!/usr/bin/env python3
"""用真实 M3FD 样本做一次单 batch smoke test。"""

import argparse
import inspect
import sys
from pathlib import Path

import torch
from mmengine import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models as models_lib
import tasks as task_lib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/root/CAVER/configs/chapter3_m3fd_sam2_large.py")
    parser.add_argument("--model-name", type=str, default="SAM2PriorAlignmentYOLODetector")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--disable-teacher", action="store_true")
    return parser.parse_args()


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=device.type == "cuda")
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    return data


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config, use_predefined_variables=False)
    if args.disable_teacher:
        cfg.model.teacher_adapter_cfg.enable = False

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境未检测到可用 GPU。")
    device = torch.device(args.device)

    task = task_lib.build_task(cfg)

    dataset = task.build_train_dataset(cfg=cfg)
    batch = task.get_collate_fn(split="train")([dataset[args.sample_index]])
    batch = move_to_device(batch, device)
    model_kwargs = dict(cfg.get("model", {}))
    model_kwargs.pop("name", None)
    model_kwargs.pop("pretrained", None)

    module_class = getattr(models_lib, args.model_name)
    model = module_class(pretrained=args.pretrained, **model_kwargs)
    model.to(device)

    print("model_class:", module_class.__name__)
    print("model_source_head:", inspect.getsource(module_class).splitlines()[0])
    print("device:", device)
    print("teacher_enabled:", bool(cfg.model.teacher_adapter_cfg.enable))
    print("sample_index:", args.sample_index)

    model.train()
    with torch.no_grad():
        train_outputs = model(data=task.get_model_inputs(batch))
    train_loss, loss_str = task.compute_loss(train_outputs, batch)
    print("train_loss:", float(train_loss))
    print("loss_str:", loss_str)
    print("teacher_prior_shape:", None if train_outputs["teacher_prior"] is None else tuple(train_outputs["teacher_prior"].shape))
    print("student_prior_shape:", tuple(train_outputs["student_prior"].shape))

    model.eval()
    with torch.no_grad():
        eval_outputs = model(data=task.get_model_inputs(batch))
    predictions = task.predict(eval_outputs, batch)
    print("teacher_prior_eval:", eval_outputs["teacher_prior"])
    print("pred_count:", len(predictions[0]["boxes"]))
    print("cls_shapes:", [tuple(item.shape) for item in eval_outputs["cls_logits"]])
    print("bbox_shapes:", [tuple(item.shape) for item in eval_outputs["bbox_preds"]])


if __name__ == "__main__":
    main()
