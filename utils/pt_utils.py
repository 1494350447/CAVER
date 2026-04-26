import os
import random

import numpy as np
import torch
from torch import nn


def worker_init_fn(worker_id, base_seed):
    set_seed_for_lib(base_seed + worker_id)


def set_seed_for_lib(seed):
    random.seed(seed)
    np.random.seed(seed)
    # 为了禁止hash随机化，使得实验可复现。
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def initialize_seed_cudnn(seed, deterministic):
    assert isinstance(deterministic, bool) and isinstance(seed, int)
    if seed >= 0:
        print(f"We will use the fixed seed: {seed} !!!")
        set_seed_for_lib(seed)
    else:
        print(f"We will not use the fixed seed !!!")
    if not deterministic:
        print("We will use `torch.backends.cudnn.benchmark`")
    else:
        print("We will not use `torch.backends.cudnn.benchmark`")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def is_on_gpu(x):
    """
    判定x是否是gpu上的实例，可以检测tensor和module
    :param x: (torch.Tensor, nn.Module)目标对象
    :return: 是否在gpu上
    """
    # https://blog.csdn.net/WYXHAHAHA123/article/details/86596981
    if isinstance(x, torch.Tensor):
        return "cuda" in x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).is_cuda
    else:
        raise NotImplementedError


def get_device(x):
    """
    返回x的设备信息，可以处理tensor和module
    :param x: (torch.Tensor, nn.Module) 目标对象
    :return: 所在设备
    """
    # https://blog.csdn.net/WYXHAHAHA123/article/details/86596981
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).device
    else:
        raise NotImplementedError


class ModelEMA:
    """维护学生模型的指数滑动平均权重。"""

    def __init__(self, model, decay=0.9998, exclude_prefixes=("teacher_adapter.teacher.",)):
        self.decay = float(decay)
        self.exclude_prefixes = tuple(exclude_prefixes)
        self.shadow_state = {}
        self.backup_state = {}
        self._capture(model=model)

    def _should_track(self, key):
        return not any(key.startswith(prefix) for prefix in self.exclude_prefixes)

    def _capture(self, model):
        for key, value in model.state_dict().items():
            if self._should_track(key):
                self.shadow_state[key] = value.detach().clone()

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for key, value in model_state.items():
            if not self._should_track(key):
                continue
            if key not in self.shadow_state:
                self.shadow_state[key] = value.detach().clone()
                continue
            if torch.is_floating_point(value):
                self.shadow_state[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow_state[key].copy_(value.detach())

    @torch.no_grad()
    def apply_to(self, model):
        self.backup_state = {}
        model_state = model.state_dict()
        for key, value in self.shadow_state.items():
            self.backup_state[key] = model_state[key].detach().clone()
            model_state[key].copy_(value)

    @torch.no_grad()
    def restore(self, model):
        if not self.backup_state:
            return
        model_state = model.state_dict()
        for key, value in self.backup_state.items():
            model_state[key].copy_(value)
        self.backup_state = {}
