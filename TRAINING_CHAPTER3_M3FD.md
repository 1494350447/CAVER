# 第三章模型训练说明（M3FD + SAM2 Large Teacher）

本文档对应当前仓库中的第三章检测模型落地实现，目标是让训练、续训、评估和参数修改都有统一入口。

## 1. 当前推荐入口

推荐直接使用脚本：

```bash
bash tools/train_chapter3_m3fd.sh
```

脚本默认行为：

- 使用配置文件 `configs/chapter3_m3fd_sam2_large.py`
- 使用模型 `SAM2PriorAlignmentYOLODetector`
- 数据集为本地 `M3FD`
- teacher 为官方 `SAM2 Large`
- 输入尺寸固定为 `1024x768`
- 默认输出目录为 `output_chapter3_m3fd_full`
- 默认 fresh train 时启用 `timm:resnet50d.ra4_e3600_r224_in1k` 作为 backbone 预训练

## 2. 当前已验证的训练事实

- 当前 32GB GPU 上，`batch_size=3` 已实测可训练。
- `batch_size=4` 和 `batch_size=8` 会 OOM。
- 当前默认关闭 AMP，即 `use_amp=False`，因为现阶段半精度反传不稳定。
- 每个 epoch 结束后会自动验证一次，日志里会输出：
  - `mAP50`
  - `mAP50_95`
  - `Precision`
  - `Recall`
- 训练时会生成可视化图片，保存到实验目录下的 `imgs/`。

## 3. 一键脚本使用方式

### 3.1 新训练

```bash
bash tools/train_chapter3_m3fd.sh
```

### 3.2 从已有权重继续训练

```bash
MODE=resume LOAD_FROM=/abs/path/to/state.pth bash tools/train_chapter3_m3fd.sh
```

### 3.3 自动寻找最新 checkpoint 后继续训练

```bash
MODE=resume AUTO_RESUME=1 bash tools/train_chapter3_m3fd.sh
```

### 3.4 只做评估

```bash
MODE=eval LOAD_FROM=/abs/path/to/state.pth bash tools/train_chapter3_m3fd.sh
```

### 3.5 指定预训练 backbone

```bash
PRETRAINED=timm:resnet50d.ra4_e3600_r224_in1k bash tools/train_chapter3_m3fd.sh
```

### 3.6 指定 GPU

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/train_chapter3_m3fd.sh
```

## 4. 一键脚本参数说明

脚本文件：`tools/train_chapter3_m3fd.sh`

脚本通过环境变量控制行为。

| 参数名 | 默认值 | 说明 |
|---|---:|---|
| `MODE` | `train` | 运行模式。可选 `train`、`resume`、`eval`。 |
| `PYTHON_BIN` | `python` | Python 可执行文件。 |
| `CONFIG` | `configs/chapter3_m3fd_sam2_large.py` | 使用的配置文件。 |
| `MODEL_NAME` | `SAM2PriorAlignmentYOLODetector` | 模型类名，必须能在 `models/__init__.py` 中解析。 |
| `OUTPUT_ROOT` | `output_chapter3_m3fd_full` | 实验输出根目录。 |
| `CUDA_VISIBLE_DEVICES` | `0` | 使用哪块 GPU。 |
| `OMP_NUM_THREADS` | `1` | CPU OpenMP 线程数。一般保持 1 更稳。 |
| `PRETRAINED` | `timm:resnet50d.ra4_e3600_r224_in1k` | backbone 预训练来源。仅 fresh train 时生效。 |
| `LOAD_FROM` | 空 | 载入的 checkpoint 路径。 |
| `AUTO_RESUME` | `1` | 当 `MODE=resume/eval` 且未手工给 `LOAD_FROM` 时，是否自动寻找最新的 `state.pth`。 |
| `INFO` | 空 | 会拼接到实验名中，方便区分实验。 |
| `SHOW_BAR` | `0` | 是否在测试时显示 tqdm 进度条。`1` 表示打开。 |
| `EXTRA_ARGS` | 空 | 透传给 `main.py` 的额外参数字符串。 |

## 5. `main.py` 命令行参数说明

训练入口文件：`main.py`

| 参数名 | 是否必填 | 说明 |
|---|---|---|
| `--config` | 是 | mmengine 配置文件路径。 |
| `--model-name` | 是 | 模型名。当前第三章模型为 `SAM2PriorAlignmentYOLODetector`。 |
| `--output-root` | 否 | 输出目录根路径。 |
| `--load-from` | 否 | 加载权重文件。当前行为是“加载模型参数”，适合 warm start 或继续训练。 |
| `--pretrained` | 否 | 传给 backbone 的预训练参数。可为本地路径，也可为 `timm` 或 `timm:具体权重名`。 |
| `--info` | 否 | 附加实验标识，会进入实验名。 |
| `--evaluate` | 否 | 若带上，则只评估不训练。 |
| `--show-bar` | 否 | 测试/验证时显示进度条。 |
| `--cooldown-epoch-num` | 否 | 额外的冷却 epoch 数，默认 0。 |

## 6. 配置文件总览

主配置文件：`configs/chapter3_m3fd_sam2_large.py`

它继承自 `configs/base.py`，因此最终生效参数由两部分共同决定。

---

## 7. `task` 段参数说明

### 7.1 `task.name`

- 当前固定为 `bimodal_detection`
- 表示启用双模态检测任务，不再走旧分割任务流程

### 7.2 `task.num_classes`

- 当前为 `6`
- 对应 M3FD 六类：
  - `People`
  - `Car`
  - `Bus`
  - `Motorcycle`
  - `Lamp`
  - `Truck`

### 7.3 `task.strides`

- 当前为 `(8, 16, 32)`
- 表示三层检测特征图对应原图步长
- 必须与检测头输出尺度一致

### 7.4 `task.center_radius`

- 当前为 `2.5`
- 用于 anchor-free 正样本中心区域约束
- 值越大，正样本区域越宽

### 7.5 `task.loss`

| 参数名 | 当前值 | 说明 |
|---|---:|---|
| `cls_weight` | `1.0` | 分类损失权重 |
| `reg_weight` | `1.0` | 框回归损失权重 |
| `distill_weight` | `1.0` | 学生先验和 teacher prior 的蒸馏损失权重 |
| `focal_alpha` | `0.25` | Focal Loss 的 alpha |
| `focal_gamma` | `2.0` | Focal Loss 的 gamma |

### 7.6 `task.inference`

| 参数名 | 当前值 | 说明 |
|---|---:|---|
| `score_thr` | `0.05` | 预测框筛选分数阈值 |
| `nms_iou_thr` | `0.6` | NMS 的 IoU 阈值 |
| `pre_nms_topk` | `1000` | 每张图进入 NMS 前保留的最高分候选数 |
| `max_per_img` | `300` | NMS 后每张图最多保留多少个预测框 |

---

## 8. `model` 段参数说明

### 8.1 主模型结构参数

| 参数名 | 当前值 | 说明 |
|---|---:|---|
| `num_classes` | `6` | 检测类别数 |
| `backbone_name` | `resnet50d` | 共享双流 backbone 名称 |
| `neck_channels` | `256` | FPN/PAN 与中间特征统一通道数 |
| `patch_size` | `4` | SPD-DRE 中空间网格划分大小 |
| `channel_groups` | `4` | F-DDSA 与部分分组计算使用的分组数 |
| `prior_channels` | `128` | 学生先验和 teacher prior 的通道数 |
| `num_frequency_experts` | `3` | HFSD 中频率专家组数 |
| `frequency_coord_dim` | `16` | 频域坐标编码隐藏维 |
| `strides` | `(8,16,32)` | 检测头尺度步长，需与任务层保持一致 |

### 8.2 `teacher_adapter_cfg`

| 参数名 | 当前值 | 说明 |
|---|---:|---|
| `enable` | `True` | 是否启用 teacher 蒸馏 |
| `variant` | `sam2_large` | 当前只支持原版官方 `SAM2 Large` |
| `repo_root` | `/root/autodl-fs/third_party/sam2_official` | 官方 SAM2 源码根目录 |
| `model_cfg` | `configs/sam2/sam2_hiera_l.yaml` | teacher 配置文件，相对 `sam2/` |
| `checkpoint_path` | `/root/autodl-fs/checkpoints/sam2/sam2_hiera_large.pt` | 官方 checkpoint 路径 |
| `freeze` | `True` | teacher 是否冻结 |
| `compile_image_encoder` | `False` | 是否让 SAM2 image encoder 走编译路径 |

---

## 9. `args` 段参数说明

### 9.1 继承自 `base.py` 且当前生效的参数

| 参数名 | 当前值 | 说明 |
|---|---:|---|
| `base_seed` | `42` | 随机种子 |
| `deterministic` | `True` | 是否启用确定性 cudnn 路径 |
| `epoch_num` | `100` | 训练 epoch 数 |
| `batch_size` | `3` | 当前 32GB GPU 验证可用的 batch size |
| `num_workers` | `4` | dataloader worker 数 |
| `print_freq` | `20` | 每隔多少 iter 输出一次训练日志 |
| `val_freq` | `1` | 每多少个 epoch 验证一次 |
| `use_amp` | `False` | 是否启用混合精度 |
| `iter_num` | `10000` | 仅用于非 epoch 模式；当前保留但不主导训练流程 |
| `epoch_based` | `True` | 当前按 epoch 组织训练，而不是按固定 iter 截断 |

### 9.2 参数建议

- `batch_size`
  - 当前建议保持 `3`
  - 如果换更大显存卡，可再尝试增大
- `val_freq`
  - 如果你想更快训练，可改成 `5`
  - 如果你想每轮都看指标，保持 `1`
- `use_amp`
  - 当前建议保持 `False`
  - 等半精度稳定性单独验证后再开

---

## 10. `optimizers` 段参数说明

| 参数名 | 当前值 | 说明 |
|---|---:|---|
| `lr` | `1e-4` | 初始学习率 |
| `strategy` | `all` | 当前对所有可训练参数统一建优化器 |
| `optimizer` | `adamw` | 当前使用 AdamW |
| `optimizer_candidates.adamw.betas` | `(0.9, 0.999)` | AdamW 一阶、二阶动量参数 |
| `optimizer_candidates.adamw.eps` | `1e-8` | 数值稳定项 |
| `optimizer_candidates.adamw.weight_decay` | `5e-4` | 权重衰减 |
| `optimizer_candidates.adamw.amsgrad` | `False` | 是否启用 AMSGrad |

---

## 11. `schedulers` 段参数说明

| 参数名 | 当前值 | 说明 |
|---|---:|---|
| `sche_usebatch` | `True` | 按 batch 而不是按 epoch 更新学习率 |
| `strategy` | `cos` | 当前使用余弦调度 |
| `scheduler_candidates.cos.warmup_length` | `5` | warmup 长度 |
| `scheduler_candidates.cos.min_coef` | `0.001` | 学习率最低系数 |
| `scheduler_candidates.cos.max_coef` | `1` | 学习率最高系数 |

---

## 12. `data` 段参数说明

### 12.1 训练集

| 参数名 | 当前值 | 说明 |
|---|---:|---|
| `data.train.name` | `["M3FD_TRAIN"]` | 训练集注册名 |
| `data.train.shape.h` | `768` | 训练时 resize 高度 |
| `data.train.shape.w` | `1024` | 训练时 resize 宽度 |
| `image_root` | `/root/CAVER/M3FD/Vis` | RGB 图像目录 |
| `depth_root` | `/root/CAVER/M3FD/Ir` | 红外图像目录 |
| `ann_file` | `/root/CAVER/data/m3fd_detection/train_coco.json` | 训练 COCO 标注 |
| `depth_file_key` | `depth_file_name` | COCO 中红外图像文件名字段 |

### 12.2 验证/测试集

| 参数名 | 当前值 | 说明 |
|---|---:|---|
| `data.test.name` | `["M3FD_VAL"]` | 验证集注册名 |
| `data.test.shape.h` | `768` | 测试时 resize 高度 |
| `data.test.shape.w` | `1024` | 测试时 resize 宽度 |
| `image_root` | `/root/CAVER/M3FD/Vis` | RGB 图像目录 |
| `depth_root` | `/root/CAVER/M3FD/Ir` | 红外图像目录 |
| `ann_file` | `/root/CAVER/data/m3fd_detection/val_coco.json` | 验证 COCO 标注 |
| `depth_file_key` | `depth_file_name` | COCO 中红外图像文件名字段 |

---

## 13. 输出目录说明

每次训练都会新建一个 `exp_x` 目录，典型结构如下：

```text
output_chapter3_m3fd_full/
└── SAM2PriorAlignmentYOLODetector_.../
    └── exp_x/
        ├── cfg.py
        ├── log.txt
        ├── lr.png
        ├── results.csv
        ├── trainer.txt
        ├── imgs/
        ├── pth/
        │   └── state.pth
        └── pre/
```

各文件作用：

- `cfg.py`: 运行时保存下来的最终配置
- `log.txt`: 训练和验证日志
- `lr.png`: 当前学习率曲线
- `results.csv`: 每轮验证和最终测试指标
- `imgs/`: 训练阶段可视化图片
- `pth/state.pth`: 当前模型权重
- `pre/`: 测试阶段导出的预测结果

---

## 14. 预训练权重建议

### 14.1 最推荐

```text
timm:resnet50d.ra4_e3600_r224_in1k
```

原因：

- 与当前 `resnet50d` backbone 完全匹配
- 直接走 timm 官方权重接口
- 对你现在这个共享双流 ResNet 结构最自然

### 14.2 更稳妥的备选

```text
timm
```

或

```text
timm:resnet50d.a1_in1k
```

### 14.3 使用注意

- 如果你传了 `--load-from`，那么 `--pretrained` 只在模型初始化时起作用，随后会被 checkpoint 覆盖。
- 所以：
  - 新训练时推荐设置 `PRETRAINED`
  - 续训时主要看 `LOAD_FROM`

---

## 15. 可视化图片说明

当前检测任务会自动输出：

- `rgb`
- `depth`
- `gt_boxes`
- `pred_boxes`
- `student_prior`
- `teacher_prior`（训练态 teacher 开启时）

默认会保存：

- 前 3 个 iteration 的图：`iter-0.png`、`iter-1.png`、`iter-2.png`
- 每个 epoch 末的图：`epoch-x.png`

---

## 16. 日志指标说明

当前日志分三类：

### 16.1 训练日志

示例：

```text
[20/1120 20/112000 0/100] [3, 3, 768, 1024] Lr:... cls:... reg:... distill:... pos:...
```

字段说明：

- `20/1120`: 当前 batch / 当前 epoch 总 batch
- `20/112000`: 当前全局 iter / 总 iter
- `0/100`: 当前 epoch / 总 epoch
- `[3, 3, 768, 1024]`: 当前 RGB batch shape
- `Lr`: 当前学习率
- `M`: 平均 loss
- `C`: 当前 batch loss
- `cls`: 分类损失
- `reg`: 回归损失
- `distill`: 蒸馏损失
- `pos`: 当前 batch 内正样本点数

### 16.2 验证日志

示例：

```text
Val@Epoch1 [M3FD_VAL] mAP50:... mAP50_95:... Precision:... Recall:...
```

### 16.3 最终测试日志

示例：

```text
Test [M3FD_VAL] mAP50:... mAP50_95:... Precision:... Recall:...
```

---

## 17. 重要提醒

### 17.1 关于 `MODE=resume`

当前 `--load-from` 的核心作用是“恢复模型权重”。它适合：

- 从上次权重继续训练
- 迁移到新代码逻辑下继续跑
- 作为 warm start

但它不是严格意义上的“优化器状态级断点续训”。也就是说：

- optimizer 状态会重新初始化
- scaler 状态会重新初始化
- 学习率调度会从新 run 的第 0 步开始

如果后面你想做真正意义上的“从第 N 个 epoch 原地恢复”，我们可以再把 checkpoint 扩展成完整 resume 格式。

### 17.2 关于 teacher 参数

现在新保存的 checkpoint 会自动排除 `teacher_adapter.teacher.*` 这一整套官方 SAM2 参数，原因是：

- teacher 推理期不参与部署
- teacher 可按配置和官方 checkpoint 重新构建
- 这样 checkpoint 更轻，加载更稳

---

## 18. 最常用命令总结

### 新训练

```bash
bash tools/train_chapter3_m3fd.sh
```

### 从最新 checkpoint 继续

```bash
MODE=resume AUTO_RESUME=1 bash tools/train_chapter3_m3fd.sh
```

### 指定 checkpoint 继续

```bash
MODE=resume LOAD_FROM=/abs/path/to/state.pth bash tools/train_chapter3_m3fd.sh
```

### 只评估

```bash
MODE=eval LOAD_FROM=/abs/path/to/state.pth bash tools/train_chapter3_m3fd.sh
```

### 使用不同实验标记

```bash
INFO=ablation1 bash tools/train_chapter3_m3fd.sh
```
