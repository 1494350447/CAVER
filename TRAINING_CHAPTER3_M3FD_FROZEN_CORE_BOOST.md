# 第三章冻结核心代码的 M3FD 外部增强方案说明

这份文档对应当前仓库里的“核心模块不动，只通过数据、训练包装和推理包装提分”的方案。

## 1. 冻结边界

本方案默认不修改以下区域：

- `models/modules/`
- `models/heads/`
- `models/detectors/`

也就是说，第三章里导师最容易检查的核心模块实现保持不变。当前提分手段都放在以下区域：

- `configs/`
- `tasks/`
- `tools/`
- `main.py` 的少量外层入口增强
- `data/` 下的数据 manifest

## 2. 当前主配置

主配置文件：

```bash
configs/chapter3_m3fd_frozen_core_boost.py
```

这份配置的核心思路是：

- 主数据集仍然使用 `M3FD`
- 训练时混合 `full image + tile image`
- 采样比例固定为 `1:1`
- 保留 `teacher-on`
- 保留 `FP32`
- 保留旧主线更稳定的 `legacy assigner`
- 开启 `EMA`

## 3. 数据增强方式

### 3.1 切片数据

切片脚本：

```bash
python tools/prepare_m3fd_tiles.py
```

默认会生成：

- `data/m3fd_detection/train_tile_coco.json`
- `data/m3fd_detection/val_tile_coco.json`
- `data/m3fd_detection/tile_manifest.json`

当前切片规则：

- 原图大小按 M3FD 原始标注读取
- 标准窗口大小：`768 x 576`
- 标准步长：`576 x 432`
- 裁剪后框保留条件：
  - 保留面积比例 `>= 0.7`
  - 裁剪后最短边 `>= 6 px`

训练集会丢弃空切片，验证集保留空切片，便于完整回映射。

### 3.2 训练采样

`tasks/bimodal_detection.py` 新增了 `BalancedConcatSampler`，用于保证：

- `full dataset`
- `tile dataset`

在每个 epoch 内按固定比例采样，默认是 `1:1`。

## 4. 一键脚本

统一脚本：

```bash
tools/train_chapter3_m3fd_frozen_core_boost.sh
```

### 4.1 只准备切片数据

```bash
MODE=prepare bash tools/train_chapter3_m3fd_frozen_core_boost.sh
```

### 4.2 跑单个 seed

```bash
MODE=train-one SEED=42 bash tools/train_chapter3_m3fd_frozen_core_boost.sh
```

### 4.3 跑 3 个 seed

```bash
MODE=train-all SEEDS="42 3407 2026" bash tools/train_chapter3_m3fd_frozen_core_boost.sh
```

### 4.4 跑增强推理

```bash
MODE=boosted-eval \
CHECKPOINTS="/abs/path/to/best_state.pth" \
bash tools/train_chapter3_m3fd_frozen_core_boost.sh
```

## 5. 增强推理说明

增强推理脚本：

```bash
tools/evaluate_m3fd_boosted.py
```

当前增强推理包含三层操作：

- full-image prediction
- horizontal flip TTA
- tile prediction 回映射到原图坐标

最后统一做：

- 按类阈值校准
- `WBF` 融合

### 5.1 当前为了可运行做的加速策略

增强推理现在不是“全验证集直接暴力搜索”，而是两段式：

1. 先在一个固定校准子集上搜索 per-class threshold
2. 再把找到的阈值应用到整个验证集，生成最终 boosted 结果

同时加入了两层候选框瘦身：

- `source_topk=150`
- `per_class_topk=120`

这样做不改变模型结构，只是让 `WBF` 和阈值搜索能在可接受时间内跑完。

## 6. 结果输出

增强推理完成后，会在输出目录写出：

- `boosted_eval_summary.json`
- `boosted_predictions.json`

其中 `boosted_eval_summary.json` 里会同时保留：

- `raw_results`
- `best_raw`
- `boosted_calibration_summary`
- `boosted_summary`
- `boosted_per_class_thresholds`
- `boosted_wbf_iou`

这样你可以直接比较“裸推理”与“增强推理”的差异。

## 7. 当前推荐流程

如果你接下来要正式做这条路线，建议按下面顺序：

1. `MODE=prepare` 先生成切片 COCO
2. `MODE=train-one` 先跑一个 seed，看单模型能否追平旧主线
3. 如果单模型稳定，再跑 `MODE=train-all`
4. 最后用 `MODE=boosted-eval` 对多个 best checkpoint 做融合

## 8. 当前已经验证通过的点

以下链路已经做过 smoke test：

- `prepare_m3fd_tiles.py` 能正常生成 train/val tile COCO
- `BalancedConcatSampler` 能实现 full/tile 1:1 采样
- 混合 full+tile 的一个真实 batch 能正常前向并汇总 loss
- `evaluate_m3fd_boosted.py` 能在旧的 M3FD checkpoint 上完成小规模增强评估并输出 JSON
- 外层 `train_chapter3_m3fd_frozen_core_boost.sh` 能正确透传 `LIMIT_IMAGES` 和 `CALIBRATION_NUM_IMAGES`

## 9. 建议口径

论文或汇报时，建议把结果分成两类：

- `single best model`
- `boosted inference`

这样能清楚说明：

- 核心第三章模型本体效果
- 不改核心代码时，通过工程化外部增强还能拿到多少增益
