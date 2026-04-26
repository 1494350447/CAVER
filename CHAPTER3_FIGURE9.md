# 第三章图 3.9 出图说明（M3FD-only 模块证据图）

这次图 3.9 的定位已经改成“`M3FD` 数据集上的模块证据图”，不再强依赖 `w/o SPD-DRE / w/o F-DDSA / w/o HFSD` 三条真实消融 checkpoint。

也就是说，这张图的职责是：

- 给表 3.7、表 3.8 提供定性支撑；
- 把第三章三段式逻辑可视化出来：
  - `SPD-DRE`：提取对齐
  - `F-DDSA`：融合对齐
  - `HFSD`：任务对齐

而不是再去伪造一张“看起来像严格消融，但实际上没有真实对比 checkpoint”的图。

## 这次工具链支持什么

- `tools/generate_chapter3_figure9.py`
  - `render`：根据 JSON spec 直接渲染最终论文图。
  - `mine`：先从 `M3FD_VAL` 自动筛候选样例，再人工挑 1 张放论文。
- `assets/chapter3_figure9/spec_template.json`
  - 已改成 `M3FD-only` 的证据图模板。
  - 默认 `figure_mode = "evidence"`。
  - 已写入当前建议使用的 full model 路径：
    - config: `output_chapter3_m3fd_recover/.../exp_1/cfg.py`
    - checkpoint: `output_chapter3_m3fd_recover/.../exp_1/pth/best_state.pth`

## 图 3.9 的版式

仍然是 `3 行 × 7 列`，但第 6 列不再是伪造的 `w/o 当前模块`，而是：

1. `RGB`
2. `IR`
3. `Teacher / Student Prior`
4. 模块关键中间响应 A
5. 模块关键中间响应 B
6. `GT + Zoom`
7. `Full Model Prediction`

三行分别对应：

- `SPD-DRE`
- `F-DDSA`
- `HFSD`

## 三行分别要证明什么

- `SPD-DRE`
  - 选 `M3FD` 中弱小目标、低对比、易被背景淹没的样例。
  - 优先看 `People / Motorcycle / Lamp`。
  - 目标是证明：先验引导后，空间与频率解耦响应更聚焦目标区域。
- `F-DDSA`
  - 选 `RGB` 与 `IR` 响应形态不一致、容易错位或重影的样例。
  - 优先看车辆、路灯、热反射共存场景。
  - 目标是证明：`focus_map` 更聚焦主体，`divergence_map` 更覆盖补偿区域。
- `HFSD`
  - 选边界复杂、密集、类别容易混淆的样例。
  - 优先看 `Car / Bus / Truck / People` 混合场景。
  - 目标是证明：`cls_features` 更强调主体语义，`reg_features` 更强调边缘和轮廓。

## 中间量对应关系

本工具继续直接复用当前模型已有输出，不新增模型接口：

- `SPD-DRE`
  - `aux.rgb_spd[0]['spatial']`
  - `aux.rgb_spd[0]['frequency']`
  - `aux.ir_spd[0]['spatial']`
  - `aux.ir_spd[0]['frequency']`
  - 最终做双模态平均，显示“空间解耦响应”和“频率解耦响应”。
- `F-DDSA`
  - `aux.neck['pan'][0]['focus_map']`
  - `aux.neck['pan'][0]['divergence_map']`
- `HFSD`
  - `cls_features[0]`
  - `reg_features[0]`
- `Prior`
  - `student_prior`
  - `teacher_prior`

注意：

- 模型在 `eval()` 下默认不返回 `teacher_prior`。
- 当前脚本已经兼容这件事，会额外显式调用 `teacher_adapter` 把 `teacher_prior` 取出来。

## 如何自动筛候选样例

现在 `mine` 已经支持“只依赖 full model”工作，不再强制要求 compare checkpoint。

### 1. 筛 `SPD-DRE` 候选

```bash
python /root/CAVER/tools/generate_chapter3_figure9.py mine \
  --module spd_dre \
  --full-config /root/CAVER/output_chapter3_m3fd_recover/SAM2PriorAlignmentYOLODetector_768x1024_BS3_E30_AMPn_LR6e-05_OTall_OPadamw_LTcos_INFOm3fd_recover_resume/exp_1/cfg.py \
  --full-checkpoint /root/CAVER/output_chapter3_m3fd_recover/SAM2PriorAlignmentYOLODetector_768x1024_BS3_E30_AMPn_LR6e-05_OTall_OPadamw_LTcos_INFOm3fd_recover_resume/exp_1/pth/best_state.pth \
  --dataset-name M3FD_VAL \
  --split test \
  --output /root/CAVER/assets/chapter3_figure9/spd_dre_candidates.json \
  --device cuda \
  --limit 840 \
  --topk 15 \
  --score-thr 0.03 \
  --nms-iou-thr 0.5
```

### 2. 筛 `F-DDSA` 候选

```bash
python /root/CAVER/tools/generate_chapter3_figure9.py mine \
  --module fddsa \
  --full-config /root/CAVER/output_chapter3_m3fd_recover/SAM2PriorAlignmentYOLODetector_768x1024_BS3_E30_AMPn_LR6e-05_OTall_OPadamw_LTcos_INFOm3fd_recover_resume/exp_1/cfg.py \
  --full-checkpoint /root/CAVER/output_chapter3_m3fd_recover/SAM2PriorAlignmentYOLODetector_768x1024_BS3_E30_AMPn_LR6e-05_OTall_OPadamw_LTcos_INFOm3fd_recover_resume/exp_1/pth/best_state.pth \
  --dataset-name M3FD_VAL \
  --split test \
  --output /root/CAVER/assets/chapter3_figure9/fddsa_candidates.json \
  --device cuda \
  --limit 840 \
  --topk 15 \
  --score-thr 0.03 \
  --nms-iou-thr 0.5
```

### 3. 筛 `HFSD` 候选

```bash
python /root/CAVER/tools/generate_chapter3_figure9.py mine \
  --module hfsd \
  --full-config /root/CAVER/output_chapter3_m3fd_recover/SAM2PriorAlignmentYOLODetector_768x1024_BS3_E30_AMPn_LR6e-05_OTall_OPadamw_LTcos_INFOm3fd_recover_resume/exp_1/cfg.py \
  --full-checkpoint /root/CAVER/output_chapter3_m3fd_recover/SAM2PriorAlignmentYOLODetector_768x1024_BS3_E30_AMPn_LR6e-05_OTall_OPadamw_LTcos_INFOm3fd_recover_resume/exp_1/pth/best_state.pth \
  --dataset-name M3FD_VAL \
  --split test \
  --output /root/CAVER/assets/chapter3_figure9/hfsd_candidates.json \
  --device cuda \
  --limit 840 \
  --topk 15 \
  --score-thr 0.03 \
  --nms-iou-thr 0.5
```

输出 JSON 里会包含：

- `image_id`
- `file_name`
- `score`
- `prediction`
- `evidence`

这里的 `score` 不是 AP，而是“这个样例是否适合拿来说明当前模块优势”的综合排序分数。

## 当前 evidence 模式下的筛选逻辑

- `SPD-DRE`
  - 更看重 `teacher_prior` 与 `student_prior` 的一致性；
  - 更看重弱目标区域里 `spatial / frequency` 的响应对比度；
  - 同时兼顾当前图像上的检测质量。
- `F-DDSA`
  - 更看重 `focus_map` 在目标主体上的聚焦程度；
  - 更看重 `divergence_map` 在目标外环补偿区域上的响应；
  - 同时兼顾当前图像上的检测质量。
- `HFSD`
  - 更看重 `cls_features` 的主体关注度；
  - 更看重 `reg_features` 在边缘 / 外环区域上的响应；
  - 同时兼顾当前图像上的检测质量。

所以这一步的目的不是“替代人工判断”，而是把 840 张验证图先缩小到每个模块 10 到 15 张可读性更强的候选。

## 如何渲染最终图

先复制模板：

```bash
cp /root/CAVER/assets/chapter3_figure9/spec_template.json /root/CAVER/assets/chapter3_figure9/spec_final.json
```

然后只改这几个字段：

- `rows[0/1/2].sample.image_id`
- `rows[0/1/2].zoom_box`
- 如有需要，再微调 `row_title`

渲染命令：

```bash
python /root/CAVER/tools/generate_chapter3_figure9.py render \
  --spec /root/CAVER/assets/chapter3_figure9/spec_final.json \
  --output /root/CAVER/assets/chapter3_figure9/figure3_9.png \
  --device cuda
```

如果你想放进论文时更清楚，可以拉大单元尺寸：

```bash
python /root/CAVER/tools/generate_chapter3_figure9.py render \
  --spec /root/CAVER/assets/chapter3_figure9/spec_final.json \
  --output /root/CAVER/assets/chapter3_figure9/figure3_9_large.png \
  --device cuda \
  --cell-width 460 \
  --cell-height 300 \
  --row-label-width 170
```

## evidence 模式和旧 compare 模式的关系

当前默认是 `figure_mode = "evidence"`：

- 没有 `compare_run` 也能完整出图；
- 第 6 列固定渲染 `GT + Zoom`；
- 第 7 列固定渲染 `Full Model Prediction`。

如果你以后真的补跑了：

- `w/o SPD-DRE`
- `w/o F-DDSA`
- `w/o HFSD`

那么这个工具仍然兼容旧 compare 模式。只要在 spec 里补上 `compare_run`，并把 `figure_mode` 改成 `comparison`，就可以继续生成严格消融对比图。

## 我建议的最终使用顺序

1. 先分别跑 3 次 `mine`，拿到三个模块各自的候选 JSON。
2. 人工各挑 1 张最能说明问题的图：
   - `SPD-DRE` 看弱目标和 prior 一致性；
   - `F-DDSA` 看 focus / divergence 是否明显；
   - `HFSD` 看 cls / reg 是否有清晰分工。
3. 把 `image_id` 和 `zoom_box` 填进 `spec_final.json`。
4. 运行 `render` 输出最终 `figure3_9.png`。
5. 放在表 3.7、表 3.8 后面，正文明确写它是“模块证据图”，不是“严格消融预测对比图”。

## 建议的正文表述

图题建议：

- `图3.9 M3FD数据集上第三章核心模块的可解释性与定性证据图`

正文建议围绕三句话写：

- `SPD-DRE` 说明先验引导的空频解耦能够缓解弱目标被模态竞争淹没的问题。
- `F-DDSA` 说明聚焦-发散动态对齐能够抑制双模态空间错位带来的响应漂移和虚警。
- `HFSD` 说明频域任务分工使分类更关注主体语义，而回归更关注边缘与几何轮廓。

这样它和表格的关系就很清楚：

- 表格负责定量结论；
- 图 3.9 负责解释“为什么会提升”。
