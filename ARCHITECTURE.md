# CAVER Architecture Map

This repository is now organized as a lightweight experiment shell with a task layer on top of the original CAVER model code.

## Top-Level Structure

- `main.py`
  - The single training and evaluation entrypoint.
  - Owns argument parsing, config loading, experiment directory creation, logging, optimizer scheduling, checkpoint saving, and calling the active task.
- `configs/`
  - Experiment configs.
  - `configs/base.py` defines the default task, optimizer, scheduler, and runtime settings.
- `method/`
  - Model registry and model implementations.
  - `method/__init__.py` is the model registration boundary used by `main.py`.
  - Models are still instantiated with `ModuleClass(pretrained=...)`.
- `tasks/`
  - Task layer that isolates dataset construction, batch-to-device transfer, loss computation, visualization, and evaluation.
  - `tasks/base.py` defines the extension interface.
  - `tasks/bimodal_saliency.py` contains the current RGB-D / RGB-T saliency task implementation.
- `datasets.py`
  - Dataset registry for named dataset entries.
  - The current saliency task expects `image`, `depth`, and `mask`.
- `utils/`
  - Generic helpers for optimization, scheduling, logging, plotting, tensor ops, and dataset path utilities.

## Stable Interfaces Kept for Future Models

- Model registration still happens through `method/__init__.py`.
- The runtime still calls models as:

```python
model = getattr(method, cfg.model_name)(pretrained=cfg.pretrained)
outputs = model(data=dict(image=images, depth=depths))
```

- The training entrypoint is still a single script: `python main.py --config ... --model-name ...`.

## What Is Now Task-Specific

The following logic no longer lives in `main.py` and should not be treated as global runtime behavior:

- Training dataset construction
- Test dataset construction
- Loss computation
- Prediction visualization
- Evaluation logic
- Metric names used for CSV export

All of these now belong to the active task implementation in `tasks/`.

## How to Add a New Model

1. Add a new model class under `method/`.
2. Export it from `method/__init__.py`.
3. Keep the `forward(self, data)` style unless you are intentionally changing the runtime contract.
4. Point `--model-name` to the exported symbol.

If the new model is still a dual-modal dense prediction model, it can reuse the existing `bimodal_saliency` task unchanged as long as it consumes `image` and `depth` and returns the saliency logits expected by that task.

## How to Add a New Task

1. Create a new task class under `tasks/` by subclassing `BaseTask`.
2. Implement:
   - `build_train_dataset`
   - `build_test_dataset`
   - `get_model_inputs`
   - `compute_loss`
   - `evaluate_once`
3. Register the task in `tasks/__init__.py`.
4. Set `task = dict(name="...")` in the config.

This is the intended extension point for future dual-modal detection work.

## Guidance for Future Dual-Modal Detection Work

The current saliency task is useful as a code skeleton, but it is not a detection framework yet.

The first interfaces you will likely need to redefine are:

- Dataset outputs: from `{image, depth, mask}` to `{image, depth, targets}`
- Model outputs: from a single saliency logit map to detection head outputs
- Loss: from segmentation losses to detection target assignment and multi-term losses
- Evaluation: from SOD metrics to detection metrics and post-processing

The safest runtime contract to keep is the outer `model(data=...)` call, because it keeps the trainer decoupled from model internals.
