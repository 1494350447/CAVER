from importlib import import_module

from .base import BaseTask


TASKS = {
    "bimodal_saliency": ("tasks.bimodal_saliency", "BimodalSaliencyTask"),
    "bimodal_detection": ("tasks.bimodal_detection", "BimodalDetectionTask"),
}


def build_task(cfg) -> BaseTask:
    task_cfg = cfg.get("task", None)
    task_name = "bimodal_saliency"
    if task_cfg is not None:
        task_name = task_cfg.get("name", task_name)

    if task_name not in TASKS:
        available_tasks = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task <{task_name}>. Available tasks: {available_tasks}")

    module_name, class_name = TASKS[task_name]
    module = import_module(module_name)
    task_class = getattr(module, class_name)
    return task_class(cfg=cfg)
