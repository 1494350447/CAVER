from importlib import import_module


MODEL_REGISTRY = {
    "SAM2PriorAlignmentYOLODetector": ("models.detectors.sam2_prior_alignment_detector", "SAM2PriorAlignmentYOLODetector"),
}


def __getattr__(name):
    if name not in MODEL_REGISTRY:
        raise AttributeError(f"module 'models' has no attribute {name}")
    module_name, class_name = MODEL_REGISTRY[name]
    module = import_module(module_name)
    return getattr(module, class_name)


__all__ = list(MODEL_REGISTRY.keys())
