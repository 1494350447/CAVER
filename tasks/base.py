from abc import ABC, abstractmethod

import torch


class BaseTask(ABC):
    name = ""
    metric_names = ()

    def __init__(self, cfg):
        self.cfg = cfg

    def get_collate_fn(self, split):
        return None

    def get_train_sampler(self, train_dataset, cfg=None):
        return None

    def move_to_device(self, data):
        if torch.is_tensor(data):
            return data.cuda(non_blocking=True)
        if isinstance(data, dict):
            return {key: self.move_to_device(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self.move_to_device(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self.move_to_device(item) for item in data)
        return data

    def move_batch_to_device(self, batch):
        return self.move_to_device(batch)

    @abstractmethod
    def build_train_dataset(self, cfg):
        raise NotImplementedError

    @abstractmethod
    def build_test_dataset(self, dataset_name, cfg):
        raise NotImplementedError

    @abstractmethod
    def get_model_inputs(self, batch):
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, model_outputs, batch):
        raise NotImplementedError

    def get_visualization_data(self, model_outputs, batch):
        return None

    def format_eval_details(self, eval_results):
        return []

    @abstractmethod
    def evaluate_once(self, model, data_loader, save_path="", show_bar=True, calibrate_inference=False):
        raise NotImplementedError
