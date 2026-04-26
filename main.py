import argparse
import csv
import inspect
import os
import shutil
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
from mmengine import Config

import models as models_lib
import tasks as task_lib
from utils import constructor, pt_utils, py_utils
from utils.recorder import AvgMeter, MsgLogger, TimeRecoder


def get_checkpoint_state_dict(model):
    state_dict = model.state_dict()
    return {key: value for key, value in state_dict.items() if not key.startswith("teacher_adapter.teacher.")}


def get_primary_metric_name(task):
    return getattr(task, "primary_metric_name", task.metric_names[0])


def is_better_result(candidate, best, primary_metric_name):
    if best is None:
        return True
    candidate_primary = float(candidate.get(primary_metric_name, float("-inf")))
    best_primary = float(best.get(primary_metric_name, float("-inf")))
    if candidate_primary != best_primary:
        return candidate_primary > best_primary
    candidate_map50 = float(candidate.get("mAP50", float("-inf")))
    best_map50 = float(best.get("mAP50", float("-inf")))
    if candidate_map50 != best_map50:
        return candidate_map50 > best_map50
    return float(candidate.get("Precision", float("-inf"))) > float(best.get("Precision", float("-inf")))


def freeze_backbone_bn(model):
    actual_model = model.module if hasattr(model, "module") else model
    backbone = getattr(actual_model, "backbone", None)
    if backbone is None:
        return
    for module in backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def save_model_state(model, save_path, ema=None):
    if ema is not None:
        ema.apply_to(model)
    try:
        torch.save(get_checkpoint_state_dict(model), save_path)
    finally:
        if ema is not None:
            ema.restore(model)


def testing(
    model,
    task,
    msg_logger,
    cfg,
    save_predictions=True,
    row_name=None,
    log_prefix="Test",
    calibrate_inference=False,
):
    msg_logger(name="log", msg="\n", show=False)

    if row_name is None:
        row_name = cfg.exp_name
    csv_row = [row_name]
    aggregate_results = {}
    for dataset_name in cfg.data.test.name:
        test_dataset, dataset_info = task.build_test_dataset(dataset_name=dataset_name, cfg=cfg)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=cfg.args.batch_size,
            num_workers=cfg.args.num_workers,
            pin_memory=True,
            collate_fn=task.get_collate_fn(split="test"),
        )
        print(f"{log_prefix} on {dataset_name} with {len(test_dataset)} samples")
        pred_save_path = os.path.join(cfg.path.save, dataset_name) if save_predictions else ""
        test_results = task.evaluate_once(
            model=model,
            save_path=pred_save_path,
            data_loader=test_loader,
            show_bar=cfg.show_bar,
            calibrate_inference=calibrate_inference,
        )
        metric_msg = " ".join(f"{name}:{test_results[name]}" for name in task.metric_names)
        msg_logger(name="log", msg=f"{log_prefix} [{dataset_name}] {metric_msg}")
        for detail_line in task.format_eval_details(test_results):
            msg_logger(name="log", msg=f"{log_prefix} [{dataset_name}] {detail_line}")
        msg_logger(name="log", msg=f"Results on {dataset_info}:\n{test_results}", show=False)

        aggregate_results[dataset_name] = test_results
        csv_row.extend([test_results[name] for name in task.metric_names])

    with open(cfg.path.csv, encoding="utf-8", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_row)
    return aggregate_results


def training(model, task, msg_logger, cfg):
    train_dataset = task.build_train_dataset(cfg=cfg)
    train_sampler = task.get_train_sampler(train_dataset=train_dataset, cfg=cfg)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.args.batch_size,
        num_workers=cfg.args.num_workers,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        collate_fn=task.get_collate_fn(split="train"),
        worker_init_fn=None
        if cfg.args.base_seed < 0
        else partial(pt_utils.worker_init_fn, base_seed=cfg.args.base_seed),
    )
    print(f"Training on {tuple(cfg.data.train.name)} with {len(train_dataset)} samples")
    if train_sampler is not None:
        print(f"train_sampler: {type(train_sampler).__name__}")

    num_iter_per_epoch = len(train_loader)
    num_iter = cfg.args.epoch_num * num_iter_per_epoch

    optimizer = constructor.make_optim_with_cfg(model=model, optimizer_cfg=cfg.optimizers)
    print(f"optimizer:\n{optimizer}")
    lr_adjustor = constructor.LRAdjustor(
        initial_lr_groups=[group["lr"] for group in optimizer.param_groups],
        total_num=num_iter if cfg.schedulers.sche_usebatch else cfg.args.epoch_num,
        num_iters_per_epoch=num_iter_per_epoch,
        scheduler_cfg=cfg.schedulers,
    )
    num_iter_in_cooldown = cfg.cooldown_epoch_num * num_iter_per_epoch
    py_utils.plot_lr_curve_for_scheduler(
        optimizer=optimizer,
        scheduler=lr_adjustor,
        num_steps=num_iter + num_iter_in_cooldown,
        save_path=os.path.join(cfg.path.exp, "lr.png"),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.args.use_amp)
    val_freq = int(cfg.args.get("val_freq", 0))
    ema_cfg = cfg.get("ema", {})
    ema = None
    if ema_cfg.get("enable", False):
        ema = pt_utils.ModelEMA(model=model, decay=ema_cfg.get("decay", 0.9998))
    freeze_backbone_bn_flag = bool(cfg.get("freeze_backbone_bn", False))
    primary_metric_name = get_primary_metric_name(task)
    best_results = None
    best_epoch = -1

    loss_recorder = AvgMeter()
    time_logger = TimeRecoder()
    for epoch_idx in range(cfg.args.epoch_num + cfg.cooldown_epoch_num):
        time_logger.start(msg=cfg.exp_name)
        loss_recorder.reset()
        model.train()
        if freeze_backbone_bn_flag:
            freeze_backbone_bn(model)

        for batch_idx, batch in enumerate(train_loader):
            curr_iter = epoch_idx * num_iter_per_epoch + batch_idx
            lr_adjustor(optimizer=optimizer, curr_idx=curr_iter)

            device_batch = task.move_batch_to_device(batch)
            with torch.cuda.amp.autocast(enabled=cfg.args.use_amp):
                model_outputs = model(data=task.get_model_inputs(device_batch))

            losses, losses_str = task.compute_loss(model_outputs=model_outputs, batch=device_batch)
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

            item_loss = losses.item()
            batch_size = batch["image"].size(0)
            loss_recorder.update(value=item_loss, num=batch_size)

            fixed_step = batch_idx == 0 or (batch_idx + 1) == num_iter_per_epoch
            interval_step = cfg.args.print_freq > 0 and (
                curr_iter % cfg.args.print_freq == 0 or curr_iter == num_iter - 1
            )
            if fixed_step or interval_step:
                lr_string = ",".join([f"{x:10.3e}" for x in [group["lr"] for group in optimizer.param_groups]])
                msg = (
                    f"[{batch_idx}/{num_iter_per_epoch} {curr_iter}/{num_iter + num_iter_in_cooldown} "
                    f"{epoch_idx}/{cfg.args.epoch_num + cfg.cooldown_epoch_num}] "
                    f"{list(batch['image'].shape)} Lr:{lr_string} M:{loss_recorder.avg:.5f}/C:{item_loss:.5f} "
                    f"{losses_str}"
                )
                msg_logger(name="log", msg=msg, show=True)

            if curr_iter < 3:
                vis_data = task.get_visualization_data(model_outputs=model_outputs, batch=batch)
                if vis_data:
                    py_utils.cvplot_results(
                        vis_data,
                        save_path=os.path.join(cfg.vis_path, f"iter-{curr_iter}.png"),
                    )

        vis_data = task.get_visualization_data(model_outputs=model_outputs, batch=batch)
        if vis_data:
            py_utils.cvplot_results(
                vis_data,
                save_path=os.path.join(cfg.vis_path, f"epoch-{epoch_idx}.png"),
            )
        save_model_state(model=model, save_path=cfg.path.state, ema=ema)
        time_logger.now(pre_msg="An Epoch End...")

        if val_freq > 0 and (epoch_idx + 1) % val_freq == 0:
            if ema is not None:
                ema.apply_to(model)
            try:
                val_results = testing(
                    model=model,
                    task=task,
                    msg_logger=msg_logger,
                    cfg=cfg,
                    save_predictions=False,
                    row_name=f"{cfg.exp_name}_epoch{epoch_idx + 1:03d}",
                    log_prefix=f"Val@Epoch{epoch_idx + 1}",
                    calibrate_inference=(epoch_idx + 1) >= int(cfg.get("val_sweep_start_epoch", 5)),
                )
            finally:
                if ema is not None:
                    ema.restore(model)

            dataset_name = cfg.data.test.name[0]
            curr_results = val_results[dataset_name]
            if is_better_result(curr_results, best_results, primary_metric_name=primary_metric_name):
                best_results = dict(curr_results)
                best_epoch = epoch_idx + 1
                save_model_state(model=model, save_path=cfg.path.best, ema=ema)
                best_msg = (
                    f"NewBest@Epoch{best_epoch} "
                    + " ".join(f"{name}:{best_results[name]}" for name in task.metric_names)
                )
                msg_logger(name="log", msg=best_msg, show=True)

    return dict(best_results=best_results, best_epoch=best_epoch)


def initialize_result_csv(cfg, metric_names):
    with open(cfg.path.csv, encoding="utf-8", mode="w", newline="") as f:
        writer = csv.writer(f)

        first_row = ["model_name"]
        for dataset_name in cfg.data.test.name:
            first_row.extend([dataset_name] + [" "] * (len(metric_names) - 1))
        writer.writerow(first_row)

        second_row = [" "] + list(metric_names) * len(cfg.data.test.name)
        writer.writerow(second_row)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="output")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--pretrained", type=str, help="Pretrained params of the backbone of your model.")
    parser.add_argument("--info", type=str)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--show-bar", action="store_true")
    parser.add_argument("--cooldown-epoch-num", type=int, default=0)
    parser.add_argument("--base-seed", type=int)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config, use_predefined_variables=False)
    cfg.output_root = args.output_root
    cfg.model_name = args.model_name
    cfg.load_from = args.load_from if args.load_from is not None else cfg.get("load_from", None)
    cfg.pretrained = args.pretrained if args.pretrained is not None else cfg.get("pretrained", None)
    cfg.info = args.info if args.info is not None else cfg.get("info", None)
    cfg.evaluate = args.evaluate
    cfg.show_bar = args.show_bar
    cfg.cooldown_epoch_num = args.cooldown_epoch_num
    if args.base_seed is not None:
        cfg.args.base_seed = int(args.base_seed)

    cfg.exp_name = py_utils.construct_exp_name(config=cfg)
    if cfg.cooldown_epoch_num > 0:
        cfg.exp_name += f"_CD{cfg.cooldown_epoch_num}"

    cfg.path = py_utils.construct_path(output_root=cfg.output_root, exp_name=cfg.exp_name)
    cfg.vis_path = os.path.join(cfg.path.exp, "imgs")

    os.makedirs(cfg.path.exp, exist_ok=True)
    os.makedirs(cfg.path.save, exist_ok=True)
    os.makedirs(cfg.path.pth, exist_ok=True)

    with open(cfg.path.log, encoding="utf-8", mode="w") as f:
        f.write(f"=== {datetime.now()} ===\n")
    with open(cfg.path.cfg, encoding="utf-8", mode="w") as f:
        f.write(cfg.pretty_text)
    shutil.copy(__file__, cfg.path.trainer)

    if os.path.exists(cfg.vis_path):
        shutil.rmtree(cfg.vis_path)
    os.makedirs(cfg.vis_path)
    return cfg


def main():
    cfg = parse_config()
    pt_utils.initialize_seed_cudnn(seed=cfg.args.base_seed, deterministic=cfg.args.deterministic)
    print(f"[{datetime.now()}] {cfg.path.exp} with base_seed {cfg.args.base_seed}")

    msg_logger = MsgLogger(log=cfg.path.log)
    task = task_lib.build_task(cfg)
    initialize_result_csv(cfg=cfg, metric_names=task.metric_names)
    msg_logger(name="log", msg=f"task: {task.name}")

    model_kwargs = dict(cfg.get("model", {}))
    model_kwargs.pop("name", None)
    model_kwargs.pop("pretrained", None)

    if hasattr(models_lib, cfg.model_name):
        module_class = getattr(models_lib, cfg.model_name)
    else:
        import method as model_lib

        if hasattr(model_lib, cfg.model_name):
            module_class = getattr(model_lib, cfg.model_name)
        else:
            raise ModuleNotFoundError(f"Please add <{cfg.model_name}> into models/__init__.py or method/__init__.py.")

    model = module_class(pretrained=cfg.pretrained, **model_kwargs)
    msg_logger(name="log", msg=inspect.getsource(module_class))

    if cfg.load_from:
        checkpoint = torch.load(cfg.load_from, map_location="cpu")
        incompatible = model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded from {cfg.load_from}")
        if incompatible.missing_keys:
            print(f"Missing keys when loading checkpoint: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys when loading checkpoint: {len(incompatible.unexpected_keys)}")

    model.cuda()
    if cfg.get("freeze_backbone_bn", False):
        freeze_backbone_bn(model)
    if not cfg.evaluate:
        train_summary = training(model=model, task=task, msg_logger=msg_logger, cfg=cfg)
        if os.path.isfile(cfg.path.best):
            best_checkpoint = torch.load(cfg.path.best, map_location="cpu")
            incompatible = model.load_state_dict(best_checkpoint, strict=False)
            print(f"Loaded best checkpoint from {cfg.path.best}")
            if incompatible.missing_keys:
                print(f"Missing keys when loading best checkpoint: {len(incompatible.missing_keys)}")
            if incompatible.unexpected_keys:
                print(f"Unexpected keys when loading best checkpoint: {len(incompatible.unexpected_keys)}")
        if train_summary["best_results"] is not None:
            best_msg = (
                f"BestSummary@Epoch{train_summary['best_epoch']} "
                + " ".join(f"{name}:{train_summary['best_results'][name]}" for name in task.metric_names)
            )
            msg_logger(name="log", msg=best_msg, show=True)

    testing(
        model=model,
        task=task,
        msg_logger=msg_logger,
        cfg=cfg,
        row_name=f"{cfg.exp_name}_best" if os.path.isfile(cfg.path.best) else cfg.exp_name,
        calibrate_inference=not cfg.evaluate,
    )
    print(f"{datetime.now()}: End training...")


if __name__ == "__main__":
    main()
