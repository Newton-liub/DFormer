"""Optional, rank-zero SwanLab logging; credentials use SwanLab's login store."""


def start_tracking(config, args, engine):
    if not config.get("swanlab_enabled", False):
        return None
    if engine.distributed:
        import torch.distributed as distributed

        if distributed.get_rank() != 0:
            return None
    import swanlab

    core_keys = (
        "dataset_name", "backbone", "pretrained_model", "decoder", "num_classes",
        "seed", "optimizer", "lr", "lr_power", "weight_decay", "batch_size",
        "nepochs", "niters_per_epoch", "warm_up_epoch", "train_scale_array",
        "image_height", "image_width", "eval_scale_array", "eval_flip",
        "train_source", "eval_source", "gt_transform", "x_is_single_channel",
    )
    core = {key: config[key] for key in core_keys if key in config}
    core["cli"] = vars(args).copy()
    return swanlab.init(
        project=config.get("swanlab_project", "dformer-research"),
        experiment_name=config.get("experiment_name", config.backbone),
        config=core,
        mode=config.get("swanlab_mode", "online"),
        logdir=config.get("swanlab_log_dir", config.log_dir + "/swanlab"),
    )


def log_epoch(run, epoch, updates, loss, learning_rate, miou=None):
    if run is None:
        return
    values = {
        "train/loss": loss.item() if hasattr(loss, "item") else float(loss),
        "train/learning_rate": float(learning_rate),
        "epoch": epoch,
        "update": updates,
    }
    if miou is not None:
        values["val/mIoU"] = float(miou)
    run.log(values, step=updates)
