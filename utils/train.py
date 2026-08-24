import argparse
import datetime
import os
import pprint
import random
import time
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from val_mm import evaluate, evaluate_msf

from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.experiment_tracker import ExperimentTracker, build_run_name, gpu_safety_violation
from utils.init_func import configure_optimizers, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.pyt_utils import all_reduce_tensor

# from eval import evaluate_mid


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", default=2, type=int, help="used gpu number")
# parser.add_argument('-d', '--devices', default='0,1', type=str)
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", type=int, default=None, help="override config.nepochs")
parser.add_argument("--batch-size", "--batch", dest="batch_size", type=int, default=None, help="override config.batch_size")
parser.add_argument("--workers", "--worker", "--num-workers", dest="workers", type=int, default=None, help="override config.num_workers")
parser.add_argument("--val-batch-size", "--val-batch", dest="val_batch_size", type=int, default=None, help="override validation batch size")
parser.add_argument("--max-train-iters", type=int, default=None, help="stop normally after this many train steps")
parser.add_argument("--log-interval", type=int, default=0, help="train metric interval in steps; 0 uses 10% of an epoch")
parser.add_argument("--min-free-vram-gib", type=float, default=0.0, help="fail when free VRAM falls below GiB; 0 disables")
parser.add_argument("--min-free-vram-ratio", type=float, default=0.0, help="fail when free/total VRAM falls below ratio; 0 disables")
parser.add_argument("--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled")
parser.add_argument("--swanlab-project", default="DFormer-liu")
parser.add_argument("--swanlab-workspace", default="Newton_liub")
parser.add_argument("--swanlab-run-name", default=None)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir")
parser.add_argument("--continue_fpath")
parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile_mode", default="default")
parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--mst", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--val_amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--pad_SUNRGBD", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--use_seed", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--local-rank", default=0)
# parser.add_argument('--save_path', '-p', default=None)

# os.environ['MASTER_PORT'] = '169710'
torch.set_float32_matmul_precision("high")
import torch._dynamo

torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.automatic_dynamic_shapes = False


def is_eval(epoch, config):
    return epoch > int(config.checkpoint_start_epoch) or epoch == 1 or epoch % 10 == 0


class gpu_timer:
    def __init__(self, beta=0.6) -> None:
        self.start_time = None
        self.stop_time = None
        self.mean_time = None
        self.beta = beta
        self.first_call = True

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            print("Use start() before stop(). ")
        torch.cuda.synchronize()
        self.stop_time = time.perf_counter()
        elapsed = self.stop_time - self.start_time
        self.start_time = None
        if self.first_call:
            self.mean_time = elapsed
            self.first_call = False
        else:
            self.mean_time = self.beta * self.mean_time + (1 - self.beta) * elapsed
        return elapsed


def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True  # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True, warn_only=True)


with Engine(custom_parser=parser) as engine, ExperimentTracker() as tracker:
    args = parser.parse_args()

    config = getattr(import_module(args.config), "C")
    if args.epochs is not None:
        if args.epochs <= 0:
            parser.error("--epochs must be positive")
        original_nepochs = config.nepochs
        terminal_eval_follows_nepochs = getattr(config, "checkpoint_start_epoch", None) == original_nepochs
        config.nepochs = args.epochs
        if terminal_eval_follows_nepochs:
            config.checkpoint_start_epoch = config.nepochs
    if args.batch_size is not None:
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        config.batch_size = args.batch_size
        config.niters_per_epoch = config.num_train_imgs // config.batch_size + 1
    if args.workers is not None:
        if args.workers < 0:
            parser.error("--workers cannot be negative")
        config.num_workers = args.workers
    if args.val_batch_size is not None:
        if args.val_batch_size <= 0:
            parser.error("--val-batch-size must be positive")
        config.val_batch_size = args.val_batch_size
    if args.max_train_iters is not None and args.max_train_iters <= 0:
        parser.error("--max-train-iters must be positive")
    if args.log_interval < 0:
        parser.error("--log-interval cannot be negative")
    if args.min_free_vram_gib < 0:
        parser.error("--min-free-vram-gib cannot be negative")
    if not 0.0 <= args.min_free_vram_ratio <= 1.0:
        parser.error("--min-free-vram-ratio must be between 0 and 1")
    if args.checkpoint_dir:
        config.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)
    # check if pad_SUNRGBD is used correctly
    if args.pad_SUNRGBD and config.dataset_name != "SUNRGBD":
        args.pad_SUNRGBD = False
        logger.warning("pad_SUNRGBD is only used for SUNRGBD dataset")
    if (args.pad_SUNRGBD) and (not config.backbone.startswith("DFormerv2")):
        raise ValueError("DFormerv1 is not recommended with pad_SUNRGBD")
    if (not args.pad_SUNRGBD) and config.backbone.startswith("DFormerv2") and config.dataset_name == "SUNRGBD":
        raise ValueError("DFormerv2 is not recommended without pad_SUNRGBD")
    config.pad = args.pad_SUNRGBD
    if args.use_seed:
        set_seed(config.seed)
        logger.info(f"set seed {config.seed}")
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        logger.info("use random seed")

    # assert not (args.compile and args.syncbn), "syncbn is not supported in compile mode"
    if not args.compile and args.compile_mode != "default":
        logger.warning("compile_mode is only valid when compile is enabled, ignoring compile_mode")

    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)

    if args.gpus == 2:
        if args.mst and args.compile and args.compile_mode == "reduce-overhead":
            val_dl_factor = 0.25
        elif args.mst and not args.val_amp:
            val_dl_factor = 1.5
        elif args.mst and args.val_amp:
            val_dl_factor = 1.3
        else:
            val_dl_factor = 2
    elif args.gpus == 4:
        if args.mst and args.compile and args.compile_mode == "reduce-overhead":
            val_dl_factor = 0.25
        elif args.mst and not args.val_amp:
            val_dl_factor = 1.5
        elif args.mst and args.val_amp:
            val_dl_factor = 0.6
        else:
            val_dl_factor = 2
    else:
        val_dl_factor = 1.5

    val_dl_factor = 1  # TODO: remove this line
    default_val_batch_size = (
        int(config.batch_size * val_dl_factor) if config.dataset_name != "SUNRGBD" else int(args.gpus)
    )
    val_batch_size = int(getattr(config, "val_batch_size", default_val_batch_size))
    val_loader, val_sampler = get_val_loader(
        engine,
        RGBXDataset,
        config,
        val_batch_size=val_batch_size,
    )
    logger.info(f"val dataset len:{len(val_loader) * int(args.gpus)}")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + "/tb"
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("config: \n" + pp.pformat(config))

    logger.info("args parsed:")
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))

    is_primary = (not engine.distributed) or engine.local_rank == 0
    run_name = build_run_name(
        config.dataset_name,
        config.backbone,
        explicit_name=args.swanlab_run_name,
        repo_root=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    )
    logger.info(f"experiment run name: {run_name}")
    tracker.start(
        mode=args.swanlab_mode,
        is_primary=is_primary,
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        name=run_name,
        log_dir=config.log_dir,
        config={
            "config_module": args.config,
            "dataset": config.dataset_name,
            "backbone": config.backbone,
            "epochs": config.nepochs,
            "batch_size": config.batch_size,
            "val_batch_size": val_batch_size,
            "workers": config.num_workers,
            "optimizer": config.optimizer,
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
            "amp": args.amp,
            "max_train_iters": args.max_train_iters,
            "min_free_vram_gib": args.min_free_vram_gib,
            "min_free_vram_ratio": args.min_free_vram_ratio,
        },
    )

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)

    if args.syncbn:
        BatchNorm2d = nn.SyncBatchNorm
        logger.info("using syncbn")
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")

    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )
    # weight=torch.load('checkpoints/NYUv2_DFormer_Large.pth')['model']
    # w_list=list(weight.keys())
    # # for k in w_list:
    # #     weight[k[7:]] = weight[k]
    # print('load model')
    # model.load_state_dict(weight)

    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    # params_list = configure_optimizers(model, base_lr, config.weight_decay)

    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params_list,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "SGDM":
        optimizer = torch.optim.SGD(
            params_list,
            lr=base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        base_lr,
        config.lr_power,
        total_iteration,
        config.niters_per_epoch * config.warm_up_epoch,
    )
    if engine.distributed:
        logger.info(".............distributed training.............")
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=False,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()

    logger.info("begin trainning:")
    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
    }
    # val_pre = ValPre()
    # val_dataset = RGBXDataset(data_setting, 'val', val_pre)
    # test_loader, test_sampler = get_test_loader(engine, RGBXDataset,config)
    all_dev = [0]
    # segmentor = SegEvaluator(val_dataset, config.num_classes, config.norm_mean,
    #                                 config.norm_std, None,
    #                                 config.eval_scale_array, config.eval_flip,
    #                                 all_dev, config,args.verbose, args.save_path,args.show_image)
    uncompiled_model = model
    if args.compile:
        compiled_model = torch.compile(model, backend="inductor", mode=args.compile_mode)
    else:
        compiled_model = model
    miou, best_miou = 0.0, 0.0
    train_timer = gpu_timer()
    eval_timer = gpu_timer()

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    log_interval = args.log_interval or max(1, int(config.niters_per_epoch * 0.1))
    train_steps_completed = 0
    short_run_reached = False
    for epoch in range(engine.state.epoch, config.nepochs + 1):
        model = compiled_model
        model.train()
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        # bar_format = "{desc}[{elapsed}<{remaining},{rate_fmt}]"
        # pbar = tqdm(
        #     range(config.niters_per_epoch),
        #     file=sys.stdout,
        #     bar_format=bar_format,
        #     # range(5),
        #     # file=sys.stdout,
        #     # bar_format=bar_format,
        # )
        dataloader = iter(train_loader)

        sum_loss = 0
        i = 0
        train_timer.start()
        for idx in range(config.niters_per_epoch):
            engine.update_iteration(epoch, idx)
            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            global_step = current_idx + 1
            lr = lr_policy.get_lr(current_idx)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            next_train_step = train_steps_completed + 1
            should_log = next_train_step == 1 or next_train_step % log_interval == 0
            safety_monitoring = args.min_free_vram_gib > 0 or args.min_free_vram_ratio > 0
            measure_step = should_log or safety_monitoring
            if measure_step:
                torch.cuda.synchronize()
                step_started_at = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

            # minibatch = dataloader.next()
            minibatch = next(dataloader)
            imgs = minibatch["data"]
            gts = minibatch["label"]
            modal_xs = minibatch["modal_x"]

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            if args.amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(imgs, modal_xs, gts)
            else:
                loss = model(imgs, modal_xs, gts)

            if measure_step and not bool(torch.isfinite(loss.detach()).all().item()):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, iteration {idx + 1}")

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

            if args.amp:
                # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # TODO: check if set_to_none=True impact the performance
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if not args.amp:
                if epoch == 1:
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            logger.warning(f"{name} has no grad, please check")

            telemetry_suffix = ""
            if measure_step:
                torch.cuda.synchronize()
                step_seconds = time.perf_counter() - step_started_at
                images_per_second = config.batch_size / step_seconds
                max_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                max_memory_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
                free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
                free_vram_mb = free_vram_bytes / 1024**2
                total_vram_mb = total_vram_bytes / 1024**2
                free_vram_ratio = free_vram_bytes / total_vram_bytes
                safety_error = gpu_safety_violation(
                    free_vram_bytes,
                    total_vram_bytes,
                    min_free_gib=args.min_free_vram_gib,
                    min_free_ratio=args.min_free_vram_ratio,
                )
                if safety_error is not None:
                    raise RuntimeError(f"GPU safety threshold violated at step {global_step}: {safety_error}")
                amp_scale = scaler.get_scale() if args.amp else 1.0
                telemetry_suffix = (
                    f" step={step_seconds:.3f}s throughput={images_per_second:.2f} img/s"
                    f" allocated={max_memory_mb:.0f} MiB reserved={max_memory_reserved_mb:.0f} MiB"
                    f" free={free_vram_mb:.0f}/{total_vram_mb:.0f} MiB"
                    f" free_ratio={free_vram_ratio:.3f} amp_scale={amp_scale:.1f}"
                )

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = (
                    "Epoch {}/{}".format(epoch, config.nepochs)
                    + " Iter {}/{}:".format(idx + 1, config.niters_per_epoch)
                    + " lr=%.4e" % lr
                    + " loss=%.4f total_loss=%.4f" % (reduce_loss.item(), (sum_loss / (idx + 1)))
                    + telemetry_suffix
                )

            else:
                loss_value = loss.item()
                sum_loss += loss_value
                print_str = (
                    f"Epoch {epoch}/{config.nepochs} "
                    + f"Iter {idx + 1}/{config.niters_per_epoch}: "
                    + f"lr={lr:.4e} loss={loss_value:.4f} total_loss={(sum_loss / (idx + 1)):.4f}"
                    + telemetry_suffix
                )

            train_steps_completed += 1
            loss_value_for_log = reduce_loss.item() if engine.distributed else loss_value
            if should_log and is_primary:
                print(print_str)
                tracker.log(
                    {
                        "train/loss": loss_value_for_log,
                        "train/loss_mean": sum_loss / (idx + 1),
                        "train/learning_rate": lr,
                        "train/step_seconds": step_seconds,
                        "train/images_per_second": images_per_second,
                        "train/max_memory_mb": max_memory_mb,
                        "train/max_memory_reserved_mb": max_memory_reserved_mb,
                        "train/free_vram_mb": free_vram_mb,
                        "train/total_vram_mb": total_vram_mb,
                        "train/free_vram_ratio": free_vram_ratio,
                        "train/amp_scale": amp_scale,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            del loss
            if args.max_train_iters is not None and train_steps_completed >= args.max_train_iters:
                short_run_reached = True
                logger.info(f"short run reached --max-train-iters={args.max_train_iters}")
                break
            # pbar.set_description(print_str, refresh=False)
        logger.info(print_str)
        train_epoch_seconds = train_timer.stop()
        epoch_loss = sum_loss / (idx + 1)
        if is_primary:
            tracker.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "train/epoch_seconds": train_epoch_seconds,
                },
                step=global_step,
            )
        if short_run_reached:
            logger.info("short run completed normally; validation and checkpoint saving were skipped")
            break

        # if (engine.distributed and (engine.local_rank == 0)) or (
        #     not engine.distributed
        # ):
        #     tb.add_scalar("train_loss", sum_loss / len(pbar), epoch)

        if is_eval(epoch, config):
            eval_timer.start()
            torch.cuda.empty_cache()
            # if args.compile and args.mst and (not args.sliding):
            #     model = uncompiled_model
            # TODO: FIX this
            if engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.val_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            if args.mst:
                                all_metrics = evaluate_msf(
                                    model,
                                    val_loader,
                                    config,
                                    device,
                                    [0.5, 0.75, 1.0, 1.25, 1.5],
                                    True,
                                    engine,
                                    sliding=args.sliding,
                                )
                            else:
                                all_metrics = evaluate(
                                    model,
                                    val_loader,
                                    config,
                                    device,
                                    engine,
                                    sliding=args.sliding,
                                )
                    else:
                        if args.mst:
                            all_metrics = evaluate_msf(
                                model,
                                val_loader,
                                config,
                                device,
                                [0.5, 0.75, 1.0, 1.25, 1.5],
                                True,
                                engine,
                                sliding=args.sliding,
                            )
                        else:
                            all_metrics = evaluate(
                                model,
                                val_loader,
                                config,
                                device,
                                engine,
                                sliding=args.sliding,
                            )
                    if engine.local_rank == 0:
                        metric = all_metrics[0]
                        for other_metric in all_metrics[1:]:
                            metric.update_hist(other_metric.hist)
                        ious, miou = metric.compute_iou()
                        acc, macc = metric.compute_pixel_acc()
                        f1, mf1 = metric.compute_f1()
                        if miou > best_miou:
                            best_miou = miou
                            engine.save_and_link_checkpoint(
                                config.log_dir,
                                config.log_dir,
                                config.log_dir_link,
                                infor="_miou_" + str(miou),
                                metric=miou,
                            )
                        print("miou", miou, "best", best_miou)
            elif not engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.val_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            if args.mst:
                                metric = evaluate_msf(
                                    model,
                                    val_loader,
                                    config,
                                    device,
                                    [0.5, 0.75, 1.0, 1.25, 1.5],
                                    True,
                                    engine,
                                    sliding=args.sliding,
                                )
                            else:
                                metric = evaluate(
                                    model,
                                    val_loader,
                                    config,
                                    device,
                                    engine,
                                    sliding=args.sliding,
                                )
                    else:
                        if args.mst:
                            metric = evaluate_msf(
                                model,
                                val_loader,
                                config,
                                device,
                                [0.5, 0.75, 1.0, 1.25, 1.5],
                                True,
                                engine,
                                sliding=args.sliding,
                            )
                        else:
                            metric = evaluate(
                                model,
                                val_loader,
                                config,
                                device,
                                engine,
                                sliding=args.sliding,
                            )
                    ious, miou = metric.compute_iou()
                    acc, macc = metric.compute_pixel_acc()
                    f1, mf1 = metric.compute_f1()
                    # print('miou',miou)
                # print('acc, macc, f1, mf1, ious, miou',acc, macc, f1, mf1, ious, miou)
                # print('miou',miou)
                if miou > best_miou:
                    best_miou = miou
                    engine.save_and_link_checkpoint(
                        config.log_dir,
                        config.log_dir,
                        config.log_dir_link,
                        infor="_miou_" + str(miou),
                        metric=miou,
                    )
                print("miou", miou, "best", best_miou)
            logger.info(f"Epoch {epoch} validation result: mIoU {miou}, best mIoU {best_miou}")
            if is_primary:
                tracker.log(
                    {
                        "validation/miou": float(miou),
                        "validation/best_miou": float(best_miou),
                        "validation/mean_accuracy": float(macc),
                        "validation/mean_f1": float(mf1),
                        "validation/epoch": epoch,
                    },
                    step=epoch * config.niters_per_epoch,
                )
            eval_timer.stop()

        checkpoint_step = int(getattr(config, "checkpoint_step", 0))
        save_epoch_checkpoints = getattr(config, "save_epoch_checkpoints", False)
        if save_epoch_checkpoints and checkpoint_step > 0 and (
            epoch % checkpoint_step == 0 or epoch == config.nepochs
        ) and ((engine.distributed and engine.local_rank == 0) or not engine.distributed):
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            epoch_checkpoint = os.path.join(config.checkpoint_dir, f"epoch-{epoch}.pth")
            engine.save_checkpoint(epoch_checkpoint)
            if getattr(config, "save_latest_checkpoint", False):
                engine.save_checkpoint(os.path.join(config.checkpoint_dir, "latest.pth"))

        eval_count = 0
        for i in range(engine.state.epoch + 1, config.nepochs + 1):
            if is_eval(i, config):
                eval_count += 1
        left_time = train_timer.mean_time * (config.nepochs - engine.state.epoch) + eval_timer.mean_time * eval_count
        eta = (datetime.datetime.now() + datetime.timedelta(seconds=left_time)).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"Avg train time: {train_timer.mean_time:.2f}s, avg eval time: {eval_timer.mean_time:.2f}s, left eval count: {eval_count}, ETA: {eta}"
        )
