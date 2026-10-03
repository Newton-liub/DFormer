"""Independent Natural/Grid/Replay continuation config for MUSeg.

The factory starts from a deep copy of the shared DFormerv2-S base, then explicitly
restores every model, data, and training field used by the Natural Missing contract.
It never imports a mutable E1/A2 config object.
"""

from __future__ import annotations

import copy
import os
import os.path as osp

import numpy as np

from .DFormerv2_S_Base import C as _BASE_C


_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
_DATA_ROOT = osp.abspath(
    os.environ.get("DFORMER_DATA_ROOT", osp.join(_PROJECT_ROOT, "..", "dataset"))
)
_DATASET_ROOT = osp.join(_DATA_ROOT, "MUSeg_DFormer")
_DEPTH16_ROOT = osp.abspath(
    os.environ.get("MUSEG_DEPTH16_ROOT", osp.join(_DATA_ROOT, "MUSeg_DFormer", "Depth16"))
)
_SPLIT_ROOT = osp.join(_PROJECT_ROOT, "data", "splits", "MUSeg", "dev-v1")
_OUTPUT_ROOT = osp.abspath(
    os.environ.get("DFORMER_OUTPUT_ROOT", osp.join(_PROJECT_ROOT, "outputs"))
)
_PRETRAINED = osp.abspath(
    os.environ.get(
        "DFORMER_PRETRAINED",
        osp.join(osp.dirname(_PROJECT_ROOT), "pretrained", "DFormerv2_Small_pretrained.pth"),
    )
)

_STRATEGIES = ("Natural", "Grid", "Replay")
_TRAIN_SPLIT_SHA256 = "a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470"
_VAL_SPLIT_SHA256 = "1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83"


def make_config(strategy: str = "Natural"):
    """Return a fresh, isolated config for one of the three paired strategies."""
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")

    config = copy.deepcopy(_BASE_C)

    # Restore the complete segmentation/model identity rather than inheriting any
    # experiment fields that may have leaked into the shared base object.
    config.dataset_name = "MUSeg_DFormer"
    config.num_classes = 15
    config.background = 255
    config.backbone = "DFormerv2_S"
    config.decoder = "ham"
    config.decoder_embed_dim = 512
    config.aux_rate = 0.0
    config.drop_path_rate = 0.25
    config.bn_eps = 1e-3
    config.bn_momentum = 0.1
    config.pad = False
    config.eval_scale_array = [1.0]
    config.eval_flip = False
    config.eval_crop_size = [480, 640]
    config.eval_stride_rate = 2 / 3
    config.mst = False
    config.sliding = False
    config.image_height = 480
    config.image_width = 640
    config.x_modal = ["d"]
    config.x_is_single_channel = True
    config.rgb_format = ".jpg"
    config.gt_format = ".png"
    config.gt_transform = True
    config.x_format = ".png"
    config.class_names = [
        "person",
        "cable",
        "tube",
        "indicator",
        "metal fixture",
        "container",
        "tools & materials",
        "door",
        "electrical equipment",
        "electronic equipment",
        "mining equipment",
        "anchoring equipment",
        "support equipment",
        "rescue equipment",
        "rail area",
    ]
    config.norm_mean = np.array([0.485, 0.456, 0.406])
    config.norm_std = np.array([0.229, 0.224, 0.225])
    config.channel_order = "RGB"
    config.normalization_identity = "rgb-imagenet-rgb-order-v1"
    config.pretrained_model = _PRETRAINED

    # Use only the development train/validation split identities. The training
    # entry point never resolves or reads an official-test list.
    config.root_dir = _DATA_ROOT
    config.dataset_path = _DATASET_ROOT
    config.rgb_root_folder = osp.join(_DATASET_ROOT, "RGB")
    config.gt_root_folder = osp.join(_DATASET_ROOT, "Label")
    config.x_root_folder = osp.join(_DATASET_ROOT, "Depth")
    config.depth16_root = _DEPTH16_ROOT
    config.split_root = _SPLIT_ROOT
    config.train_source = osp.join(_SPLIT_ROOT, "train-dev.txt")
    config.val_source = osp.join(_SPLIT_ROOT, "val-dev.txt")
    config.eval_source = config.val_source
    config.test_source = None
    config.expected_split_sha256 = {
        "train": _TRAIN_SPLIT_SHA256,
        "val": _VAL_SPLIT_SHA256,
    }
    config.expected_split_samples = {"train": 1277, "val": 318}
    config.num_train_imgs = 1277
    config.num_eval_imgs = 318
    config.experiment_phase = "development"

    # The non-corruption augmentation and common optimizer/scheduler settings follow
    # the actual E1 C0 runner. Only the requested LR, strategy, and continuation
    # checkpoint identity differ.
    config.seed = 772961337
    config.optimizer = "AdamW"
    config.lr = 1e-6
    config.lr_power = 0.9
    config.momentum = 0.9
    config.weight_decay = 0.01
    config.batch_size = 10
    config.niters_per_epoch = 128
    config.nepochs = 20
    config.num_workers = 8
    config.train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    config.warm_up_epoch = 1
    config.warmup_successful_updates = 128
    config.successful_update_budget = 2560
    config.max_attempts = 2560  # Any non-finite or AMP-skipped update aborts the run.
    config.recovery_interval_successful_updates = 640
    config.fixed_final_checkpoint = "update-2560.pth"
    config.save_interval = 5
    config.checkpoint_step = 5
    config.checkpoint_start_epoch = config.nepochs
    config.save_epoch_checkpoints = True
    config.save_latest_checkpoint = True
    config.checkpoint_candidate_manifest = None
    config.checkpoint_retention_policy = {
        "policy": "fixed-successful-update-recovery",
        "interval_successful_updates": 640,
        "retain_recovery_points": True,
        "fixed_final_checkpoint": "update-2560.pth",
        "preflight_checkpoints_formal_eligible": False,
    }
    config.optimizer_telemetry_policy = {
        "schema_version": "museg-optimizer-telemetry-v1",
        "invariant": "attempted_steps=successful_updates;skipped_steps=0",
    }
    config.training_validation_enabled = False
    config.eval_start_epoch = config.nepochs
    config.eval_interval = config.nepochs
    config.val_batch_size = 1

    # Match the E1 C0 execution defaults without carrying its gate/config block.
    config.syncbn = True
    config.amp = True
    config.amp_dtype = "float16"
    config.grad_scaler = {
        "initial_scale": 1024.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
    }
    config.tf32_matmul_precision = "high"
    config.tf32_matmul = True
    config.tf32_cudnn = True
    # The 2026-10-04 Natural diagnosis reproduced non-finite NMF backward
    # under autocast at attempt 1598. All three strategies use the same local
    # FP32 numerical repair; the NMF algorithm and surrounding AMP stay unchanged.
    config.nmf_training_precision = "fp32_local"

    # Explicitly disable all legacy training branches and gates.
    config.mmfr_a2 = None
    config.mmfr_a2_execution_profile = None
    config.e1_batch1 = None
    config.e1_batch1b = None
    config.e1_roe = None
    config.mmfr_av1 = None
    config.mmfr_f_lite = None
    config.mmfr_roe = None
    config.training_validation_enabled = False

    config.run_id = f"NaturalMissing-{strategy}"
    config.log_dir = osp.join(_OUTPUT_ROOT, config.run_id, "development", f"seed-{config.seed}")
    config.tb_dir = osp.join(config.log_dir, "tb")
    config.log_dir_link = config.log_dir
    config.checkpoint_dir = osp.join(config.log_dir, "checkpoint")
    config.log_file = osp.join(config.log_dir, "train.log")
    config.link_log_file = osp.join(config.log_dir, "log_last.log")
    config.val_log_file = osp.join(config.log_dir, "val.log")
    config.link_val_log_file = osp.join(config.log_dir, "val_last.log")

    config.natural_missing = {
        "enabled": True,
        "strategy": strategy,
        "metadata_enabled": True,
        "train_role": "train-dev",
        "depth16_root": _DEPTH16_ROOT,
        "successful_update_budget": 2560,
        "recovery_interval_successful_updates": 640,
    }
    return config


C = make_config(strategy="Natural")
