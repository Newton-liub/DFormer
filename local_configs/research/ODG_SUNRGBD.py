"""ODG research config: SUN RGB-D, DFormerv2-S + HAM, 37 classes.

Importing this module never trains; it only fills the shared config
object.  ``--geometry-mode`` on the entry points overrides ``C.geometry_mode``
(``original`` | ``mean`` | ``odg``), ``--micro-batch`` / ``--accum-steps``
override the batching, and ``--fulltrain`` switches from the fixed seed12345
development split to the official full train/test lists.

Boundary that the entry points enforce:
  * dev mode (default) trains on ``train-dev.txt`` and validates on ``dev.txt``
    only; the official ``test.txt`` is never read during training.
  * ``--fulltrain`` must be requested explicitly; in that mode in-training
    periodic validation is disabled so official test can never pick the best
    checkpoint.  Test is only reachable through ``research/evaluate_odg.py
    --split test``.
"""
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from local_configs.SUNRGBD.DFormerv2_S import C as _official

C = deepcopy(_official)
config = C

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEV_SPLIT_DIR = _REPO_ROOT / "research" / "splits" / "sunrgbd_seed12345"

"""Model (author v2-S + HAM, 37 classes).
The decoder width keeps the author's value: ``local_configs/SUNRGBD/DFormerv2_S.py``
sets ``C.decoder_embed_dim = 1024``.
"""
C.backbone = "DFormerv2_S"
C.decoder = "ham"
C.decoder_embed_dim = 1024
C.pretrained_model = "checkpoints/pretrained/DFormerv2_Small_pretrained.pth"
# Only the encoder is loaded from the official pretrained file; the HAM head and
# the optimizer/scaler are created fresh (the builder already behaves this way).
C.geometry_mode = "odg"
# Keep BatchNorm statistics trainable (no frozen "norm eval" behaviour).
C.norm_eval = False

"""Data paths."""
C.dataset_path = str(Path(os.environ.get("DFORMER_DATASET_ROOT", "datasets")) / "SUNRGBD")
C.rgb_root_folder = str(Path(C.dataset_path) / "RGB")
C.gt_root_folder = str(Path(C.dataset_path) / "labels")
C.x_root_folder = str(Path(C.dataset_path) / "Depth")

C.train_dev_source = str(_DEV_SPLIT_DIR / "train-dev.txt")
C.dev_eval_source = str(_DEV_SPLIT_DIR / "dev.txt")
C.full_train_source = str(Path(C.dataset_path) / "train.txt")
C.full_test_source = str(Path(C.dataset_path) / "test.txt")
C.train_source = C.train_dev_source
C.eval_source = C.dev_eval_source
C.dev_split_dir = str(_DEV_SPLIT_DIR)
C.dev_split_ready = True

# Counts are documentation only; the entry points recount the actual manifests.
C.num_train_imgs = 4757
C.num_eval_imgs = 528
C.full_num_train_imgs = 5285
C.full_num_eval_imgs = 5050

C.seed = 12345
C.optimizer = "AdamW"
C.lr = 8e-5
C.weight_decay = 0.01
C.lr_power = 0.9
C.momentum = 0.9
C.warm_up_epoch = 10
C.schedule_epochs = 300
C.effective_batch = 16
C.micro_batch = 16
C.accum_steps = 1
C.nepochs = C.schedule_epochs
C.batch_size = C.effective_batch
# Documentation only: the entry points recompute iterations from the real manifest.
C.niters_per_epoch = -(-C.num_train_imgs // C.effective_batch)
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.num_workers = 8

"""Periodic validation: fixed dev split, single scale, no flip, every 10 epochs."""
C.val_every = 10
C.val_msf = False
C.val_flip = False
# Artificial connected holes are opt-in and never run by default.
C.artificial_hole_ratio = 0.0
C.artificial_hole_seed = 12345

"""Stage checkpoints / pause."""
C.stage_epochs = (30, 100)
C.stop_after_epoch = 0
C.best_metric = "dev_miou"

C.checkpoint_start_epoch = 200
C.checkpoint_step = 25
C.drop_path_rate = 0.1
C.aux_rate = 0.0
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Evaluation defaults (explicit five-scale + flip lives in evaluate_odg.py)."""
C.eval_scale_array = [0.5, 0.75, 1, 1.25, 1.5]
C.eval_flip = True
C.eval_crop_size = [480, 480]
C.image_height = 480
C.image_width = 480

"""Outputs."""
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
C.experiment_name = os.environ.get("DFORMER_EXPERIMENT_NAME", "odg-sunrgbd")
C.log_dir = str(Path("outputs") / C.experiment_name / run_id)
C.log_dir_link = C.log_dir
C.checkpoint_dir = C.log_dir
C.tb_dir = str(Path(C.log_dir) / "tb")
C.log_file = str(Path(C.log_dir) / "train.log")
C.link_log_file = str(Path(C.log_dir) / "train-last.log")
C.val_log_file = str(Path(C.log_dir) / "validation.log")
C.link_val_log_file = str(Path(C.log_dir) / "validation-last.log")
C.swanlab_enabled = True
C.swanlab_project = "dformer-research"
C.swanlab_mode = "online"
C.swanlab_log_dir = str(Path(C.log_dir) / "swanlab")
