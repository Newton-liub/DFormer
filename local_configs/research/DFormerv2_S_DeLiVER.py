"""DeLiVER research config: RGB + raw single-channel ``depth/``, 25 classes.

Approved protocol (``doc/plans/2026-10-10-deliver-integration.md``, 2026-10-10):

* input = RGB image + the ``depth/`` native single-channel uint8 image, normalised
  with the author rule ``(D8/255 - 0.48)/0.28`` and replicated to three channels.
  ``hha/``, ``lidar/`` and ``event/`` are **not** read, so this is **not** the
  official DeLiVER HHA RGB-D protocol: scores from it are not comparable with the
  official numbers and ``depth/`` has no claimed physical unit.
* the official ``train``/``val``/``test`` directories are kept unchanged and are
  not re-split.  In the research entries ``--split dev`` maps to the official
  **val** list and ``--split test`` to the official **test** list; the official
  test list is final-evaluation only, never a tuning signal.
* DFormerv2-S + the author HAM head (width 1024), 25 classes, geometry mode
  ``original``.  No candidate method is scored here.
* ``pad=False``: DeLiVER images keep their native grid, so the SUN 531x730 padding
  is not used and ``depth_support`` is all-observed except for random training
  crops.

Importing this module never trains.  It does inherit the author SUN config's
import-time run-directory creation, which is then overridden below.
"""
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from local_configs.SUNRGBD.DFormerv2_S import C as _official
from research.deliver import DELIVER_CLASS_NAMES, DELIVER_IGNORE, DELIVER_NUM_CLASSES

C = deepcopy(_official)
config = C

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_DIR = _REPO_ROOT / "research" / "splits" / "deliver_official"

"""Model: author DFormerv2-S + HAM at the author decoder width (1024)."""
C.backbone = "DFormerv2_S"
C.decoder = "ham"
C.decoder_embed_dim = 1024
# The encoder checkpoint lives in a gitignored directory, so a fresh checkout or
# worktree may not carry it.  Point ``$DFORMER_PRETRAINED`` at an absolute path in
# that case; the default is the main-line relative path.
C.pretrained_model = os.environ.get(
    "DFORMER_PRETRAINED", "checkpoints/pretrained/DFormerv2_Small_pretrained.pth"
)
# Only the official encoder weights are loaded; the HAM head, optimizer and scaler
# are created fresh.  No SUN/MUSeg segmentation checkpoint is reused.
C.geometry_mode = "original"
# Keep BatchNorm statistics trainable (no frozen "norm eval" behaviour).
C.norm_eval = False

"""Data protocol and paths.

``data_module`` selects the DeLiVER reader (``research/deliver.py``); the shared
research entry points read it instead of hard-wiring ``research.data``.  Override
the parent directory with ``$DFORMER_DATASET_ROOT``.
"""
C.data_module = "research.deliver"
C.dataset_name = "DeLiVER"
C.dataset_path = str(Path(os.environ.get("DFORMER_DATASET_ROOT", "datasets")) / "DELIVER")
C.rgb_root_folder = str(Path(C.dataset_path) / "img")
C.gt_root_folder = str(Path(C.dataset_path) / "semantic")
C.x_root_folder = str(Path(C.dataset_path) / "depth")
C.rgb_format = ".png"
C.gt_format = ".png"
C.x_format = ".png"
# The DeLiVERDataset maps the raw ids itself, so the author ``gt - 1`` transform
# must stay off.
C.gt_transform = False
C.x_is_single_channel = True
# Recorded input conventions (documentation; the reader is authoritative).
C.rgb_order = "RGB"
C.depth_protocol = "depth/ native single-channel uint8, (D8/255-0.48)/0.28, no unit claim"
C.official_protocol_equivalent = False

"""Official splits: train for training, val for periodic validation, test final only."""
C.train_dev_source = str(_SPLIT_DIR / "train.txt")
C.dev_eval_source = str(_SPLIT_DIR / "val.txt")
C.full_train_source = C.train_dev_source
C.full_test_source = str(_SPLIT_DIR / "test.txt")
C.train_source = C.train_dev_source
C.eval_source = C.dev_eval_source
C.dev_split_dir = str(_SPLIT_DIR)
C.dev_split_ready = True

# Documentation only; the entry points recount the actual manifests.
C.num_train_imgs = 3983
C.num_eval_imgs = 2005
C.full_num_train_imgs = 3983
C.full_num_eval_imgs = 1897

"""Classes (order and palette from the official loader)."""
C.num_classes = DELIVER_NUM_CLASSES
C.class_names = list(DELIVER_CLASS_NAMES)
C.background = DELIVER_IGNORE

"""Training defaults.  Runnable defaults only: not an approved experiment contract
and not the official DeLiVER recipe.  The micro-batch is chosen at run time."""
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
C.niters_per_epoch = -(-C.num_train_imgs // C.effective_batch)
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.num_workers = 8

"""Image geometry: native resolution on the 480x480 training crop, no padding."""
C.pad = False
C.image_height = 480
C.image_width = 480

"""Periodic validation: official val, single scale, no flip, every 10 epochs."""
C.val_every = 10
C.val_msf = False
C.val_flip = False
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

"""Evaluation defaults (five-scale + flip stays opt-in in evaluate_odg.py)."""
C.eval_scale_array = [0.5, 0.75, 1, 1.25, 1.5]
C.eval_flip = True
C.eval_crop_size = [480, 480]  # unused by the research evaluation entry
C.eval_iter = 25
C.eval_stride_rate = 2 / 3

"""Outputs."""
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
C.experiment_name = os.environ.get("DFORMER_EXPERIMENT_NAME", "deliver")
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
