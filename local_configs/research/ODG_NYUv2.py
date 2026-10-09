"""ODG research config: NYU Depth v2, DFormerv2-S + HAM, 40 classes.

Entry-point preparation only -- this round does not train a second dataset.
No fixed development split exists for NYU yet, so ``C.dev_split_ready`` is
``False`` and the dev-mode list paths point at a directory that does not exist
yet; dev-mode training refuses to run until a dev split is created, and
``--fulltrain`` (official train/test) is required otherwise.  The official
``test.txt`` is deliberately *not* reused as a dev placeholder.  Importing this
module never trains.
"""
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from local_configs.NYUDepthv2.DFormerv2_S import C as _official

C = deepcopy(_official)
config = C

_REPO_ROOT = Path(__file__).resolve().parents[2]

"""Model (author v2-S + HAM, 40 classes).
The decoder width is the author's own value: ``local_configs/NYUDepthv2/
DFormerv2_S.py`` sets ``C.decoder_embed_dim = 512``, so the research config keeps
512 rather than overriding it.
"""
C.backbone = "DFormerv2_S"
C.decoder = "ham"
C.decoder_embed_dim = 512
C.pretrained_model = "checkpoints/pretrained/DFormerv2_Small_pretrained.pth"
C.geometry_mode = "odg"
C.norm_eval = False

"""Data paths (author layout; override the parent with $DFORMER_DATASET_ROOT).

No NYU development split exists yet: the dev-mode sources point at the (absent)
``research/splits/nyuv2_seed12345`` files and ``dev_split_ready`` is ``False``,
so dev-mode training refuses.  ``--fulltrain`` uses the official lists.  The
official ``test.txt`` is not used as a dev placeholder.
"""
C.dataset_path = str(Path(os.environ.get("DFORMER_DATASET_ROOT", "datasets")) / "NYUDepthv2")
C.rgb_root_folder = str(Path(C.dataset_path) / "RGB")
C.gt_root_folder = str(Path(C.dataset_path) / "Label")
C.x_root_folder = str(Path(C.dataset_path) / "Depth")

C.dev_split_dir = str(_REPO_ROOT / "research" / "splits" / "nyuv2_seed12345")
C.train_dev_source = str(Path(C.dev_split_dir) / "train-dev.txt")
C.dev_eval_source = str(Path(C.dev_split_dir) / "dev.txt")
C.full_train_source = str(Path(C.dataset_path) / "train.txt")
C.full_test_source = str(Path(C.dataset_path) / "test.txt")
C.train_source = C.train_dev_source
C.eval_source = C.dev_eval_source
C.dev_split_ready = False

C.num_train_imgs = 795
C.num_eval_imgs = 654
C.full_num_train_imgs = 795
C.full_num_eval_imgs = 654

C.seed = 12345
C.optimizer = "AdamW"
C.lr = 6e-5
C.weight_decay = 0.01
C.lr_power = 0.9
C.momentum = 0.9
C.warm_up_epoch = 10
C.schedule_epochs = 500
C.effective_batch = 16
C.micro_batch = 16
C.accum_steps = 1
C.nepochs = C.schedule_epochs
C.batch_size = C.effective_batch
# Documentation only: the entry points recompute iterations from the real manifest.
C.niters_per_epoch = -(-C.num_train_imgs // C.effective_batch)
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.num_workers = 8

C.val_every = 10
C.val_msf = False
C.val_flip = False
C.artificial_hole_ratio = 0.0
C.artificial_hole_seed = 12345

C.stage_epochs = (30, 100)
C.stop_after_epoch = 0
C.best_metric = "dev_miou"

C.checkpoint_start_epoch = 250
C.checkpoint_step = 25
C.drop_path_rate = 0.25
C.aux_rate = 0.0
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

C.eval_scale_array = [1]
C.eval_flip = True
C.eval_crop_size = [480, 640]
C.image_height = 480
C.image_width = 640

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
C.experiment_name = os.environ.get("DFORMER_EXPERIMENT_NAME", "odg-nyuv2")
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
