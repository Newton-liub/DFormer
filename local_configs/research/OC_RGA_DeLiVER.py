"""OC-RGA research config: DeLiVER RGB + raw single-channel ``depth/``, 25 classes.

Deep copy of :mod:`local_configs.research.DFormerv2_S_DeLiVER` (the approved
DeLiVER access protocol: RGB plus the native ``depth/`` uint8 image normalised
with ``(D8/255 - 0.48)/0.28``; ``hha/``, ``lidar/`` and ``event/`` are never
read, so this is **not** the official DeLiVER HHA RGB-D protocol and its scores
are not comparable with the official numbers).

The first-round contract fixed by the OC-RGA v0.3 design (section 11) lives here:

* 120 epoch matching schedule, 5 epoch warmup, poly 0.9; 50 epoch is a shared
  observation point and never compresses the schedule;
* effective batch 16, ``micro_batch * accum_steps == 16`` chosen at run time;
* AdamW base lr 8e-5, weight decay 0.01, betas (0.9, 0.999); the new geometry
  branch runs at lr x2 and its positive-parameter logits are not decayed;
* FP16 AMP with FP32 probability maths and a shared FP32 attention softmax;
* 480x480 training crop, 640 long-side single-scale shared evaluation protocol.

Everything here is a runnable default, not an approved GPU budget.
"""
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from local_configs.research.DFormerv2_S_DeLiVER import C as _deliver

C = deepcopy(_deliver)
config = C

_REPO_ROOT = Path(__file__).resolve().parents[2]

"""Model: author DFormerv2-S + HAM1024, 25 classes, OC-RGA research wrapper.

``model_mode`` is chosen on the command line (``masked_baseline`` | ``oc_rga`` |
``mean_relation`` | ``diag_relation`` | ``direct_post``).  The wrapper builds the
geometry package; the backbone itself never runs the author/ODG depth relation.
"""
C.oc_study = True
C.model_mode = "oc_rga"
C.num_heads_per_stage = (4, 4, 8, 16)
C.oc_geometry_cell = 4
C.oc_rank = 20
# Only stage 2 blocks are checkpointed, identically for both arms (design 8.3).
C.use_checkpoint = False
C.checkpoint_stages = (2,)

"""Probability-geometry constants (mirrored in research/oc_rga.py)."""
C.oc_tau = 4.0
C.oc_sigma_ref = 0.02
C.oc_ell_min = 0.15
C.oc_ell_max = 1.50
C.oc_beta_init = 0.10

"""Joint training objective schedule (design 10.5)."""
C.loss_lambda_cond_start = 0.01
C.loss_lambda_cond_end = 0.05
C.loss_lambda_obs_start = 0.002
C.loss_lambda_obs_end = 0.010
C.loss_lambda_ramp_epochs = 5.0
C.loss_prior_weight = 0.25
C.loss_huber_delta = 0.2
C.loss_holdout_targets = 256
C.loss_holdout_sparse_prob = 0.5
C.loss_holdout_sparse_low = 0.50
C.loss_holdout_sparse_high = 0.95

"""First-round shared degradation distribution (design 9.2), exactly one per crop."""
C.degrade_clean = 0.40
C.degrade_local_holes = 0.25
C.degrade_pixel_missing = 0.15
C.degrade_nonzero = 0.15
C.degrade_full_missing = 0.05
C.degrade_hole_ratio_low = 0.25
C.degrade_hole_ratio_high = 0.75
C.degrade_local_window_low = 0.10
C.degrade_local_window_high = 0.40
C.degrade_noise_std_low = 0.10
C.degrade_noise_std_high = 0.40
C.degrade_shift_area_low = 0.10
C.degrade_shift_area_high = 0.35
C.degrade_shift_offset_low = 0.20
C.degrade_shift_offset_high = 0.80
C.degrade_shift_outlier_rate = 0.10

"""Optimiser multipliers (design 11.1)."""
C.oc_geometry_lr_multiplier = 2.0

"""Schedule: 120 epochs, 5 warmup, 50 is a pause and observation point."""
C.optimizer = "AdamW"
C.lr = 8e-5
C.weight_decay = 0.01
C.lr_power = 0.9
C.schedule_epochs = 120
C.warm_up_epoch = 5
C.nepochs = C.schedule_epochs
C.effective_batch = 16
C.micro_batch = 4
C.accum_steps = 4
C.stage_epochs = (30, 40, 50, 80, 120)
C.stop_after_epoch = 0
C.save_every_epoch = True

"""Validation: official val, clean, single scale, the shared 640 protocol."""
C.val_every = 10
C.val_msf = False
C.val_flip = False
C.eval_long_side = 640
# Extra robustness conditions evaluated only at the pre-registered epochs.
C.robust_conditions_epoch40 = ("holes50", "noise020")
C.robust_conditions_epoch50 = (
    "clean", "holes25", "holes50", "holes75", "noise020", "localshift", "entire_missing",
)

"""Evaluation defaults."""
C.eval_ham_seed = 12345
C.eval_corruption_seed = 12345

"""Outputs: a single OC-RGA delivery root, one directory per mode/run."""
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
C.experiment_name = os.environ.get("DFORMER_EXPERIMENT_NAME", "oc-rga-deliver")
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

"""Data module: the OC-RGA protocol (image support / observation / reference)."""
C.data_module = "research.oc_rga_data"
