"""Frozen A-v1 contract; GPU execution requires the explicit runner command.

2026-10-01: preflight, conditional formal training and one Quick-Val authorized.
The current no-GPU preparation session does not execute that authorization.
"""

import os
from copy import deepcopy
from pathlib import Path

from .DFormerv2_S_MMFR_E1_Batch1A_C0 import C as SOURCE_C0

C = deepcopy(SOURCE_C0)
_ROOT = Path(__file__).resolve().parents[2]
C.run_id = "MMFR-A-v1-action-utility-v1"
C.e1_batch1 = None
C.pretrained_model = None  # The complete verified C0 model is loaded weights-only.
C.mmfr_av1 = {
    "enabled": True,
    "protocol": C.run_id,
    "protocol_document": "MMFR/01_research/mmfr_a_v1_action_utility_protocol.md",
    "source_config": "local_configs.MUSeg.DFormerv2_S_MMFR_E1_Batch1A_C0",
    "source_checkpoint": os.environ.get("MMFR_AV1_SOURCE_CHECKPOINT", str(
        _ROOT / "cloud" / "MMFR_E1_Batch1A_local_transfer_20260922"
        / "MMFR_E1_Batch1A_local_transfer_20260922" / "C0"
        / "checkpoint" / "update-2560.pth")),
    "source_checkpoint_sha256": "ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a",
    "initialization_seed": 2026093000,
    "phase_corruption_seeds": {"proposal": 2026093001, "gate": 2026093002},
    "run_name": "MMFR-A-v1-action-utility-v1-formal-v1",
    "phase_successful_updates": {"proposal": 1920, "gate": 640},
    "required_successful_updates": 2560,
    "margin": 0.01,  # Per-image mean CE difference; pre-frozen, not tuned.
    "lambda_clean": 0.1,  # Same protection weight in both phases; not tuned.
    "optimizer": "AdamW",
    "new_module_lr": 3e-5,
    "weight_decay": 0.01,
    "no_decay": "existing build_e1_optimizer_param_groups bias/norm rules",
    "schedule": {
        "unit": "successful_optimizer_update",
        "reset_per_phase": True,
        "warmup_successful_updates": 128,
        "poly_power": 0.9,
        "lr_rule": "lr*k/128 if k<=128 else lr*((N-k)/(N-128))**0.9; k=1..N",
        "validation_selection": False,
    },
    # Data/corruption coordinates only: these are not an epoch-extension policy.
    "phase_data_schedule": {
        "proposal": {"nepochs": 15, "niters_per_epoch": 128},
        "gate": {"nepochs": 5, "niters_per_epoch": 128},
    },
    "curriculum": "existing v3 phase-local progress 0..1; no E1 0.84..0.88 remap",
    "train_seed": 772961337,
    "p_clean": 0.25,
    "batch_size": 10,
    "num_workers": 8,
    "accumulation_steps": 1,
    "train_resolution": [480, 640],
    "amp": True,
    "amp_dtype": "float16",
    "tf32_matmul": True,
    "tf32_cudnn": True,
    "float32_matmul_precision": "high",
    "grad_scaler": {
        "initial_scale": 1024.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
    },
    "reject_skipped_optimizer_step": True,
    "checkpoint": {
        "transition": "proposal-update-1920.pth",
        "fixed_final": "update-2560.pth",
        "recovery_interval_successful_updates": 640,
        "recovery_state": "model/optimizer/scheduler/GradScaler/RNG/data cursor/phase/counters",
        "phase_transition": "same Proposal weights; fresh Gate optimizer/scheduler/GradScaler; isolated phase seed",
        "source_restart": "weights only; no source optimizer/scheduler/GradScaler/RNG",
        "validation_selector": False,
    },
    "quickval_contract": {
        "conditions": ["clean", "entire_missing@1.0", "spatial_dropout@0.75", "misalignment@0.75"],
        "behaviors": ["off", "full", "learned"],
        "checkpoint": "update-2560.pth",
        "max_rounds": 1,
        "learned_hard_delta_pp_min": 0.50,
        "learned_clean_delta_pp_min": -0.20,
        "learned_hard_must_exceed_full": True,
        "authorized": True,
    },
    "cloud_preflight": {
        "device": "RTX 4090",
        "phases": ["proposal", "gate"],
        "successful_updates_per_phase_max": 3,
        "stop_after_each_phase": True,
        "gate_requires_proposal_pass": True,
        "auto_start_formal_training": True,
        "preflight_updates_count_toward_formal_budget": False,
        "execution_authorized_this_round": True,
    },
    "formal_training_authorized": True,
    "validation_enabled": False,
    "official_test": "sealed_unread",
}
# Pin execution fields rather than silently inheriting a legacy run contract.
C.optimizer = "AdamW"
C.lr = 3e-5
C.lr_power = 0.9
C.warm_up_epoch = None  # Update-index schedule above is authoritative.
C.weight_decay = 0.01
C.batch_size = 10
C.num_workers = 8
C.accumulation_steps = 1
C.image_height = 480
C.image_width = 640
C.seed = 772961337
C.amp = True
C.tf32_matmul = True
C.tf32_cudnn = True
C.training_validation_enabled = False
C.mmfr_a2["frozen"]["active_schedule_authority"] = "mmfr_av1"
# Retain the audited corruption algorithm and C0 auxiliary keys, not its losses.
# build_phase_batch overrides the inherited corruption seed per A-v1 phase.
C.mmfr_a2["corruption"]["p_clean"] = C.mmfr_av1["p_clean"]

# Avoid inherited E1 output/checkpoint identities, schedules and selector paths.
C.log_dir = str(_ROOT / "outputs" / C.mmfr_av1["run_name"])
C.checkpoint_dir = str(Path(C.log_dir) / "checkpoint")
C.log_dir_link = C.log_dir
C.tb_dir = str(Path(C.log_dir) / "tb")
C.log_file = str(Path(C.log_dir) / "train.log")
C.link_log_file = C.log_file
C.val_log_file = None
C.link_val_log_file = None
C.eval_source = None
C.val_source = None
C.test_source = None
C.nepochs = None
C.niters_per_epoch = None
C.eval_start_epoch = None
C.eval_interval = None
C.save_epoch_checkpoints = False
C.save_latest_checkpoint = False
C.checkpoint_retention_policy = None
C.checkpoint_candidate_manifest = None
