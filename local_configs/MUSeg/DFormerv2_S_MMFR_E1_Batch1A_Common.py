"""Shared frozen configuration for MMFR E1 Batch 1A C0/F-lite Gate-B and training.

Importing this module does not authorize training.  It binds the already-frozen
20-epoch/2560-update contract so the implementation can be qualified before a
separate run authorization is granted.
"""

import os.path as osp

from .DFormerv2_S_MMFR_A2_DepthCorrupt_v3 import C

E1_BATCH1A_PROTOCOL = "MMFR-E1-Batch1A-C0-F-v1"
E1_SOURCE_CHECKPOINT_SHA256 = "2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597"
E1_SOURCE_MODEL_KEY_COUNT = 812
E1_SOURCE_CHECKPOINT = osp.join(
    osp.abspath(osp.join(osp.dirname(__file__), "..", "..")),
    "experiments",
    "MMFR_A2_v3",
    "checkpoints",
    "selector-epoch-420.pth",
)
E1_FEATURE_ADAPTER = {
    "stages": [1, 2, 3],
    "bottleneck_ratio": 4,
    "normalization": "none",
    "activation": "GELU",
    "down_init": "trunc_normal_std_0.02",
    "up_init": "zeros",
    "uses_reliability": False,
    "uses_condition": False,
    "uses_severity": False,
    "uses_oracle": False,
    "expected_trainable_parameters": 173152,
}

# Frozen common Batch 1A schedule. The training entry point keeps the new-module
# groups at three times the base LR through each warmup/poly update.
C.nepochs = 20
C.niters_per_epoch = 128
C.lr = 1e-5
C.lr_power = 0.9
C.warm_up_epoch = 1
C.weight_decay = 0.01
C.batch_size = 10
C.num_workers = 8
C.eval_start_epoch = C.nepochs
C.eval_interval = C.nepochs
C.save_interval = 5
C.save_epoch_checkpoints = True
C.save_latest_checkpoint = True
C.checkpoint_retention_policy = {}
C.training_validation_enabled = False

# ``mmfr_a2.frozen`` was serialized by the A2-v3 source config before the E1
# schedule overrides above. Keep it as historical source metadata, but label its
# scope explicitly so a run manifest cannot mistake the inherited 500-epoch /
# 6e-5 values for the active E1 contract. ``e1_batch1`` is authoritative here.
C.mmfr_a2 = dict(C.mmfr_a2)
C.mmfr_a2["frozen"] = dict(C.mmfr_a2.get("frozen") or {})
C.mmfr_a2["frozen"]["metadata_scope"] = "source-a2-v3-history-only"
C.mmfr_a2["frozen"]["active_schedule_authority"] = "e1_batch1"


def configure_e1_batch1a_candidate(candidate: str):
    if candidate not in ("C0", "F-lite"):
        raise ValueError(f"unsupported E1 Batch 1A candidate {candidate!r}")
    feature_adapter = dict(E1_FEATURE_ADAPTER) if candidate == "F-lite" else None
    C.run_id = f"MMFR-E1-Batch1A-{candidate}"
    C.e1_batch1 = {
        "enabled": True,
        "protocol": E1_BATCH1A_PROTOCOL,
        "candidate": candidate,
        "source_checkpoint": osp.abspath(E1_SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": E1_SOURCE_CHECKPOINT_SHA256,
        "source_model_key_count": E1_SOURCE_MODEL_KEY_COUNT,
        "weights_only_restart": True,
        "restore_optimizer": False,
        "restore_scheduler": False,
        "restore_grad_scaler": False,
        "restore_rng": False,
        "optimizer_groups": [
            "base_decay",
            "base_no_decay",
            "new_decay",
            "new_no_decay",
        ],
        "base_lr": 1e-5,
        "new_module_lr": 3e-5,
        "weight_decay": 0.01,
        "warmup_successful_updates": 128,
        "poly_power": 0.9,
        "required_successful_updates": 2560,
        "grad_scaler": {
            "initial_scale": 1024.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
        },
        "reject_skipped_optimizer_step": True,
        "curriculum_progress_start": 0.84,
        "curriculum_progress_end": 0.88,
        "recovery_interval_successful_updates": 640,
        "fixed_final_checkpoint": "update-2560.pth",
        "training_validation_enabled": False,
        "feature_adapter": feature_adapter,
        "official_test": "sealed_unread",
    }
    return C
