"""MMFR E1 Batch 1B R-OE-lite candidate configuration."""

from .DFormerv2_S_MMFR_E1_Batch1A_Common import (
    C,
    E1_SOURCE_CHECKPOINT,
    E1_SOURCE_CHECKPOINT_SHA256,
    E1_SOURCE_MODEL_KEY_COUNT,
)

E1_BATCH1B_PROTOCOL = "MMFR-E1-Batch1B-R-OE-lite-v2"
E1_ROE_SUBSTITUTE = {
    "architecture": "observable-empty-geometry-substitute",
    "channels": [122, 398, 256, 256, 398, 122, 1],
    "pool": "avgpool2d-kernel2-stride2-ceil-mode-true",
    "activation": "GELU",
    "resize": "bilinear-align-corners-false",
    "normalization": "none",
    "skip_connections": False,
    "output": "straight-through-clamp-0-255",
    "head_bias": 127.5,
    "uses_depth": False,
    "uses_reliability": False,
    "uses_condition": False,
    "uses_severity": False,
    "uses_oracle": False,
    "expected_trainable_parameters": 3302785,
}

C.run_id = "MMFR-E1-Batch1B-R-OE-lite-v2"
C.e1_batch1 = {
    "enabled": True,
    "protocol": E1_BATCH1B_PROTOCOL,
    "candidate": "R-OE-lite",
    "matched_control": "MMFR-E1-Batch1A-C0",
    "source_checkpoint": E1_SOURCE_CHECKPOINT,
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
    "roe_substitute": dict(E1_ROE_SUBSTITUTE),
    "official_test": "sealed_unread",
    "tf32_training_behavior": "batch-1a-actual-preserved",
}
