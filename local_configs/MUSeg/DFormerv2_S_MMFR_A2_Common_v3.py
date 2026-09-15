"""Shared MMFR-A2 **v3** train-integration configuration.

The v3 identities are copied from the audited v2 scientific hyperparameters but use a
new protocol and a new explicit validity-state semantic.  v2 files and evidence remain
unchanged.  A2 v3 fixes the validity transport decision as A + MID-A: intensity
corruptions act only on currently valid Depth measurements, and misalignment translates
Depth values and validity together with one integer shift.
"""

import os
import os.path as osp

import numpy as np

from .DFormerv2_S_MVE import C

_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
_DEFAULT_DATA_ROOT = osp.join(osp.dirname(_PROJECT_ROOT), "dataset")
_DATA_ROOT = osp.abspath(os.environ.get("DFORMER_DATA_ROOT", _DEFAULT_DATA_ROOT))
_OUTPUT_ROOT = osp.abspath(os.environ.get("DFORMER_OUTPUT_ROOT", osp.join(_PROJECT_ROOT, "outputs")))

MMFR_A2_PROTOCOL = "MMFR-A2-train-integration-v3"
MMFR_A2_SUPERSEDES = "MMFR-A2-train-integration-v2"
MMFR_A2_SUPERSEDED_REASON = (
    "new v3 identity after the audited A + MID-A decision: explicit sequential validity "
    "state, validity-preserving intensity corruption and integer Depth/state transport"
)
MMFR_A2_SCHEDULE_VERSION = "mmfr-a2-v3-adamw-6e-5-warmup10-poly0.9-500e-v1"
MMFR_A2_CLEAN_CONTROL_IDENTITY = "MMFR-A2-clean-control-v3"
MMFR_A2_DEPTH_CORRUPTION_IDENTITY = "MMFR-A2-depth-corruption-train-v3"
MMFR_A2_CORRUPTION_BASIS = "MMFR-A1-corruption-basis-v3"
MMFR_A2_TARGET_COMPOSITION = "R_depth_sup_v3(p) = V_state_final(p) * R_depth_synthetic(p)"
MMFR_A2_SUPERVISED_CHANNELS = ["depth"]
MMFR_A2_PROTOCOL_NARRATIVE = "liu-test-exp/方案1/MMFR-多形式模态失效可靠性学习-独立审计总规划.md"
MMFR_A2_NARRATIVE_STATE = "v3-audited-A-plus-MID-A"
MMFR_A2_PROTOCOL_TEMPLATE = "protocols/mmfr-a2-train-integration-v3.template.json"
MMFR_A2_STAGE_PLAN_DOCUMENT = "doc/plans/2026-09-MUSeg-多形式模态失效可靠性学习/03-MMFR-A2训练接入与公平对照协议.md"
MMFR_A2_STAGE_PLAN_DOCUMENT_STATUS = "v3-only-identity-derived-from-audited-A-plus-MID-A"

MMFR_A2_CORRUPTION = {
    "basis": MMFR_A2_CORRUPTION_BASIS,
    "seed": 2026091402,
    "p_clean": 0.25,
    "max_specs": 2,
    "modalities": ["depth"],
    "kinds": [
        "entire_missing",
        "spatial_dropout",
        "gaussian_noise",
        "blur",
        "quantization",
        "misalignment",
    ],
    "target_composition": MMFR_A2_TARGET_COMPOSITION,
    "severity_encoding": "single",
    "relative_scale_spatial_corruptions": True,
    "blur_sigma_fraction_of_min_side_at_severity_1": 1.0 / 80.0,
    "misalignment_max_shift_fraction_of_axis_at_severity_1": 1.0 / 30.0,
    "noise_sigma_uint8_at_severity_1": 48.0,
    "quantization_min_levels_at_severity_1": 2,
    "application_point": "post-mirror-scale-crop-pad-pre-gpu-main-process",
    "rng_engine": "numpy-PCG64-SeedSequence-per-sample",
    "seed_words": [
        "train_seed",
        "epoch_1_based",
        "iteration_0_based",
        "global_rank",
        "sample_slot_0_based",
        "sample_id_sha256_u32_be_0",
        "sample_id_sha256_u32_be_1",
        "sample_id_sha256_u32_be_2",
        "sample_id_sha256_u32_be_3",
    ],
    "curriculum_progress": "((epoch-1)*N_iter+iteration)/(N_epoch*N_iter-1) clipped to [0,1]",
    "clean_sample_semantics": "original normalized tensors reused without reconstruction",
    "pad_semantics": "RGB and Depth both all-channel exact zero; corrupted pads restored to exact 0",
    "validity_state_initial": "depth_valid_pre = (raw_depth_pre_uint8 > 0) & valid_mask",
    "validity_state_updates": {
        "gaussian_noise": "unchanged; operate only where V_state=1",
        "blur": "unchanged; mask-normalized G*(D*V)/(G*V+epsilon), output only where V_state=1",
        "quantization": "unchanged; operate only where V_state=1",
        "spatial_dropout": "V_state &= keep_mask",
        "entire_missing": "V_state = 0",
        "misalignment": "translate Depth and V_state with the same integer dy/dx, then apply valid_mask",
    },
    "misalignment_validity_transport": "MID-A",
    "final_validity_contract": "depth_valid_post == V_state_final == (raw_depth_uint8 > 0) & valid_mask",
    "valid_depth_uint8_contract": "V_state=1 implies uint8 in [1,255]; V_state=0 implies uint8=0",
    "telemetry_categories": [
        "natural-invalid",
        "synthetic-invalid",
        "implicit-quality",
        "valid-clean",
    ],
}

MMFR_A2_NORMALIZATION = {
    "rgb_mean": [0.485, 0.456, 0.406],
    "rgb_std": [0.229, 0.224, 0.225],
    "depth_mean": [0.48, 0.48, 0.48],
    "depth_std": [0.28, 0.28, 0.28],
    "raw_signal_range": [0.0, 1.0],
    "padding_normalized_value": 0.0,
    "clean_path_exact_noop": True,
    "round_trip": "exhaustive uint8 0..255 normalize->float32->inverse round qualification",
}

MMFR_A2_LOSS = {
    "segmentation": "safe_masked_mean(cross_entropy(input, label))",
    "lambda_reliability": 0.1,
    "lambda_consistency": 0.0,
    "total": "seg_loss + 0.1 * continuous_bce_loss(supervised_channels only)",
    "single_forward_per_sample": True,
    "clean_teacher": False,
    "clean_corrupt_consistency": False,
    "geometry_adapter_enabled": False,
    "reliability_head_feeds_backbone": False,
    "reliability_head_feeds_geometry_prior": False,
    "supervised_channels": list(MMFR_A2_SUPERVISED_CHANNELS),
    "rgb_channel_role": "all-ones scaffold channel: unscored, no calibration metric, not a trained estimator",
    "target_composition": MMFR_A2_TARGET_COMPOSITION,
    "invalidity_populations_kept_separate": True,
    "optimizer_semantics": "inherited-upstream-optimizer-semantics",
}

MMFR_A2_EXECUTION_PROFILE = {
    "formal_training_environment": "cloud-single-gpu",
    "local_machine_role": "inference-and-small-preflight-only",
    "primary_gpu": "NVIDIA GeForce RTX 4090 24GB",
    "fallback_gpu": "NVIDIA GeForce RTX 5090 32GB",
    "price_snapshot_cny_per_hour": {"rtx4090": 1.88, "rtx5090": 2.78},
    "rtx5090_to_rtx4090_price_ratio": 2.78 / 1.88,
    "rtx5090_break_even_throughput_ratio": 2.78 / 1.88,
    "selection_rule": (
        "prefer RTX 4090 when available and the frozen batch fits; use RTX 5090 only "
        "when 4090 is unavailable, 24GB is insufficient, or a paired cloud probe "
        "measures throughput ratio above the price ratio"
    ),
    "frozen_global_batch_size": 10,
    "num_workers": 8,
    "amp": True,
    "syncbn": True,
    "torch_compile_initially_enabled": False,
    "hardware_must_not_change_scientific_hyperparameters": True,
}

# Data responsibilities remain the frozen train-dev/val-dev development roles.
C.root_dir = _DATA_ROOT
C.dataset_path = osp.join(C.root_dir, "MUSeg_DFormer")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.split_root = osp.join(_PROJECT_ROOT, "data", "splits", "MUSeg", "dev-v1")
C.experiment_phase = "development"
C.train_source = osp.join(C.split_root, "train-dev.txt")
C.val_source = osp.join(C.split_root, "val-dev.txt")
C.test_source = osp.join(C.split_root, "official-test.txt")
C.eval_source = C.val_source
C.expected_split_sha256 = {
    "train": "a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470",
    "val": "1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83",
    "test": "12d9834215fcbfe696ad88321539c224850ff6fb66a01f48a02b1df478f48a4b",
}
C.expected_split_samples = {"train": 1277, "val": 318, "test": 1576}
C.num_train_imgs = 1277
C.num_eval_imgs = 318

# Starting point and all scientific hyperparameters are unchanged from v2.
C.pretrained_model = osp.abspath(
    os.environ.get(
        "DFORMER_PRETRAINED",
        osp.join(osp.dirname(_PROJECT_ROOT), "pretrained", "DFormerv2_Small_pretrained.pth"),
    )
)
C.optimizer = "AdamW"
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 10
C.val_batch_size = 1
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10
C.seed = 772961337
C.channel_order = "RGB"
C.normalization_identity = "rgb-imagenet-rgb-order-v1"
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

C.eval_start_epoch = 10
C.eval_interval = 10
C.save_interval = 10
C.checkpoint_start_epoch = C.nepochs
C.checkpoint_step = C.save_interval
C.save_epoch_checkpoints = False
C.save_latest_checkpoint = True
C.checkpoint_candidate_manifest = "checkpoint-candidates.json"
C.checkpoint_retention_policy = {
    "selector_geometry": "original-full",
    "selector_scale": 1.0,
    "selector_flip": False,
    "top_k": 3,
    "retain_latest": True,
    "tie_break": "earlier_epoch",
}
C.optimizer_telemetry_policy = {
    "schema_version": "museg-optimizer-telemetry-v1",
    "required_counters": ["attempted_steps", "completed_optimizer_steps", "skipped_optimizer_steps"],
    "invariant": "attempted_steps=completed_optimizer_steps+skipped_optimizer_steps",
}
C.mst = False
C.sliding = False
C.mmfr_a2_execution_profile = dict(MMFR_A2_EXECUTION_PROFILE)
C.mmfr_a2 = {
    "protocol": MMFR_A2_PROTOCOL,
    "supersedes": MMFR_A2_SUPERSEDES,
    "mode": "clean-control",
    "corruption": None,
    "reliability_head": None,
    "lambda_consistency": 0.0,
}


def configure_mmfr_identity_v3(
    identity,
    mode,
    analysis_identity,
    corruption=None,
    reliability_head=None,
):
    """Bind one frozen v3 identity without changing the inherited hyperparameters."""
    if mode not in ("clean-control", "depth-corruption"):
        raise ValueError(f"unsupported MMFR-A2 mode {mode!r}")
    if mode == "depth-corruption":
        if not corruption or not reliability_head:
            raise ValueError("depth-corruption requires both corruption and reliability_head")
        if str(corruption.get("basis")) != MMFR_A2_CORRUPTION_BASIS:
            raise ValueError(f"depth-corruption corruption basis must be {MMFR_A2_CORRUPTION_BASIS}")
        if str(corruption.get("target_composition")) != MMFR_A2_TARGET_COMPOSITION:
            raise ValueError("depth-corruption must freeze the v3 supervised target composition")
        if str(corruption.get("severity_encoding")) != "single":
            raise ValueError("depth-corruption must encode severity exactly once")
        if str(corruption.get("misalignment_validity_transport")) != "MID-A":
            raise ValueError("depth-corruption must use MID-A validity transport")
        if list(reliability_head.get("supervised_channels") or ()) != list(MMFR_A2_SUPERVISED_CHANNELS):
            raise ValueError("depth-corruption reliability supervision must be Depth-only")
    elif corruption or reliability_head:
        raise ValueError("clean-control must not enable corruption or the reliability head")

    C.run_id = str(identity)
    C.log_dir = osp.join(_OUTPUT_ROOT, C.run_id, C.experiment_phase, f"seed-{C.seed}")
    C.tb_dir = osp.join(C.log_dir, "tb")
    C.log_dir_link = C.log_dir
    C.checkpoint_dir = osp.join(C.log_dir, "checkpoint")
    C.log_file = osp.join(C.log_dir, "train.log")
    C.link_log_file = osp.join(C.log_dir, "log_last.log")
    C.val_log_file = osp.join(C.log_dir, "val.log")
    C.link_val_log_file = osp.join(C.log_dir, "val_last.log")
    C.mmfr_a2 = {
        "protocol": MMFR_A2_PROTOCOL,
        "supersedes": MMFR_A2_SUPERSEDES,
        "superseded_reason": MMFR_A2_SUPERSEDED_REASON,
        "mode": mode,
        "corruption": dict(corruption) if corruption else None,
        "reliability_head": dict(reliability_head) if reliability_head else None,
        "lambda_consistency": MMFR_A2_LOSS["lambda_consistency"],
        "frozen": {
            "protocol_narrative": MMFR_A2_PROTOCOL_NARRATIVE,
            "narrative_state": MMFR_A2_NARRATIVE_STATE,
            "protocol_template": MMFR_A2_PROTOCOL_TEMPLATE,
            "stage_plan_document": MMFR_A2_STAGE_PLAN_DOCUMENT,
            "stage_plan_document_status": MMFR_A2_STAGE_PLAN_DOCUMENT_STATUS,
            "schedule_version": MMFR_A2_SCHEDULE_VERSION,
            "analysis_identity": analysis_identity,
            "run_id": C.run_id,
            "mode": mode,
            "seed": C.seed,
            "epochs": C.nepochs,
            "iterations_per_epoch": C.niters_per_epoch,
            "optimizer": C.optimizer,
            "base_lr": C.lr,
            "poly_power": C.lr_power,
            "weight_decay": C.weight_decay,
            "batch_size": C.batch_size,
            "warmup_epochs": C.warm_up_epoch,
            "train_scale_array": list(C.train_scale_array),
            "data": {
                "train_role": "train-dev",
                "train_samples": C.expected_split_samples["train"],
                "val_role": "val-dev",
                "val_samples": C.expected_split_samples["val"],
                "official_test": "sealed_unread",
            },
            "selector": dict(C.checkpoint_retention_policy),
            "corruption": dict(corruption) if corruption else None,
            "reliability_head": dict(reliability_head) if reliability_head else None,
            "loss": dict(MMFR_A2_LOSS),
            "normalization": dict(MMFR_A2_NORMALIZATION),
            "execution_profile": dict(MMFR_A2_EXECUTION_PROFILE),
            "optimizer_semantics": "inherited-upstream-optimizer-semantics",
        },
    }
    return C
