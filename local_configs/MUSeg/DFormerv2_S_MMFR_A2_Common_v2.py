"""Shared MMFR-A2 **v2** train-integration configuration.

Both v2 identities (``MMFR-A2-clean-control-v2`` and
``MMFR-A2-depth-corruption-train-v2``) inherit this module, so the official pretrained
starting point, model seed, optimizer, 500-epoch schedule, scale augmentation, checkpoint
selector and input contract are identical by construction. The child configs only bind
the A2 mode and their own run id / output directory, so neither identity overwrites the
Quick-B0 run or the frozen v1 run.

This module is the v2 revision of ``DFormerv2_S_MMFR_A2_Common``. The v1 file is frozen
and is not imported, edited or aliased here: the two protocols stay separately
reproducible. Everything that the revision instruction does **not** ask to change (data
responsibilities, official pretrained, seed, optimizer, learning rate, 500 epochs,
warmup, batch size, workers, scale augmentation, clean ``val-dev`` selector, cloud
execution profile, price rule) is copied unchanged from the v1 common file, so the two
identities remain a fair pair and the two protocol revisions remain comparable.

v2 changes carried by this config
---------------------------------
1. corruption basis is ``MMFR-A1-corruption-basis-v2``: severity is encoded exactly once
   (it selects the corruption parameter; the continuous burden is the realized normalized
   damage), and ``blur``/``misalignment`` severities are fractions of the actual grid
   rather than absolute pixel counts;
2. the supervised Depth surrogate is
   ``R_depth_sup(p) = depth_valid_pre(p) * R_depth_synthetic(p)``, so native MUSeg Depth
   invalidity is part of the target;
3. the reliability auxiliary loss scores **Depth only**; the RGB channel of the two-channel
   head output is an all-ones scaffold channel that is not scored and must not be reported
   as a trained RGB reliability estimator;
4. the batch helper exposes ``depth_valid_pre`` and ``depth_valid_post`` instead of the
   ambiguous single ``depth_valid``, and records ``natural_invalid_pixels``,
   ``synthetic_missing_pixels`` and ``post_corruption_invalid_pixels`` separately.
"""

import os
import os.path as osp

import numpy as np

from .DFormerv2_S_MVE import C


_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
_DEFAULT_DATA_ROOT = osp.join(osp.dirname(_PROJECT_ROOT), "dataset")
_DATA_ROOT = osp.abspath(os.environ.get("DFORMER_DATA_ROOT", _DEFAULT_DATA_ROOT))
_OUTPUT_ROOT = osp.abspath(os.environ.get("DFORMER_OUTPUT_ROOT", osp.join(_PROJECT_ROOT, "outputs")))

#: Frozen v2 protocol identity and its two analysis identities.
MMFR_A2_PROTOCOL = "MMFR-A2-train-integration-v2"
MMFR_A2_SUPERSEDES = "MMFR-A2-train-integration-v1"
MMFR_A2_SUPERSEDED_REASON = (
    "protocol revision before any formal 500-epoch run: severity double encoding, "
    "resolution-dependent blur/misalignment severity, an unqualified reliability target "
    "for natively invalid Depth and an unscored-but-scored RGB reliability channel"
)
MMFR_A2_SCHEDULE_VERSION = "mmfr-a2-v2-adamw-6e-5-warmup10-poly0.9-500e-v1"
MMFR_A2_CLEAN_CONTROL_IDENTITY = "MMFR-A2-clean-control-v2"
MMFR_A2_DEPTH_CORRUPTION_IDENTITY = "MMFR-A2-depth-corruption-train-v2"
MMFR_A2_CORRUPTION_BASIS = "MMFR-A1-corruption-basis-v2"
MMFR_A2_TARGET_COMPOSITION = "R_depth_sup(p) = depth_valid_pre(p) * R_depth_synthetic(p)"
MMFR_A2_SUPERVISED_CHANNELS = ["depth"]
#: The revised audit narrative that freezes this revision.
MMFR_A2_PROTOCOL_NARRATIVE = "liu-test-exp/方案1/MMFR-多形式模态失效可靠性学习-独立审计总规划.md"
MMFR_A2_PROTOCOL_TEMPLATE = "protocols/mmfr-a2-train-integration-v2.template.json"
#: The v1 stage document still describes the v1 protocol and has not been revised for v2.
MMFR_A2_STAGE_PLAN_DOCUMENT = "doc/plans/2026-09-MUSeg-多形式模态失效可靠性学习/03-MMFR-A2训练接入与公平对照协议.md"
MMFR_A2_STAGE_PLAN_DOCUMENT_STATUS = "v1-only-not-yet-revised-for-v2"

#: Frozen corruption record for v2. ``seed`` is the training corruption seed, deliberately
#: independent of the model seed ``C.seed``.
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
    "invalidity_populations": [
        "natural_invalid_pixels",
        "synthetic_missing_pixels",
        "post_corruption_invalid_pixels",
    ],
}

#: Frozen normalization record: RGB from the config contract, Depth from
#: ``TrainPre(sign=True)``. Unchanged from v1.
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

#: Frozen auxiliary loss record for v2.
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
}

#: Formal training is cloud-first. Unchanged from v1: the local GPU is reserved for
#: inference and small qualification probes and must not silently redefine the training
#: batch, learning rate or schedule.
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


# ---------------------------------------------------------------------------
# Data responsibilities (frozen: train-dev 1277 / val-dev 318, official test sealed)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Starting point, optimizer, schedule and input contract (identical for both identities)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Checkpoint selector and evaluation cadence. Frozen as clean ``val-dev``,
# ``original-full`` geometry, scale ``1.0``, no flip, ``earlier_epoch`` tie-break;
# failure conditions never take part in epoch selection.
# ---------------------------------------------------------------------------
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

#: Training-time validation follows the selector geometry only; the frozen
#: ``msflip-whole-original-grid-v1`` development evaluation is a separate step and is
#: not implemented by this configuration.
C.mst = False
C.sliding = False
C.mmfr_a2_execution_profile = dict(MMFR_A2_EXECUTION_PROFILE)

#: MMFR-A2 v2 block, disabled until a child config binds one identity.
C.mmfr_a2 = {
    "protocol": MMFR_A2_PROTOCOL,
    "supersedes": MMFR_A2_SUPERSEDES,
    "mode": "clean-control",
    "corruption": None,
    "reliability_head": None,
    "lambda_consistency": 0.0,
}


def configure_mmfr_identity_v2(
    identity,
    mode,
    analysis_identity,
    corruption=None,
    reliability_head=None,
):
    """Bind one frozen v2 identity to its own run id, output tree and MMFR mode.

    ``identity`` is the run id, ``analysis_identity`` the experiment-family name used in
    reports. Both v2 identities must keep the selector, seed, optimizer and data
    responsibilities inherited above; this helper only owns the run identity, the output
    paths and the MMFR v2 switch. Nothing here touches the Quick-B0 or v1 configs.
    """
    if mode not in ("clean-control", "depth-corruption"):
        raise ValueError(f"unsupported MMFR-A2 mode {mode!r}")
    if mode == "depth-corruption":
        if not corruption or not reliability_head:
            raise ValueError("depth-corruption requires both corruption and reliability_head")
        if str(corruption.get("basis")) != MMFR_A2_CORRUPTION_BASIS:
            raise ValueError(f"depth-corruption corruption basis must be {MMFR_A2_CORRUPTION_BASIS}")
        if str(corruption.get("target_composition")) != MMFR_A2_TARGET_COMPOSITION:
            raise ValueError("depth-corruption must freeze the v2 supervised target composition")
        if str(corruption.get("severity_encoding")) != "single":
            raise ValueError("depth-corruption must encode severity exactly once")
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
        },
    }
    return C
