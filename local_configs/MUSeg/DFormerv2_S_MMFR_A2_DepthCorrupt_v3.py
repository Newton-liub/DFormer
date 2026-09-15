"""MMFR-A2 v3 Depth corruption training configuration.

The v3 model trains on the unchanged six-kind Depth corruption distribution but uses the
explicit sequential validity state.  The supervised target is
``V_state_final * R_depth_synthetic``; noise, blur and quantization never act on invalid
measurements, and MID-A transports Depth and validity with one integer translation.
The reliability head remains an auxiliary Depth-only estimator and never feeds the
backbone, decoder or geometry prior.
"""

from .DFormerv2_S_MMFR_A2_Common_v3 import (
    C,
    MMFR_A2_CORRUPTION,
    MMFR_A2_DEPTH_CORRUPTION_IDENTITY,
    MMFR_A2_SUPERVISED_CHANNELS,
    MMFR_A2_TARGET_COMPOSITION,
    configure_mmfr_identity_v3,
)

_MMFR_A2_RELIABILITY_HEAD = {
    "enabled": True,
    "hidden_channels": 16,
    "weight": 0.1,
    "supervised_channels": list(MMFR_A2_SUPERVISED_CHANNELS),
    "rgb_channel_role": (
        "all-ones scaffold channel kept for the future two-channel interface: not scored, "
        "no calibration metric reported, not a trained RGB reliability estimator"
    ),
    "target_composition": MMFR_A2_TARGET_COMPOSITION,
    "validity_state_semantics": "explicit sequential V_state_final",
    "misalignment_validity_transport": "MID-A",
    "supervision": "continuous_bce_loss(logits, reliability_target, valid_mask * depth-channel mask)",
    "module_initialization": "modal_reliability module defaults; init_weight only touches the segmentation heads",
    "feeds_backbone": False,
    "feeds_geometry_prior": False,
    "geometry_adapter_enabled": False,
    "decoder_or_backbone_modified": False,
    "checkpoint_semantics": "strict save/restore of the complete model and optimizer state",
}

configure_mmfr_identity_v3(
    identity="museg-dformerv2-s-mmfr-a2-depth-corruption-train-v3",
    mode="depth-corruption",
    analysis_identity=MMFR_A2_DEPTH_CORRUPTION_IDENTITY,
    corruption=dict(MMFR_A2_CORRUPTION),
    reliability_head=dict(_MMFR_A2_RELIABILITY_HEAD),
)
