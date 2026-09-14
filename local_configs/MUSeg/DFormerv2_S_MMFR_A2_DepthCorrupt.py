"""MMFR-A2 Depth corruption training configuration.

Experiment family identity ``MMFR-A2-depth-corruption-train-v1``: DFormerv2-S trained
on the frozen Depth corruption distribution with the auxiliary reliability loss

$$
\\mathcal L = \\mathcal L_{seg}^{input} + 0.1\\,\\mathcal L_{rel},
\\qquad \\lambda_{cons} = 0.
$$

Everything below is the frozen A2 protocol: corruption applied in the main training
process after the DataLoader mirror/scale/crop/pad and before the batch reaches the
GPU, a stateless per-sample ``PCG64(SeedSequence(words))`` generator, ``p_clean=0.25``,
Depth-only specs with ``max_specs=2`` over the six A1 failure kinds, one segmentation
forward per sample, and a reliability head that only carries auxiliary supervision and
never enters the backbone or the geometry prior. ``frozen`` therefore records every
active parameter of this run, including the inherited selector, seed and schedule.

The official pretrained checkpoint, model seed, optimizer, learning rate, 500 epochs,
batch size, scale augmentation, clean ``val-dev`` ``original-full`` selector and input
contract are inherited unchanged from ``DFormerv2_S_MMFR_A2_Common``.
"""

from .DFormerv2_S_MMFR_A2_Common import (
    C,
    MMFR_A2_CORRUPTION,
    MMFR_A2_DEPTH_CORRUPTION_IDENTITY,
    configure_mmfr_identity,
)


_MMFR_A2_RELIABILITY_HEAD = {
    "enabled": True,
    "hidden_channels": 16,
    "weight": 0.1,
    "supervision": "continuous_bce_loss(logits, reliability_target, valid_mask)",
    "module_initialization": "modal_reliability module defaults; init_weight only touches the segmentation heads",
    "feeds_backbone": False,
    "feeds_geometry_prior": False,
    "geometry_adapter_enabled": False,
    "decoder_or_backbone_modified": False,
    "checkpoint_semantics": "strict save/restore of the complete model and optimizer state",
}

configure_mmfr_identity(
    identity="museg-dformerv2-s-mmfr-a2-depth-corruption-v1",
    mode="depth-corruption",
    analysis_identity=MMFR_A2_DEPTH_CORRUPTION_IDENTITY,
    corruption=dict(MMFR_A2_CORRUPTION),
    reliability_head=dict(_MMFR_A2_RELIABILITY_HEAD),
)
