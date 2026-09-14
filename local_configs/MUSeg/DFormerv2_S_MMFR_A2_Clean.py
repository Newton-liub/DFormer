"""MMFR-A2 clean control training configuration.

Experiment family identity ``MMFR-A2-clean-control-v1``: the original DFormerv2-S
structure trained on clean inputs only. The reliability head and the corruption helper
stay disabled, so this config exercises exactly the pre-A2 module set and loss path and
differs from Quick-B0 only in run identity, output directory and A2 bookkeeping.

The frozen comparison rules (official pretrained, model seed, AdamW, learning rate,
500 epochs, batch size, scale augmentation, clean ``val-dev`` ``original-full``
selector) are inherited unchanged from ``DFormerv2_S_MMFR_A2_Common``.
"""

from .DFormerv2_S_MMFR_A2_Common import C, MMFR_A2_CLEAN_CONTROL_IDENTITY, configure_mmfr_identity

configure_mmfr_identity(
    identity="museg-dformerv2-s-mmfr-a2-clean-control-v1",
    mode="clean-control",
    analysis_identity=MMFR_A2_CLEAN_CONTROL_IDENTITY,
)
