"""MMFR-A2 v2 clean control training configuration.

Experiment family identity ``MMFR-A2-clean-control-v2``: the original DFormerv2-S
structure trained on clean inputs only, with the v2 honest-clean-control semantics. The
reliability head and the corruption helper stay disabled, so this config exercises
exactly the pre-A2 module set and loss path and differs from Quick-B0 and from the v1
clean control only in run identity, output directory and v2 protocol bookkeeping.

The frozen comparison rules (official pretrained, model seed, AdamW, learning rate,
500 epochs, batch size, scale augmentation, clean ``val-dev`` ``original-full`` selector)
are inherited unchanged from ``DFormerv2_S_MMFR_A2_Common_v2``.
"""

from .DFormerv2_S_MMFR_A2_Common_v2 import (
    C,
    MMFR_A2_CLEAN_CONTROL_IDENTITY,
    configure_mmfr_identity_v2,
)

configure_mmfr_identity_v2(
    identity="museg-dformerv2-s-mmfr-a2-clean-control-v2",
    mode="clean-control",
    analysis_identity=MMFR_A2_CLEAN_CONTROL_IDENTITY,
)
