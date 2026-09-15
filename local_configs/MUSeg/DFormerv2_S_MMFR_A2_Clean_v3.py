"""MMFR-A2 v3 clean control configuration.

This identity uses the unchanged DFormerv2-S clean path.  Corruption and the reliability
head are disabled, and the normalized tensors therefore follow the exact clean no-op
path while carrying the independent v3 protocol identity.
"""

from .DFormerv2_S_MMFR_A2_Common_v3 import (
    C,
    MMFR_A2_CLEAN_CONTROL_IDENTITY,
    configure_mmfr_identity_v3,
)

configure_mmfr_identity_v3(
    identity="museg-dformerv2-s-mmfr-a2-clean-control-v3",
    mode="clean-control",
    analysis_identity=MMFR_A2_CLEAN_CONTROL_IDENTITY,
)
