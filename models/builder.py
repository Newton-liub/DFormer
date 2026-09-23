import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial

from .losses.safe_masked_loss import safe_masked_mean
from utils.engine.logger import get_logger
import warnings

# from mmcv.cnn import MODELS as MMCV_MODELS
# from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
# from mmcv.utils import Registry

# MODELS = Registry('models', parent=MMCV_MODELS)
# ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

# BACKBONES = MODELS
# NECKS = MODELS
# HEADS = MODELS
# LOSSES = MODELS
# SEGMENTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn("train_cfg and test_cfg is deprecated, please specify them in model", UserWarning)
    assert cfg.get("train_cfg") is None or train_cfg is None, (
        "train_cfg specified in both outer field and model field "
    )
    assert cfg.get("test_cfg") is None or test_cfg is None, "test_cfg specified in both outer field and model field "
    return SEGMENTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


logger = get_logger()


def _mmfr_reliability_config(cfg):
    """Return the frozen MMFR-A2 reliability block, or ``None`` when it is disabled.

    The block lives in ``cfg.mmfr_a2["reliability_head"]``. Until a config enables it,
    the model builds exactly the same modules as before A2 and keeps the scalar
    segmentation loss path unchanged.
    """
    block = cfg.get("mmfr_a2") if hasattr(cfg, "get") else getattr(cfg, "mmfr_a2", None)
    if not block:
        return None
    head = block.get("reliability_head") if hasattr(block, "get") else None
    if not head:
        return None
    if not bool(head.get("enabled", True)):
        return None
    return head


def _mmfr_supervised_channels(reliability_cfg):
    """Channels the A2 reliability auxiliary loss is allowed to score.

    ``MMFR-A2-train-integration-v1`` scores both channels (its default). The v2 protocol
    fixes ``["depth"]`` because the RGB channel is an all-ones scaffold there: scoring it
    would turn a constant into a "trained RGB reliability estimator". An empty or unknown
    list fails closed instead of silently scoring nothing.
    """
    from .modal_reliability import MODALITY_ORDER

    raw = reliability_cfg.get("supervised_channels", list(MODALITY_ORDER))
    if isinstance(raw, str):
        raw = [raw]
    channels = tuple(str(name) for name in raw)
    if not channels:
        raise ValueError("mmfr_a2 reliability_head.supervised_channels must not be empty")
    for name in channels:
        if name not in MODALITY_ORDER:
            raise ValueError(
                f"unsupported supervised reliability channel {name!r}; expected names from {MODALITY_ORDER}"
            )
    if len(set(channels)) != len(channels):
        raise ValueError(f"supervised_channels must be distinct, got {channels}")
    return channels


def _e1_feature_adapter_config(cfg):
    """Return the frozen E1 F-lite adapter block, or ``None`` for C0/non-E1 models."""

    block = cfg.get("e1_batch1") if hasattr(cfg, "get") else getattr(cfg, "e1_batch1", None)
    if not block or not bool(block.get("enabled", False)):
        return None
    candidate = str(block.get("candidate", ""))
    if candidate in ("C0", "R-OE-lite"):
        return None
    if candidate != "F-lite":
        raise ValueError(f"unsupported E1 Batch 1 candidate {candidate!r}")
    adapter = dict(block.get("feature_adapter") or {})
    expected = {
        "stages": [1, 2, 3],
        "bottleneck_ratio": 4,
        "normalization": "none",
        "activation": "GELU",
        "down_init": "trunc_normal_std_0.02",
        "up_init": "zeros",
    }
    for key, value in expected.items():
        if adapter.get(key) != value:
            raise ValueError(
                f"E1 F-lite feature_adapter.{key} must be {value!r}, got {adapter.get(key)!r}"
            )
    forbidden = ("reliability", "condition", "severity", "oracle")
    if any(bool(adapter.get(f"uses_{name}", False)) for name in forbidden):
        raise ValueError("E1 F-lite must not consume reliability, condition, severity or oracle inputs")
    return adapter



def _e1_roe_config(cfg):
    """Return and validate the frozen Batch 1B R-OE-lite block."""

    block = cfg.get("e1_batch1") if hasattr(cfg, "get") else getattr(cfg, "e1_batch1", None)
    if not block or not bool(block.get("enabled", False)):
        return None
    if str(block.get("candidate", "")) != "R-OE-lite":
        return None
    substitute = dict(block.get("roe_substitute") or {})
    expected = {
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
    for key, value in expected.items():
        if substitute.get(key) != value:
            raise ValueError(
                f"E1 R-OE-lite roe_substitute.{key} must be {value!r}, "
                f"got {substitute.get(key)!r}"
            )
    return substitute


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        cfg=None,
        criterion=nn.CrossEntropyLoss(reduction="none", ignore_index=255),
        norm_layer=nn.BatchNorm2d,
        syncbn=False,
    ):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer
        self.cfg = cfg

        if cfg.backbone == "DFormer-Large":
            from .encoders.DFormer import DFormer_Large as backbone

            self.channels = [96, 192, 288, 576]
        elif cfg.backbone == "DFormer-Base":
            from .encoders.DFormer import DFormer_Base as backbone

            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == "DFormer-Small":
            from .encoders.DFormer import DFormer_Small as backbone

            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == "DFormer-Tiny":
            from .encoders.DFormer import DFormer_Tiny as backbone

            self.channels = [32, 64, 128, 256]

        elif cfg.backbone == "DFormerv2_L":
            from .encoders.DFormerv2 import DFormerv2_L as backbone

            self.channels = [112, 224, 448, 640]
        elif cfg.backbone == "DFormerv2_B":
            from .encoders.DFormerv2 import DFormerv2_B as backbone

            self.channels = [80, 160, 320, 512]
        elif cfg.backbone == "DFormerv2_S":
            from .encoders.DFormerv2 import DFormerv2_S as backbone

            self.channels = [64, 128, 256, 512]
        else:
            raise NotImplementedError

        if syncbn:
            norm_cfg = dict(type="SyncBN", requires_grad=True)
        else:
            norm_cfg = dict(type="BN", requires_grad=True)

        if cfg.drop_path_rate is not None:
            self.backbone = backbone(drop_path_rate=cfg.drop_path_rate, norm_cfg=norm_cfg)
        else:
            self.backbone = backbone(drop_path_rate=0.1, norm_cfg=norm_cfg)

        self.aux_head = None

        if cfg.decoder == "ham":
            logger.info("Using Ham Decoder")
            print(cfg.num_classes)
            from .decoders.ham_head import LightHamHead as DecoderHead

            # from mmseg.models.decode_heads.ham_head import LightHamHead as DecoderHead
            self.decode_head = DecoderHead(
                in_channels=self.channels[1:],
                num_classes=cfg.num_classes,
                in_index=[1, 2, 3],
                norm_cfg=norm_cfg,
                channels=cfg.decoder_embed_dim,
            )
            from .decoders.fcnhead import FCNHead

            if cfg.aux_rate != 0:
                self.aux_index = 2
                self.aux_rate = cfg.aux_rate
                print("aux rate is set to", str(self.aux_rate))
                self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info("No decoder(FCN-32s)")
            from .decoders.fcnhead import FCNHead

            self.decode_head = FCNHead(
                in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer
            )

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

        # MMFR-A2 auxiliary reliability head. It is created only when the config asks
        # for it, keeps its module default initialization (``init_weights`` only touches
        # the segmentation heads) and never feeds the backbone, geometry prior or
        # decoder: A2 uses it for auxiliary supervision only.
        self.reliability_estimator = None
        self.reliability_weight = 0.0
        self.reliability_supervised_channels = ()
        reliability_cfg = _mmfr_reliability_config(cfg)
        if reliability_cfg is not None:
            from .modal_reliability import ModalReliabilityEstimator

            hidden_channels = int(reliability_cfg.get("hidden_channels", 16))
            self.reliability_weight = float(reliability_cfg.get("weight", 0.1))
            self.reliability_supervised_channels = _mmfr_supervised_channels(reliability_cfg)
            self.reliability_estimator = ModalReliabilityEstimator(hidden_channels=hidden_channels)
            logger.info(
                "MMFR-A2 reliability auxiliary head enabled: hidden_channels=%d weight=%s supervised_channels=%s",
                hidden_channels,
                self.reliability_weight,
                ",".join(self.reliability_supervised_channels),
            )

        # R-OE-lite is initialized in an isolated torch RNG scope. Its construction
        # must not consume the global state used later by DataLoader shuffling, so a
        # matched C0 can be reused without changing the first-epoch permutation.
        self.roe_substitute = None
        self.last_roe_route = None
        roe_cfg = _e1_roe_config(cfg)
        if roe_cfg is not None:
            from .roe_substitute import ObservableEmptyGeometrySubstitute

            cpu_rng_state = torch.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                self.roe_substitute = ObservableEmptyGeometrySubstitute()
            finally:
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
            actual_trainable_parameters = sum(
                int(parameter.numel())
                for parameter in self.roe_substitute.parameters()
                if parameter.requires_grad
            )
            expected_trainable_parameters = int(roe_cfg["expected_trainable_parameters"])
            if actual_trainable_parameters != expected_trainable_parameters:
                raise ValueError(
                    "E1 R-OE-lite substitute parameter count mismatch: "
                    f"expected {expected_trainable_parameters}, got {actual_trainable_parameters}"
                )
            logger.info(
                "MMFR E1 R-OE-lite substitute enabled: parameters=%d",
                actual_trainable_parameters,
            )

        self.feature_adapter = None
        feature_adapter_cfg = _e1_feature_adapter_config(cfg)
        if feature_adapter_cfg is not None:
            from .feature_adapter import FeatureLiteAdapter

            self.feature_adapter = FeatureLiteAdapter(
                self.channels,
                bottleneck_ratio=int(feature_adapter_cfg["bottleneck_ratio"]),
            )
            actual_trainable_parameters = sum(
                int(parameter.numel())
                for parameter in self.feature_adapter.parameters()
                if parameter.requires_grad
            )
            expected_trainable_parameters = int(
                feature_adapter_cfg["expected_trainable_parameters"]
            )
            if actual_trainable_parameters != expected_trainable_parameters:
                raise ValueError(
                    "E1 F-lite feature adapter parameter count mismatch: "
                    f"expected {expected_trainable_parameters}, got {actual_trainable_parameters}"
                )
            logger.info(
                "MMFR E1 F-lite feature adapter enabled: stages=1,2,3 ratio=%d parameters=%d",
                int(feature_adapter_cfg["bottleneck_ratio"]),
                actual_trainable_parameters,
            )

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info("Loading pretrained model: {}".format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info("Initing weights ...")
        init_weight(
            self.decode_head,
            nn.init.kaiming_normal_,
            self.norm_layer,
            cfg.bn_eps,
            cfg.bn_momentum,
            mode="fan_in",
            nonlinearity="relu",
        )
        if self.aux_head:
            init_weight(
                self.aux_head,
                nn.init.kaiming_normal_,
                self.norm_layer,
                cfg.bn_eps,
                cfg.bn_momentum,
                mode="fan_in",
                nonlinearity="relu",
            )

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        # print('builder',rgb.shape,modal_x.shape)
        x = self.backbone(rgb, modal_x)
        if len(x) == 2:  # if output is (rgb,depth) only use rgb
            x = x[0]
        if self.feature_adapter is not None:
            x = self.feature_adapter(x)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[-2:], mode="bilinear", align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode="bilinear", align_corners=False)
            return out, aux_fm
        return out

    def _route_roe_modal_x(self, rgb, modal_x, raw_depth=None, geometry_mask=None):
        """Route only observable-empty samples through the frozen R-OE substitute."""

        if self.roe_substitute is None:
            return modal_x
        if raw_depth is None or geometry_mask is None:
            raise ValueError(
                "R-OE-lite requires current raw_depth and V_geom for observable-empty routing"
            )
        if raw_depth.ndim != 4 or raw_depth.shape[1] != 1:
            raise ValueError(
                f"R-OE-lite detector expects raw_depth [B,1,H,W], got {tuple(raw_depth.shape)}"
            )
        if geometry_mask.ndim != 4 or geometry_mask.shape[1] != 1:
            raise ValueError(
                f"R-OE-lite detector expects V_geom [B,1,H,W], got {tuple(geometry_mask.shape)}"
            )
        if raw_depth.shape[0] != rgb.shape[0] or raw_depth.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("R-OE-lite raw_depth must match the RGB batch and spatial shape")
        if geometry_mask.shape != raw_depth.shape:
            raise ValueError("R-OE-lite V_geom must match raw_depth exactly")
        if modal_x.ndim != 4 or modal_x.shape[1] != 3:
            raise ValueError(
                f"R-OE-lite requires the existing three-channel Depth path, got {tuple(modal_x.shape)}"
            )
        if not torch.isfinite(raw_depth).all():
            raise ValueError("R-OE-lite raw_depth contains non-finite values")
        if bool((raw_depth < 0.0).any().item()) or bool((raw_depth > 1.0).any().item()):
            raise ValueError("R-OE-lite raw_depth must lie in [0, 1]")

        v_geom = geometry_mask.to(dtype=torch.bool)
        raw_u8 = torch.round(raw_depth.to(torch.float32) * 255.0).to(torch.int64)
        geometry_pixels = v_geom.flatten(1).sum(dim=1)
        nonzero_pixels = (raw_u8 > 0).logical_and(v_geom).flatten(1).sum(dim=1)
        trigger = (geometry_pixels > 0) & (nonzero_pixels == 0)
        self.last_roe_route = {
            "oe_semantics": "observable-empty",
            "trigger": trigger.detach(),
            "geometry_pixels": geometry_pixels.detach(),
            "raw_depth_nonzero_pixels": nonzero_pixels.detach(),
            "substitute_forward": int(trigger.sum().item()),
        }
        if not bool(trigger.any().item()):
            return modal_x

        trigger_indices = torch.nonzero(trigger, as_tuple=False).flatten()
        substitute_raw = self.roe_substitute(
            rgb.index_select(0, trigger_indices),
            v_geom.index_select(0, trigger_indices),
        )
        substitute_depth = substitute_raw.repeat(1, 3, 1, 1)
        substitute_depth = (substitute_depth / 255.0 - 0.48) / 0.28
        substitute_depth = substitute_depth.to(dtype=modal_x.dtype)
        return torch.index_copy(modal_x, 0, trigger_indices, substitute_depth)

    def forward(
        self,
        rgb,
        modal_x=None,
        label=None,
        raw_rgb=None,
        raw_depth=None,
        reliability_target=None,
        reliability_valid_mask=None,
        depth_valid=None,
        reliability_telemetry_masks=None,
    ):
        # Fail closed on partial or mismatched A2 supervision. Inference keeps the
        # historical path because reliability supervision is a training-only input.
        reliability_inputs = {
            "raw_rgb": raw_rgb,
            "raw_depth": raw_depth,
            "reliability_target": reliability_target,
            "reliability_valid_mask": reliability_valid_mask,
            "depth_valid": depth_valid,
            "reliability_telemetry_masks": reliability_telemetry_masks,
        }
        provided_reliability = [name for name, value in reliability_inputs.items() if value is not None]
        if self.reliability_estimator is None and provided_reliability:
            raise ValueError(
                "reliability supervision was provided while the MMFR-A2 reliability head is disabled: "
                + ", ".join(provided_reliability)
            )
        if self.reliability_estimator is not None and label is not None:
            missing_reliability = [name for name, value in reliability_inputs.items() if value is None]
            if missing_reliability:
                raise ValueError(
                    "MMFR-A2 training requires the complete reliability supervision batch; missing: "
                    + ", ".join(missing_reliability)
                )

        if self.roe_substitute is not None:
            modal_x = self._route_roe_modal_x(
                rgb,
                modal_x,
                raw_depth=raw_depth,
                geometry_mask=reliability_valid_mask,
            )

        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        if label is not None:
            target = label.long()
            valid_mask = target != self.cfg.background
            loss = safe_masked_mean(self.criterion(out, target), valid_mask)
            if self.aux_head:
                loss += self.aux_rate * safe_masked_mean(self.criterion(aux_fm, target), valid_mask)
            if self.reliability_estimator is not None:
                loss = loss + self.reliability_weight * self.reliability_auxiliary_loss(
                    raw_rgb,
                    raw_depth,
                    reliability_target,
                    reliability_valid_mask,
                    depth_valid,
                    reliability_telemetry_masks,
                )
            return loss
        return out

    def reliability_auxiliary_loss(
        self,
        raw_rgb,
        raw_depth,
        reliability_target,
        reliability_valid_mask=None,
        depth_valid=None,
        reliability_telemetry_masks=None,
    ):
        """Continuous BCE between the auxiliary reliability estimate and the A1 target.

        The A2 total loss is ``seg_loss + 0.1 * reliability_loss``; ``valid_mask`` only
        excludes crop/pad pixels. Only ``reliability_supervised_channels`` contribute:
        the v1 protocol scores both channels, while the v2 protocol scores Depth only and
        keeps the RGB channel as an unscored scaffold. The predicted reliability is
        returned nowhere else, and no clean/corrupt consistency term is added
        (``lambda_cons = 0``).
        """
        from .modal_reliability import MODALITY_ORDER, RELIABILITY_CHANNELS, continuous_bce_loss

        if self.reliability_estimator is None:
            raise RuntimeError("reliability auxiliary loss requested without a reliability head")
        if raw_rgb is None or raw_depth is None:
            raise ValueError("reliability supervision requires raw_rgb and raw_depth")
        if reliability_target is None:
            raise ValueError("reliability supervision requires reliability_target")
        # ``SignalFeatureExtractor`` consumes a floating point validity map. A2 v2 passes
        # the post-corruption Depth validity, i.e. the validity of the input the model
        # actually sees; the pre-corruption validity shapes the target instead. The A2
        # auxiliary loss deliberately skips the estimator's four-level pyramid; that
        # pyramid remains reserved for the later learned adapter stage. The fixed signal
        # operators include products of squared gradients, which under the outer CUDA FP16
        # autocast can underflow to zero before the cosine denominator is formed and
        # produce NaNs on real images, so the small auxiliary branch stays in FP32 while
        # the segmentation path remains AMP.
        channel_weight = torch.zeros(
            RELIABILITY_CHANNELS, 1, 1, dtype=torch.float32, device=raw_rgb.device
        )
        for name in self.reliability_supervised_channels:
            channel_weight[MODALITY_ORDER.index(name)] = 1.0
        if reliability_valid_mask is None:
            supervision_mask = channel_weight
        else:
            supervision_mask = reliability_valid_mask.to(torch.float32) * channel_weight
        if float(supervision_mask.sum().item()) <= 0.0:
            raise RuntimeError(
                "no supervised reliability pixel remains after applying supervised_channels="
                f"{self.reliability_supervised_channels}; refusing to continue with a silent zero loss"
            )
        feature_validity = None if depth_valid is None else depth_valid.to(torch.float32)
        with torch.autocast(device_type=raw_rgb.device.type, enabled=False):
            features = self.reliability_estimator.features(
                raw_rgb.to(torch.float32),
                raw_depth.to(torch.float32),
                feature_validity,
            )
            logits = self.reliability_estimator.head(features)
            aggregate_loss = continuous_bce_loss(
                logits,
                reliability_target.to(torch.float32),
                supervision_mask,
            )
            if reliability_telemetry_masks is not None:
                masks = reliability_telemetry_masks.to(torch.float32)
                expected_shape = (logits.shape[0], 4, logits.shape[2], logits.shape[3])
                if tuple(masks.shape) != expected_shape:
                    raise ValueError(
                        f"reliability_telemetry_masks must have shape {expected_shape}, got {tuple(masks.shape)}"
                    )
                depth_index = MODALITY_ORDER.index("depth")
                depth_logits = logits[:, depth_index : depth_index + 1]
                depth_target = reliability_target[:, depth_index : depth_index + 1].to(torch.float32)
                category_losses = []
                for index in range(masks.shape[1]):
                    category_mask = masks[:, index : index + 1]
                    count = category_mask.sum()
                    if float(count.item()) == 0.0:
                        category_losses.append(torch.full((), float("nan"), device=logits.device))
                    else:
                        category_losses.append(
                            continuous_bce_loss(
                                depth_logits,
                                depth_target,
                                category_mask,
                            ).detach()
                        )
                self.last_reliability_telemetry = {
                    "category_order": (
                        "natural-invalid",
                        "synthetic-invalid",
                        "implicit-quality",
                        "valid-clean",
                    ),
                    "losses": torch.stack(category_losses),
                    "pixel_counts": masks.sum(dim=(0, 2, 3)).detach(),
                }
            else:
                self.last_reliability_telemetry = None
            return aggregate_loss
