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
    if candidate == "C0":
        return None
    if candidate != "F-lite":
        raise ValueError(f"unsupported E1 Batch 1A candidate {candidate!r}")
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

        if cfg.decoder == "MLPDecoder":
            logger.info("Using MLP Decoder")
            from .decoders.MLPDecoder import DecoderHead

            self.decode_head = DecoderHead(
                in_channels=self.channels,
                num_classes=cfg.num_classes,
                norm_layer=norm_layer,
                embed_dim=cfg.decoder_embed_dim,
            )

        elif cfg.decoder == "ham":
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

        elif cfg.decoder == "UPernet":
            logger.info("Using Upernet Decoder")
            from .decoders.UPernet import UPerHead

            self.decode_head = UPerHead(
                in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512
            )
            from .decoders.fcnhead import FCNHead

            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == "deeplabv3+":
            logger.info("Using Decoder: DeepLabV3+")
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head

            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead

            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        elif cfg.decoder == "nl":
            logger.info("Using Decoder: nl+")
            from .decoders.nl_head import NLHead as Head

            self.decode_head = Head(
                in_channels=self.channels[1:],
                in_index=[1, 2, 3],
                num_classes=cfg.num_classes,
                norm_cfg=norm_cfg,
                channels=512,
            )
            from .decoders.fcnhead import FCNHead

            self.aux_index = 2
            self.aux_rate = 0.4
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

    def encode_decode(self, rgb, modal_x, oracle_corruption_mask=None, geometry_oracle=None):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        # print('builder',rgb.shape,modal_x.shape)
        x = self.backbone(
            rgb,
            modal_x,
            oracle_corruption_mask=oracle_corruption_mask,
            geometry_oracle=geometry_oracle,
        )
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

    def forward(
        self,
        rgb,
        modal_x=None,
        label=None,
        oracle_corruption_mask=None,
        geometry_oracle=None,
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

        if self.aux_head:
            out, aux_fm = self.encode_decode(
                rgb,
                modal_x,
                oracle_corruption_mask=oracle_corruption_mask,
                geometry_oracle=geometry_oracle,
            )
        else:
            out = self.encode_decode(
                rgb,
                modal_x,
                oracle_corruption_mask=oracle_corruption_mask,
                geometry_oracle=geometry_oracle,
            )
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
