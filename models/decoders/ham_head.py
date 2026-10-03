import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize

# from ..builder import HEADS
from .decode_head import BaseDecodeHead


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault("SPATIAL", True)

        self.S = args.setdefault("MD_S", 1)
        self.D = args.setdefault("MD_D", 512)
        self.R = args.setdefault("MD_R", 64)

        self.train_steps = args.setdefault("TRAIN_STEPS", 6)
        self.eval_steps = args.setdefault("EVAL_STEPS", 7)

        self.inv_t = args.setdefault("INV_T", 100)
        self.eta = args.setdefault("ETA", 0.9)

        self.rand_init = args.setdefault("RAND_INIT", True)
        self.diagnostic_observer = None

        print("spatial", self.spatial)
        print("S", self.S)
        print("D", self.D)
        print("R", self.R)
        print("train_steps", self.train_steps)
        print("eval_steps", self.eval_steps)
        print("inv_t", self.inv_t)
        print("eta", self.eta)
        print("rand_init", self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        observer = self.diagnostic_observer
        if observer is None:
            # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
            coef = torch.bmm(x.transpose(1, 2), bases)
            coef = F.softmax(self.inv_t * coef, dim=-1)

            steps = self.train_steps if self.training else self.eval_steps
            for _ in range(steps):
                bases, coef = self.local_step(x, bases, coef)
            return bases, coef

        coef = torch.bmm(x.transpose(1, 2), bases)
        observer("initial.coef.bmm", coef)
        scaled_coef = self.inv_t * coef
        observer("initial.coef.scaled", scaled_coef)
        coef = F.softmax(scaled_coef, dim=-1)
        observer("initial.coef.softmax", coef)

        steps = self.train_steps if self.training else self.eval_steps
        for iteration in range(1, steps + 1):
            bases, coef = self.local_step(
                x,
                bases,
                coef,
                diagnostic_iteration=iteration,
            )

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        if self.diagnostic_observer is not None:
            self.diagnostic_observer("input", x)
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)
        if self.diagnostic_observer is not None:
            self.diagnostic_observer("input.reshaped", x)

        if not self.rand_init and not hasattr(self, "bases"):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer("bases", bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)
        if self.diagnostic_observer is not None:
            self.diagnostic_observer("initial.bases", bases)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))
        if self.diagnostic_observer is not None:
            self.diagnostic_observer("output.reconstruction.bmm", x)

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)
        if self.diagnostic_observer is not None:
            self.diagnostic_observer("output", x)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def forward(self, x, return_bases=False):
        if (
            (
                getattr(self, "_mmfr_av1_training_active", False)
                or (self.training and getattr(self, "_natural_missing_nmf_fp32", False))
            )
            and x.is_cuda
            and torch.is_autocast_enabled()
        ):
            with torch.autocast(device_type="cuda", enabled=False):
                return super().forward(x.float(), return_bases=return_bases)
        return super().forward(x, return_bases=return_bases)

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef, diagnostic_iteration=None):
        observer = self.diagnostic_observer
        if observer is None:
            # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
            numerator = torch.bmm(x.transpose(1, 2), bases)
            # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
            denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
            # Multiplicative Update
            coef = coef * numerator / (denominator + 1e-6)

            # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
            numerator = torch.bmm(x, coef)
            # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
            denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
            # Multiplicative Update
            bases = bases * numerator / (denominator + 1e-6)
            return bases, coef

        iteration = "unspecified" if diagnostic_iteration is None else int(diagnostic_iteration)
        coef_prefix = f"iteration.{iteration}.coef"
        numerator = torch.bmm(x.transpose(1, 2), bases)
        observer(f"{coef_prefix}.numerator.bmm", numerator)
        gram = bases.transpose(1, 2).bmm(bases)
        observer(f"{coef_prefix}.denominator.gram.bmm", gram)
        denominator = coef.bmm(gram)
        observer(f"{coef_prefix}.denominator.bmm", denominator)
        multiplied = coef * numerator
        observer(f"{coef_prefix}.multiply", multiplied)
        denominator_with_epsilon = denominator + 1e-6
        observer(f"{coef_prefix}.denominator.add_epsilon", denominator_with_epsilon)
        coef = multiplied / denominator_with_epsilon
        observer(f"{coef_prefix}.divide", coef)

        bases_prefix = f"iteration.{iteration}.bases"
        numerator = torch.bmm(x, coef)
        observer(f"{bases_prefix}.numerator.bmm", numerator)
        gram = coef.transpose(1, 2).bmm(coef)
        observer(f"{bases_prefix}.denominator.gram.bmm", gram)
        denominator = bases.bmm(gram)
        observer(f"{bases_prefix}.denominator.bmm", denominator)
        multiplied = bases * numerator
        observer(f"{bases_prefix}.multiply", multiplied)
        denominator_with_epsilon = denominator + 1e-6
        observer(f"{bases_prefix}.denominator.add_epsilon", denominator_with_epsilon)
        bases = multiplied / denominator_with_epsilon
        observer(f"{bases_prefix}.divide", bases)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        observer = self.diagnostic_observer
        if observer is None:
            # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
            numerator = torch.bmm(x.transpose(1, 2), bases)
            # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
            denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
            # multiplication update
            coef = coef * numerator / (denominator + 1e-6)
            return coef

        prefix = "final_coef"
        numerator = torch.bmm(x.transpose(1, 2), bases)
        observer(f"{prefix}.numerator.bmm", numerator)
        gram = bases.transpose(1, 2).bmm(bases)
        observer(f"{prefix}.denominator.gram.bmm", gram)
        denominator = coef.bmm(gram)
        observer(f"{prefix}.denominator.bmm", denominator)
        multiplied = coef * numerator
        observer(f"{prefix}.multiply", multiplied)
        denominator_with_epsilon = denominator + 1e-6
        observer(f"{prefix}.denominator.add_epsilon", denominator_with_epsilon)
        coef = multiplied / denominator_with_epsilon
        observer(f"{prefix}.divide", coef)
        return coef


class Hamburger(nn.Module):
    def __init__(self, ham_channels=512, ham_kwargs=dict(), norm_cfg=None, **kwargs):
        super().__init__()

        self.ham_in = ConvModule(ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


# @HEADS.register_module()
class LightHamHead(BaseDecodeHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO:
        Add other MD models (Ham).
    """

    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super(LightHamHead, self).__init__(input_transform="multiple_select", **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [
            resize(level, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
            for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        output = self.cls_seg(output)
        return output
