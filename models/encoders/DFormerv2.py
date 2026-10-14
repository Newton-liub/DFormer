"""
DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation
Code: https://github.com/VCIP-RGBD/DFormer

Author: yinbow
Email: bowenyin@mail.nankai.edu.cn

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from mmengine.runner.checkpoint import load_state_dict
from mmengine.runner.checkpoint import load_checkpoint
from typing import Tuple
import sys
import os
from collections import OrderedDict


_RESEARCH_ATTENTION_PROBS = None


def _research_attention_probs(qk, bias, valid):
    """Lazy bridge to :func:`research.oc_rga.research_attention_probs`.

    Imported on first use so ``models/`` keeps working for callers that never
    touch the OC-RGA research path.
    """
    global _RESEARCH_ATTENTION_PROBS
    if _RESEARCH_ATTENTION_PROBS is None:
        from research.oc_rga import research_attention_probs as _fn

        _RESEARCH_ATTENTION_PROBS = _fn
    return _RESEARCH_ATTENTION_PROBS(qk, bias, valid)


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        """
        input shape (b c h w)
        """
        x = x.permute(0, 2, 3, 1).contiguous()  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)
        return x


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        input (b h w c)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.SyncBatchNorm(out_dim)

    def forward(self, x):
        """
        x: B H W C
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (b oh ow oc)
        return x


def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class GeoPriorGen(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_depth_decay(self, H: int, W: int, depth_grid):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H: int, W: int):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(self, H, W, depth_grid):
        """
        generate 1d depth decay mask, the result is l*l
        """
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        assert mask.shape[2:] == (W, H, H)
        return mask

    def generate_1d_decay(self, l: int):
        """
        generate 1d decay mask, the result is l*l
        """
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False, depth_relation=None,
                return_components=False):
        """
        depth_map: depth patches
        HW_tuple: (H, W)
        H * W == l

        ``return_components`` is an opt-in research hook used by the direction-A
        attention control: when true the already-weighted spatial (``bs*``) and
        signed depth (``bd*``) parts are returned in addition to the author's
        merged prior, so the control can rebuild the spatial-only reference
        ``p0 = softmax(qk + bs)`` without touching the merged prior.  The default
        return value, dtype, the ``weight`` parameter and the float addition
        order of the merged prior are all unchanged.
        """
        if depth_relation is None:
            # Original mode preserves the author's interpolation and depth-difference path.
            depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)

        if split_or_not:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)

            if depth_relation is None:
                mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
                mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)
            else:
                mask_d_h, mask_d_w = depth_relation

            mask_h_spatial = self.generate_1d_decay(HW_tuple[0])
            mask_w_spatial = self.generate_1d_decay(HW_tuple[1])

            bs_h = self.weight[0] * mask_h_spatial.unsqueeze(0).unsqueeze(2)
            bd_h = self.weight[1] * mask_d_h
            bs_w = self.weight[0] * mask_w_spatial.unsqueeze(0).unsqueeze(2)
            bd_w = self.weight[1] * mask_d_w

            geo_prior = ((sin, cos), (bs_h + bd_h, bs_w + bd_w))
            parts = {"bs_h": bs_h, "bd_h": bd_h, "bs_w": bs_w, "bd_w": bd_w}

        else:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)
            mask_spatial = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])

            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map) if depth_relation is None else depth_relation
            bs = self.weight[0] * mask_spatial
            bd = self.weight[1] * mask_d

            geo_prior = ((sin, cos), bs + bd)
            parts = {"bs": bs, "bd": bd}

        if return_components:
            return geo_prior, parts
        return geo_prior


class Decomposed_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos, split_or_not=False, bias_parts=None, research=None):
        """
        ``research`` is the OC-RGA / mask-aware-baseline path: a dict holding the
        already assembled ``bias_w``/``bias_h`` and the optional ``valid_w`` /
        ``valid_h`` token masks.  It replaces only the softmax probability
        operator with the shared FP32 branch; Q/K/V generation, RoPE, the W -> H
        wiring, the V aggregation layout and the LePE/out_proj tail are the
        author's.  ``bias_parts`` (direction A, inference only) and the default
        author path are untouched.
        """
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        if research is not None:
            p_w = _research_attention_probs(qk_mat_w, research["bias_w"], research.get("valid_w"))
            with torch.autocast(device_type=qk_mat_w.device.type, enabled=False):
                v = torch.matmul(p_w, v.float())
        elif bias_parts is None:
            bias_w = mask_w.transpose(1, 2)
            if getattr(self, "match_bias_dtype", False):
                bias_w = bias_w.to(qk_mat_w.dtype)
            qk_mat_w = qk_mat_w + bias_w
            qk_mat_w = torch.softmax(qk_mat_w, -1)
            v = torch.matmul(qk_mat_w, v)
        else:
            # Direction-A research path for the W axis.  The softmax is replaced
            # by the FP32 projection/control and the V aggregation stays FP32, so
            # the probability-ratio budget survives the subsequent H path.
            control = getattr(self, "influence_control", None)
            if control is None:
                raise RuntimeError("bias_parts was supplied without an installed influence control")
            p_w = control.attend(qk_mat_w, bias_parts["bs_w"].transpose(1, 2),
                                 bias_parts["bd_w"].transpose(1, 2), axis="w")
            with torch.autocast(device_type=qk_mat_w.device.type, enabled=False):
                v = torch.matmul(p_w, v.float())

        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        if research is not None:
            p_h = _research_attention_probs(qk_mat_h, research["bias_h"], research.get("valid_h"))
            with torch.autocast(device_type=qk_mat_h.device.type, enabled=False):
                output = torch.matmul(p_h, v.float())
        elif bias_parts is None:
            bias_h = mask_h.transpose(1, 2)
            if getattr(self, "match_bias_dtype", False):
                bias_h = bias_h.to(qk_mat_h.dtype)
            qk_mat_h = qk_mat_h + bias_h
            qk_mat_h = torch.softmax(qk_mat_h, -1)
            output = torch.matmul(qk_mat_h, v)
        else:
            control = getattr(self, "influence_control", None)
            if control is None:
                raise RuntimeError("bias_parts was supplied without an installed influence control")
            p_h = control.attend(qk_mat_h, bias_parts["bs_h"].transpose(1, 2),
                                 bias_parts["bd_h"].transpose(1, 2), axis="h")
            with torch.autocast(device_type=qk_mat_h.device.type, enabled=False):
                output = torch.matmul(p_h, v.float())

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        if bias_parts is not None or research is not None:
            # H aggregation is FP32 by protocol; rejoin the module dtype before
            # the unchanged "+ lepe / out_proj" tail.
            output = output.to(lepe.dtype)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class Full_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos, split_or_not=False, bias_parts=None, research=None):
        """
        x: (b h w c)
        rel_pos: mask: (n l l)
        ``research``: OC-RGA / mask-aware-baseline path (``bias_full`` plus an
        optional ``valid_full`` token mask).  Only the softmax operator changes.
        """
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        assert h * w == mask.size(3)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        vr = vr.flatten(2, 3)
        qk_mat = qr @ kr.transpose(-1, -2)
        if research is not None:
            p_full = _research_attention_probs(qk_mat, research["bias_full"], research.get("valid_full"))
            with torch.autocast(device_type=qk_mat.device.type, enabled=False):
                output = torch.matmul(p_full, vr.float())
        elif bias_parts is None:
            bias = mask.to(qk_mat.dtype) if getattr(self, "match_bias_dtype", False) else mask
            qk_mat = qk_mat + bias
            qk_mat = torch.softmax(qk_mat, -1)
            output = torch.matmul(qk_mat, vr)
        else:
            # Direction-A research path for the full axis: the FP32 projection
            # replaces the softmax; Q/K generation, flatten order and output
            # reshape are unchanged.
            control = getattr(self, "influence_control", None)
            if control is None:
                raise RuntimeError("bias_parts was supplied without an installed influence control")
            bs_full = bias_parts["bs"]
            bd_full = bias_parts["bd"]
            if bs_full.dim() == qk_mat.dim() - 1:
                bs_full = bs_full.unsqueeze(0)
            if bd_full.dim() == qk_mat.dim() - 1:
                bd_full = bd_full.unsqueeze(0)
            p_full = control.attend(qk_mat, bs_full, bd_full, axis="full")
            with torch.autocast(device_type=qk_mat.device.type, enabled=False):
                output = torch.matmul(p_full, vr.float())

        output = output.transpose(1, 2).reshape(bsz, h, w, -1)
        if bias_parts is not None or research is not None:
            output = output.to(lepe.dtype)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        """
        input shape: (b h w c)
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class RGBD_Block(nn.Module):
    def __init__(
        self,
        split_or_not: str,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        drop_path=0.0,
        layerscale=False,
        layer_init_values=1e-5,
        init_value=2,
        heads_range=4,
        oc_geometry=False,
    ):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        if split_or_not:
            self.Attention = Decomposed_GSA(embed_dim, num_heads)
        else:
            self.Attention = Full_GSA(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        # FFN
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.cnn_pos_encode = DWConv2d(embed_dim, 3, 1, 1)
        # the function to generate the geometry prior for the current block
        self.Geo = GeoPriorGen(embed_dim, num_heads, init_value, heads_range)
        # OC-RGA turns this block into an observational-geometry block.  The OC
        # model replaces ``self.Geo`` with ``OCSpatialPrior``; the mask-aware
        # baseline keeps the author ``GeoPriorGen`` and only replaces its signed
        # depth relation.  Only the OC model owns an injection scale, so the
        # baseline carries no unused trainable parameter.
        self.oc_geometry = bool(oc_geometry)
        if self.oc_geometry:
            from research.oc_rga import OCInjectionScale

            self.oc_injection = OCInjectionScale(num_heads)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def observational_research(self, h: int, w: int, split_or_not: bool, stage):
        """Assemble ``(rel_pos, research)`` for the OC-RGA / baseline path.

        ``rel_pos`` keeps the author RoPE/SDPA contract; ``research`` carries the
        fully assembled per-axis bias and the valid-token masks.  Everything the
        attention needs is built here, so neither attention module has to know
        which arm it serves.
        """
        if split_or_not:
            if "Ehat_w" in stage:
                rel_pos = self.Geo((h, w), split_or_not=True)
                bs_h, bs_w = rel_pos[1]
                beta = self.oc_injection().view(1, -1, 1, 1, 1)
                bias_w = (bs_w + beta * stage["Ehat_w"].transpose(1, 2)).transpose(1, 2)
                bias_h = (bs_h + beta * stage["Ehat_h"].transpose(1, 2)).transpose(1, 2)
            else:
                # Mask-aware baseline (design 11.2).  The pooled relation
                # ``|ybar_i - ybar_j| m_i m_j`` is premultiplied here by the author
                # per-head depth decay, because ``GeoPriorGen`` only applies the
                # signed ``weight[1]`` to a supplied ``depth_relation`` and would
                # otherwise drop the decay entirely.  The signed coefficient is
                # therefore still applied exactly once, by the author module.
                decay5 = self.Geo.decay.reshape(1, -1, 1, 1, 1).to(stage["base_w"].dtype)
                rel_pos, parts = self.Geo(
                    (h, w), None, split_or_not=True,
                    depth_relation=(stage["base_h"] * decay5, stage["base_w"] * decay5),
                    return_components=True,
                )
                bias_w = (parts["bs_w"] + parts["bd_w"]).transpose(1, 2)
                bias_h = (parts["bs_h"] + parts["bd_h"]).transpose(1, 2)
            valid = stage["valid"]
            return rel_pos, {
                "bias_w": bias_w,
                "bias_h": bias_h,
                "valid_w": valid,
                "valid_h": valid.transpose(1, 2),
            }

        if "Ehat_full" in stage:
            rel_pos = self.Geo((h, w), split_or_not=False)
            bs = rel_pos[1]
            beta = self.oc_injection().view(1, -1, 1, 1)
            bias = bs + beta * stage["Ehat_full"]
        else:
            # Full-attention baseline: same premultiplied decay, ``weight[1]``
            # applied once by ``GeoPriorGen``.
            decay4 = self.Geo.decay.reshape(1, -1, 1, 1).to(stage["base_full"].dtype)
            rel_pos, parts = self.Geo(
                (h, w), None, split_or_not=False,
                depth_relation=stage["base_full"] * decay4, return_components=True,
            )
            bias = parts["bs"] + parts["bd"]
        valid = stage["valid"].reshape(stage["valid"].shape[0], -1)
        return rel_pos, {"bias_full": bias, "valid_full": valid}

    def forward(self, x: torch.Tensor, x_e: torch.Tensor, split_or_not=False, depth_relation=None,
                oc_stage=None):
        x = x + self.cnn_pos_encode(x)
        b, h, w, d = x.size()

        if oc_stage is not None:
            # OC-RGA / mask-aware-baseline path.  Q/K/V, RoPE, LePE, the
            # residual/drop-path composition and the FFN are the author's; only
            # the attention probability operator is the shared FP32 one.
            rel_pos, research = self.observational_research(h, w, split_or_not, oc_stage)
            attended = self.Attention(self.layer_norm1(x), rel_pos, split_or_not, research=research)
            if self.layerscale:
                x = x + self.drop_path(self.gamma_1 * attended)
            else:
                x = x + self.drop_path(attended)
            x = x + self.drop_path(self.ffn(self.layer_norm2(x)))
            return x

        control = getattr(self, "influence_control", None)
        if control is None:
            geo_prior = self.Geo((h, w), x_e, split_or_not=split_or_not, depth_relation=depth_relation)
            if self.layerscale:
                x = x + self.drop_path(self.gamma_1 * self.Attention(self.layer_norm1(x), geo_prior, split_or_not))
                x = x + self.drop_path(self.gamma_2 * self.ffn(self.layer_norm2(x)))
            else:
                x = x + self.drop_path(self.Attention(self.layer_norm1(x), geo_prior, split_or_not))
                x = x + self.drop_path(self.ffn(self.layer_norm2(x)))
            return x

        # Direction-A frozen research branch.  It asks GeoPriorGen for the
        # already-weighted spatial/depth parts instead of the merged prior.  It is
        # inference-only by design: the projection has no validated backward yet,
        # so running it in training mode or with gradients enabled is refused
        # rather than silently turning it into an unvalidated training mechanism.
        if self.training or torch.is_grad_enabled():
            raise RuntimeError(
                "direction-A influence control is frozen inference only "
                "(training=%s grad_enabled=%s)" % (self.training, torch.is_grad_enabled())
            )
        geo_prior, parts = self.Geo((h, w), x_e, split_or_not=split_or_not,
                                    depth_relation=depth_relation, return_components=True)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.Attention(self.layer_norm1(x), geo_prior, split_or_not,
                                                                 bias_parts=parts))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.layer_norm2(x)))
        else:
            x = x + self.drop_path(self.Attention(self.layer_norm1(x), geo_prior, split_or_not,
                                                  bias_parts=parts))
            x = x + self.drop_path(self.ffn(self.layer_norm2(x)))
        return x


class BasicLayer(nn.Module):
    """
    A basic RGB-D layer in DFormerv2.
    """

    def __init__(
        self,
        embed_dim,
        out_dim,
        depth,
        num_heads,
        init_value: float,
        heads_range: float,
        ffn_dim=96.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        split_or_not=False,
        downsample: PatchMerging = None,
        use_checkpoint=False,
        layerscale=False,
        layer_init_values=1e-5,
        oc_geometry=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.split_or_not = split_or_not

        # build blocks
        self.blocks = nn.ModuleList(
            [
                RGBD_Block(
                    split_or_not,
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layerscale,
                    layer_init_values,
                    init_value=init_value,
                    heads_range=heads_range,
                    oc_geometry=oc_geometry,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_e, observations=None, oc_stage=None):
        b, h, w, d = x.size()
        # All blocks in this stage share fixed head decay; weights remain per-block.
        relation = None if observations is None else observations.relation((h, w), self.blocks[0].Geo.decay, self.split_or_not)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_e, self.split_or_not, relation, oc_stage, use_reentrant=False)
            else:
                x = blk(x, x_e, split_or_not=self.split_or_not, depth_relation=relation, oc_stage=oc_stage)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x


class dformerv2(nn.Module):
    def __init__(
        self,
        out_indices=(0, 1, 2, 3),
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        projection=1024,
        norm_cfg=None,
        layerscales=[False, False, False, False],
        layer_init_values=1e-6,
        norm_eval=True,
        geometry_mode="original",
        oc_geometry=False,
        checkpoint_stages=None,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.norm_eval = norm_eval
        if geometry_mode not in ("original", "mean", "odg"):
            raise ValueError("geometry_mode must be original, mean or odg")
        self.geometry_mode = geometry_mode

        # patch embedding
        self.patch_embed = PatchEmbed(
            in_chans=3, embed_dim=embed_dims[0], norm_layer=norm_layer if self.patch_norm else None
        )

        # drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                split_or_not=(i_layer != 3),
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                # Per-stage control is opt-in: the default keeps the author's
                # single global ``use_checkpoint`` for every stage.
                use_checkpoint=(use_checkpoint if checkpoint_stages is None
                                else (i_layer in tuple(checkpoint_stages))),
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
                oc_geometry=oc_geometry,
            )
            for block in layer.blocks:
                block.Attention.match_bias_dtype = geometry_mode != "original"
            self.layers.append(layer)

        self.extra_norms = nn.ModuleList()
        for i in range(3):
            self.extra_norms.append(nn.LayerNorm(embed_dims[i + 1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            _state_dict = torch.load(pretrained)
            if "model" in _state_dict.keys():
                _state_dict = _state_dict["model"]
            if "state_dict" in _state_dict.keys():
                _state_dict = _state_dict["state_dict"]
            state_dict = OrderedDict()

            for k, v in _state_dict.items():
                if k.startswith("backbone."):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            print("load " + pretrained)
            load_state_dict(self, state_dict, strict=False)
            # load_checkpoint(self, pretrained, strict=False)
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward(self, x, x_e, depth_support=None, image_support=None, geometry_builder=None):
        # Construct distributions from transformed INPUT observations, not stage scalars.
        x_e = x_e[:, 0:1, :, :]
        observations = None
        if self.geometry_mode != "original":
            from research.geometry import ObservationGeometry
            observations = ObservationGeometry(x_e, depth_support, mode=self.geometry_mode)
        x = self.patch_embed(x)
        # OC-RGA builds its whole geometry package once, right after the shared RGB
        # stem and before any stage-0 attention.  The builder is a registered
        # module supplied by the research wrapper; it never sees labels or
        # reference depth.  Without a builder the author path is unchanged.
        bundle = None
        if geometry_builder is not None:
            bundle = geometry_builder(x, x_e, depth_support, image_support)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            oc_stage = None if bundle is None else bundle.stage(i)
            x_out, x = layer(x, x_e, observations, oc_stage=oc_stage)
            if i in self.out_indices:
                if i != 0:
                    x_out = self.extra_norms[i - 1](x_out)
                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        if bundle is not None:
            return tuple(outs), bundle
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


def DFormerv2_S(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        heads_ranges=[4, 4, 6, 6],
        **kwargs,
    )
    return model


def DFormerv2_B(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 25, 8],
        num_heads=[5, 5, 10, 16],
        heads_ranges=[5, 5, 6, 6],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs,
    )
    return model


def DFormerv2_L(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20],
        heads_ranges=[6, 6, 6, 6],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs,
    )
    return model
