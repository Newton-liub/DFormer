"""
DFormer++ encoder (journal version of DFormer, TPAMI 2026).

Depth-guided local-global attention built upon the TransNeXt aggregated-attention design.
Optional CUDA extension `swattention` accelerates local window attention; a pure-PyTorch
unfold fallback is used when the extension is unavailable.
"""

from collections import OrderedDict
from functools import partial
from itertools import repeat
import collections.abc
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner.checkpoint import load_state_dict

try:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    def _ntuple(n):
        def parse(x):
            if isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes)):
                return tuple(x)
            return tuple(repeat(x, n))

        return parse

    to_2tuple = _ntuple(2)

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        with torch.no_grad():
            return tensor.normal_(mean, std).clamp_(a, b)

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            return x.div(keep_prob) * random_tensor

CUDA_NUM_THREADS = 128

try:
    import swattention

    _HAS_SWATTENTION = True
except ImportError:
    swattention = None
    _HAS_SWATTENTION = False
    warnings.warn(
        "swattention CUDA extension not found; DFormer++ falls back to native local attention.",
        stacklevel=2,
    )


if _HAS_SWATTENTION:

    class sw_qkrpb_cuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, rpb, height, width, kernel_size):
            attn_weight = swattention.qk_rpb_forward(
                query, key, rpb, height, width, kernel_size, CUDA_NUM_THREADS
            )
            ctx.save_for_backward(query, key)
            ctx.height, ctx.width, ctx.kernel_size = height, width, kernel_size
            return attn_weight

        @staticmethod
        def backward(ctx, d_attn_weight):
            query, key = ctx.saved_tensors
            height, width, kernel_size = ctx.height, ctx.width, ctx.kernel_size
            d_query, d_key, d_rpb = swattention.qk_rpb_backward(
                d_attn_weight.contiguous(),
                query,
                key,
                height,
                width,
                kernel_size,
                CUDA_NUM_THREADS,
            )
            return d_query, d_key, d_rpb, None, None, None

    class sw_av_cuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, attn_weight, value, height, width, kernel_size):
            output = swattention.av_forward(
                attn_weight, value, height, width, kernel_size, CUDA_NUM_THREADS
            )
            ctx.save_for_backward(attn_weight, value)
            ctx.height, ctx.width, ctx.kernel_size = height, width, kernel_size
            return output

        @staticmethod
        def backward(ctx, d_output):
            attn_weight, value = ctx.saved_tensors
            height, width, kernel_size = ctx.height, ctx.width, ctx.kernel_size
            d_attn_weight, d_value = swattention.av_backward(
                d_output.contiguous(),
                attn_weight,
                value,
                height,
                width,
                kernel_size,
                CUDA_NUM_THREADS,
            )
            return d_attn_weight, d_value, None, None, None


def get_seqlen_and_mask(input_resolution, window_size, device):
    attn_map = F.unfold(
        torch.ones([1, 1, input_resolution[0], input_resolution[1]], device=device),
        window_size,
        dilation=1,
        padding=(window_size // 2, window_size // 2),
        stride=1,
    )
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x, H, W):
        x = self.norm(x)
        x = self.fc1(x)
        b,_,dim=x.shape
        x = x.view(b,H,W,dim).permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1).view(b, H*W, dim)
        x = self.act(x)
        x = self.fc2(x)

        return x

@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None,
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table


@torch.no_grad()
def get_seqlen_scale(input_resolution, window_size, device):
    return torch.nn.functional.avg_pool2d(
        torch.ones(1, input_resolution[0], input_resolution[1], device=device) * (window_size ** 2), window_size,
        stride=1, padding=window_size // 2, ).reshape(-1, 1)


class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, is_extrapolation=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        self.is_extrapolation = is_extrapolation

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W
            self.trained_pool_H, self.trained_pool_W = input_resolution[0] // self.sr_ratio, input_resolution[
                1] // self.sr_ratio
            self.trained_pool_len = self.trained_pool_H * self.trained_pool_W

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2
        self.use_cuda = _HAS_SWATTENTION
        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)

        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.
        self.temperature_e = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_e = nn.Linear(dim // 2, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.query_embedding_e = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv_e = nn.Linear(dim // 2, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_e = nn.Linear(dim, dim // 2)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.sr_e = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.norm_e = nn.LayerNorm(dim // 2)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative_bias_local:
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0, std=0.0004))

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def _local_attn_cuda(self, q_norm_scaled, k_local, v_local, H, W):
        attn_local = sw_qkrpb_cuda.apply(
            q_norm_scaled.contiguous(),
            F.normalize(k_local, dim=-1).contiguous(),
            self.relative_pos_bias_local,
            H,
            W,
            self.window_size,
        )
        return attn_local, v_local

    def _local_attn_native(self, q_norm_scaled, k_local, v_local, H, W):
        B, _, N, _ = k_local.shape
        k_flat = F.normalize(k_local, dim=-1).permute(0, 2, 1, 3).reshape(B, N, -1)
        v_flat = v_local.permute(0, 2, 1, 3).reshape(B, N, -1)
        kv_local = torch.cat([k_flat, v_flat], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local, v_local = (
            self.unfold(kv_local)
            .reshape(B, 2 * self.num_heads, self.head_dim, self.local_len, N)
            .permute(0, 1, 4, 2, 3)
            .chunk(2, dim=1)
        )
        _, padding_mask = get_seqlen_and_mask((H, W), self.window_size, device=q_norm_scaled.device)
        attn_local = (q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) + self.relative_pos_bias_local.unsqueeze(1)
        attn_local = attn_local.masked_fill(padding_mask, float("-inf"))
        return attn_local, v_local

    def forward(
        self,
        x,
        x_e,
        H,
        W,
        relative_pos_index,
        relative_coords_table,
        relative_pos_index_e,
        relative_coords_table_e,
        seq_length_scale,
    ):
        B, N, C = x.shape
        pool_H, pool_W = H // self.sr_ratio, W // self.sr_ratio
        pool_len = pool_H * pool_W

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        # Use softplus function ensuring that the temperature is not lower than 0.
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * seq_length_scale

        q_norm_e = F.normalize(self.q_e(x_e).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled_e = (q_norm_e + self.query_embedding_e) * F.softplus(self.temperature_e) * seq_length_scale

        q_norm_scaled = q_norm_scaled + q_norm_scaled_e

        # Generate keys / values and fuse RGB-D embeddings
        k_local, v_local = self.kv(x).reshape(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).chunk(2, dim=1)
        k_local_e, v_local_e = (
            self.kv_e(x_e).reshape(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).chunk(2, dim=1)
        )
        k_local = k_local + k_local_e
        v_local = v_local + v_local_e

        if self.use_cuda:
            attn_local, v_local_win = self._local_attn_cuda(q_norm_scaled, k_local, v_local, H, W)
        else:
            attn_local, v_local_win = self._local_attn_native(q_norm_scaled, k_local, v_local, H, W)

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = F.adaptive_avg_pool2d(self.act(self.sr(x_)), (pool_H, pool_W)).reshape(B, -1, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        x_e_ = x_e.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_e_ = (
            F.adaptive_avg_pool2d(self.act(self.sr_e(x_e_)), (pool_H, pool_W)).reshape(B, -1, pool_len).permute(0, 2, 1)
        )
        x_e_ = self.norm_e(x_e_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        kv_pool_e = self.kv_e(x_e_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool_e, v_pool_e = kv_pool_e.chunk(2, dim=1)

        k_pool = k_pool + k_pool_e
        v_pool = v_pool + v_pool_e

        if self.is_extrapolation:
            pool_bias = (
                self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table)))
                .transpose(0, 1)[:, relative_pos_index.view(-1)]
                .view(-1, N, pool_len)
            )
        else:
            pool_bias = (
                self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table)))
                .transpose(0, 1)[:, relative_pos_index.view(-1)]
                .view(-1, self.trained_len, self.trained_pool_len)
            )
            pool_bias = pool_bias.reshape(-1, self.trained_len, self.trained_pool_H, self.trained_pool_W)
            pool_bias = F.interpolate(pool_bias, (pool_H, pool_W), mode="bilinear")
            pool_bias = (
                pool_bias.reshape(-1, self.trained_len, pool_len)
                .transpose(-1, -2)
                .reshape(-1, pool_len, self.trained_H, self.trained_W)
            )
            pool_bias = F.interpolate(pool_bias, (H, W), mode="bilinear").reshape(-1, pool_len, N).transpose(-1, -2)

        attn_pool = torch.matmul(q_norm_scaled, F.normalize(k_pool, dim=-1).transpose(-2, -1)) + pool_bias

        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)
        attn_local = torch.matmul(q_norm, self.learnable_tokens) + self.learnable_bias + attn_local
        if self.use_cuda:
            x_local = sw_av_cuda.apply(attn_local.type_as(v_local_win), v_local_win.contiguous(), H, W, self.window_size)
        else:
            x_local = (attn_local.unsqueeze(-2) @ v_local_win.transpose(-2, -1)).squeeze(-2)
        x_pool = torch.matmul(attn_pool, v_pool)
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        x, x_e = self.proj(x), self.proj_e(x)
        x, x_e = self.proj_drop(x), self.proj_drop(x_e)

        return x, x_e


class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, attn_drop=0.,
                 proj_drop=0., is_extrapolation=False, drop_last_depth=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.is_extrapolation = is_extrapolation

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_e = nn.Linear(dim//2, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.drop_last_depth = drop_last_depth
        if not self.drop_last_depth:
            self.proj_e = nn.Linear(dim, dim//2)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

    def forward(self, x, x_e, H, W, relative_pos_index, relative_coords_table, relative_pos_index_e, relative_coords_table_e, seq_length_scale):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        qkv_e = self.qkv_e(x_e).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k, v = qkv.chunk(3, dim=1)
        q_e, k_e, v_e = qkv_e.chunk(3, dim=1)
        q, k, v = q+q_e, k+k_e, v+v_e 

        if self.is_extrapolation:
            # Use MLP to generate continuous relative positional bias
            rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                       relative_pos_index.view(-1)].view(-1, N, N)
        else:
            # Use MLP to generate continuous relative positional bias
            rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                       relative_pos_index.view(-1)].view(-1, self.trained_len, self.trained_len)
            # bilinear interpolation:
            rel_bias = rel_bias.reshape(-1, self.trained_len, self.trained_H, self.trained_W)
            rel_bias = F.interpolate(rel_bias, (H, W), mode='bilinear')
            rel_bias = rel_bias.reshape(-1, self.trained_len, N).transpose(-1, -2).reshape(-1, N, self.trained_H,
                                                                                           self.trained_W)
            rel_bias = F.interpolate(rel_bias, (H, W), mode='bilinear').reshape(-1, N, N).transpose(-1, -2)

        attn = torch.matmul(((F.normalize(q, dim=-1) + self.query_embedding) * F.softplus(
            self.temperature) * seq_length_scale) , F.normalize(k, dim=-1).transpose(-2, -1)) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn , v).transpose(1, 2).reshape(B, N, C)

        if not self.drop_last_depth:
            x, x_e = self.proj(x), self.proj_e(x)
            x, x_e = self.proj_drop(x), self.proj_drop(x_e)
        else:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x, x_e


class Block(nn.Module):

    def __init__(self, dim, num_heads, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, is_extrapolation=False, drop_last_depth=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_e = norm_layer(dim//2)
        self.drop_last_depth = drop_last_depth
        self.dim = dim
        if sr_ratio == 1:
            self.attn = Attention(
                dim,
                input_resolution,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                is_extrapolation=is_extrapolation, drop_last_depth=drop_last_depth)
            # self.attn_e = Attention(
            #     dim//2,
            #     input_resolution,
            #     num_heads=num_heads,
            #     qkv_bias=qkv_bias,
            #     attn_drop=attn_drop,
            #     proj_drop=drop,
            #     is_extrapolation=is_extrapolation)
        else:
            self.attn = AggregatedAttention(
                dim,
                input_resolution,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                is_extrapolation=is_extrapolation)
            # self.attn_e = AggregatedAttention(
            #     dim//2,
            #     input_resolution,
            #     window_size=window_size,
            #     num_heads=num_heads,
            #     qkv_bias=qkv_bias,
            #     attn_drop=attn_drop,
            #     proj_drop=drop,
            #     sr_ratio=sr_ratio,
            #     is_extrapolation=is_extrapolation)
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)

        if not self.drop_last_depth:
            self.norm2_e = norm_layer(dim//2)
            self.mlp_e = MLP(dim=dim//2, mlp_ratio=mlp_ratio)
        

        # self.linear_rgb = nn.Linear(dim*3//2, dim)
        # if not self.drop_last_depth:
        #     self.linear_depth = nn.Linear(dim*3//2, dim//2)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, x_e, H, W, relative_pos_index, relative_coords_table, relative_pos_index_e, relative_coords_table_e, seq_length_scale):
        attn_x, attn_x_e = self.attn(self.norm1(x), self.norm1_e(x_e),  H, W, relative_pos_index, relative_coords_table, relative_pos_index_e, relative_coords_table_e, seq_length_scale)
        
        # attn = torch.cat([attn_x, attn_x_e], dim=2)
        # attn_x = self.linear_rgb(attn)
        if not self.drop_last_depth:
            # attn_x_e = self.linear_depth(attn)
            x_e = x_e + self.drop_path(attn_x_e)
        x = x + self.drop_path(attn_x)
      
        

        
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        if not self.drop_last_depth:
            x_e = x_e + self.drop_path(self.mlp_e(self.norm2_e(x_e), H, W))

        return x, x_e


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class TransNeXt(nn.Module):
    '''
    The parameter "img size" is primarily utilized for generating relative spatial coordinates,
    which are used to compute continuous relative positional biases. As this TransNeXt implementation can accept multi-scale inputs,
    it is recommended to set the "img size" parameter to a value close to the resolution of the inference images.
    It is not advisable to set the "img size" parameter to a value exceeding 800x800.
    The "pretrain size" refers to the "img size" used during the initial pre-training phase,
    which is used to scale the relative spatial coordinates for better extrapolation by the MLP.
    For models trained on ImageNet-1K at a resolution of 224x224,
    as well as downstream task models fine-tuned based on these pre-trained weights,
    the "pretrain size" parameter should be set to 224x224.
    '''

    def __init__(self, img_size=224, pretrain_size=None, window_size=[3, 3, 3, None],
                 patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, pretrained=None, is_extrapolation=False):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.window_size = window_size
        self.sr_ratios = sr_ratios
        self.is_extrapolation = is_extrapolation
        self.pretrain_size = pretrain_size or img_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if not self.is_extrapolation:
                relative_pos_index, relative_coords_table = get_relative_position_cpb(
                    query_size=to_2tuple(img_size // (2 ** (i + 2))),
                    key_size=to_2tuple(img_size // ((2 ** (i + 2)) * sr_ratios[i])),
                    pretrain_size=to_2tuple(pretrain_size // (2 ** (i + 2))))

                self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
                self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

                relative_pos_index_e, relative_coords_table_e = get_relative_position_cpb(
                    query_size=to_2tuple(img_size // (2 ** (i + 2))),
                    key_size=to_2tuple(img_size // ((2 ** (i + 2)) * sr_ratios[i])),
                    pretrain_size=to_2tuple(pretrain_size // (2 ** (i + 2))))

                self.register_buffer(f"relative_pos_index_e{i + 1}", relative_pos_index_e, persistent=False)
                self.register_buffer(f"relative_coords_table_e{i + 1}", relative_coords_table_e, persistent=False)

            patch_embed = OverlapPatchEmbed(patch_size=patch_size * 2 - 1 if i == 0 else 3,
                                            stride=patch_size if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            patch_embed_e = OverlapPatchEmbed(patch_size=patch_size * 2 - 1 if i == 0 else 3,
                                            stride=patch_size if i == 0 else 2,
                                            in_chans=1 if i == 0 else embed_dims[i - 1]//2,
                                            embed_dim=embed_dims[i]//2)
            # drop_last_depth=)
            # for j in range(depths[i]):
                # print('stage',str(i),' and ',str(j),((i==3)&(j==(depths[3]-1))))
            block = nn.ModuleList([Block(
                dim=embed_dims[i], input_resolution=to_2tuple(img_size // (2 ** (i + 2))), window_size=window_size[i],
                num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], is_extrapolation=is_extrapolation, drop_last_depth=((i==3) & (j==(depths[3]-1))))
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            if not ((i==3)):
                norm_e = norm_layer(embed_dims[i]//2)
                setattr(self, f"norm_e{i + 1}", norm_e)
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"patch_embed_e{i + 1}", patch_embed_e)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        for n, m in self.named_modules():
            self._init_weights(m, n)
        if pretrained:
            self.init_weights(pretrained)

    def _init_weights(self, m: nn.Module, name: str = ''):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def init_weights(self, pretrained=None):
        if not isinstance(pretrained, str):
            return
        _state_dict = torch.load(pretrained, map_location="cpu")
        if "state_dict_ema" in _state_dict:
            _state_dict = _state_dict["state_dict_ema"]
        elif "state_dict" in _state_dict:
            _state_dict = _state_dict["state_dict"]
        elif "model" in _state_dict:
            _state_dict = _state_dict["model"]

        state_dict = OrderedDict()
        for k, v in _state_dict.items():
            if k.startswith("backbone."):
                state_dict[k[9:]] = v
            else:
                state_dict[k] = v
        if state_dict and list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        load_state_dict(self, state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'query_embedding', 'relative_pos_bias_local', 'cpb', 'temperature'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, x_e):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            patch_embed_e = getattr(self, f"patch_embed_e{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            if not (i==3):
                norm_e = getattr(self, f"norm_e{i + 1}")
            x, H, W = patch_embed(x)
            x_e, _, _ = patch_embed_e(x_e) 
            sr_ratio = self.sr_ratios[i]
            if self.is_extrapolation:
                relative_pos_index, relative_coords_table = get_relative_position_cpb(query_size=(H, W),
                                                                                      key_size=(
                                                                                          H // sr_ratio,
                                                                                          W // sr_ratio),
                                                                                      pretrain_size=to_2tuple(
                                                                                          self.pretrain_size // (
                                                                                                  2 ** (i + 2))),
                                                                                      device=x.device)
                relative_pos_index_e, relative_coords_table_e = get_relative_position_cpb(query_size=(H, W),
                                                                                      key_size=(
                                                                                          H // sr_ratio,
                                                                                          W // sr_ratio),
                                                                                      pretrain_size=to_2tuple(
                                                                                          self.pretrain_size // (
                                                                                                  2 ** (i + 2))),
                                                                                      device=x_e.device)
            else:
                relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
                relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")
                relative_pos_index_e = getattr(self, f"relative_pos_index_e{i + 1}")
                relative_coords_table_e = getattr(self, f"relative_coords_table_e{i + 1}")

            with torch.no_grad():
                if i != (self.num_stages - 1):
                    local_seq_length = get_seqlen_scale((H, W), self.window_size[i], device=x.device)
                    seq_length_scale = torch.log(local_seq_length + (H // sr_ratio) * (W // sr_ratio))
                else:
                    seq_length_scale = torch.log(torch.as_tensor((H // sr_ratio) * (W // sr_ratio), device=x.device))

            for blk in block:
                x, x_e = blk(x, x_e, H, W, relative_pos_index, relative_coords_table, relative_pos_index_e, relative_coords_table_e, seq_length_scale)

            x = norm(x)
            if not (i==3):
                x_e = norm_e(x_e)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x, x_e):
        x_e = x_e[:, 0, :, :].unsqueeze(1)
        x = self.forward_features(x, x_e)
        return x, None


def _resolve_size_args(kwargs):
    """Normalize constructor kwargs used by the segmentation builder."""
    import numpy as np

    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    pretrained = kwargs.pop("pretrained", None)
    img_size = kwargs.pop("img_size", 224)
    pretrain_size = kwargs.pop("pretrain_size", 224)
    kwargs.pop("norm_cfg", None)
    if isinstance(img_size, (list, tuple)):
        img_size = np.asarray(img_size)
    elif hasattr(img_size, "tolist") and not isinstance(img_size, np.ndarray):
        img_size = np.asarray(img_size)
    return drop_path_rate, pretrained, img_size, pretrain_size


class DFormerPP_T(TransNeXt):
    """DFormer++-Tiny."""

    def __init__(self, **kwargs):
        drop_path_rate, pretrained, img_size, pretrain_size = _resolve_size_args(kwargs)
        super().__init__(
            window_size=[3, 3, 3, None],
            patch_size=4,
            embed_dims=[48, 96, 192, 384],
            num_heads=[2, 4, 8, 16],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 15, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
            img_size=img_size,
            pretrain_size=pretrain_size,
        )


class DFormerPP_S(TransNeXt):
    """DFormer++-Small."""

    def __init__(self, **kwargs):
        drop_path_rate, pretrained, img_size, pretrain_size = _resolve_size_args(kwargs)
        super().__init__(
            window_size=[3, 3, 3, None],
            patch_size=4,
            embed_dims=[72, 144, 288, 576],
            num_heads=[3, 6, 12, 24],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 15, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
            img_size=img_size,
            pretrain_size=pretrain_size,
        )


class DFormerPP_B(TransNeXt):
    """DFormer++-Base."""

    def __init__(self, **kwargs):
        drop_path_rate, pretrained, img_size, pretrain_size = _resolve_size_args(kwargs)
        super().__init__(
            window_size=[3, 3, 3, None],
            patch_size=4,
            embed_dims=[72, 144, 288, 576],
            num_heads=[3, 6, 12, 24],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[5, 5, 22, 5],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
            img_size=img_size,
            pretrain_size=pretrain_size,
        )


# Convenience aliases
DFormerPP_Tiny = DFormerPP_T
DFormerPP_Small = DFormerPP_S
DFormerPP_Base = DFormerPP_B