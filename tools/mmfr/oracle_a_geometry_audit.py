#!/usr/bin/env python3
"""Read-only CPU audit of the DFormerv2 geometry path for ``MMFR-Oracle-A``.

Answers, with numbers instead of prose:

1. what the checkpoint's learned geometry weights ``Geo.weight = [w_position, w_depth]``
   actually are, per stage;
2. how the raw Depth enters the geometry prior (normalisation, interpolation, the
   ``|d_i - d_j| * decay`` term) and how large that term is relative to the
   position/spatial term;
3. how invalid Depth behaves inside that term: the Depth value invalid pixels carry, the
   pair-term distribution for valid-valid / valid-invalid / invalid-invalid pairs, and the
   share of the total prior magnitude that invalid pairs are responsible for;
4. the gate statistics the two Oracle-A modes produce for the same view.

Nothing is trained, no GPU is required, no evaluator or corruption code is modified.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import tools.evaluate_museg_10condition as E10
import tools.evaluate_museg_checkpoint as EV
from models.encoders.DFormerv2 import build_oracle_pairwise_gates, build_oracle_stage_reliability
from utils.dataloader.oracle_a_validity import build_view_invalidity, raw_invalidity

DEFAULT_CONFIG = "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v3"
DEFAULT_DATASET_ROOT = r"D:\0Project\dataset\MUSeg_DFormer"
DEFAULT_SPLIT = r"data\splits\MUSeg\dev-v1\val-dev.txt"
STAGE_DIVISORS = (4, 8, 16, 32)
MAX_PAIRS_SAMPLED = 4_000_000


def _sample_pairs(mask_rows: np.ndarray, mask_columns: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    count = min(int(mask_rows.size) * int(mask_columns.size), MAX_PAIRS_SAMPLED)
    rows = mask_rows[rng.integers(0, mask_rows.size, size=count)]
    columns = mask_columns[rng.integers(0, mask_columns.size, size=count)]
    return rows, columns


def audit_weights(model: torch.nn.Module) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for layer_index, layer in enumerate(model.backbone.layers):
        values = [block.Geo.weight.detach().reshape(-1).tolist() for block in layer.blocks]
        array = np.asarray(values, dtype=np.float64)
        stages[f"layer_{layer_index}"] = {
            "block_count": len(values),
            "position_weight": {
                "min": float(array[:, 0].min()),
                "max": float(array[:, 0].max()),
                "mean": float(array[:, 0].mean()),
            },
            "depth_weight": {
                "min": float(array[:, 1].min()),
                "max": float(array[:, 1].max()),
                "mean": float(array[:, 1].mean()),
            },
            "initialised_to": [1.0, 1.0],
        }
    return stages


def audit_view(
    model: torch.nn.Module,
    depth_view: torch.Tensor,
    invalidity_view: np.ndarray,
    stage_divisor: int,
    layer_index: int,
    block_index: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    padded_height, padded_width = int(depth_view.shape[-2]), int(depth_view.shape[-1])
    height, width = padded_height // stage_divisor, padded_width // stage_divisor
    depth_plane = depth_view.unsqueeze(0)[:, 0:1].float()
    interpolated = F.interpolate(depth_plane, size=(height, width), mode="bilinear", align_corners=False)
    generator = model.backbone.layers[layer_index].blocks[block_index].Geo
    tensor = torch.from_numpy(np.ascontiguousarray(invalidity_view))
    strict = build_oracle_stage_reliability(
        {"mode": "strict", "invalidity": tensor, "depth_geometry_off": False}, (height, width), tensor
    )
    continuous = build_oracle_stage_reliability(
        {"mode": "continuous", "invalidity": tensor, "depth_geometry_off": False}, (height, width), tensor
    )
    split_or_not = layer_index != 3
    depth_values = interpolated[0, 0].reshape(-1).numpy()
    strict_values = strict[0, 0].reshape(-1).numpy()
    valid_index = np.nonzero(strict_values == 1)[0]
    invalid_index = np.nonzero(strict_values == 0)[0]
    if valid_index.size == 0 or invalid_index.size == 0:
        pair_statistics: dict[str, Any] = {"error": "one pair class is empty"}
    else:
        classes = {
            "valid-valid": (valid_index, valid_index),
            "valid-invalid": (valid_index, invalid_index),
            "invalid-invalid": (invalid_index, invalid_index),
        }
        pair_statistics = {}
        for name, (rows_index, columns_index) in classes.items():
            rows, columns = _sample_pairs(rows_index, columns_index, rng)
            difference = np.abs(depth_values[rows] - depth_values[columns])
            # Same arithmetic as generate_depth_decay / generate_1d_depth_decay, i.e. the
            # raw depth term before the learned weight is applied.
            # decay is per head; report the head-mean decay magnitude.
            decay_mean = float(generator.decay.detach().abs().mean().item())
            raw_term = difference * decay_mean
            pair_statistics[name] = {
                "sampled_pairs": int(difference.size),
                "abs_depth_difference_mean": float(difference.mean()),
                "abs_depth_difference_median": float(np.median(difference)),
                "abs_depth_difference_p90": float(np.percentile(difference, 90)),
                "raw_depth_term_mean": float(raw_term.mean()),
                "raw_depth_term_min": float(raw_term.min()),
                "raw_depth_term_max": float(raw_term.max()),
            }
    with torch.inference_mode():
        _, baseline_mask = generator.forward((height, width), depth_plane, split_or_not=split_or_not)
    if isinstance(baseline_mask, tuple):
        position_term = generator.weight[0].detach() * generator.generate_1d_decay(height).unsqueeze(0).unsqueeze(2)
        depth_term = baseline_mask[0] - position_term
    else:
        position_term = generator.weight[0].detach() * generator.generate_pos_decay(height, width)
        depth_term = baseline_mask - position_term
    return {
        "stage_divisor": stage_divisor,
        "stage_grid_hw": [height, width],
        "layer_index": layer_index,
        "block_index": block_index,
        "split_or_not": split_or_not,
        "learned_position_weight": float(generator.weight[0].detach().item()),
        "learned_depth_weight": float(generator.weight[1].detach().item()),
        "decay_abs_mean": float(generator.decay.detach().abs().mean().item()),
        "decay_abs_min": float(generator.decay.detach().abs().min().item()),
        "decay_abs_max": float(generator.decay.detach().abs().max().item()),
        "strict_valid_token_fraction": float((strict == 1).float().mean().item()),
        "continuous_reliability_mean": float(continuous.mean().item()),
        "position_term_abs_mean": float(position_term.abs().mean().item()),
        "depth_term_abs_mean": float(depth_term.abs().mean().item()),
        "depth_term_share_of_prior": float(
            depth_term.abs().mean().item()
            / max(1.0e-12, position_term.abs().mean().item() + depth_term.abs().mean().item())
        ),
        "invalid_depth_token_normalized_value": float(
            depth_values[invalid_index].mean() if invalid_index.size else float("nan")
        ),
        "valid_depth_token_normalized_value": float(
            depth_values[valid_index].mean() if valid_index.size else float("nan")
        ),
        "pair_statistics": pair_statistics,
    }


def collect_block_inputs(model: torch.nn.Module, rgb_view: torch.Tensor, depth_plane: torch.Tensor) -> dict[Any, torch.Tensor]:
    """Capture the input features of every ``RGBD_Block`` from one real backbone forward."""
    captures: dict[Any, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(model.backbone.layers):
        for block_index, block in enumerate(layer.blocks):
            def hook(module, args, kwargs, key=(layer_index, block_index)):
                tensor = args[0] if args else kwargs.get("x")
                if tensor is not None:
                    captures[key] = tensor.detach().clone()

            handles.append(block.register_forward_pre_hook(hook, with_kwargs=True))
    with torch.inference_mode():
        model.backbone(rgb_view.unsqueeze(0), depth_plane)
    for handle in handles:
        handle.remove()
    return captures


def attention_effect_for_block(
    model: torch.nn.Module,
    x: torch.Tensor,
    depth_plane: torch.Tensor,
    invalidity_view: np.ndarray,
    layer_index: int,
    block_index: int,
    stage_divisor: int,
    query_samples: int = 32,
) -> dict[str, Any]:
    """Measure how much the strict Oracle gate reshapes one block's attention weights."""
    block = model.backbone.layers[layer_index].blocks[block_index]
    attention = block.Attention
    with torch.inference_mode():
        x = x + block.cnn_pos_encode(x)
        normalised = block.layer_norm1(x)
        height, width = int(x.shape[1]), int(x.shape[2])
        tensor = torch.from_numpy(np.ascontiguousarray(invalidity_view))
        oracle = {"mode": "strict", "invalidity": tensor, "depth_geometry_off": False}
        oracle_off = {"mode": "continuous", "invalidity": None, "depth_geometry_off": True}
        split_or_not = layer_index != 3
        (sin, cos), reference_bias = block.Geo((height, width), depth_plane, split_or_not=split_or_not)
        (_, _), gated_bias = block.Geo(
            (height, width), depth_plane, split_or_not=split_or_not, geometry_oracle=oracle
        )
        (_, _), off_bias = block.Geo(
            (height, width), depth_plane, split_or_not=split_or_not, geometry_oracle=oracle_off
        )
        from models.encoders.DFormerv2 import angle_transform

        q = attention.q_proj(normalised)
        k = attention.k_proj(normalised) * attention.scaling
        q = q.view(1, height, width, attention.num_heads, attention.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(1, height, width, attention.num_heads, attention.key_dim).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)
        reliability = build_oracle_stage_reliability(oracle, (height, width), depth_plane)[0, 0]
        invalid = reliability == 0

        def row_statistics(
            query_rotated: torch.Tensor,
            key_rotated: torch.Tensor,
            bias: torch.Tensor,
            bias_gated: torch.Tensor,
            bias_off: torch.Tensor,
            key_invalid_source: torch.Tensor,
        ) -> dict[str, Any]:
            row_count = int(query_rotated.shape[1])
            position_count = int(query_rotated.shape[2])
            row_index = torch.linspace(0, row_count - 1, steps=min(query_samples, row_count)).long()
            column_index = torch.linspace(
                0, position_count - 1, steps=min(query_samples, position_count)
            ).long()
            queries = query_rotated[0][row_index][:, :, column_index, :]
            keys = key_rotated[0][row_index]
            logits = queries @ keys.transpose(-1, -2)
            original = torch.softmax(logits + bias[0][row_index][:, :, column_index, :], -1)
            gated = torch.softmax(logits + bias_gated[0][row_index][:, :, column_index, :], -1)
            off = torch.softmax(logits + bias_off[0][row_index][:, :, column_index, :], -1)
            key_invalid = key_invalid_source[row_index].reshape(row_index.numel(), 1, 1, -1).float()
            if key_invalid.shape[-1] != original.shape[-1]:
                raise RuntimeError("attention key mask does not match the key axis")
            return {
                "attention_rows": int(original.numel() // original.shape[-1]),
                "mean_total_variation": float((0.5 * (original - gated).abs().sum(-1)).mean().item()),
                "mean_total_variation_depth_geometry_off": float(
                    (0.5 * (original - off).abs().sum(-1)).mean().item()
                ),
                "mean_attention_mass_on_invalid_keys_original": float((original * key_invalid).sum(-1).mean().item()),
                "mean_attention_mass_on_invalid_keys_gated": float((gated * key_invalid).sum(-1).mean().item()),
                "uniform_mass_on_invalid_keys": float(key_invalid.mean().item()),
            }

        if split_or_not:
            mask_h, mask_w = reference_bias
            mask_h_gated, mask_w_gated = gated_bias
            mask_h_off, mask_w_off = off_bias
            qr_w = qr.transpose(1, 2)
            kr_w = kr.transpose(1, 2)
            w_statistics = row_statistics(
                qr_w,
                kr_w,
                mask_w.transpose(1, 2),
                mask_w_gated.transpose(1, 2),
                mask_w_off.transpose(1, 2),
                key_invalid_source=invalid,
            )
            qr_h = qr.permute(0, 3, 1, 2, 4)
            kr_h = kr.permute(0, 3, 1, 2, 4)
            h_statistics = row_statistics(
                qr_h,
                kr_h,
                mask_h.transpose(1, 2),
                mask_h_gated.transpose(1, 2),
                mask_h_off.transpose(1, 2),
                key_invalid_source=invalid.transpose(0, 1),
            )
        else:
            query_index = torch.linspace(0, invalid.numel() - 1, steps=min(64, invalid.numel())).long()
            q_flat = qr.flatten(2, 3)[0]
            k_flat = kr.flatten(2, 3)[0]
            logits = q_flat[:, query_index] @ k_flat.transpose(-1, -2)
            original = torch.softmax(logits + reference_bias[0][:, query_index, :], -1)
            gated = torch.softmax(logits + gated_bias[0][:, query_index, :], -1)
            off = torch.softmax(logits + off_bias[0][:, query_index, :], -1)
            key_invalid = invalid.reshape(1, 1, -1).float()
            w_statistics = {
                "attention_rows": int(original.numel() // original.shape[-1]),
                "mean_total_variation": float((0.5 * (original - gated).abs().sum(-1)).mean().item()),
                "mean_total_variation_depth_geometry_off": float(
                    (0.5 * (original - off).abs().sum(-1)).mean().item()
                ),
                "mean_attention_mass_on_invalid_keys_original": float((original * key_invalid).sum(-1).mean().item()),
                "mean_attention_mass_on_invalid_keys_gated": float((gated * key_invalid).sum(-1).mean().item()),
                "uniform_mass_on_invalid_keys": float(invalid.float().mean().item()),
            }
            h_statistics = None
    return {
        "layer_index": layer_index,
        "block_index": block_index,
        "stage_divisor": stage_divisor,
        "stage_grid_hw": [height, width],
        "split_or_not": split_or_not,
        "invalid_token_fraction": float((reliability == 0).float().mean().item()),
        "full_attention": w_statistics,
        "w_direction": w_statistics if split_or_not else None,
        "h_direction": h_statistics,
        "query_sample_note": "strided query rows; the key set is the full row/column of the query index",
    }


def attention_effect_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def _collect(direction: str, key: str) -> list[float]:
        values = []
        for record in records:
            entry = record.get(direction)
            if entry:
                values.append(float(entry[key]))
        return values

    summary: dict[str, Any] = {"block_count": len(records)}
    for direction in ("w_direction", "h_direction", "full_attention"):
        values = _collect(direction, "mean_total_variation")
        if not values:
            continue
        summary[direction] = {
            "mean_total_variation_mean": float(np.mean(values)),
            "mean_total_variation_max": float(np.max(values)),
            "mean_total_variation_min": float(np.min(values)),
            "mean_total_variation_depth_geometry_off_mean": float(
                np.mean(_collect(direction, "mean_total_variation_depth_geometry_off"))
            ),
            "mean_total_variation_depth_geometry_off_max": float(
                np.max(_collect(direction, "mean_total_variation_depth_geometry_off"))
            ),
            "mass_on_invalid_keys_original_mean": float(
                np.mean(_collect(direction, "mean_attention_mass_on_invalid_keys_original"))
            ),
            "mass_on_invalid_keys_gated_mean": float(
                np.mean(_collect(direction, "mean_attention_mass_on_invalid_keys_gated"))
            ),
            "uniform_mass_mean": float(np.mean(_collect(direction, "uniform_mass_on_invalid_keys"))),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", type=Path, default=Path(DEFAULT_SPLIT))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args(argv)

    config = copy.copy(importlib.import_module(args.config).C)
    channel_order = str(config.channel_order)
    entries = EV.split_entries(args.split.resolve())
    entry = entries[int(args.sample_index)]
    sample_id, rgb, depth, label = E10.read_sample(args.dataset_root.resolve(), entry, channel_order)
    del label
    cache = E10.build_rgb_view_cache(rgb, config)
    view_entry = next(item for item in cache if float(item["scale"]) == float(args.scale) and not item["flipped"])
    depth_view = E10.build_depth_view(depth, view_entry)
    invalidity_raw = raw_invalidity(depth > 0)
    invalidity_view = build_view_invalidity(
        invalidity_raw, view_entry["scaled_size_hw"], view_entry["padded_size_hw"], False
    )
    model = EV.load_model(config, args.checkpoint.resolve(), torch.device("cpu"))
    rng = np.random.Generator(np.random.PCG64(20260919))

    per_stage = [
        audit_view(model, depth_view, invalidity_view, divisor, layer_index, 0, rng)
        for divisor, layer_index in zip(STAGE_DIVISORS, (0, 1, 2, 3))
    ]
    rgb_view = next(
        item["rgb"]
        for item in cache
        if float(item["scale"]) == float(args.scale) and not item["flipped"]
    )
    depth_plane = depth_view.unsqueeze(0)[:, 0:1].float()
    captures = collect_block_inputs(model, rgb_view, depth_plane)
    attention_records = [
        attention_effect_for_block(
            model,
            captures[(layer_index, block_index)],
            depth_plane,
            invalidity_view,
            layer_index,
            block_index,
            STAGE_DIVISORS[layer_index],
        )
        for layer_index in range(len(model.backbone.layers))
        for block_index in range(len(model.backbone.layers[layer_index].blocks))
    ]
    del captures
    attention = {
        "per_block": attention_records,
        "summary": attention_effect_summary(attention_records),
    }
    report = {
        "schema_version": "mmfr-oracle-a-geometry-audit-v1",
        "sample_id": sample_id,
        "entry": entry,
        "scale": float(args.scale),
        "flipped": False,
        "raw_grid_hw": [int(depth.shape[0]), int(depth.shape[1])],
        "raw_valid_fraction": float((depth > 0).mean()),
        "padded_view_hw": [int(depth_view.shape[-2]), int(depth_view.shape[-1])],
        "depth_normalization": {"mean": 0.48, "std": 0.28, "invalid_raw_value": 0.0,
                                "invalid_normalized_value": (0.0 / 255.0 - 0.48) / 0.28,
                                "padding_normalized_value": 0.0},
        "geometry_weights": audit_weights(model),
        "stages": per_stage,
        "attention_effect": attention,
        "official_test_included": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stages": [
        {k: stage[k] for k in ("stage_divisor", "stage_grid_hw", "learned_position_weight",
                               "learned_depth_weight", "strict_valid_token_fraction",
                               "position_term_abs_mean", "depth_term_abs_mean", "depth_term_share_of_prior")}
        for stage in per_stage
    ], "attention_effect": report["attention_effect"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
