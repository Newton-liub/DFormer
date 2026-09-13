#!/usr/bin/env python3
"""Run staged qualification and GPU preflight for the MUSeg DVG-B1 protocol."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_museg_checkpoint import (
    apply_channel_order,
    build_msflip_views,
    configure_fp32_forward,
    load_model,
    metrics_from_confusion,
)
from tools.mve.dvc_a1_core import (
    CONDITION_IDS,
    aggregate_condition_records,
    build_corruption_masks,
    confusion_matrix,
    image_miou_from_confusion,
    location_group,
    mine_id,
    paired_effect,
    quantized_condition_depth,
    semantic_boundary_iou,
)
from tools.mve.dvg_b1_core import (
    build_view_corruption_mask,
    run_cpu_qualification,
    stage_sizes_from_padded_view,
)
from tools.mve.dvg_b1_protocol import DvgProtocolError, file_sha256, load_dvg_protocol
from models.encoders.DFormerv2 import build_dvg_b1_stage_reliability

P0_PROTOCOL_SHA256 = "e7b9ed0a3c84053736f70a7807bdd4f270ee5bf84a6b85ca5cfd053f9cc47e46"
P1_QUALIFICATION_SHA256 = "e101c972a25ada6749e7d8408172938d104ecfb4eee9266f1fbab7fc632b3dc3"
P2_NOOP_SHA256 = "a9cdbe8ca3b0ef210184499a06702103e71855e8b3529472403aae00fe31df24"
P3_GATE_PREFLIGHT_SHA256 = "bd2ec0f677b89a5a3129c1823c876b5b510262f28d8d96499d023af826c1360c"
RUNTIME_IDENTITY_PATHS = (
    "models/builder.py",
    "models/encoders/DFormerv2.py",
    "tools/mve/dvg_b1_protocol.py",
    "tools/mve/dvg_b1_core.py",
    "tools/mve/run_dvg_b1.py",
    "protocols/dvg-b1-oracle-gsa-v1.template.json",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _archive_existing_artifact(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    sha256 = file_sha256(path)
    archive_path = path.parent / "attempts" / f"{path.stem}-{sha256}.json"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        archive_path.write_bytes(path.read_bytes())
    return {"path": str(archive_path), "sha256": sha256}


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _verify_p0_assets(protocol_path: Path) -> tuple[Any, str]:
    protocol = load_dvg_protocol(protocol_path)
    protocol_sha256 = file_sha256(protocol.path)
    if protocol_sha256 != P0_PROTOCOL_SHA256:
        raise DvgProtocolError(
            f"P0 protocol SHA-256 changed: expected {P0_PROTOCOL_SHA256}, got {protocol_sha256}"
        )
    raw = protocol.raw
    if raw["execution_gates"]["current_maximum_authorized_stage"] != "p0":
        raise DvgProtocolError("P0 protocol unexpectedly authorizes a later execution stage")
    scope = raw["evaluation_scope"]
    assets = {
        Path(str(scope["source_protocol_path"])): str(scope["source_protocol_sha256"]),
        Path(str(scope["source_mask_manifest_path"])): str(scope["source_mask_manifest_sha256"]),
        Path(str(scope["source_allowlist_path"])): str(scope["source_allowlist_sha256"]),
        Path(str(scope["source_allowlist_summary_path"])): str(scope["source_allowlist_summary_sha256"]),
        Path(str(scope["allowlist_summary_path"])): str(scope["allowlist_summary_sha256"]),
        Path(str(raw["split"]["path"])): str(raw["split"]["sha256"]),
    }
    for path, expected in assets.items():
        if not path.is_file():
            raise DvgProtocolError(f"required P0 asset is missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise DvgProtocolError(f"P0 asset SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    if raw["official_test"] != {
        "state": "sealed_unread",
        "included": False,
        "reject_identity_substrings": ["official", "test"],
    }:
        raise DvgProtocolError("official test boundary changed after P0")
    return protocol, protocol_sha256


def _runtime_identity() -> dict[str, Any]:
    source_hashes: dict[str, str] = {}
    for relative_path in RUNTIME_IDENTITY_PATHS:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise DvgProtocolError(f"P1 runtime source is missing: {relative_path}")
        source_hashes[relative_path] = file_sha256(path)
    status = _git_output("status", "--porcelain=v1")
    return {
        "git_head": _git_output("rev-parse", "HEAD").strip(),
        "git_status_porcelain": status.splitlines(),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_sha256": source_hashes,
    }


def run_qualification(protocol_path: Path, output_path: Path) -> tuple[dict[str, Any], Path]:
    protocol, protocol_sha256 = _verify_p0_assets(protocol_path)
    archived_prior_artifact = _archive_existing_artifact(output_path)
    result = run_cpu_qualification(scales=tuple(float(value) for value in protocol.raw["evaluator"]["scales"]))
    result = {
        **result,
        "created_at_utc": _utc_now(),
        "protocol_path": str(protocol.path),
        "protocol_sha256": protocol_sha256,
        "archived_prior_artifact": archived_prior_artifact,
        "runtime_identity": _runtime_identity(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    _atomic_write_json(output_path, result)
    return result, output_path


def run_noop_equivalence(
    protocol_path: Path,
    output_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], Path]:
    protocol, protocol_sha256 = _verify_p0_assets(protocol_path)
    archived_prior_artifact = _archive_existing_artifact(output_path)
    if device.type != "cuda":
        raise ValueError("this authorized P2 run is restricted to the local CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the authorized P2 run")

    evidence_root = Path(str(protocol.raw["evidence"]["output_root"])).resolve()
    qualification_path = evidence_root / "qualification.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if qualification.get("status") != "passed" or qualification.get("failures") != []:
        raise DvgProtocolError("latest P1 qualification is not a clean pass")
    qualification_sha256 = file_sha256(qualification_path)

    checkpoint_path = Path(str(protocol.raw["model"]["checkpoint_path"])).resolve()
    if file_sha256(checkpoint_path) != str(protocol.raw["model"]["checkpoint_sha256"]):
        raise DvgProtocolError("epoch 420 checkpoint SHA-256 mismatch")

    allowlist_path = Path(str(protocol.raw["evaluation_scope"]["source_allowlist_path"])).resolve()
    entries = [line.strip() for line in allowlist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries or any("official" in entry.lower() or "test" in entry.lower() for entry in entries):
        raise DvgProtocolError("DVG-B1 allowlist is empty or crosses the official-test boundary")
    sample_id = Path(entries[0]).stem
    dataset = protocol.raw["dataset"]
    dataset_root = Path(str(dataset["root"])).resolve()
    rgb_path = dataset_root / str(dataset["rgb_directory"]) / f"{sample_id}.jpg"
    depth16_path = dataset_root / str(dataset["depth16_directory"]) / f"{sample_id}.png"
    depth_path = dataset_root / str(dataset["quantized_depth_directory"]) / f"{sample_id}.png"
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth16 = cv2.imread(str(depth16_path), cv2.IMREAD_UNCHANGED)
    depth8 = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    if rgb_bgr is None or depth16 is None or depth8 is None:
        raise FileNotFoundError(f"missing P2 input modality for {sample_id}")
    if depth16.dtype != np.uint16 or depth16.ndim != 2 or rgb_bgr.shape[:2] != depth16.shape or depth8.shape != depth16.shape:
        raise ValueError("P2 RGB/Depth16/Depth8 inputs do not share the frozen geometry and dtypes")
    depth_max_raw = int(protocol.raw["input_contract"]["depth"]["depth_max_raw"])
    q0_depth8 = np.rint(depth16.astype(np.float64) * 255.0 / depth_max_raw).astype(np.uint8)
    if not np.array_equal(q0_depth8, depth8):
        raise DvgProtocolError("q=0 quantized depth is not array-identical to production Depth8")

    config = copy.copy(importlib.import_module(str(protocol.raw["model"]["config_module"])).C)
    rgb_contract = protocol.raw["input_contract"]["rgb"]
    config.channel_order = str(rgb_contract["channel_order"])
    config.normalization_identity = str(rgb_contract["normalization_identity"])
    config.norm_mean = np.asarray(rgb_contract["mean"], dtype=np.float32)
    config.norm_std = np.asarray(rgb_contract["std"], dtype=np.float32)
    if config.channel_order != "RGB" or config.normalization_identity != "rgb-imagenet-rgb-order-v1":
        raise DvgProtocolError("P2 runtime RGB contract differs from the frozen protocol")

    rgb = apply_channel_order(rgb_bgr, config.channel_order)
    clean_view = build_msflip_views(rgb, depth8, config, scales=(0.5,))[0]
    q0_view = build_msflip_views(rgb, q0_depth8, config, scales=(0.5,))[0]
    if clean_view["flipped"] or q0_view["flipped"]:
        raise RuntimeError("P2 selected an unexpected flipped view")
    if not torch.equal(clean_view["depth"], q0_view["depth"]):
        raise DvgProtocolError("q=0 and production Depth8 view tensors are not identical")

    configure_fp32_forward(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = load_model(config, checkpoint_path, device)
    rgb_tensor = clean_view["rgb"].unsqueeze(0).to(device=device, dtype=torch.float32)
    depth_tensor = clean_view["depth"].unsqueeze(0).to(device=device, dtype=torch.float32)
    q0_depth_tensor = q0_view["depth"].unsqueeze(0).to(device=device, dtype=torch.float32)
    zero_corruption = torch.zeros(
        (1, 1, int(rgb_tensor.shape[-2]), int(rgb_tensor.shape[-1])),
        device=device,
        dtype=torch.float32,
    )

    captured: dict[str, tuple[torch.Tensor, ...]] = {}

    def capture_backbone(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, tuple) or len(output) != 4:
            raise RuntimeError("DFormerv2 backbone did not return four stage outputs")
        captured["stages"] = tuple(value.detach() for value in output)

    def compare_tensor(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
        equal = torch.equal(reference, candidate)
        record: dict[str, Any] = {
            "equal": equal,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
            "device": str(candidate.device),
        }
        if not equal and reference.shape == candidate.shape:
            mismatch = torch.ne(reference, candidate).flatten()
            mismatch_indices = torch.nonzero(mismatch, as_tuple=False)
            if mismatch_indices.numel():
                flat_index = int(mismatch_indices[0, 0].item())
                coordinate = [int(value) for value in np.unravel_index(flat_index, tuple(reference.shape))]
                record["first_difference_index"] = coordinate
                record["reference_value"] = float(reference[tuple(coordinate)].item())
                record["candidate_value"] = float(candidate[tuple(coordinate)].item())
        return record

    variants = (
        ("original-call-omitted", depth_tensor, "omitted", None),
        ("oracle-corruption-mask-none", depth_tensor, "explicit", None),
        ("clean", depth_tensor, "explicit", zero_corruption),
        ("q=0", q0_depth_tensor, "explicit", zero_corruption.clone()),
        ("all-trusted-mask", depth_tensor, "explicit", zero_corruption.clone()),
    )
    baseline_stages: tuple[torch.Tensor, ...] | None = None
    baseline_logits: torch.Tensor | None = None
    comparisons: dict[str, Any] = {}
    hook = model.backbone.register_forward_hook(capture_backbone)
    torch.cuda.reset_peak_memory_stats(device)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    try:
        for name, variant_depth, call_style, mask in variants:
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, device)
            captured.clear()
            with torch.inference_mode():
                if call_style == "omitted":
                    logits = model(rgb_tensor, variant_depth)
                else:
                    logits = model(rgb_tensor, variant_depth, oracle_corruption_mask=mask)
            stages = captured.get("stages")
            if stages is None:
                raise RuntimeError(f"P2 failed to capture stage outputs for {name}")
            if baseline_stages is None:
                baseline_stages = tuple(value.clone() for value in stages)
                baseline_logits = logits.detach().clone()
                comparisons[name] = {
                    "role": "reference",
                    "stage_shapes": [list(value.shape) for value in stages],
                    "logits_shape": list(logits.shape),
                    "dtype": str(logits.dtype),
                    "device": str(logits.device),
                }
                continue
            assert baseline_logits is not None
            stage_records = [compare_tensor(reference, candidate) for reference, candidate in zip(baseline_stages, stages)]
            logits_record = compare_tensor(baseline_logits, logits.detach())
            comparisons[name] = {
                "stage_outputs": stage_records,
                "final_pre_softmax_logits": logits_record,
                "all_equal": all(record["equal"] for record in stage_records) and logits_record["equal"],
            }
        torch.cuda.synchronize(device)
    finally:
        hook.remove()

    failures = [name for name, record in comparisons.items() if record.get("all_equal") is False]
    result = {
        "schema_version": "museg-dvg-b1-noop-equivalence-v1",
        "protocol_id": "DVG-B1-oracle-gsa-v1",
        "status": "passed" if not failures else "protocol-blocked",
        "device": str(device),
        "checkpoint_loaded": True,
        "checkpoint_strict_load": True,
        "model_forward_executed": True,
        "gpu_used": True,
        "official_test_included": False,
        "created_at_utc": _utc_now(),
        "protocol_path": str(protocol.path),
        "protocol_sha256": protocol_sha256,
        "qualification_path": str(qualification_path),
        "qualification_sha256": qualification_sha256,
        "archived_prior_artifact": archived_prior_artifact,
        "rng_control": {
            "reason": "LightHamHead NMF2D uses random bases in eval; replay identical RNG state to isolate the no-op interface",
            "cpu_rng_state_replayed_before_each_forward": True,
            "cuda_rng_state_replayed_before_each_forward": True,
            "decoder_rand_init": bool(model.decode_head.hamburger.ham.rand_init),
        },
        "sample_id": sample_id,
        "sample_entry": entries[0],
        "input_files": {
            "rgb": {"path": str(rgb_path), "sha256": file_sha256(rgb_path)},
            "depth16": {"path": str(depth16_path), "sha256": file_sha256(depth16_path)},
            "depth8": {"path": str(depth_path), "sha256": file_sha256(depth_path)},
        },
        "view": {
            "scale": float(clean_view["scale"]),
            "flipped": bool(clean_view["flipped"]),
            "scaled_size_hw": list(clean_view["scaled_size_hw"]),
            "padded_size_hw": list(clean_view["padded_size_hw"]),
            "q0_depth8_array_equal": True,
            "q0_depth_view_tensor_equal": True,
        },
        "comparisons": comparisons,
        "failures": failures,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        },
        "runtime_identity": _runtime_identity(),
    }
    _atomic_write_json(output_path, result)
    return result, output_path


def _restore_view_logits(
    logits: torch.Tensor,
    *,
    scaled_size_hw: tuple[int, int],
    padded_size_hw: tuple[int, int],
    flipped: bool,
    original_size_hw: tuple[int, int],
) -> torch.Tensor:
    if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("P3 model produced non-finite or non-FP32 logits")
    if tuple(logits.shape[-2:]) != padded_size_hw:
        logits = F.interpolate(logits, size=padded_size_hw, mode="bilinear", align_corners=False)
    logits = logits[..., : scaled_size_hw[0], : scaled_size_hw[1]]
    if flipped:
        logits = torch.flip(logits, dims=(-1,))
    return F.interpolate(logits, size=original_size_hw, mode="bilinear", align_corners=False)


def _sampled_geo_audit(
    module: Any,
    *,
    stage_index: int,
    hw: tuple[int, int],
    depth_map: torch.Tensor,
    split_or_not: bool,
    reliability: torch.Tensor,
    expected_reliability: torch.Tensor,
    expected_depth_input: torch.Tensor,
    gate_applied: bool,
    output: Any,
) -> dict[str, Any]:
    height, width = hw
    reliability_equal = torch.equal(reliability, expected_reliability)
    depth_input_equal = torch.equal(depth_map, expected_depth_input)
    stage_depth = F.interpolate(depth_map, size=hw, mode="bilinear", align_corners=False)
    flat_index = int(torch.argmin(reliability[0, 0]).item())
    row, column = divmod(flat_index, width)
    decay = module.decay
    weight_spatial = module.weight[0, 0, 0, 0]
    weight_depth = module.weight[1, 0, 0, 0]

    def vector_record(
        *,
        topology: str,
        first_index: list[int],
        second_index: list[int],
        gate: torch.Tensor,
        spatial_distance: int,
        depth_difference: torch.Tensor,
        actual: torch.Tensor,
    ) -> dict[str, Any]:
        spatial_base = torch.as_tensor(float(spatial_distance), device=decay.device, dtype=decay.dtype) * decay
        depth_base = depth_difference * decay
        spatial_contribution = weight_spatial * spatial_base
        raw_depth_contribution = weight_depth * depth_base
        gated_depth_contribution = weight_depth * (gate * depth_base)
        expected_depth_contribution = gated_depth_contribution if gate_applied else raw_depth_contribution
        expected = spatial_contribution + expected_depth_contribution
        exact = torch.equal(actual, expected)
        return {
            "topology": topology,
            "first_index": first_index,
            "second_index": second_index,
            "gate": float(gate.item()),
            "spatial_distance": spatial_distance,
            "depth_difference": float(depth_difference.item()),
            "spatial_contribution_finite": bool(torch.isfinite(spatial_contribution).all()),
            "raw_depth_contribution_finite": bool(torch.isfinite(raw_depth_contribution).all()),
            "gated_depth_contribution_finite": bool(torch.isfinite(gated_depth_contribution).all()),
            "gate_applied_in_production": gate_applied,
            "production_matches_spatial_plus_expected_depth_exactly": exact,
            "production_reconstruction_max_abs_error": float((actual - expected).abs().max().item()),
            "depth_gate_effect_max_abs": float(
                (gated_depth_contribution - raw_depth_contribution).abs().max().item()
            ),
        }

    samples: list[dict[str, Any]] = []
    if split_or_not:
        (sin, cos), (mask_h, mask_w) = output
        anchor_reliability = reliability[0, 0, row, column]
        h_gates = anchor_reliability * reliability[0, 0, :, column]
        h_differences = (stage_depth[0, 0, row, column] - stage_depth[0, 0, :, column]).abs()
        other_row = int(torch.argmax((1.0 - h_gates) * h_differences).item())
        samples.append(
            vector_record(
                topology="H",
                first_index=[row, column],
                second_index=[other_row, column],
                gate=anchor_reliability * reliability[0, 0, other_row, column],
                spatial_distance=abs(row - other_row),
                depth_difference=(stage_depth[0, 0, row, column] - stage_depth[0, 0, other_row, column]).abs(),
                actual=mask_h[0, :, column, row, other_row],
            )
        )
        w_gates = anchor_reliability * reliability[0, 0, row, :]
        w_differences = (stage_depth[0, 0, row, column] - stage_depth[0, 0, row, :]).abs()
        other_column = int(torch.argmax((1.0 - w_gates) * w_differences).item())
        samples.append(
            vector_record(
                topology="W",
                first_index=[row, column],
                second_index=[row, other_column],
                gate=anchor_reliability * reliability[0, 0, row, other_column],
                spatial_distance=abs(column - other_column),
                depth_difference=(stage_depth[0, 0, row, column] - stage_depth[0, 0, row, other_column]).abs(),
                actual=mask_w[0, :, row, column, other_column],
            )
        )
        topology_shapes = {
            "H": [1, 1, width, height, height],
            "W": [1, 1, height, width, width],
        }
    else:
        (sin, cos), mask = output
        flat_reliability = reliability[0, 0].flatten()
        flat_depth = stage_depth[0, 0].flatten()
        anchor_reliability = flat_reliability[flat_index]
        gates = anchor_reliability * flat_reliability
        differences = (flat_depth[flat_index] - flat_depth).abs()
        other_flat = int(torch.argmax((1.0 - gates) * differences).item())
        other_row, other_column = divmod(other_flat, width)
        samples.append(
            vector_record(
                topology="Full",
                first_index=[row, column],
                second_index=[other_row, other_column],
                gate=anchor_reliability * flat_reliability[other_flat],
                spatial_distance=abs(row - other_row) + abs(column - other_column),
                depth_difference=(flat_depth[flat_index] - flat_depth[other_flat]).abs(),
                actual=mask[0, :, flat_index, other_flat],
            )
        )
        topology_shapes = {"Full": [1, 1, height * width, height * width]}

    minimum = float(reliability.min().item())
    maximum = float(reliability.max().item())
    return {
        "stage_index": stage_index,
        "stage_size_hw": [height, width],
        "split_topology": bool(split_or_not),
        "expected_split_topology": stage_index != 3,
        "topology_matches_stage": bool(split_or_not) == (stage_index != 3),
        "gate_applied_in_production": gate_applied,
        "reliability_shape": list(reliability.shape),
        "reliability_minimum": minimum,
        "reliability_maximum": maximum,
        "reliability_zero_count": int(torch.count_nonzero(reliability == 0).item()),
        "reliability_partial_count": int(
            torch.count_nonzero((reliability > 0) & (reliability < 1)).item()
        ),
        "reliability_non_all_one": minimum < 1.0,
        "reliability_matches_precomputed_exactly": reliability_equal,
        "depth_input_matches_corrupted_view_exactly": depth_input_equal,
        "pairwise_gate_broadcast_shapes": topology_shapes,
        "pairwise_gate_minimum_derived": minimum * minimum,
        "pairwise_gate_maximum_derived": maximum * maximum,
        "pairwise_gate_non_all_one": minimum < 1.0,
        "sin_finite": bool(torch.isfinite(sin).all()),
        "cos_finite": bool(torch.isfinite(cos).all()),
        "sampled_contribution_checks": samples,
        "sampled_production_reconstruction_exact": all(
            row["production_matches_spatial_plus_expected_depth_exactly"] for row in samples
        ),
        "sampled_contributions_finite": all(
            row["spatial_contribution_finite"]
            and row["raw_depth_contribution_finite"]
            and row["gated_depth_contribution_finite"]
            for row in samples
        ),
        "sampled_depth_gate_effect_nonzero": any(
            row["depth_gate_effect_max_abs"] > 0.0 for row in samples
        ),
    }


def run_gate_preflight(
    protocol_path: Path,
    output_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], Path]:
    protocol, protocol_sha256 = _verify_p0_assets(protocol_path)
    archived_prior_artifact = _archive_existing_artifact(output_path)
    if device.type != "cuda":
        raise ValueError("this authorized P3 run is restricted to the local CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the authorized P3 run")

    evidence_root = Path(str(protocol.raw["evidence"]["output_root"])).resolve()
    prerequisite_paths = {
        "qualification": evidence_root / "qualification.json",
        "noop_equivalence": evidence_root / "noop-equivalence.json",
    }
    prerequisites: dict[str, Any] = {}
    for name, path in prerequisite_paths.items():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "passed" or value.get("failures") != []:
            raise DvgProtocolError(f"P3 prerequisite {name} is not a clean pass")
        prerequisites[name] = {"path": str(path), "sha256": file_sha256(path)}

    checkpoint_path = Path(str(protocol.raw["model"]["checkpoint_path"])).resolve()
    if file_sha256(checkpoint_path) != str(protocol.raw["model"]["checkpoint_sha256"]):
        raise DvgProtocolError("epoch 420 checkpoint SHA-256 mismatch")

    scope = protocol.raw["evaluation_scope"]
    allowlist_path = Path(str(scope["source_allowlist_path"])).resolve()
    entries = [line.strip() for line in allowlist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entry_by_id = {Path(entry).stem: entry for entry in entries}
    if len(entries) != int(scope["included_sample_count"]) or len(entry_by_id) != len(entries):
        raise DvgProtocolError("P3 source allowlist count or uniqueness changed")
    if any("official" in entry.lower() or "test" in entry.lower() for entry in entries):
        raise DvgProtocolError("P3 source allowlist crosses the official-test boundary")

    manifest_path = Path(str(scope["source_mask_manifest_path"])).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "museg-dvc-a1-mask-manifest-v3"
        or manifest.get("protocol_id") != str(scope["source_protocol_id"])
        or manifest.get("status") != "passed"
        or manifest.get("official_test_included") is not False
    ):
        raise DvgProtocolError("P3 source mask manifest identity is invalid")
    rows = [row for row in manifest.get("samples", []) if str(row.get("sample_id", "")) in entry_by_id]
    eligible = [
        row
        for row in rows
        if int(row["condition_counts"]["boundary-q75"]) > 0
        and int(row["condition_counts"]["nonboundary-q50"]) > 0
    ]
    if not eligible:
        raise DvgProtocolError("P3 cannot locate a sample with both nonempty damaged conditions")
    selected_row = sorted(
        eligible,
        key=lambda row: (
            -min(
                int(row["condition_counts"]["boundary-q75"]),
                int(row["condition_counts"]["nonboundary-q50"]),
            ),
            -int(row["condition_counts"]["boundary-q75"]),
            str(row["sample_id"]),
        ),
    )[0]
    sample_id = str(selected_row["sample_id"])
    sample_entry = entry_by_id[sample_id]

    dataset = protocol.raw["dataset"]
    dataset_root = Path(str(dataset["root"])).resolve()
    input_paths = {
        "rgb": dataset_root / str(dataset["rgb_directory"]) / f"{sample_id}.jpg",
        "depth16": dataset_root / str(dataset["depth16_directory"]) / f"{sample_id}.png",
        "depth8": dataset_root / str(dataset["quantized_depth_directory"]) / f"{sample_id}.png",
        "label": dataset_root / str(dataset["label_directory"]) / f"{sample_id}.png",
    }
    rgb_bgr = cv2.imread(str(input_paths["rgb"]), cv2.IMREAD_COLOR)
    depth16 = cv2.imread(str(input_paths["depth16"]), cv2.IMREAD_UNCHANGED)
    depth8 = cv2.imread(str(input_paths["depth8"]), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(str(input_paths["label"]), cv2.IMREAD_GRAYSCALE)
    if any(value is None for value in (rgb_bgr, depth16, depth8, label)):
        raise FileNotFoundError(f"missing P3 input modality for {sample_id}")
    assert rgb_bgr is not None and depth16 is not None and depth8 is not None and label is not None
    if (
        depth16.dtype != np.uint16
        or depth16.ndim != 2
        or rgb_bgr.shape[:2] != depth16.shape
        or depth8.shape != depth16.shape
        or label.shape != depth16.shape
    ):
        raise ValueError("P3 modalities do not share the frozen geometry and dtypes")

    corruption = protocol.raw["corruption"]
    masks = build_corruption_masks(
        depth16,
        sample_id,
        seed=int(corruption["seed"]),
        relative_jump_threshold=float(corruption["relative_jump_threshold"]),
    )
    source_hashes = {str(key): str(value) for key, value in selected_row["condition_mask_sha256"].items()}
    if dict(masks.condition_sha256) != source_hashes:
        raise DvgProtocolError("P3 recomputed condition masks differ from the frozen v3 manifest")
    if any(int(masks.condition_counts[name]) <= 0 for name in ("boundary-q75", "nonboundary-q50")):
        raise DvgProtocolError("P3 selected damaged condition is unexpectedly empty")

    config = copy.copy(importlib.import_module(str(protocol.raw["model"]["config_module"])).C)
    rgb_contract = protocol.raw["input_contract"]["rgb"]
    config.channel_order = str(rgb_contract["channel_order"])
    config.normalization_identity = str(rgb_contract["normalization_identity"])
    config.norm_mean = np.asarray(rgb_contract["mean"], dtype=np.float32)
    config.norm_std = np.asarray(rgb_contract["std"], dtype=np.float32)
    if config.channel_order != "RGB" or config.normalization_identity != "rgb-imagenet-rgb-order-v1":
        raise DvgProtocolError("P3 runtime RGB contract differs from the frozen protocol")
    rgb = apply_channel_order(rgb_bgr, config.channel_order)
    q0_depth8 = quantized_condition_depth(
        depth16,
        masks.masks["clean"],
        depth_max_raw=int(protocol.raw["input_contract"]["depth"]["depth_max_raw"]),
    )
    if not np.array_equal(q0_depth8, depth8):
        raise DvgProtocolError("P3 q=0 quantized depth is not array-identical to production Depth8")
    clean_reference_views = build_msflip_views(rgb, depth8, config)

    configure_fp32_forward(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = load_model(config, checkpoint_path, device)
    audit_context: dict[str, Any] = {"active": None}
    hooks = []

    def make_geo_hook(stage_index: int):
        def hook(module: Any, args: Any, kwargs: Any, output: Any) -> None:
            active = audit_context.get("active")
            if active is None:
                return
            hw = tuple(int(value) for value in args[0])
            depth_map = args[1]
            production_reliability = kwargs.get("oracle_reliability")
            gate_applied = bool(active["gate_applied"])
            if gate_applied and production_reliability is None:
                active["errors"].append(f"stage-{stage_index}: missing Oracle reliability")
                return
            if not gate_applied and production_reliability is not None:
                active["errors"].append(f"stage-{stage_index}: unexpected Oracle reliability in baseline")
                return
            expected_reliability = active["expected_reliability"][stage_index]
            reliability = production_reliability if production_reliability is not None else expected_reliability
            active["records"].append(
                _sampled_geo_audit(
                    module,
                    stage_index=stage_index,
                    hw=hw,
                    depth_map=depth_map,
                    split_or_not=bool(kwargs.get("split_or_not", False)),
                    reliability=reliability,
                    expected_reliability=expected_reliability,
                    expected_depth_input=active["expected_depth_input"],
                    gate_applied=gate_applied,
                    output=output,
                )
            )

        return hook

    for stage_index, layer in enumerate(model.backbone.layers):
        hooks.append(layer.blocks[0].Geo.register_forward_hook(make_geo_hook(stage_index), with_kwargs=True))

    condition_records: dict[str, Any] = {}
    failures: list[str] = []
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for condition in ("clean", "boundary-q75", "nonboundary-q50"):
            raw_mask = masks.masks[condition]
            condition_depth = quantized_condition_depth(
                depth16,
                raw_mask,
                depth_max_raw=int(protocol.raw["input_contract"]["depth"]["depth_max_raw"]),
            )
            views = build_msflip_views(rgb, condition_depth, config)
            if len(views) != int(protocol.raw["evaluator"]["view_count_per_sample"]):
                raise DvgProtocolError(f"P3 {condition} did not produce exactly 10 evaluator views")
            fused_baseline: torch.Tensor | None = None
            fused_gated: torch.Tensor | None = None
            view_records: list[dict[str, Any]] = []
            for view_index, (view, clean_reference_view) in enumerate(zip(views, clean_reference_views)):
                if (
                    float(view["scale"]) != float(clean_reference_view["scale"])
                    or bool(view["flipped"]) != bool(clean_reference_view["flipped"])
                    or tuple(view["scaled_size_hw"]) != tuple(clean_reference_view["scaled_size_hw"])
                    or tuple(view["padded_size_hw"]) != tuple(clean_reference_view["padded_size_hw"])
                    or not torch.equal(view["rgb"], clean_reference_view["rgb"])
                ):
                    raise DvgProtocolError(f"P3 {condition} view geometry or RGB differs from clean reference")
                rgb_tensor = view["rgb"].unsqueeze(0).to(device=device, dtype=torch.float32)
                depth_tensor = view["depth"].unsqueeze(0).to(device=device, dtype=torch.float32)
                scaled_size = tuple(int(value) for value in view["scaled_size_hw"])
                padded_size = tuple(int(value) for value in view["padded_size_hw"])
                flipped = bool(view["flipped"])
                view_mask_array = build_view_corruption_mask(
                    raw_mask,
                    scaled_size_hw=scaled_size,
                    padded_size_hw=padded_size,
                    flipped=flipped,
                )
                depth_delta = torch.ne(view["depth"][0], clean_reference_view["depth"][0]).numpy()
                mask_support = view_mask_array > 0.0
                depth_delta_count = int(np.count_nonzero(depth_delta))
                mask_support_count = int(np.count_nonzero(mask_support))
                depth_delta_outside_mask_count = int(np.count_nonzero(depth_delta & ~mask_support))
                depth_delta_mask_overlap_count = int(np.count_nonzero(depth_delta & mask_support))
                view_mask = torch.from_numpy(view_mask_array).unsqueeze(0).unsqueeze(0).to(device)
                stage_sizes = stage_sizes_from_padded_view(padded_size)
                expected_reliability: list[torch.Tensor] = []
                aggregation_timings: list[dict[str, Any]] = []
                for stage_index, stage_size in enumerate(stage_sizes):
                    torch.cuda.synchronize(device)
                    aggregation_started = time.perf_counter()
                    stage_reliability = build_dvg_b1_stage_reliability(view_mask, stage_size)
                    torch.cuda.synchronize(device)
                    aggregation_timings.append(
                        {
                            "stage_index": stage_index,
                            "stage_size_hw": list(stage_size),
                            "elapsed_seconds": round(time.perf_counter() - aggregation_started, 6),
                        }
                    )
                    expected_reliability.append(stage_reliability)

                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state(device)
                baseline_active = None
                if condition != "clean":
                    baseline_active = {
                        "gate_applied": False,
                        "records": [],
                        "errors": [],
                        "expected_reliability": expected_reliability,
                        "expected_depth_input": depth_tensor[:, :1],
                    }
                audit_context["active"] = baseline_active
                torch.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state(cuda_rng_state, device)
                torch.cuda.synchronize(device)
                baseline_started = time.perf_counter()
                with torch.inference_mode():
                    baseline_logits = model(rgb_tensor, depth_tensor)
                torch.cuda.synchronize(device)
                baseline_elapsed = time.perf_counter() - baseline_started

                gated_active = None
                if condition != "clean":
                    gated_active = {
                        "gate_applied": True,
                        "records": [],
                        "errors": [],
                        "expected_reliability": expected_reliability,
                        "expected_depth_input": depth_tensor[:, :1],
                    }
                audit_context["active"] = gated_active
                torch.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state(cuda_rng_state, device)
                torch.cuda.synchronize(device)
                gated_started = time.perf_counter()
                with torch.inference_mode():
                    gated_logits = model(
                        rgb_tensor,
                        depth_tensor,
                        oracle_corruption_mask=view_mask,
                    )
                torch.cuda.synchronize(device)
                gated_elapsed = time.perf_counter() - gated_started
                audit_context["active"] = None

                baseline_restored = _restore_view_logits(
                    baseline_logits,
                    scaled_size_hw=scaled_size,
                    padded_size_hw=padded_size,
                    flipped=flipped,
                    original_size_hw=tuple(label.shape),
                )
                gated_restored = _restore_view_logits(
                    gated_logits,
                    scaled_size_hw=scaled_size,
                    padded_size_hw=padded_size,
                    flipped=flipped,
                    original_size_hw=tuple(label.shape),
                )
                fused_baseline = baseline_restored if fused_baseline is None else fused_baseline + baseline_restored
                fused_gated = gated_restored if fused_gated is None else fused_gated + gated_restored

                clean_noop_equal = torch.equal(baseline_logits, gated_logits) if condition == "clean" else None
                baseline_geo_records = [] if baseline_active is None else baseline_active["records"]
                baseline_geo_errors = [] if baseline_active is None else baseline_active["errors"]
                gated_geo_records = [] if gated_active is None else gated_active["records"]
                gated_geo_errors = [] if gated_active is None else gated_active["errors"]
                record = {
                    "view_index": view_index,
                    "scale": float(view["scale"]),
                    "flipped": flipped,
                    "scaled_size_hw": list(scaled_size),
                    "padded_size_hw": list(padded_size),
                    "raw_mask_nonzero_count": int(raw_mask.sum()),
                    "view_mask_nonzero_count": mask_support_count,
                    "view_mask_minimum": float(view_mask_array.min()),
                    "view_mask_maximum": float(view_mask_array.max()),
                    "depth_delta_from_clean_count": depth_delta_count,
                    "depth_delta_mask_overlap_count": depth_delta_mask_overlap_count,
                    "depth_delta_outside_mask_count": depth_delta_outside_mask_count,
                    "depth_delta_is_nonempty": depth_delta_count > 0,
                    "depth_delta_is_subset_of_mask_support": depth_delta_outside_mask_count == 0,
                    "stage_aggregation_gpu_cpu_opencv_gpu_timings": aggregation_timings,
                    "stage_aggregation_total_seconds": round(
                        sum(row["elapsed_seconds"] for row in aggregation_timings), 6
                    ),
                    "baseline_forward_seconds": round(baseline_elapsed, 6),
                    "gated_forward_seconds": round(gated_elapsed, 6),
                    "baseline_logits_finite": bool(torch.isfinite(baseline_logits).all()),
                    "gated_logits_finite": bool(torch.isfinite(gated_logits).all()),
                    "restored_size_hw": list(baseline_restored.shape[-2:]),
                    "expected_original_size_hw": list(label.shape),
                    "baseline_vs_gated_logits_equal": torch.equal(baseline_logits, gated_logits),
                    "clean_noop_equal": clean_noop_equal,
                    "baseline_first_block_geo_audits": baseline_geo_records,
                    "gated_first_block_geo_audits": gated_geo_records,
                    "baseline_geo_audit_errors": baseline_geo_errors,
                    "gated_geo_audit_errors": gated_geo_errors,
                }
                view_records.append(record)

                if condition == "clean":
                    if mask_support_count != 0:
                        failures.append(f"{condition}/view-{view_index}: clean mask support is not empty")
                    if depth_delta_count != 0:
                        failures.append(f"{condition}/view-{view_index}: clean Depth differs from the clean reference")
                    if not clean_noop_equal:
                        failures.append(f"{condition}/view-{view_index}: explicit clean mask is not a strict no-op")
                    if any(not torch.equal(value, torch.ones_like(value)) for value in expected_reliability):
                        failures.append(f"{condition}/view-{view_index}: clean stage reliability is not all one")
                else:
                    if mask_support_count == 0:
                        failures.append(f"{condition}/view-{view_index}: transformed corruption mask became empty")
                    if depth_delta_count == 0:
                        failures.append(f"{condition}/view-{view_index}: corruption did not change the Depth view")
                    if depth_delta_outside_mask_count != 0:
                        failures.append(
                            f"{condition}/view-{view_index}: Depth changes extend outside transformed mask support"
                        )

                    paired_audits = (
                        ("baseline", baseline_geo_records, baseline_geo_errors, False),
                        ("gated", gated_geo_records, gated_geo_errors, True),
                    )
                    for audit_name, audit_records, audit_errors, expected_gate_applied in paired_audits:
                        if len(audit_records) != 4 or audit_errors:
                            failures.append(
                                f"{condition}/view-{view_index}/{audit_name}: four-stage Geo audit was not captured cleanly"
                            )
                            continue
                        for expected_stage_index, geo in enumerate(audit_records):
                            stage_index = int(geo["stage_index"])
                            expected_stage_size = list(stage_sizes[expected_stage_index])
                            common_required = (
                                stage_index == expected_stage_index
                                and geo["stage_size_hw"] == expected_stage_size
                                and geo["topology_matches_stage"]
                                and geo["gate_applied_in_production"] is expected_gate_applied
                                and geo["reliability_matches_precomputed_exactly"]
                                and geo["depth_input_matches_corrupted_view_exactly"]
                                and geo["sin_finite"]
                                and geo["cos_finite"]
                                and geo["sampled_contributions_finite"]
                                and geo["sampled_production_reconstruction_exact"]
                            )
                            gated_required = (
                                geo["reliability_non_all_one"]
                                and geo["pairwise_gate_non_all_one"]
                                and geo["sampled_depth_gate_effect_nonzero"]
                            )
                            if not common_required or (expected_gate_applied and not gated_required):
                                failures.append(
                                    f"{condition}/view-{view_index}/{audit_name}/stage-{stage_index}: "
                                    "depth geometry audit failed"
                                )
                if record["restored_size_hw"] != record["expected_original_size_hw"]:
                    failures.append(f"{condition}/view-{view_index}: logits did not restore to original label grid")
                if not record["baseline_logits_finite"] or not record["gated_logits_finite"]:
                    failures.append(f"{condition}/view-{view_index}: non-finite logits")

            assert fused_baseline is not None and fused_gated is not None
            fused_baseline = fused_baseline / float(len(views))
            fused_gated = fused_gated / float(len(views))
            fused_equal = torch.equal(fused_baseline, fused_gated)
            if condition == "clean" and not fused_equal:
                failures.append("clean: fused original-grid logits are not a strict no-op")
            if condition != "clean" and fused_equal:
                failures.append(f"{condition}: fused original-grid logits did not respond to the gate")
            if not bool(torch.isfinite(fused_baseline).all()) or not bool(torch.isfinite(fused_gated).all()):
                failures.append(f"{condition}: fused original-grid logits are non-finite")
            condition_records[condition] = {
                "raw_mask_count": int(raw_mask.sum()),
                "raw_mask_sha256": str(masks.condition_sha256[condition]),
                "source_manifest_mask_sha256": source_hashes[condition],
                "source_mask_hash_matches": str(masks.condition_sha256[condition]) == source_hashes[condition],
                "view_count": len(view_records),
                "views": view_records,
                "fused_baseline_finite": bool(torch.isfinite(fused_baseline).all()),
                "fused_gated_finite": bool(torch.isfinite(fused_gated).all()),
                "fused_size_hw": list(fused_gated.shape[-2:]),
                "fused_baseline_vs_gated_equal": fused_equal,
            }
    finally:
        audit_context["active"] = None
        for hook in hooks:
            hook.remove()

    result = {
        "schema_version": "museg-dvg-b1-gate-preflight-v1",
        "protocol_id": "DVG-B1-oracle-gsa-v1",
        "status": "passed" if not failures else "protocol-blocked",
        "device": str(device),
        "checkpoint_loaded": True,
        "checkpoint_strict_load": True,
        "model_forward_executed": True,
        "gpu_used": True,
        "official_test_included": False,
        "created_at_utc": _utc_now(),
        "protocol_path": str(protocol.path),
        "protocol_sha256": protocol_sha256,
        "prerequisites": prerequisites,
        "archived_prior_artifact": archived_prior_artifact,
        "rng_control": {
            "reason": "LightHamHead NMF2D uses random bases in eval; replay identical paired RNG state before baseline and gated forwards to isolate the Oracle gate",
            "paired_cpu_rng_state_replayed_before_each_forward": True,
            "paired_cuda_rng_state_replayed_before_each_forward": True,
            "decoder_rand_init": bool(model.decode_head.hamburger.ham.rand_init),
        },
        "sample_selection_rule": "maximise min(boundary-q75 count, nonboundary-q50 count), then boundary-q75 count, then sample id",
        "sample_id": sample_id,
        "sample_entry": sample_entry,
        "location_group": location_group(sample_id),
        "source_manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        "input_files": {
            name: {"path": str(path), "sha256": file_sha256(path)} for name, path in input_paths.items()
        },
        "original_label_size_hw": list(label.shape),
        "q0_depth8_array_equal_to_production": True,
        "condition_counts": dict(masks.condition_counts),
        "condition_mask_sha256": dict(masks.condition_sha256),
        "condition_records": condition_records,
        "failures": failures,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        },
        "runtime_identity": _runtime_identity(),
    }
    _atomic_write_json(output_path, result)
    return result, output_path


def _p4_sample_paths(protocol: Any, sample_id: str) -> dict[str, Path]:
    dataset = protocol.raw["dataset"]
    root = Path(str(dataset["root"])).resolve()
    return {
        "rgb": root / str(dataset["rgb_directory"]) / f"{sample_id}.jpg",
        "depth16": root / str(dataset["depth16_directory"]) / f"{sample_id}.png",
        "depth8": root / str(dataset["quantized_depth_directory"]) / f"{sample_id}.png",
        "label": root / str(dataset["label_directory"]) / f"{sample_id}.png",
    }


def _read_p4_sample(
    protocol: Any,
    sample_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = _p4_sample_paths(protocol, sample_id)
    rgb_bgr = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
    depth16 = cv2.imread(str(paths["depth16"]), cv2.IMREAD_UNCHANGED)
    depth8 = cv2.imread(str(paths["depth8"]), cv2.IMREAD_GRAYSCALE)
    label_raw = cv2.imread(str(paths["label"]), cv2.IMREAD_GRAYSCALE)
    if any(value is None for value in (rgb_bgr, depth16, depth8, label_raw)):
        raise FileNotFoundError(f"missing or undecodable P4 modality for {sample_id}")
    assert rgb_bgr is not None and depth16 is not None and depth8 is not None and label_raw is not None
    if depth16.dtype != np.uint16 or depth16.ndim != 2:
        raise ValueError(f"P4 Depth16 for {sample_id} is not two-dimensional uint16")
    if rgb_bgr.shape[:2] != depth16.shape or depth8.shape != depth16.shape or label_raw.shape != depth16.shape:
        raise ValueError(f"P4 modality geometry mismatch for {sample_id}")
    depth_max_raw = int(protocol.raw["input_contract"]["depth"]["depth_max_raw"])
    if int(depth16.max()) > depth_max_raw:
        raise ValueError(f"P4 Depth16 exceeds {depth_max_raw} for {sample_id}")
    allowed_labels = set(range(16)) | {255}
    unexpected_labels = sorted(set(int(value) for value in np.unique(label_raw)) - allowed_labels)
    if unexpected_labels:
        raise ValueError(f"P4 Label contains unsupported values for {sample_id}: {unexpected_labels}")
    rgb_contract = protocol.raw["input_contract"]["rgb"]
    rgb = apply_channel_order(rgb_bgr, str(rgb_contract["channel_order"]))
    label = label_raw.astype(np.int64) - 1
    label[(label_raw == 0) | (label_raw == 255)] = int(protocol.raw["input_contract"]["label"]["evaluator_ignore"])
    boundary_target = label_raw.astype(np.int64) - 1
    boundary_target[label_raw == 0] = len(protocol.raw["input_contract"]["label"]["foreground_ids"])
    boundary_target[label_raw == 255] = int(protocol.raw["input_contract"]["label"]["evaluator_ignore"])
    return rgb, depth16, depth8, label, boundary_target


def _p4_config(protocol: Any) -> Any:
    config = copy.copy(importlib.import_module(str(protocol.raw["model"]["config_module"])).C)
    rgb_contract = protocol.raw["input_contract"]["rgb"]
    config.channel_order = str(rgb_contract["channel_order"])
    config.normalization_identity = str(rgb_contract["normalization_identity"])
    config.norm_mean = np.asarray(rgb_contract["mean"], dtype=np.float32)
    config.norm_std = np.asarray(rgb_contract["std"], dtype=np.float32)
    if config.channel_order != "RGB" or config.normalization_identity != "rgb-imagenet-rgb-order-v1":
        raise DvgProtocolError("P4 runtime RGB contract differs from the frozen protocol")
    return config


def _verify_p4_prerequisites(protocol_path: Path, device: torch.device) -> tuple[Any, str, list[str], dict[str, Any]]:
    protocol, protocol_sha256 = _verify_p0_assets(protocol_path)
    if device.type != "cuda":
        raise ValueError("this authorized P4 run is restricted to the local CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the authorized P4 run")

    evidence_root = Path(str(protocol.raw["evidence"]["output_root"])).resolve()
    expected_prerequisites = {
        "qualification": (evidence_root / "qualification.json", P1_QUALIFICATION_SHA256),
        "noop_equivalence": (evidence_root / "noop-equivalence.json", P2_NOOP_SHA256),
        "gate_preflight": (evidence_root / "gate-preflight.json", P3_GATE_PREFLIGHT_SHA256),
    }
    prerequisites: dict[str, Any] = {}
    for name, (path, expected_sha256) in expected_prerequisites.items():
        if not path.is_file():
            raise DvgProtocolError(f"P4 prerequisite is missing: {path}")
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise DvgProtocolError(
                f"P4 prerequisite {name} SHA-256 changed: expected {expected_sha256}, got {actual_sha256}"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("protocol_id") != "DVG-B1-oracle-gsa-v1"
            or value.get("status") != "passed"
            or value.get("official_test_included") is not False
            or value.get("failures") != []
        ):
            raise DvgProtocolError(f"P4 prerequisite {name} is not the canonical clean pass")
        prerequisites[name] = {"path": str(path), "sha256": actual_sha256}

    checkpoint_path = Path(str(protocol.raw["model"]["checkpoint_path"])).resolve()
    if file_sha256(checkpoint_path) != str(protocol.raw["model"]["checkpoint_sha256"]):
        raise DvgProtocolError("P4 epoch 420 checkpoint SHA-256 mismatch")

    scope = protocol.raw["evaluation_scope"]
    allowlist_path = Path(str(scope["source_allowlist_path"])).resolve()
    entries = [line.strip() for line in allowlist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample_ids = [Path(entry).stem for entry in entries]
    groups = {location_group(sample_id) for sample_id in sample_ids}
    if len(entries) != int(scope["included_sample_count"]) or len(set(entries)) != len(entries):
        raise DvgProtocolError("P4 allowlist count or uniqueness differs from the frozen scope")
    if len(groups) != int(scope["included_location_group_count"]):
        raise DvgProtocolError("P4 location-group count differs from the frozen scope")
    if any(not entry.startswith("RGB/") or "official" in entry.lower() or "test" in entry.lower() for entry in entries):
        raise DvgProtocolError("P4 allowlist crosses the official-test boundary")

    manifest_path = Path(str(scope["source_mask_manifest_path"])).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "museg-dvc-a1-mask-manifest-v3"
        or manifest.get("protocol_id") != str(scope["source_protocol_id"])
        or manifest.get("status") != "passed"
        or manifest.get("official_test_included") is not False
        or int(manifest.get("sample_count", -1)) != len(entries)
        or int(manifest.get("location_group_count", -1)) != len(groups)
        or manifest.get("gate_passed") is not True
    ):
        raise DvgProtocolError("P4 source mask manifest identity is invalid")
    rows = manifest.get("samples")
    if not isinstance(rows, list):
        raise DvgProtocolError("P4 source mask manifest lacks sample rows")
    rows_by_id = {str(row.get("sample_id", "")): row for row in rows if isinstance(row, Mapping)}
    if set(rows_by_id) != set(sample_ids):
        raise DvgProtocolError("P4 source mask manifest sample identities differ from the allowlist")
    return protocol, protocol_sha256, entries, {
        "prerequisites": prerequisites,
        "source_manifest_path": manifest_path,
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_rows_by_id": rows_by_id,
    }


def _build_p4_mask_manifest(
    protocol: Any,
    protocol_sha256: str,
    entries: Sequence[str],
    source_rows_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    expected_hashes: dict[str, dict[str, str]] = {}
    corruption = protocol.raw["corruption"]
    depth_max_raw = int(protocol.raw["input_contract"]["depth"]["depth_max_raw"])
    for entry in entries:
        sample_id = Path(entry).stem
        _, depth16, depth8, _, _ = _read_p4_sample(protocol, sample_id)
        masks = build_corruption_masks(
            depth16,
            sample_id,
            seed=int(corruption["seed"]),
            relative_jump_threshold=float(corruption["relative_jump_threshold"]),
        )
        source_row = source_rows_by_id[sample_id]
        source_hashes = source_row.get("condition_mask_sha256")
        if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(CONDITION_IDS):
            raise DvgProtocolError(f"P4 source mask hashes are incomplete for {sample_id}")
        computed_hashes = {condition: str(masks.condition_sha256[condition]) for condition in CONDITION_IDS}
        for condition in CONDITION_IDS:
            if computed_hashes[condition] != str(source_hashes[condition]):
                raise DvgProtocolError(f"P4 mask hash differs from frozen v3 for {sample_id} in {condition}")
        q0_depth8 = quantized_condition_depth(depth16, masks.masks["clean"], depth_max_raw=depth_max_raw)
        if not np.array_equal(q0_depth8, depth8):
            raise DvgProtocolError(f"P4 q=0 Depth8 differs from production Depth8 for {sample_id}")
        expected_hashes[sample_id] = computed_hashes
        rows.append(
            {
                "sample_id": sample_id,
                "location_group": location_group(sample_id),
                "condition_counts": {condition: int(masks.condition_counts[condition]) for condition in CONDITION_IDS},
                "condition_mask_sha256": computed_hashes,
                "source_condition_mask_sha256": {condition: str(source_hashes[condition]) for condition in CONDITION_IDS},
                "source_hashes_match": True,
                "q0_depth8_array_equal_to_production": True,
            }
        )
    manifest = {
        "schema_version": "museg-dvg-b1-mask-manifest-v1",
        "protocol_id": "DVG-B1-oracle-gsa-v1",
        "protocol_sha256": protocol_sha256,
        "status": "passed",
        "generated_at_utc": _utc_now(),
        "official_test_included": False,
        "sample_count": len(rows),
        "location_group_count": len({row["location_group"] for row in rows}),
        "condition_ids": list(CONDITION_IDS),
        "source_mask_manifest_sha256": file_sha256(Path(str(protocol.raw["evaluation_scope"]["source_mask_manifest_path"]))),
        "all_source_hashes_match": True,
        "all_q0_depth8_equal_to_production": True,
        "samples": rows,
    }
    return manifest, expected_hashes


def _combine_p4_groups(
    baseline_records: Sequence[Mapping[str, Any]],
    gated_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    baseline = {str(row["location_group"]): row for row in aggregate_condition_records(baseline_records)["per_group"]}
    gated = {str(row["location_group"]): row for row in aggregate_condition_records(gated_records)["per_group"]}
    if set(baseline) != set(gated):
        raise DvgProtocolError("P4 baseline and gated group identities differ")
    combined: list[dict[str, Any]] = []
    for group in sorted(baseline):
        left = baseline[group]
        right = gated[group]
        boundary_gain = None
        if left["boundary_iou"] is not None and right["boundary_iou"] is not None:
            boundary_gain = (float(right["boundary_iou"]) - float(left["boundary_iou"])) * 100.0
        miou_gain = None
        if left["image_miou"] is not None and right["image_miou"] is not None:
            miou_gain = (float(right["image_miou"]) - float(left["image_miou"])) * 100.0
        combined.append(
            {
                "location_group": group,
                "mine": str(left["mine"]),
                "image_count": int(left["image_count"]),
                "baseline_boundary_iou": left["boundary_iou"],
                "gated_boundary_iou": right["boundary_iou"],
                "boundary_iou_gain_percentage_points": boundary_gain,
                "baseline_image_miou": left["image_miou"],
                "gated_image_miou": right["image_miou"],
                "miou_gain_percentage_points": miou_gain,
            }
        )
    return combined


def _p4_metric_effect(
    per_group: Sequence[Mapping[str, Any]],
    *,
    baseline_key: str,
    gated_key: str,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    values = {
        str(row["location_group"]): {
            "gated": row[gated_key],
            "baseline": row[baseline_key],
        }
        for row in per_group
    }
    return paired_effect(
        values,
        "gated",
        "baseline",
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        bootstrap_replicates=int(statistics["bootstrap_replicates"]),
    )


def _run_p4_condition(
    *,
    protocol: Any,
    protocol_sha256: str,
    config: Any,
    model: torch.nn.Module,
    device: torch.device,
    entries: Sequence[str],
    condition: str,
    expected_mask_sha256: Mapping[str, str],
    full_started: float,
) -> dict[str, Any]:
    condition_started = time.monotonic()
    baseline_records: list[dict[str, Any]] = []
    gated_records: list[dict[str, Any]] = []
    baseline_hist = np.zeros((int(config.num_classes), int(config.num_classes)), dtype=np.int64)
    gated_hist = np.zeros_like(baseline_hist)
    total_baseline_seconds = 0.0
    total_gated_seconds = 0.0
    total_aggregation_seconds = 0.0
    total_depth_delta = 0
    total_depth_delta_outside_mask = 0
    total_view_mask_support = 0
    paired_equal_view_count = 0
    fused_equal_sample_count = 0
    nonempty_raw_mask_sample_count = 0
    zero_raw_mask_sample_count = 0
    reliability_non_all_one_view_stage_count = 0
    view_stage_count = 0
    depth_max_raw = int(protocol.raw["input_contract"]["depth"]["depth_max_raw"])
    corruption = protocol.raw["corruption"]
    total_samples = len(entries)
    condition_index = CONDITION_IDS.index(condition) + 1

    for sample_index, entry in enumerate(entries):
        sample_id = Path(entry).stem
        rgb, depth16, depth8, label, boundary_target = _read_p4_sample(protocol, sample_id)
        masks = build_corruption_masks(
            depth16,
            sample_id,
            seed=int(corruption["seed"]),
            relative_jump_threshold=float(corruption["relative_jump_threshold"]),
        )
        raw_mask = masks.masks[condition]
        mask_sha256 = str(masks.condition_sha256[condition])
        if mask_sha256 != expected_mask_sha256[sample_id]:
            raise DvgProtocolError(f"P4 runtime mask hash differs from frozen manifest for {sample_id} in {condition}")
        raw_mask_count = int(raw_mask.sum())
        if raw_mask_count:
            nonempty_raw_mask_sample_count += 1
        else:
            zero_raw_mask_sample_count += 1
        condition_depth = quantized_condition_depth(depth16, raw_mask, depth_max_raw=depth_max_raw)
        views = build_msflip_views(rgb, condition_depth, config)
        clean_reference_views = build_msflip_views(rgb, depth8, config)
        if len(views) != int(protocol.raw["evaluator"]["view_count_per_sample"]) or len(views) != len(clean_reference_views):
            raise DvgProtocolError(f"P4 {sample_id}/{condition} did not produce exactly 10 paired views")

        fused_baseline: torch.Tensor | None = None
        fused_gated: torch.Tensor | None = None
        sample_baseline_seconds = 0.0
        sample_gated_seconds = 0.0
        sample_aggregation_seconds = 0.0
        sample_depth_delta = 0
        sample_depth_delta_outside_mask = 0
        sample_view_mask_support = 0
        sample_equal_views = 0
        sample_reliability_minimum = 1.0
        sample_reliability_maximum = 1.0
        sample_non_all_one_view_stages = 0

        for view_index, (view, clean_view) in enumerate(zip(views, clean_reference_views)):
            if (
                float(view["scale"]) != float(clean_view["scale"])
                or bool(view["flipped"]) != bool(clean_view["flipped"])
                or tuple(view["scaled_size_hw"]) != tuple(clean_view["scaled_size_hw"])
                or tuple(view["padded_size_hw"]) != tuple(clean_view["padded_size_hw"])
                or not torch.equal(view["rgb"], clean_view["rgb"])
            ):
                raise DvgProtocolError(f"P4 paired geometry or RGB mismatch for {sample_id}/{condition}/view-{view_index}")
            scaled_size = tuple(int(value) for value in view["scaled_size_hw"])
            padded_size = tuple(int(value) for value in view["padded_size_hw"])
            flipped = bool(view["flipped"])
            view_mask_array = build_view_corruption_mask(
                raw_mask,
                scaled_size_hw=scaled_size,
                padded_size_hw=padded_size,
                flipped=flipped,
            )
            mask_support = view_mask_array > 0.0
            depth_delta = torch.ne(view["depth"][0], clean_view["depth"][0]).numpy()
            depth_delta_count = int(np.count_nonzero(depth_delta))
            depth_delta_outside = int(np.count_nonzero(depth_delta & ~mask_support))
            if depth_delta_outside:
                raise DvgProtocolError(
                    f"P4 Depth changes leave mask support for {sample_id}/{condition}/view-{view_index}"
                )
            if condition == "clean" and (int(np.count_nonzero(mask_support)) != 0 or depth_delta_count != 0):
                raise DvgProtocolError(f"P4 clean input is not unchanged for {sample_id}/view-{view_index}")
            if raw_mask_count == 0 and (int(np.count_nonzero(mask_support)) != 0 or depth_delta_count != 0):
                raise DvgProtocolError(f"P4 empty raw mask is not a no-op for {sample_id}/{condition}/view-{view_index}")

            rgb_tensor = view["rgb"].unsqueeze(0).to(device=device, dtype=torch.float32)
            depth_tensor = view["depth"].unsqueeze(0).to(device=device, dtype=torch.float32)
            view_mask = torch.from_numpy(view_mask_array).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
            stage_sizes = stage_sizes_from_padded_view(padded_size)
            for stage_size in stage_sizes:
                torch.cuda.synchronize(device)
                aggregation_started = time.perf_counter()
                reliability = build_dvg_b1_stage_reliability(view_mask, stage_size)
                torch.cuda.synchronize(device)
                aggregation_elapsed = time.perf_counter() - aggregation_started
                if not bool(torch.isfinite(reliability).all()):
                    raise RuntimeError(f"P4 non-finite reliability for {sample_id}/{condition}/view-{view_index}")
                minimum = float(reliability.min().item())
                maximum = float(reliability.max().item())
                if minimum < 0.0 or maximum > 1.0:
                    raise DvgProtocolError(f"P4 reliability leaves [0,1] for {sample_id}/{condition}/view-{view_index}")
                sample_reliability_minimum = min(sample_reliability_minimum, minimum)
                sample_reliability_maximum = max(sample_reliability_maximum, maximum)
                if minimum < 1.0:
                    sample_non_all_one_view_stages += 1
                sample_aggregation_seconds += aggregation_elapsed
                view_stage_count += 1

            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(device)
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, device)
            torch.cuda.synchronize(device)
            baseline_started = time.perf_counter()
            with torch.inference_mode():
                baseline_logits = model(rgb_tensor, depth_tensor)
            torch.cuda.synchronize(device)
            baseline_elapsed = time.perf_counter() - baseline_started

            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, device)
            torch.cuda.synchronize(device)
            gated_started = time.perf_counter()
            with torch.inference_mode():
                gated_logits = model(rgb_tensor, depth_tensor, oracle_corruption_mask=view_mask)
            torch.cuda.synchronize(device)
            gated_elapsed = time.perf_counter() - gated_started
            if not bool(torch.isfinite(baseline_logits).all()) or not bool(torch.isfinite(gated_logits).all()):
                raise RuntimeError(f"P4 non-finite logits for {sample_id}/{condition}/view-{view_index}")
            equal_logits = torch.equal(baseline_logits, gated_logits)
            if (condition == "clean" or raw_mask_count == 0) and not equal_logits:
                raise DvgProtocolError(f"P4 no-op pair differs for {sample_id}/{condition}/view-{view_index}")

            baseline_restored = _restore_view_logits(
                baseline_logits,
                scaled_size_hw=scaled_size,
                padded_size_hw=padded_size,
                flipped=flipped,
                original_size_hw=tuple(label.shape),
            )
            gated_restored = _restore_view_logits(
                gated_logits,
                scaled_size_hw=scaled_size,
                padded_size_hw=padded_size,
                flipped=flipped,
                original_size_hw=tuple(label.shape),
            )
            if tuple(baseline_restored.shape[-2:]) != tuple(label.shape) or tuple(gated_restored.shape[-2:]) != tuple(label.shape):
                raise DvgProtocolError(f"P4 logits did not restore to the original grid for {sample_id}/{condition}")
            fused_baseline = baseline_restored if fused_baseline is None else fused_baseline + baseline_restored
            fused_gated = gated_restored if fused_gated is None else fused_gated + gated_restored

            sample_baseline_seconds += baseline_elapsed
            sample_gated_seconds += gated_elapsed
            sample_depth_delta += depth_delta_count
            sample_depth_delta_outside_mask += depth_delta_outside
            sample_view_mask_support += int(np.count_nonzero(mask_support))
            sample_equal_views += int(equal_logits)

        assert fused_baseline is not None and fused_gated is not None
        fused_baseline = fused_baseline / float(len(views))
        fused_gated = fused_gated / float(len(views))
        fused_equal = torch.equal(fused_baseline, fused_gated)
        if (condition == "clean" or raw_mask_count == 0) and not fused_equal:
            raise DvgProtocolError(f"P4 fused no-op pair differs for {sample_id}/{condition}")
        baseline_prediction = fused_baseline.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
        gated_prediction = fused_gated.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
        baseline_sample_hist = confusion_matrix(
            baseline_prediction,
            label,
            int(config.num_classes),
            int(config.background),
        )
        gated_sample_hist = confusion_matrix(
            gated_prediction,
            label,
            int(config.num_classes),
            int(config.background),
        )
        baseline_hist += baseline_sample_hist
        gated_hist += gated_sample_hist
        baseline_boundary = semantic_boundary_iou(
            baseline_prediction,
            boundary_target,
            num_classes=int(config.num_classes),
            ignore_label=int(config.background),
            background_label=int(config.num_classes),
            distance_ratio=float(protocol.raw["boundary_iou"]["distance_ratio_of_image_diagonal"]),
        )
        gated_boundary = semantic_boundary_iou(
            gated_prediction,
            boundary_target,
            num_classes=int(config.num_classes),
            ignore_label=int(config.background),
            background_label=int(config.num_classes),
            distance_ratio=float(protocol.raw["boundary_iou"]["distance_ratio_of_image_diagonal"]),
        )
        baseline_image_miou = image_miou_from_confusion(baseline_sample_hist)
        gated_image_miou = image_miou_from_confusion(gated_sample_hist)
        common = {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "location_group": location_group(sample_id),
            "mine": mine_id(sample_id),
            "height": int(label.shape[0]),
            "width": int(label.shape[1]),
        }
        baseline_records.append(
            {
                **common,
                "boundary_iou": baseline_boundary["value"],
                "boundary_distance_pixels": baseline_boundary["distance_pixels"],
                "boundary_iou_per_class": baseline_boundary["per_class"],
                "image_miou": baseline_image_miou,
            }
        )
        gated_records.append(
            {
                **common,
                "boundary_iou": gated_boundary["value"],
                "boundary_distance_pixels": gated_boundary["distance_pixels"],
                "boundary_iou_per_class": gated_boundary["per_class"],
                "image_miou": gated_image_miou,
            }
        )
        total_baseline_seconds += sample_baseline_seconds
        total_gated_seconds += sample_gated_seconds
        total_aggregation_seconds += sample_aggregation_seconds
        total_depth_delta += sample_depth_delta
        total_depth_delta_outside_mask += sample_depth_delta_outside_mask
        total_view_mask_support += sample_view_mask_support
        paired_equal_view_count += sample_equal_views
        fused_equal_sample_count += int(fused_equal)
        reliability_non_all_one_view_stage_count += sample_non_all_one_view_stages

        boundary_gain = None
        if baseline_boundary["value"] is not None and gated_boundary["value"] is not None:
            boundary_gain = (float(gated_boundary["value"]) - float(baseline_boundary["value"])) * 100.0
        miou_gain = None
        if baseline_image_miou is not None and gated_image_miou is not None:
            miou_gain = (float(gated_image_miou) - float(baseline_image_miou)) * 100.0
        gated_records[-1]["paired_audit"] = {
            "raw_mask_count": raw_mask_count,
            "mask_sha256": mask_sha256,
            "view_count": len(views),
            "paired_equal_view_count": sample_equal_views,
            "fused_baseline_vs_gated_equal": fused_equal,
            "depth_delta_from_clean_total": sample_depth_delta,
            "depth_delta_outside_mask_total": sample_depth_delta_outside_mask,
            "view_mask_support_total": sample_view_mask_support,
            "stage_reliability_minimum": sample_reliability_minimum,
            "stage_reliability_maximum": sample_reliability_maximum,
            "non_all_one_view_stage_count": sample_non_all_one_view_stages,
            "stage_aggregation_seconds": round(sample_aggregation_seconds, 6),
            "baseline_forward_seconds": round(sample_baseline_seconds, 6),
            "gated_forward_seconds": round(sample_gated_seconds, 6),
            "boundary_iou_gain_percentage_points": boundary_gain,
            "miou_gain_percentage_points": miou_gain,
        }

        completed = sample_index + 1
        overall_completed = (condition_index - 1) * total_samples + completed
        overall_total = len(CONDITION_IDS) * total_samples
        elapsed = max(time.monotonic() - full_started, 1e-9)
        rate = overall_completed / elapsed
        eta_seconds = (overall_total - overall_completed) / rate
        print(
            f"\r[DVG-B1 P4] condition {condition_index}/{len(CONDITION_IDS)} {condition}: "
            f"{completed}/{total_samples} samples | overall {overall_completed}/{overall_total} "
            f"({overall_completed / overall_total:.1%}) | elapsed {elapsed / 60:.1f} min | "
            f"ETA {eta_seconds / 60:.1f} min",
            end="",
            flush=True,
        )
        if overall_completed == overall_total:
            print()

    per_group = _combine_p4_groups(baseline_records, gated_records)
    if len(per_group) != int(protocol.raw["evaluation_scope"]["included_location_group_count"]):
        raise DvgProtocolError(f"P4 {condition} does not contain exactly 138 location groups")
    total_views = len(entries) * int(protocol.raw["evaluator"]["view_count_per_sample"])
    if condition == "clean" and (paired_equal_view_count != total_views or fused_equal_sample_count != len(entries)):
        raise DvgProtocolError("P4 clean strict no-op completeness failed")
    if total_depth_delta_outside_mask != 0:
        raise DvgProtocolError(f"P4 {condition} has Depth changes outside transformed mask support")
    return {
        "schema_version": "museg-dvg-b1-condition-result-v1",
        "protocol_id": "DVG-B1-oracle-gsa-v1",
        "protocol_sha256": protocol_sha256,
        "condition": condition,
        "status": "completed",
        "generated_at_utc": _utc_now(),
        "official_test_included": False,
        "sample_count": len(entries),
        "location_group_count": len(per_group),
        "view_count_per_sample": int(protocol.raw["evaluator"]["view_count_per_sample"]),
        "paired_forward_count": total_views,
        "baseline": {
            "metrics_percent": metrics_from_confusion(baseline_hist, config.class_names),
            "per_image": baseline_records,
        },
        "oracle_gated": {
            "metrics_percent": metrics_from_confusion(gated_hist, config.class_names),
            "per_image": gated_records,
        },
        "per_group": per_group,
        "gate_audit_summary": {
            "nonempty_raw_mask_sample_count": nonempty_raw_mask_sample_count,
            "zero_raw_mask_sample_count": zero_raw_mask_sample_count,
            "paired_equal_view_count": paired_equal_view_count,
            "fused_equal_sample_count": fused_equal_sample_count,
            "depth_delta_from_clean_total": total_depth_delta,
            "depth_delta_outside_mask_total": total_depth_delta_outside_mask,
            "view_mask_support_total": total_view_mask_support,
            "view_stage_count": view_stage_count,
            "reliability_non_all_one_view_stage_count": reliability_non_all_one_view_stage_count,
            "paired_rng_replayed": True,
            "all_logits_finite": True,
            "all_original_grids_restored": True,
            "all_mask_hashes_match_source": True,
        },
        "timing_seconds": {
            "baseline_forward_total": round(total_baseline_seconds, 6),
            "gated_forward_total": round(total_gated_seconds, 6),
            "stage_aggregation_total": round(total_aggregation_seconds, 6),
            "condition_total": round(time.monotonic() - condition_started, 3),
        },
    }


def _p4_position_specificity(
    condition_results: Mapping[str, Mapping[str, Any]],
    *,
    baseline_key: str,
    gated_key: str,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    boundary_rows = {str(row["location_group"]): row for row in condition_results["boundary-q50"]["per_group"]}
    nonboundary_rows = {
        str(row["location_group"]): row for row in condition_results["nonboundary-q50"]["per_group"]
    }
    if set(boundary_rows) != set(nonboundary_rows):
        raise DvgProtocolError("P4 position-specificity group identities differ")
    values = {
        group: {
            "boundary-q50-gain": (
                None
                if boundary_rows[group][baseline_key] is None or boundary_rows[group][gated_key] is None
                else float(boundary_rows[group][gated_key]) - float(boundary_rows[group][baseline_key])
            ),
            "nonboundary-q50-gain": (
                None
                if nonboundary_rows[group][baseline_key] is None or nonboundary_rows[group][gated_key] is None
                else float(nonboundary_rows[group][gated_key]) - float(nonboundary_rows[group][baseline_key])
            ),
        }
        for group in sorted(boundary_rows)
    }
    return paired_effect(
        values,
        "boundary-q50-gain",
        "nonboundary-q50-gain",
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        bootstrap_replicates=int(statistics["bootstrap_replicates"]),
    )


def _adjudicate_p4(
    boundary_effect: Mapping[str, Any],
    miou_effect: Mapping[str, Any],
    protocol: Any,
) -> dict[str, Any]:
    rules = protocol.raw["adjudication"]
    supported = rules["oracle_supported"]
    boundary_point = float(boundary_effect["point_percentage_points"])
    boundary_lower = float(boundary_effect["interval95_percentage_points"][0])
    miou_point = float(miou_effect["point_percentage_points"])
    boundary_threshold = float(supported["boundary_iou_point_min_percentage_points"])
    lower_threshold = float(supported["boundary_iou_interval_lower_strictly_above_percentage_points"])
    miou_threshold = float(supported["miou_point_min_percentage_points"])
    checks = {
        "boundary_iou_point_at_least_threshold": boundary_point >= boundary_threshold,
        "boundary_iou_interval_lower_strictly_above_zero": boundary_lower > lower_threshold,
        "miou_point_nonnegative": miou_point >= miou_threshold,
    }
    if all(checks.values()):
        decision = "oracle-supported"
    elif boundary_point < boundary_threshold or miou_point < miou_threshold:
        decision = "oracle-not-supported"
    else:
        decision = "inconclusive"
    return {
        "decision": decision,
        "rule_status": str(rules["rule_status"]),
        "checks": checks,
        "thresholds_percentage_points": {
            "boundary_iou_point_minimum": boundary_threshold,
            "boundary_iou_interval_lower_strictly_above": lower_threshold,
            "miou_point_minimum": miou_threshold,
        },
        "primary_boundary_iou_effect": dict(boundary_effect),
        "primary_miou_effect": dict(miou_effect),
    }


def run_full_evaluation(
    protocol_path: Path,
    output_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], Path]:
    protocol, protocol_sha256, entries, prerequisite_context = _verify_p4_prerequisites(protocol_path, device)
    archived_prior_artifact = _archive_existing_artifact(output_path)
    evidence_root = Path(str(protocol.raw["evidence"]["output_root"])).resolve()
    conditions = tuple(str(value) for value in protocol.raw["corruption"]["conditions"])
    if conditions != CONDITION_IDS:
        raise DvgProtocolError("P4 condition order or identities differ from the frozen protocol")

    manifest, expected_hashes = _build_p4_mask_manifest(
        protocol,
        protocol_sha256,
        entries,
        prerequisite_context["source_rows_by_id"],
    )
    manifest_path = evidence_root / "mask-manifest.json"
    _archive_existing_artifact(manifest_path)
    _atomic_write_json(manifest_path, manifest)

    config = _p4_config(protocol)
    configure_fp32_forward(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint_path = Path(str(protocol.raw["model"]["checkpoint_path"])).resolve()
    model = load_model(config, checkpoint_path, device)
    torch.cuda.reset_peak_memory_stats(device)
    full_started = time.monotonic()
    condition_results: dict[str, dict[str, Any]] = {}
    for condition in CONDITION_IDS:
        condition_result = _run_p4_condition(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            config=config,
            model=model,
            device=device,
            entries=entries,
            condition=condition,
            expected_mask_sha256={sample_id: hashes[condition] for sample_id, hashes in expected_hashes.items()},
            full_started=full_started,
        )
        condition_path = evidence_root / "conditions" / f"{condition}.json"
        _archive_existing_artifact(condition_path)
        _atomic_write_json(condition_path, condition_result)
        condition_results[condition] = condition_result

    statistics = protocol.raw["statistics"]
    expected_groups = int(protocol.raw["evaluation_scope"]["included_location_group_count"])
    effects_by_condition: dict[str, Any] = {}
    for condition, result in condition_results.items():
        boundary_effect = _p4_metric_effect(
            result["per_group"],
            baseline_key="baseline_boundary_iou",
            gated_key="gated_boundary_iou",
            statistics=statistics,
        )
        miou_effect = _p4_metric_effect(
            result["per_group"],
            baseline_key="baseline_image_miou",
            gated_key="gated_image_miou",
            statistics=statistics,
        )
        if int(boundary_effect["paired_group_count"]) != expected_groups or int(miou_effect["paired_group_count"]) != expected_groups:
            raise DvgProtocolError(f"P4 {condition} effects do not contain exactly {expected_groups} paired groups")
        effects_by_condition[condition] = {
            "oracle_gated_minus_corrupted_baseline_boundary_iou": boundary_effect,
            "oracle_gated_minus_corrupted_baseline_miou": miou_effect,
        }

    boundary_specificity = _p4_position_specificity(
        condition_results,
        baseline_key="baseline_boundary_iou",
        gated_key="gated_boundary_iou",
        statistics=statistics,
    )
    miou_specificity = _p4_position_specificity(
        condition_results,
        baseline_key="baseline_image_miou",
        gated_key="gated_image_miou",
        statistics=statistics,
    )
    if int(boundary_specificity["paired_group_count"]) != expected_groups or int(miou_specificity["paired_group_count"]) != expected_groups:
        raise DvgProtocolError("P4 position-specificity effects do not contain exactly 138 paired groups")

    primary_boundary = effects_by_condition["boundary-q75"][
        "oracle_gated_minus_corrupted_baseline_boundary_iou"
    ]
    primary_miou = effects_by_condition["boundary-q75"]["oracle_gated_minus_corrupted_baseline_miou"]
    adjudication = _adjudicate_p4(primary_boundary, primary_miou, protocol)
    condition_files = {
        condition: {
            "path": str(evidence_root / "conditions" / f"{condition}.json"),
            "sha256": file_sha256(evidence_root / "conditions" / f"{condition}.json"),
            "baseline_metrics_percent": result["baseline"]["metrics_percent"],
            "oracle_gated_metrics_percent": result["oracle_gated"]["metrics_percent"],
        }
        for condition, result in condition_results.items()
    }
    result = {
        "schema_version": "museg-dvg-b1-full-evaluation-v1",
        "protocol_id": "DVG-B1-oracle-gsa-v1",
        "protocol_path": str(protocol.path),
        "protocol_sha256": protocol_sha256,
        "status": "completed",
        "scientific_decision": adjudication["decision"],
        "created_at_utc": _utc_now(),
        "official_test_included": False,
        "checkpoint_loaded": True,
        "checkpoint_strict_load": True,
        "model_forward_executed": True,
        "gpu_used": True,
        "archived_prior_artifact": archived_prior_artifact,
        "prerequisites": prerequisite_context["prerequisites"],
        "source_manifest": {
            "path": str(prerequisite_context["source_manifest_path"]),
            "sha256": str(prerequisite_context["source_manifest_sha256"]),
        },
        "scope": {
            "sample_count": len(entries),
            "location_group_count": expected_groups,
            "conditions": list(CONDITION_IDS),
            "view_count_per_sample": int(protocol.raw["evaluator"]["view_count_per_sample"]),
            "paired_baseline_gated_forward_pairs": len(entries) * len(CONDITION_IDS) * int(protocol.raw["evaluator"]["view_count_per_sample"]),
        },
        "rng_control": {
            "reason": "LightHamHead NMF2D uses random bases in eval; replay identical CPU/CUDA RNG state for each baseline/gated view pair",
            "paired_cpu_rng_state_replayed_before_each_forward": True,
            "paired_cuda_rng_state_replayed_before_each_forward": True,
            "decoder_rand_init": bool(model.decode_head.hamburger.ham.rand_init),
        },
        "condition_files": condition_files,
        "mask_manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        "effects_by_condition": effects_by_condition,
        "position_specificity": {
            "boundary_iou": boundary_specificity,
            "miou": miou_specificity,
        },
        "adjudication": adjudication,
        "duration_seconds": round(time.monotonic() - full_started, 3),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        },
        "runtime_identity": _runtime_identity(),
    }
    _atomic_write_json(output_path, result)
    return result, output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("qualification", "noop-equivalence", "gate-preflight", "full"),
        default="qualification",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = _utc_now()
    protocol_path = args.protocol.resolve()
    artifact_names = {
        "qualification": "qualification.json",
        "noop-equivalence": "noop-equivalence.json",
        "gate-preflight": "gate-preflight.json",
        "full": "full-evaluation.json",
    }
    artifact_name = artifact_names[args.mode]
    result: dict[str, Any] | None = None
    output_path: Path | None = args.output.resolve() if args.output is not None else None
    try:
        protocol = load_dvg_protocol(protocol_path)
        if output_path is None:
            output_path = Path(str(protocol.raw["evidence"]["output_root"])).resolve() / artifact_name
        if args.mode == "qualification":
            if args.device != "cpu":
                raise ValueError("P1 qualification is restricted to CPU")
            result, output_path = run_qualification(protocol_path, output_path)
        elif args.mode == "noop-equivalence":
            result, output_path = run_noop_equivalence(
                protocol_path,
                output_path,
                torch.device(args.device),
            )
        elif args.mode == "gate-preflight":
            result, output_path = run_gate_preflight(
                protocol_path,
                output_path,
                torch.device(args.device),
            )
        else:
            result, output_path = run_full_evaluation(
                protocol_path,
                output_path,
                torch.device(args.device),
            )
        status = str(result["status"])
        successful_statuses = {"passed", "completed"}
        exit_code = 0 if status in successful_statuses else 2
        error = None if exit_code == 0 else "; ".join(str(value) for value in result.get("failures", []))
    except (
        DvgProtocolError,
        ImportError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        if output_path is None:
            output_path = protocol_path.parent / artifact_name
        status = "protocol-blocked"
        exit_code = 2
        error = str(exc)
        failure_schemas = {
            "qualification": "museg-dvg-b1-cpu-qualification-v1",
            "noop-equivalence": "museg-dvg-b1-noop-equivalence-v1",
            "gate-preflight": "museg-dvg-b1-gate-preflight-v1",
            "full": "museg-dvg-b1-full-evaluation-v1",
        }
        failure = {
            "schema_version": failure_schemas[args.mode],
            "protocol_id": "DVG-B1-oracle-gsa-v1",
            "status": status,
            "device": args.device,
            "checkpoint_loaded": False,
            "model_forward_executed": False,
            "gpu_used": args.device == "cuda",
            "official_test_included": False,
            "created_at_utc": _utc_now(),
            "protocol_path": str(protocol_path),
            "error": error,
        }
        _atomic_write_json(output_path, failure)

    assert output_path is not None
    artifact_sha256 = file_sha256(output_path)
    execution = {
        "schema_version": "museg-dvg-b1-execution-v1",
        "protocol_id": "DVG-B1-oracle-gsa-v1",
        "mode": args.mode,
        "device": args.device,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "status": status,
        "exit_code": exit_code,
        "checkpoint_loaded": bool(result.get("checkpoint_loaded", False)) if result is not None else False,
        "model_forward_executed": bool(result.get("model_forward_executed", False)) if result is not None else False,
        "gpu_used": bool(result.get("gpu_used", args.device == "cuda")) if result is not None else args.device == "cuda",
        "official_test_included": False,
        "protocol_path": str(protocol_path),
        "artifact_path": str(output_path),
        "artifact_sha256": artifact_sha256,
        "error": error,
    }
    suffixes = {
        "qualification": "qualification",
        "noop-equivalence": "noop-equivalence",
        "gate-preflight": "gate-preflight",
        "full": "full",
    }
    suffix = suffixes[args.mode]
    execution_name = dt.datetime.now(dt.timezone.utc).strftime(f"%Y%m%dT%H%M%S%f+0000-{suffix}.json")
    execution_path = output_path.parent / "executions" / execution_name
    _atomic_write_json(execution_path, execution)
    print(
        json.dumps(
            {
                "status": status,
                "mode": args.mode,
                "artifact_path": str(output_path),
                "artifact_sha256": artifact_sha256,
                "execution_path": str(execution_path),
                "exit_code": exit_code,
                "error": error,
            },
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
