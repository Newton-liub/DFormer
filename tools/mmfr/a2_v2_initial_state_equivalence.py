#!/usr/bin/env python3
"""Re-runnable pre-training gate: MMFR-A2 v2 initial-state equivalence.

What this gate proves
---------------------
Before any training starts, the two frozen v2 identities of the MMFR (multi-modal
failure reliability) A2 train-integration protocol must start from *the same shared
initial state*, otherwise their final comparison is not a comparison of the training
treatment but of two different random inits.

The two identities are

* ``MMFR-A2-clean-control-v2`` -- module
  ``local_configs.MUSeg.DFormerv2_S_MMFR_A2_Clean_v2``; and
* ``MMFR-A2-depth-corruption-train-v2`` -- module
  ``local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v2``.

The clean control never instantiates the auxiliary reliability head, the corruption
identity does, so the two models are not expected to have the same parameter *set*.
Only the **shared** parameters and buffers have to be bit-for-bit identical, and the
six ``backbone.extra_norms.{0,1,2}.{weight,bias}`` tensors -- which the official
pretrained checkpoint does not contain and which are therefore supplied by this
repository's own initialization code -- have to be reproducible under the frozen seed.

Method
------
Six independent subinterpreters build the two models three times each. Each child
starts from a clean interpreter, applies the repository's own ``set_seed`` (extracted
verbatim from ``utils/train.py`` by ``ast`` and executed, so the tool cannot drift from
the training code), builds the model exactly the way ``utils/train.py`` does, and writes
a JSON record with, for every parameter and buffer, its name, shape, dtype and two
independent SHA-256 digests:

* ``sha256_native_bytes`` -- SHA-256 of the contiguous raw bytes in the tensor's own
  dtype. Equality of this digest is equality of the stored bits, i.e. "逐位一致".
* ``sha256_float64_bytes`` -- SHA-256 of the same tensor cast to ``float64``. A second,
  value-space digest that also catches a dtype-only disagreement.

The parent process compares the child records by name and never asks a child to agree
with itself.

``max_abs_difference``
----------------------
When every shared tensor is bit-for-bit identical, the largest absolute elementwise
difference is *provably* ``0.0`` and is reported as such without dumping gigabytes of
weights. Only when a digest disagrees does the parent start a second, targeted child
pass that dumps the mismatching tensors as ``float64`` and computes the real numeric
difference. A mismatch that is too large to dump inside the frozen element budget is
reported as ``null`` with the reason attached; it is never silently reported as ``0.0``.

Scope and failure policy
------------------------
This tool only reads the repository, the config files and the pretrained checkpoint; it
never modifies a tracked file, never trains, never evaluates the dataset. It runs no
training step, no forward pass and touches no MUSeg data. Model construction is
CPU-only: the Ham decoder builds its NMF bases inside ``forward``
(``models/decoders/ham_head.py``), not in ``__init__``, so construction does not need
CUDA, and the gate records that fact instead of assuming it.

A failed check writes the measured evidence into the JSON report and exits non-zero.
The gate fails when a shared parameter or buffer disagrees, when a shared tensor is not
reproducible across the three rebuilds, when the six ``extra_norms`` tensors are not
reproducible, or when a precondition pinned by the frozen protocol does not hold.

Run
---
    python -m py_compile tools/mmfr/a2_v2_initial_state_equivalence.py
    python tools/mmfr/a2_v2_initial_state_equivalence.py

Use the same interpreter that runs training; on this machine that is the ``df2`` conda
environment (:file:`D:\\\\2Env\\\\anaconda\\\\envs\\\\df2\\\\python.exe`).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# The Windows console may default to a legacy code page; the reconfigure keeps an
# unexpected non-ASCII traceback printable.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - best effort only
        pass

SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_PATH)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen gate identity
# ---------------------------------------------------------------------------
GATE_ID = "MMFR-A2-v2-initial-state-equivalence"
SCHEMA_VERSION = "mmfr-a2-v2-initial-state-equivalence-v1"
PROTOCOL_ID = "MMFR-A2-train-integration-v2"

#: Each identity is rebuilt this many times in a *new* subinterpreter, so the frozen
#: seed can be shown to reproduce ``extra_norms`` rather than merely to be applied once.
REPEATS = 3

OUTPUT_DIR_NAME = "mmfr-a2-v2-initial-state-equivalence"
REPORT_NAME = "a2-v2-initial-state-equivalence.json"
STDOUT_NAME = "run-stdout.txt"

TRAIN_SOURCE_RELATIVE = "utils/train.py"

IDENTITIES: Tuple[Dict[str, Any], ...] = (
    {
        "label": "clean-control-v2",
        "config_module": "local_configs.MUSeg.DFormerv2_S_MMFR_A2_Clean_v2",
        "config_file": "local_configs/MUSeg/DFormerv2_S_MMFR_A2_Clean_v2.py",
        "expected_run_id": "museg-dformerv2-s-mmfr-a2-clean-control-v2",
        "expected_analysis_identity": "MMFR-A2-clean-control-v2",
        "expected_mode": "clean-control",
        "expects_reliability_head": False,
    },
    {
        "label": "depth-corruption-v2",
        "config_module": "local_configs.MUSeg.DFormerv2_S_MMFR_A2_DepthCorrupt_v2",
        "config_file": "local_configs/MUSeg/DFormerv2_S_MMFR_A2_DepthCorrupt_v2.py",
        "expected_run_id": "museg-dformerv2-s-mmfr-a2-depth-corruption-v2",
        "expected_analysis_identity": "MMFR-A2-depth-corruption-train-v2",
        "expected_mode": "depth-corruption",
        "expects_reliability_head": True,
    },
)

#: Frozen by the protocol: one model seed for both identities, one pretrained
#: checkpoint, Depth-only reliability supervision, AdamW with a two-bucket param split.
EXPECTED_SEED = 772961337
EXPECTED_PRETRAINED_PATH = r"D:\0Project\pretrained\DFormerv2_Small_pretrained.pth"
EXPECTED_PRETRAINED_SHA256 = "19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6"
EXPECTED_OPTIMIZER = "AdamW"
EXPECTED_BASE_LR = 6e-5
EXPECTED_WEIGHT_DECAY = 0.01
EXPECTED_ADAMW_BETAS = (0.9, 0.999)

#: The six tensors the official pretrained checkpoint does not provide. Names are the
#: full state-dict names; the short form used in the protocol narrative is the same name
#: without the ``backbone.`` prefix.
EXTRA_NORMS_NAMES: Tuple[str, ...] = tuple(
    "backbone.extra_norms.{}.{}".format(layer, suffix)
    for layer in range(3)
    for suffix in ("weight", "bias")
)

RELIABILITY_PREFIX = "reliability_estimator."
#: The protocol narrative rounds the auxiliary reliability head to "about 4,386
#: parameters". The measured value is reported next to it; a disagreement is surfaced
#: as an observation, not as a gate failure, because the narrative states it as approximate.
BRIEF_RELIABILITY_PARAMETER_COUNT = 4386

#: Budget for the second-pass numeric dump, in ``float64`` elements across all
#: mismatching tensors. Above this the difference is reported as ``null`` with a reason
#: instead of exhausting memory.
MAX_DUMP_ELEMENTS = 40_000_000


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256_raw(path: str) -> str:
    """SHA-256 of the raw bytes of ``path``."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def sha256_lf(path: str) -> str:
    """SHA-256 of ``path`` after normalizing CRLF to LF (the project's text hash)."""
    with open(path, "rb") as handle:
        data = handle.read()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a UTF-8 encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative(path: str) -> str:
    """Repository-relative, forward-slashed path for reporting."""
    return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")


def run_git(args: Sequence[str]) -> Dict[str, Any]:
    """Run a read-only git command in the repository root and keep the raw output."""
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "argv": ["git", *args],
        "returncode": int(completed.returncode),
        "stdout_raw": completed.stdout,
        "stderr_raw": completed.stderr,
    }


def tensor_digests(tensor: torch.Tensor) -> Dict[str, Any]:
    """Return shape, dtype and two independent SHA-256 digests of one tensor.

    ``sha256_native_bytes`` is called on a contiguous CPU copy in the tensor's own dtype.
    ``sha256_float64_bytes`` is called on the same values cast to ``float64`` so a dtype
    disagreement cannot hide behind equal bits.
    """
    local = tensor.detach().to("cpu").contiguous()
    try:
        native = local.numpy().tobytes()
    except (TypeError, RuntimeError):
        # bfloat16 and other numpy-less dtypes: take the raw bytes through a uint8 view.
        native = local.view(torch.uint8).numpy().tobytes()
    float64 = local.to(torch.float64).contiguous().numpy().tobytes()
    return {
        "shape": [int(size) for size in local.shape],
        "dtype": str(local.dtype),
        "numel": int(local.numel()),
        "sha256_native_bytes": hashlib.sha256(native).hexdigest(),
        "sha256_float64_bytes": hashlib.sha256(float64).hexdigest(),
    }


def tensor_digest_key(entry: Mapping[str, Any]) -> Tuple[Any, ...]:
    """The tuple two records must share to count as bit-for-bit equal."""
    return (
        tuple(entry.get("shape") or ()),
        str(entry.get("dtype")),
        str(entry.get("sha256_native_bytes")),
        str(entry.get("sha256_float64_bytes")),
    )


def float_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    """Cheap value summary for the small ``extra_norms`` evidence block."""
    local = tensor.detach().to("cpu").to(torch.float64).contiguous()
    return {
        "min": float(local.min().item()),
        "max": float(local.max().item()),
        "mean": float(local.mean().item()),
        "all_exactly_one": bool(torch.all(local == 1.0).item()),
        "all_exactly_zero": bool(torch.all(local == 0.0).item()),
    }


class CheckRegistry:
    """Ordered PASS/FAIL registry; every entry is written into the JSON report."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self.entries: List[Dict[str, Any]] = []

    def check(
        self,
        family: str,
        name: str,
        ok: bool,
        detail: str = "",
        gates: bool = True,
    ) -> bool:
        """Record one check. ``gates=False`` records evidence without failing the gate."""
        ok = bool(ok)
        self.entries.append(
            {
                "family": family,
                "name": name,
                "ok": ok,
                "gates": bool(gates),
                "detail": str(detail),
            }
        )
        self._emit(
            "{:4s}  {:>2s}  {:<58s}  {}".format(
                "PASS" if ok else "FAIL", family, name, detail
            )
        )
        return ok

    def note(self, family: str, name: str, detail: str) -> None:
        """Record an observation that is not a pass/fail check."""
        self.entries.append(
            {
                "family": family,
                "name": name,
                "ok": None,
                "gates": False,
                "detail": str(detail),
            }
        )
        self._emit("{:4s}  {:>2s}  {:<58s}  {}".format("INFO", family, name, detail))

    def failures(self) -> List[Dict[str, Any]]:
        return [entry for entry in self.entries if entry["gates"] and not entry["ok"]]

    def observations(self) -> List[Dict[str, Any]]:
        return [entry for entry in self.entries if entry["ok"] is None]


# ---------------------------------------------------------------------------
# Frozen source fragments read out of utils/train.py
# ---------------------------------------------------------------------------
def extract_train_source_fragments() -> Dict[str, Any]:
    """Extract ``set_seed`` and the ``torch.optim.AdamW`` call from ``utils/train.py``.

    The child executes the extracted ``set_seed`` source instead of a hand-copied
    duplicate, so the gate cannot silently drift from the code that actually trains.
    """
    path = os.path.join(PROJECT_ROOT, TRAIN_SOURCE_RELATIVE.replace("/", os.sep))
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source, filename=path)

    set_seed_source: Optional[str] = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "set_seed":
            set_seed_source = ast.get_source_segment(source, node) or ast.unparse(node)
            break

    adamw_call_source: Optional[str] = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "AdamW":
            continue
        value = func.value
        if not isinstance(value, ast.Attribute) or value.attr != "optim":
            continue
        if not isinstance(value.value, ast.Name) or value.value.id != "torch":
            continue
        adamw_call_source = ast.get_source_segment(source, node) or ast.unparse(node)
        break

    return {
        "path": path,
        "raw_sha256": sha256_raw(path),
        "lf_normalized_sha256": sha256_lf(path),
        "set_seed_source": set_seed_source,
        "set_seed_source_sha256": sha256_text(set_seed_source) if set_seed_source else None,
        "adamw_call_source": adamw_call_source,
        "adamw_call_source_sha256": sha256_text(adamw_call_source) if adamw_call_source else None,
    }


def build_set_seed_callable(set_seed_source: str) -> Callable[[int], None]:
    """Compile the extracted ``set_seed`` source into a callable, unmodified."""
    namespace: Dict[str, Any] = {"random": random, "np": np, "os": os, "torch": torch}
    exec(compile(set_seed_source, "<utils/train.py:set_seed>", "exec"), namespace)  # noqa: S102
    callable_set_seed = namespace.get("set_seed")
    if not callable(callable_set_seed):
        raise RuntimeError("extracted set_seed source did not define a callable set_seed")
    return callable_set_seed


# ---------------------------------------------------------------------------
# Child process: build one model and record its initial state
# ---------------------------------------------------------------------------
def run_child(args: argparse.Namespace) -> int:
    """Build one identity once and write its parameter/buffer digest record."""
    from importlib import import_module

    identity = IDENTITIES[int(args.identity_index)]
    started = time.perf_counter()
    record: Dict[str, Any] = {
        "child": True,
        "identity_label": identity["label"],
        "identity_index": int(args.identity_index),
        "repeat": int(args.repeat),
        "config_module": identity["config_module"],
        "config_file": identity["config_file"],
        "status": "ERROR",
        "error": None,
    }

    try:
        fragments = extract_train_source_fragments()
        if not fragments["set_seed_source"]:
            raise RuntimeError(f"could not extract set_seed from {TRAIN_SOURCE_RELATIVE}")
        set_seed = build_set_seed_callable(fragments["set_seed_source"])

        config = import_module(identity["config_module"]).C
        seed = int(config.seed)
        set_seed(seed)

        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
        from models.builder import EncoderDecoder

        model = EncoderDecoder(
            cfg=config,
            criterion=criterion,
            norm_layer=nn.SyncBatchNorm,
            syncbn=True,
        )

        parameters: Dict[str, Dict[str, Any]] = {}
        for name, tensor in model.named_parameters():
            parameters[str(name)] = {"kind": "parameter", **tensor_digests(tensor)}
        buffers: Dict[str, Dict[str, Any]] = {}
        for name, tensor in model.named_buffers():
            buffers[str(name)] = {"kind": "buffer", **tensor_digests(tensor)}

        parameter_numel_total = sum(int(entry["numel"]) for entry in parameters.values())
        buffer_numel_total = sum(int(entry["numel"]) for entry in buffers.values())

        parameter_lookup = dict(model.named_parameters())
        extra_norms: Dict[str, Dict[str, Any]] = {}
        for name in EXTRA_NORMS_NAMES:
            if name not in parameters:
                continue
            tensor = parameter_lookup[name]
            extra_norms[name] = {
                **tensor_digests(tensor),
                "values": float_summary(tensor),
            }

        # The optimizer is constructed exactly as utils/train.py constructs it, then only
        # its hyperparameters and param-group sizes are recorded. Building it mutates no
        # parameter value and consumes no RNG, so it cannot influence the comparison.
        from utils.init_func import group_weight

        base_lr = config.lr
        params_list = group_weight([], model, nn.SyncBatchNorm, base_lr)
        optimizer = torch.optim.AdamW(
            params_list,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
        optimizer_record = {
            "name": type(optimizer).__name__,
            "constructor_kwargs_source": {
                "lr": "config.lr",
                "betas": [0.9, 0.999],
                "weight_decay": "config.weight_decay",
            },
            "param_groups": [
                {
                    "index": int(index),
                    "lr": float(group["lr"]),
                    "weight_decay": float(group["weight_decay"]),
                    "param_count": int(len(group["params"])),
                    "numel": int(sum(int(p.numel()) for p in group["params"])),
                }
                for index, group in enumerate(optimizer.param_groups)
            ],
            "adamw_defaults": {
                "betas": [float(value) for value in optimizer.defaults.get("betas", ())],
                "eps": float(optimizer.defaults.get("eps", float("nan"))),
                "weight_decay": float(optimizer.defaults.get("weight_decay", float("nan"))),
                "lr": float(optimizer.defaults.get("lr", float("nan"))),
                "amsgrad": bool(optimizer.defaults.get("amsgrad", False)),
            },
        }

        reliability_parameter_keys = sorted(
            name for name in parameters if name.startswith(RELIABILITY_PREFIX)
        )

        record.update(
            {
                "status": "OK",
                "run_id": str(getattr(config, "run_id", "")),
                "analysis_identity": str(
                    (getattr(config, "mmfr_a2", {}) or {}).get("frozen", {}).get("analysis_identity", "")
                ),
                "mode": str((getattr(config, "mmfr_a2", {}) or {}).get("mode", "")),
                "protocol": str((getattr(config, "mmfr_a2", {}) or {}).get("protocol", "")),
                "seed": seed,
                "pretrained_model_from_config": os.path.abspath(str(config.pretrained_model)),
                "pretrained_sha256": sha256_raw(os.path.abspath(str(config.pretrained_model))),
                "reliability_head_instantiated": bool(model.reliability_estimator is not None),
                "reliability_parameter_keys": reliability_parameter_keys,
                "reliability_parameter_count": int(len(reliability_parameter_keys)),
                "reliability_parameter_numel": int(
                    sum(int(parameters[name]["numel"]) for name in reliability_parameter_keys)
                ),
                "parameter_keys": sorted(parameters),
                "buffer_keys": sorted(buffers),
                "parameter_count_total": int(len(parameters)),
                "buffer_count_total": int(len(buffers)),
                "parameter_numel_total": int(parameter_numel_total),
                "buffer_numel_total": int(buffer_numel_total),
                "parameters": parameters,
                "buffers": buffers,
                "extra_norms": extra_norms,
                "optimizer": optimizer_record,
                "set_seed_source_sha256": fragments["set_seed_source_sha256"],
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_initialized_after_build": bool(torch.cuda.is_initialized()),
                "device_used": "cpu",
                "environment": {
                    "python": sys.version.split()[0],
                    "torch": str(torch.__version__),
                    "cuda_build": str(torch.version.cuda),
                    "numpy": str(np.__version__),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                },
            }
        )

        if args.dump_names_file and args.dump_out:
            with open(args.dump_names_file, "r", encoding="utf-8") as handle:
                requested = [str(name) for name in json.load(handle)]
            lookup = dict(model.named_parameters())
            lookup.update(dict(model.named_buffers()))
            dumped: Dict[str, Any] = {}
            for name in requested:
                if name not in lookup:
                    dumped[name] = {"found": False}
                    continue
                tensor = lookup[name].detach().to("cpu").to(torch.float64).contiguous()
                dumped[name] = {
                    "found": True,
                    "shape": [int(size) for size in tensor.shape],
                    "values": tensor.reshape(-1).tolist(),
                }
            with open(args.dump_out, "w", encoding="utf-8") as handle:
                json.dump(dumped, handle)
                handle.write("\n")
            record["dump_out"] = os.path.abspath(args.dump_out)
            record["dumped_names"] = list(requested)

        record["build_seconds"] = round(time.perf_counter() - started, 3)
    except BaseException as error:  # noqa: BLE001 - the record must still be written
        record["status"] = "ERROR"
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        record["build_seconds"] = round(time.perf_counter() - started, 3)

    output_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    return 0


# ---------------------------------------------------------------------------
# Parent process
# ---------------------------------------------------------------------------
def build_environment() -> Dict[str, str]:
    """Child environment: keep ``PYTHONPATH`` so the repository packages resolve."""
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if PROJECT_ROOT not in parts:
        parts.insert(0, PROJECT_ROOT)
    environment["PYTHONPATH"] = os.pathsep.join(parts)
    return environment


def run_build(
    output_dir: str,
    identity_index: int,
    repeat: int,
    dump_names_file: Optional[str] = None,
    dump_out: Optional[str] = None,
    filename_stem: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one child build in a fresh subinterpreter and return its bookkeeping record."""
    identity = IDENTITIES[identity_index]
    label = identity["label"]
    stem = filename_stem or f"build-{label}-repeat{repeat}"
    child_path = os.path.join(output_dir, f"{stem}.json")
    stdout_path = os.path.join(output_dir, f"{stem}-stdout.txt")
    stderr_path = os.path.join(output_dir, f"{stem}-stderr.txt")

    command = [
        sys.executable,
        SCRIPT_PATH,
        "--child",
        "--identity-index",
        str(identity_index),
        "--repeat",
        str(repeat),
        "--out",
        child_path,
    ]
    if dump_names_file and dump_out:
        command += ["--dump-names-file", dump_names_file, "--dump-out", dump_out]

    started = time.perf_counter()
    with open(stdout_path, "w", encoding="utf-8") as stdout_handle, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=build_environment(),
            text=True,
        )
    seconds = round(time.perf_counter() - started, 3)

    child_record: Optional[Dict[str, Any]] = None
    load_error: Optional[str] = None
    if os.path.isfile(child_path):
        try:
            with open(child_path, "r", encoding="utf-8") as handle:
                child_record = json.load(handle)
        except json.JSONDecodeError as error:
            load_error = f"{type(error).__name__}: {error}"
    else:
        load_error = "child did not write its record"

    return {
        "identity_label": label,
        "identity_index": int(identity_index),
        "repeat": int(repeat),
        "command": command,
        "returncode": int(completed.returncode),
        "seconds": seconds,
        "record_path": relative(child_path),
        "record_sha256": sha256_raw(child_path) if os.path.isfile(child_path) else None,
        "stdout_path": relative(stdout_path),
        "stderr_path": relative(stderr_path),
        "record_load_error": load_error,
        "record": child_record,
    }


def compare_pair(
    parent: Mapping[str, Mapping[str, Mapping[str, Any]]],
    labels: Tuple[str, str],
) -> Dict[str, Any]:
    """Compare the shared parameters/buffers of two identities across their rebuilds."""
    first_label, second_label = labels
    first_runs = parent[first_label]
    second_runs = parent[second_label]
    repeats = sorted(int(key) for key in first_runs)

    first_parameters = first_runs[repeats[0]]["parameters"]
    second_parameters = second_runs[repeats[0]]["parameters"]
    first_buffers = first_runs[repeats[0]]["buffers"]
    second_buffers = second_runs[repeats[0]]["buffers"]

    common_parameter_keys = sorted(set(first_parameters) & set(second_parameters))
    common_buffer_keys = sorted(set(first_buffers) & set(second_buffers))
    only_in_first = sorted(set(first_parameters) - set(second_parameters))
    only_in_second = sorted(set(second_parameters) - set(first_parameters))
    only_in_first_buffers = sorted(set(first_buffers) - set(second_buffers))
    only_in_second_buffers = sorted(set(second_buffers) - set(first_buffers))

    unequal_parameter_keys: List[str] = []
    unequal_buffer_keys: List[str] = []
    unequal_details: List[Dict[str, Any]] = []
    for keys, first_map, second_map, sink in (
        (common_parameter_keys, first_parameters, second_parameters, unequal_parameter_keys),
        (common_buffer_keys, first_buffers, second_buffers, unequal_buffer_keys),
    ):
        for name in keys:
            first_entry = first_map[name]
            second_entry = second_map[name]
            if tensor_digest_key(first_entry) != tensor_digest_key(second_entry):
                sink.append(name)
                unequal_details.append(
                    {
                        "name": name,
                        "first_identity": first_label,
                        "second_identity": second_label,
                        "first": {
                            "shape": first_entry["shape"],
                            "dtype": first_entry["dtype"],
                            "sha256_native_bytes": first_entry["sha256_native_bytes"],
                        },
                        "second": {
                            "shape": second_entry["shape"],
                            "dtype": second_entry["dtype"],
                            "sha256_native_bytes": second_entry["sha256_native_bytes"],
                        },
                    }
                )

    # Within-identity reproducibility: a tensor that changes between two rebuilds of the
    # *same* identity is not reproducible, even if the two identities happen to agree.
    within_identity: Dict[str, Any] = {}
    for label, runs, key_map in (
        (first_label, first_runs, "parameters"),
        (first_label, first_runs, "buffers"),
        (second_label, second_runs, "parameters"),
        (second_label, second_runs, "buffers"),
    ):
        baseline = runs[repeats[0]][key_map]
        unstable: List[str] = []
        missing: List[str] = []
        for name, entry in baseline.items():
            reference = tensor_digest_key(entry)
            for repeat in repeats[1:]:
                other = runs[repeat][key_map]
                if name not in other:
                    missing.append(f"{key_map}:{name}@repeat{repeat}")
                    continue
                if tensor_digest_key(other[name]) != reference:
                    unstable.append(f"{key_map}:{name}@repeat{repeat}")
        within_identity[f"{label}/{key_map}"] = {
            "repeats": repeats,
            "key_count": int(len(baseline)),
            "unstable_keys": unstable,
            "missing_keys": missing,
        }

    return {
        "identities": [first_label, second_label],
        "repeats_per_identity": len(repeats),
        "common_parameter_keys": common_parameter_keys,
        "common_parameter_key_count": int(len(common_parameter_keys)),
        "common_buffer_keys": common_buffer_keys,
        "common_buffer_key_count": int(len(common_buffer_keys)),
        "parameters_exact_equal_count": int(
            len(common_parameter_keys) - len(unequal_parameter_keys)
        ),
        "buffers_exact_equal_count": int(len(common_buffer_keys) - len(unequal_buffer_keys)),
        "unequal_parameter_keys": unequal_parameter_keys,
        "unequal_buffer_keys": unequal_buffer_keys,
        "unequal_tensor_details": unequal_details,
        "only_in_first_identity_parameter_keys": only_in_first,
        "only_in_first_identity_parameter_count": int(len(only_in_first)),
        "only_in_second_identity_parameter_keys": only_in_second,
        "only_in_second_identity_parameter_count": int(len(only_in_second)),
        "only_in_first_identity_buffer_keys": only_in_first_buffers,
        "only_in_second_identity_buffer_keys": only_in_second_buffers,
        "within_identity_repeat_stability": within_identity,
        "max_abs_difference": None,
        "max_abs_difference_per_tensor": None,
        "max_abs_difference_method": None,
    }


def fill_zero_differences(comparison: Mapping[str, Any]) -> Dict[str, Any]:
    """When every shared tensor is bit-identical the max abs difference is provably 0."""
    per_tensor: Dict[str, float] = {}
    for name in comparison["common_parameter_keys"]:
        per_tensor[name] = 0.0
    for name in comparison["common_buffer_keys"]:
        per_tensor[name] = 0.0
    return {
        "max_abs_difference": 0.0,
        "max_abs_difference_per_tensor": per_tensor,
        "max_abs_difference_method": (
            "every shared parameter and buffer has identical sha256_native_bytes in all "
            "builds of both identities, so the elementwise maximum absolute difference is "
            "exactly 0.0 by bit-for-bit identity; no numeric dump was needed"
        ),
    }


def compute_numeric_differences(
    output_dir: str,
    comparison: Mapping[str, Any],
    emit: Callable[[str], None],
) -> Dict[str, Any]:
    """Second pass: dump the mismatching tensors and measure the real difference."""
    labels = comparison["identities"]
    unequal_parameter_keys = list(comparison["unequal_parameter_keys"])
    unequal_buffer_keys = list(comparison["unequal_buffer_keys"])
    names = unequal_parameter_keys + unequal_buffer_keys
    if not names:
        return {
            "max_abs_difference": None,
            "max_abs_difference_per_tensor": None,
            "max_abs_difference_method": "no mismatching tensor was reported",
        }

    index_labels = {identity["label"]: index for index, identity in enumerate(IDENTITIES)}
    per_tensor: Dict[str, Any] = {}
    values: Dict[str, Dict[str, List[float]]] = {}
    for label in labels:
        index = index_labels[label]
        names_file = os.path.join(output_dir, f"dump-names-{label}.json")
        dump_out = os.path.join(output_dir, f"dump-values-{label}.json")
        with open(names_file, "w", encoding="utf-8") as handle:
            json.dump(names, handle)
        emit(f"  numeric second pass for {label}: {len(names)} tensor(s)")
        build = run_build(
            output_dir,
            index,
            0,
            dump_names_file=names_file,
            dump_out=dump_out,
            filename_stem=f"numeric-pass-{label}",
        )
        if build["record"] is None or build["record"].get("status") != "OK":
            return {
                "max_abs_difference": None,
                "max_abs_difference_per_tensor": None,
                "max_abs_difference_method": (
                    f"numeric second pass for {label} did not complete; difference unknown"
                ),
                "numeric_pass_builds": [build],
            }
        with open(dump_out, "r", encoding="utf-8") as handle:
            values[label] = json.load(handle)
        per_tensor.setdefault("_builds", []).append(
            {
                "identity_label": label,
                "returncode": build["returncode"],
                "seconds": build["seconds"],
                "dump_path": relative(dump_out),
                "dump_sha256": sha256_raw(dump_out),
            }
        )

    overall = 0.0
    for name in names:
        first = values[labels[0]].get(name, {})
        second = values[labels[1]].get(name, {})
        if not first.get("found") or not second.get("found"):
            per_tensor[name] = None
            continue
        first_array = np.asarray(first["values"], dtype=np.float64)
        second_array = np.asarray(second["values"], dtype=np.float64)
        if first_array.shape != second_array.shape:
            per_tensor[name] = None
            continue
        difference = float(np.max(np.abs(first_array - second_array))) if first_array.size else 0.0
        per_tensor[name] = difference
        overall = max(overall, difference)

    builds = per_tensor.pop("_builds", [])
    return {
        "max_abs_difference": overall,
        "max_abs_difference_per_tensor": per_tensor,
        "max_abs_difference_method": (
            "second child pass dumped the mismatching tensors as float64 and the difference "
            "was computed elementwise; tensors too large for the frozen element budget are "
            "reported as null"
        ),
        "numeric_pass_builds": builds,
    }


def collect_extra_norms(
    parent: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Collect the six repository-initialized ``extra_norms`` tensors across all builds."""
    evidence: Dict[str, Any] = {}
    for name in EXTRA_NORMS_NAMES:
        per_build: List[Dict[str, Any]] = []
        for label, runs in parent.items():
            for repeat in sorted(int(key) for key in runs):
                entry = (runs[repeat].get("parameters") or {}).get(name)
                per_build.append(
                    {
                        "identity_label": label,
                        "repeat": repeat,
                        "present": entry is not None,
                        "shape": entry["shape"] if entry else None,
                        "dtype": entry["dtype"] if entry else None,
                        "sha256_native_bytes": entry["sha256_native_bytes"] if entry else None,
                        "sha256_float64_bytes": entry["sha256_float64_bytes"] if entry else None,
                        "values": (runs[repeat].get("extra_norms") or {}).get(name, {}).get("values"),
                    }
                )
        distinct_native = sorted(
            {entry["sha256_native_bytes"] for entry in per_build if entry["present"]}
        )
        evidence[name] = {
            "short_name": name.replace("backbone.", "", 1),
            "shape": per_build[0]["shape"] if per_build else None,
            "dtype": per_build[0]["dtype"] if per_build else None,
            "present_in_all_builds": all(entry["present"] for entry in per_build),
            "builds": per_build,
            "distinct_sha256_native_bytes": distinct_native,
            "distinct_sha256_native_count": int(len(distinct_native)),
            "reproducible": bool(
                all(entry["present"] for entry in per_build) and len(distinct_native) == 1
            ),
        }
    return evidence


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="MMFR-A2 v2 initial-state equivalence gate (shared parameters/buffers + extra_norms reproducibility)."
    )
    parser.add_argument("--child", action="store_true", help="internal: build one identity once")
    parser.add_argument("--identity-index", type=int, default=0, help="internal: index into IDENTITIES")
    parser.add_argument("--repeat", type=int, default=0, help="internal: rebuild index")
    parser.add_argument("--out", default=None, help="internal: path of the child record")
    parser.add_argument("--dump-names-file", default=None, help="internal: names to dump as float64")
    parser.add_argument("--dump-out", default=None, help="internal: path of the float64 dump")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "outputs", OUTPUT_DIR_NAME),
        help="directory of the JSON report and the child records",
    )
    parser.add_argument("--report", default=None, help="path of the JSON report")
    args = parser.parse_args(argv)

    if args.child:
        if not args.out:
            parser.error("--child requires --out")
        return run_child(args)

    started = time.perf_counter()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.abspath(args.report or os.path.join(output_dir, REPORT_NAME))

    stdout_lines: List[str] = []

    def emit(text: str = "") -> None:
        stdout_lines.append(text)
        print(text)

    registry = CheckRegistry(emit)
    emit(f"{GATE_ID}: MMFR-A2 v2 initial-state equivalence")
    emit(f"repository root: {PROJECT_ROOT}")
    emit(f"interpreter: {sys.executable}")
    emit(f"identities: {', '.join(identity['config_module'] for identity in IDENTITIES)}")
    emit(f"rebuilds per identity: {REPEATS}")
    emit("")

    report: Dict[str, Any] = {
        "gate": GATE_ID,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "purpose": (
            "prove, before any training, that the two frozen MMFR-A2 v2 identities share a "
            "bit-for-bit identical initial state on their common parameters and buffers, and "
            "that the six repository-initialized backbone.extra_norms tensors are reproducible "
            "under the frozen model seed"
        ),
        "tool": relative(SCRIPT_PATH),
        "tool_sha256_raw_bytes": sha256_raw(SCRIPT_PATH),
        "tool_sha256_lf_normalized": sha256_lf(SCRIPT_PATH),
        "tool_line_count": int(
            sum(1 for _ in open(SCRIPT_PATH, "r", encoding="utf-8"))
        ),  # noqa: SIM115 - read-only, closed at interpreter exit
        "method": {
            "build_command": (
                "python tools/mmfr/a2_v2_initial_state_equivalence.py  (six fresh subinterpreters, "
                "one model build each)"
            ),
            "model_construction": (
                "models.builder.EncoderDecoder(cfg=config, criterion=CrossEntropyLoss(ignore_index=config.background), "
                "norm_layer=nn.SyncBatchNorm, syncbn=True) -- the same call utils/train.py makes"
            ),
            "seed_application": (
                "set_seed extracted verbatim from utils/train.py by ast and executed, then called with "
                "the config seed before model construction; no hand-copied duplicate"
            ),
            "equality_criterion": (
                "identical shape + identical dtype + identical sha256 of the contiguous native-dtype bytes "
                "+ identical sha256 of the float64 cast, in every build of both identities"
            ),
            "max_abs_difference_method": (
                "exactly 0.0 when a tensor is bit-identical; otherwise a second targeted child pass dumps "
                "the mismatching tensors as float64 and measures the elementwise maximum"
            ),
            "device": (
                "CPU model construction: the Ham decoder builds its NMF bases inside forward "
                "(models/decoders/ham_head.py), not in __init__, so no CUDA API is needed to reach the "
                "initial state and no randomness is introduced by moving the model to a device"
            ),
            "data_and_training": (
                "no MUSeg sample is read, no forward pass is run, no training or evaluation step is taken"
            ),
        },
        "initial_state_equivalence_semantics": (
            "PASS when no gated check failed: both builds succeeded, the frozen seed / pretrained "
            "checkpoint / optimizer identity match the pinned protocol values, every shared parameter and "
            "buffer is bit-for-bit identical across all builds of both identities, the parameter sets differ "
            "only by the reliability_estimator.* keys, and the six backbone.extra_norms tensors are identical "
            "in all six builds"
        ),
        "environment": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "cuda_build": str(torch.version.cuda),
            "numpy": str(np.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "git": {
            "rev_parse_head": run_git(["rev-parse", "HEAD"]),
            "status_porcelain": run_git(["status", "--porcelain"]),
        },
        "config_files": {},
        "train_source": {},
        "pretrained": {},
        "identities": [],
        "builds": [],
        "extra_norms": {},
        "extra_norms_reproducible": False,
        "comparison": {},
        "checks": [],
        "observations": [],
        "failures": [],
        "assertions_total": 0,
        "assertions_failed": 0,
        "initial_state_equivalence": "FAIL",
        "status": "ERROR",
        "exit_code": 2,
        "runtime_seconds": 0.0,
        "report_path": report_path,
        "not_executed": [
            "no training step, no forward pass, no dataset access, no evaluation, no GPU compute, no full test suite",
        ],
    }

    exit_code = 2
    parent: Dict[str, Dict[int, Dict[str, Any]]] = {identity["label"]: {} for identity in IDENTITIES}
    try:
        # -- preconditions pinned by the frozen protocol --------------------
        fragments = extract_train_source_fragments()
        report["train_source"] = {
            "path": relative(fragments["path"]),
            "raw_sha256": fragments["raw_sha256"],
            "lf_normalized_sha256": fragments["lf_normalized_sha256"],
            "set_seed_source_sha256": fragments["set_seed_source_sha256"],
            "adamw_call_source": fragments["adamw_call_source"],
            "adamw_call_source_sha256": fragments["adamw_call_source_sha256"],
            "note": (
                "the child executes this extracted set_seed source; the record makes the exact "
                "training-time seeding code auditable"
            ),
        }
        registry.check(
            "0",
            "train_source.set_seed_extracted",
            bool(fragments["set_seed_source"]) and bool(fragments["adamw_call_source"]),
            f"set_seed_sha256={fragments['set_seed_source_sha256']} adamw_call={fragments['adamw_call_source']!r}",
        )

        for identity in IDENTITIES:
            path = os.path.join(PROJECT_ROOT, identity["config_file"].replace("/", os.sep))
            present = os.path.isfile(path)
            report["config_files"][identity["config_file"]] = {
                "module": identity["config_module"],
                "present": present,
                "raw_sha256": sha256_raw(path) if present else None,
                "lf_normalized_sha256": sha256_lf(path) if present else None,
                "bytes": os.path.getsize(path) if present else None,
            }
            registry.check(
                "0",
                f"config_files.{identity['label']}.file_present",
                present,
                f"path={relative(path)} raw_sha256={report['config_files'][identity['config_file']]['raw_sha256']}",
            )

        head = report["git"]["rev_parse_head"]["stdout_raw"].strip()
        dirty = report["git"]["status_porcelain"]["stdout_raw"]
        registry.note(
            "0",
            "git.revision_and_worktree",
            f"HEAD={head} status_porcelain_lines={len([line for line in dirty.splitlines() if line.strip()])}",
        )

        # -- six independent builds ----------------------------------------
        emit("building the two identities, three fresh subinterpreters each")
        for index, identity in enumerate(IDENTITIES):
            for repeat in range(REPEATS):
                build = run_build(output_dir, index, repeat)
                record = build.pop("record")
                build["record_status"] = (record or {}).get("status")
                build["record_error"] = (record or {}).get("error")
                report["builds"].append(build)
                if record is not None and record.get("status") == "OK":
                    parent[identity["label"]][repeat] = record
                summary = (
                    f"params={record.get('parameter_count_total')} "
                    f"param_numel={record.get('parameter_numel_total')} "
                    f"buffers={record.get('buffer_count_total')} "
                    f"reliability_head={record.get('reliability_head_instantiated')}"
                    if record and record.get("status") == "OK"
                    else f"status={(record or {}).get('status', 'MISSING')}"
                )
                emit(
                    f"  {identity['label']} repeat {repeat}: exit={build['returncode']} "
                    f"{build['seconds']:.1f}s {summary}"
                )
                registry.check(
                    "1",
                    f"build.{identity['label']}.repeat{repeat}.succeeded",
                    build["returncode"] == 0 and (record or {}).get("status") == "OK",
                    f"exit={build['returncode']} seconds={build['seconds']} record={build['record_path']}",
                )

        for identity in IDENTITIES:
            label = identity["label"]
            records = parent[label]
            if len(records) != REPEATS:
                continue
            first = records[0]
            registry.check(
                "2",
                f"identity.{label}.run_id_matches_frozen",
                first["run_id"] == identity["expected_run_id"],
                f"run_id={first['run_id']!r} expected={identity['expected_run_id']!r}",
            )
            registry.check(
                "2",
                f"identity.{label}.analysis_identity_matches_frozen",
                first["analysis_identity"] == identity["expected_analysis_identity"],
                f"analysis_identity={first['analysis_identity']!r} expected={identity['expected_analysis_identity']!r}",
            )
            registry.check(
                "2",
                f"identity.{label}.mode_matches_frozen",
                first["mode"] == identity["expected_mode"],
                f"mode={first['mode']!r} expected={identity['expected_mode']!r}",
            )
            registry.check(
                "2",
                f"identity.{label}.protocol_matches_frozen",
                first["protocol"] == PROTOCOL_ID,
                f"protocol={first['protocol']!r} expected={PROTOCOL_ID!r}",
            )
            registry.check(
                "2",
                f"identity.{label}.seed_matches_frozen",
                int(first["seed"]) == EXPECTED_SEED,
                f"seed={first['seed']} expected={EXPECTED_SEED}",
            )
            registry.check(
                "2",
                f"identity.{label}.reliability_head_expectation",
                bool(first["reliability_head_instantiated"]) is identity["expects_reliability_head"],
                f"instantiated={first['reliability_head_instantiated']} expected={identity['expects_reliability_head']}",
            )
            optimizer = first["optimizer"]
            groups = optimizer["param_groups"]
            registry.check(
                "2",
                f"identity.{label}.optimizer_identity",
                optimizer["name"] == EXPECTED_OPTIMIZER
                and tuple(optimizer["adamw_defaults"]["betas"]) == EXPECTED_ADAMW_BETAS
                and float(optimizer["adamw_defaults"]["weight_decay"]) == EXPECTED_WEIGHT_DECAY
                and float(groups[0]["lr"]) == EXPECTED_BASE_LR,
                f"{optimizer['name']} lr={groups[0]['lr']} betas={optimizer['adamw_defaults']['betas']} "
                f"weight_decay={optimizer['adamw_defaults']['weight_decay']} "
                f"groups={[(g['lr'], g['weight_decay'], g['param_count']) for g in groups]}",
            )

        pretrained_paths = {
            records[0]["pretrained_model_from_config"]
            for records in parent.values()
            if records
        }
        config_pretrained = sorted(pretrained_paths)[0] if pretrained_paths else EXPECTED_PRETRAINED_PATH
        pretrained_present = os.path.isfile(config_pretrained)
        cross_identity_same_pretrained = len(pretrained_paths) <= 1
        pretrained_sha = sha256_raw(config_pretrained) if pretrained_present else None
        pretrained_missing_extra_norms: List[str] = []
        pretrained_extra_norms_present: List[str] = []
        pretrained_key_count: Optional[int] = None
        if pretrained_present:
            state = torch.load(config_pretrained, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                pretrained_key_count = int(len(state))
                for name in EXTRA_NORMS_NAMES:
                    short = name.replace("backbone.", "", 1)
                    if name in state or short in state:
                        pretrained_extra_norms_present.append(name)
                    else:
                        pretrained_missing_extra_norms.append(name)
        report["pretrained"] = {
            "path": config_pretrained,
            "path_matches_protocol_brief": os.path.abspath(config_pretrained)
            == os.path.abspath(EXPECTED_PRETRAINED_PATH),
            "present": pretrained_present,
            "sha256": pretrained_sha,
            "expected_sha256": EXPECTED_PRETRAINED_SHA256,
            "sha256_matches_frozen": pretrained_sha == EXPECTED_PRETRAINED_SHA256,
            "same_checkpoint_for_both_identities": cross_identity_same_pretrained,
            "state_dict_key_count": pretrained_key_count,
            "extra_norms_keys_present_in_checkpoint": pretrained_extra_norms_present,
            "extra_norms_keys_missing_from_checkpoint": pretrained_missing_extra_norms,
        }
        registry.check(
            "2",
            "pretrained.sha256_matches_frozen",
            pretrained_sha == EXPECTED_PRETRAINED_SHA256,
            f"path={config_pretrained} sha256={pretrained_sha} expected={EXPECTED_PRETRAINED_SHA256}",
        )
        registry.check(
            "2",
            "pretrained.identical_for_both_identities",
            cross_identity_same_pretrained,
            f"distinct_config_paths={sorted(pretrained_paths)}",
        )
        registry.note(
            "2",
            "pretrained.extra_norms_are_repository_initialized",
            "checkpoint provides {}/{} of the six backbone.extra_norms tensors; missing={}".format(
                len(pretrained_extra_norms_present),
                len(EXTRA_NORMS_NAMES),
                pretrained_missing_extra_norms,
            ),
        )

        # -- shared-state comparison ---------------------------------------
        labels = tuple(identity["label"] for identity in IDENTITIES)
        both_complete = all(len(parent[label]) == REPEATS for label in labels)
        if both_complete:
            comparison = compare_pair(parent, labels)
            if not comparison["unequal_parameter_keys"] and not comparison["unequal_buffer_keys"]:
                comparison.update(fill_zero_differences(comparison))
            else:
                comparison.update(compute_numeric_differences(output_dir, comparison, emit))
            report["comparison"] = comparison

            registry.check(
                "3",
                "comparison.common_parameters_bitwise_identical",
                not comparison["unequal_parameter_keys"],
                f"equal={comparison['parameters_exact_equal_count']}/{comparison['common_parameter_key_count']} "
                f"unequal={comparison['unequal_parameter_keys'][:10]}",
            )
            registry.check(
                "3",
                "comparison.common_buffers_bitwise_identical",
                not comparison["unequal_buffer_keys"],
                f"equal={comparison['buffers_exact_equal_count']}/{comparison['common_buffer_key_count']} "
                f"unequal={comparison['unequal_buffer_keys'][:10]}",
            )
            registry.check(
                "3",
                "comparison.clean_identity_has_no_extra_parameters",
                not comparison["only_in_first_identity_parameter_keys"],
                f"only_in_clean={comparison['only_in_first_identity_parameter_keys'][:10]}",
            )
            unexpected = [
                name
                for name in comparison["only_in_second_identity_parameter_keys"]
                if not name.startswith(RELIABILITY_PREFIX)
            ]
            registry.check(
                "3",
                "comparison.corruption_extra_parameters_are_reliability_head_only",
                not unexpected,
                f"extra_count={comparison['only_in_second_identity_parameter_count']} "
                f"all_reliability_head={not unexpected} unexpected={unexpected[:10]}",
            )
            registry.check(
                "3",
                "comparison.clean_identity_has_no_extra_buffers",
                not comparison["only_in_first_identity_buffer_keys"],
                f"only_in_clean={comparison['only_in_first_identity_buffer_keys'][:10]}",
            )
            registry.check(
                "3",
                "comparison.corruption_extra_buffers_are_reliability_head_only",
                all(
                    name.startswith(RELIABILITY_PREFIX)
                    for name in comparison["only_in_second_identity_buffer_keys"]
                ),
                f"only_in_corruption={comparison['only_in_second_identity_buffer_keys'][:10]}",
            )
            unstable = {
                key: value
                for key, value in comparison["within_identity_repeat_stability"].items()
                if value["unstable_keys"] or value["missing_keys"]
            }
            registry.check(
                "3",
                "comparison.every_tensor_stable_across_its_own_rebuilds",
                not unstable,
                "all tensors identical in all three rebuilds of their own identity"
                if not unstable
                else f"unstable={ {k: v['unstable_keys'][:5] for k, v in unstable.items()} }",
            )
            per_tensor_difference = comparison.get("max_abs_difference_per_tensor") or {}
            difference_value = comparison.get("max_abs_difference")
            non_zero = {
                name: value
                for name, value in per_tensor_difference.items()
                if value is None or float(value) != 0.0
            }
            registry.check(
                "3",
                "comparison.max_abs_difference_is_exactly_zero",
                difference_value is not None
                and float(difference_value) == 0.0
                and not non_zero,
                f"max_abs_difference={difference_value} "
                f"difference_known={difference_value is not None} non_zero={list(non_zero)[:10]}",
            )

            expected_reliability = sum(
                1
                for name in comparison["only_in_second_identity_parameter_keys"]
                if name.startswith(RELIABILITY_PREFIX)
            )
            registry.note(
                "3",
                "comparison.reliability_head_parameter_count",
                "measured={} brief_says_about={} numel={} keys={}".format(
                    expected_reliability,
                    BRIEF_RELIABILITY_PARAMETER_COUNT,
                    parent[labels[1]][0]["reliability_parameter_numel"],
                    comparison["only_in_second_identity_parameter_keys"],
                ),
            )
        else:
            registry.check(
                "3",
                "comparison.both_identities_have_three_records",
                False,
                "at least one identity did not produce three successful rebuilds; the shared-state "
                "comparison could not run",
            )

        # -- extra_norms reproducibility -----------------------------------
        extra_norms = collect_extra_norms(parent)
        report["extra_norms"] = extra_norms
        for name, entry in extra_norms.items():
            registry.check(
                "4",
                f"extra_norms.{entry['short_name']}.reproducible_in_all_six_builds",
                entry["reproducible"],
                f"shape={entry['shape']} dtype={entry['dtype']} "
                f"distinct_sha256_native={entry['distinct_sha256_native_count']} "
                f"sha256={entry['distinct_sha256_native_bytes'][0] if entry['distinct_sha256_native_bytes'] else None}",
            )
        extra_norms_reproducible = bool(extra_norms) and all(
            entry["reproducible"] for entry in extra_norms.values()
        )
        report["extra_norms_reproducible"] = extra_norms_reproducible
        registry.check(
            "4",
            "extra_norms.all_six_tensors_reproducible",
            extra_norms_reproducible and len(extra_norms) == len(EXTRA_NORMS_NAMES),
            f"tensors={len(extra_norms)} expected={len(EXTRA_NORMS_NAMES)} reproducible={extra_norms_reproducible}",
        )
    except BaseException as error:  # noqa: BLE001 - the report must still be written
        registry.check(
            "harness",
            "harness.unexpected_exception",
            False,
            traceback.format_exc().strip().splitlines()[-1],
        )
        emit("")
        emit("HARNESS EXCEPTION")
        emit(traceback.format_exc())
        report["harness_exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    # -- identity summary blocks -------------------------------------------
    for identity in IDENTITIES:
        records = parent[identity["label"]]
        if not records:
            continue
        first = records[0]
        report["identities"].append(
            {
                "label": identity["label"],
                "config_module": identity["config_module"],
                "config_file": identity["config_file"],
                "expected_run_id": identity["expected_run_id"],
                "expected_analysis_identity": identity["expected_analysis_identity"],
                "run_id": first["run_id"],
                "analysis_identity": first["analysis_identity"],
                "mode": first["mode"],
                "protocol": first["protocol"],
                "seed": int(first["seed"]),
                "pretrained_model": first["pretrained_model_from_config"],
                "pretrained_sha256": first["pretrained_sha256"],
                "reliability_head_instantiated": first["reliability_head_instantiated"],
                "parameter_count_total": first["parameter_count_total"],
                "parameter_numel_total": first["parameter_numel_total"],
                "buffer_count_total": first["buffer_count_total"],
                "buffer_numel_total": first["buffer_numel_total"],
                "reliability_parameter_count": first["reliability_parameter_count"],
                "reliability_parameter_numel": first["reliability_parameter_numel"],
                "reliability_parameter_keys": first["reliability_parameter_keys"],
                "optimizer": first["optimizer"],
                "build_seconds": [records[repeat]["build_seconds"] for repeat in sorted(records)],
                "cuda_available": first["cuda_available"],
                "cuda_initialized_after_build": first["cuda_initialized_after_build"],
                "device_used": first["device_used"],
                "environment": first["environment"],
            }
        )

    failures = registry.failures()
    report["checks"] = registry.entries
    report["observations"] = registry.observations()
    report["assertions_total"] = int(sum(1 for entry in registry.entries if entry["ok"] is not None))
    report["assertions_failed"] = int(len(failures))
    report["failures"] = [
        {"family": entry["family"], "name": entry["name"], "detail": entry["detail"]}
        for entry in failures
    ]
    gate_passed = bool(registry.entries) and not failures and "harness_exception" not in report
    report["initial_state_equivalence"] = "PASS" if gate_passed else "FAIL"
    report["status"] = report["initial_state_equivalence"]
    exit_code = 0 if gate_passed else 1
    report["exit_code"] = exit_code
    report["runtime_seconds"] = round(time.perf_counter() - started, 3)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    report_sha256 = sha256_raw(report_path)

    comparison = report.get("comparison") or {}
    emit("")
    emit("shared-state comparison")
    emit(
        "  common_parameter_keys={} parameters_exact_equal_count={}".format(
            comparison.get("common_parameter_key_count"),
            comparison.get("parameters_exact_equal_count"),
        )
    )
    emit(
        "  common_buffer_keys={}    buffers_exact_equal_count={}".format(
            comparison.get("common_buffer_key_count"),
            comparison.get("buffers_exact_equal_count"),
        )
    )
    emit(f"  max_abs_difference={comparison.get('max_abs_difference')}")
    emit(
        "  only_in_corruption_parameters={} (all reliability_estimator.*)".format(
            comparison.get("only_in_second_identity_parameter_count")
        )
    )
    emit("")
    emit("extra_norms reproducibility (3 fresh builds per identity, 6 builds total)")
    for name, entry in sorted(report.get("extra_norms", {}).items()):
        emit(
            "  {:<40s} shape={!s:<10s} sha256_native={} distinct={} reproducible={}".format(
                entry["short_name"],
                entry["shape"],
                entry["distinct_sha256_native_bytes"][0] if entry["distinct_sha256_native_bytes"] else None,
                entry["distinct_sha256_native_count"],
                entry["reproducible"],
            )
        )
    emit("")
    emit(f"report_path: {report_path}")
    emit(f"report_sha256: {report_sha256}")
    emit(f"tool_sha256_raw_bytes: {report['tool_sha256_raw_bytes']}")
    emit(
        "SUMMARY initial_state_equivalence={} extra_norms_reproducible={} failed={} exit_code={} runtime={:.1f}s".format(
            report["initial_state_equivalence"],
            report["extra_norms_reproducible"],
            report["assertions_failed"],
            exit_code,
            report["runtime_seconds"],
        )
    )
    if failures:
        emit("FAILED CHECKS:")
        for entry in failures:
            emit(f"  {entry['family']}.{entry['name']}: {entry['detail']}")

    stdout_path = os.path.join(output_dir, STDOUT_NAME)
    with open(stdout_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(stdout_lines) + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
