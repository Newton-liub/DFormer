"""Scoped tensor diagnostics for NaturalMissing numerical investigations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _json_number(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().item()
    if isinstance(value, (int, bool)):
        return int(value) if not isinstance(value, bool) else bool(value)
    numeric = float(value)
    if math.isnan(numeric):
        return "NaN"
    if math.isinf(numeric):
        return "+Inf" if numeric > 0 else "-Inf"
    return numeric


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    """Return compact finite-value statistics without exposing tensor contents."""
    value = tensor.detach()
    result: dict[str, Any] = {
        "is_finite": None,
        "dtype": str(value.dtype),
        "shape": [int(size) for size in value.shape],
        "numel": int(value.numel()),
        "finite_count": None,
        "nan_count": None,
        "inf_count": None,
        "min": None,
        "max": None,
    }
    if value.numel() == 0:
        result.update({"is_finite": True, "finite_count": 0, "nan_count": 0, "inf_count": 0})
        return result

    try:
        finite_mask = torch.isfinite(value)
        nan_mask = torch.isnan(value)
        inf_mask = torch.isinf(value)
        counts = torch.stack(
            (finite_mask.sum(), nan_mask.sum(), inf_mask.sum())
        ).to(dtype=torch.int64).cpu().tolist()
        finite_count, nan_count, inf_count = (int(count) for count in counts)
        result.update(
            {
                "is_finite": finite_count == int(value.numel()),
                "finite_count": finite_count,
                "nan_count": nan_count,
                "inf_count": inf_count,
            }
        )
        if finite_count:
            finite_values = value[finite_mask] if finite_count != int(value.numel()) else value
            if finite_values.is_complex():
                finite_values = finite_values.abs()
            result["min"] = _json_number(finite_values.min())
            result["max"] = _json_number(finite_values.max())
    except (RuntimeError, TypeError, ValueError) as exc:
        result["stats_error"] = f"{type(exc).__name__}: {exc}"
    return result


def tensor_is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().item())


def safe_scalar(tensor: torch.Tensor) -> float | int | str | None:
    value = tensor.detach()
    if value.numel() != 1:
        return None
    return _json_number(value.reshape(()))


def _max_abs(tensor: torch.Tensor) -> float | int | str | None:
    value = tensor.detach()
    if value.numel() == 0:
        return None
    return _json_number(value.abs().max())


def parameter_gradient_summary(model: nn.Module) -> dict[str, Any]:
    """Find the first named parameter with a non-finite gradient and its magnitude."""
    first_bad = None
    bad_parameter_count = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None or tensor_is_finite(gradient):
            continue
        bad_parameter_count += 1
        if first_bad is None:
            first_bad = {
                "name": name,
                "max_abs": _max_abs(gradient),
                "stats": tensor_stats(gradient),
            }
    return {
        "is_finite": bad_parameter_count == 0,
        "bad_parameter_count": bad_parameter_count,
        "first_nonfinite_parameter": first_bad,
    }


def sample_filenames(value: Any, *, limit: int = 10) -> list[str] | None:
    """Extract a small list of sample filenames, never tensor or image contents."""
    if value is None:
        return None
    names: list[str] = []

    def visit(item: Any) -> None:
        if len(names) >= limit:
            return
        if isinstance(item, (str, Path)):
            names.append(str(item))
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
                if len(names) >= limit:
                    break

    visit(value)
    return names


class NaturalMissingDiagnosticProbe:
    """Observe one diagnostic attempt using forward and tensor gradient hooks."""

    def __init__(self, *, attempt: int):
        self.attempt = int(attempt)
        self.input_stats: dict[str, Any] = {}
        self.nmf_tensors: list[dict[str, Any]] = []
        self.first_nonfinite_activation: dict[str, Any] | None = None
        self.first_nonfinite_gradient: dict[str, Any] | None = None
        self.nonfinite_activation_count = 0
        self.nonfinite_gradient_count = 0
        self._sequence = 0
        self._handles: list[Any] = []
        self._observer_restore: list[tuple[nn.Module, Any]] = []

    def _mark_nonfinite_activation(self, name: str, stats: Mapping[str, Any]) -> None:
        self.nonfinite_activation_count += 1
        if self.first_nonfinite_activation is None:
            self.first_nonfinite_activation = {
                "sequence": self._sequence,
                "tensor": name,
                "stats": dict(stats),
            }

    def _mark_nonfinite_gradient(self, name: str, stats: Mapping[str, Any]) -> None:
        self.nonfinite_gradient_count += 1
        if self.first_nonfinite_gradient is None:
            self.first_nonfinite_gradient = {
                "sequence": self._sequence,
                "tensor": name,
                "stats": dict(stats),
            }

    def record_inputs(self, *, images: torch.Tensor, depth: torch.Tensor, labels: torch.Tensor) -> None:
        for name, tensor in (("images", images), ("depth", depth), ("labels", labels)):
            stats = tensor_stats(tensor)
            self.input_stats[name] = stats
            self._sequence += 1
            if not stats["is_finite"]:
                self._mark_nonfinite_activation(f"input.{name}", stats)

    def observe_nmf(self, module_name: str, stage: str, tensor: torch.Tensor) -> None:
        name = f"{module_name}.{stage}" if module_name else stage
        record: dict[str, Any] = {
            "stage": name,
            "forward": tensor_stats(tensor),
            "gradient": None,
        }
        self.nmf_tensors.append(record)
        self._sequence += 1
        if not record["forward"]["is_finite"]:
            self._mark_nonfinite_activation(name, record["forward"])

        if tensor.requires_grad:
            def record_gradient(gradient: torch.Tensor, *, label: str = name, target: dict[str, Any] = record):
                stats = tensor_stats(gradient)
                target["gradient"] = stats
                self._sequence += 1
                if not stats["is_finite"]:
                    self._mark_nonfinite_gradient(label, stats)
                return gradient

            self._handles.append(tensor.register_hook(record_gradient))

    def _observe_module_input(self, module_name: str, inputs: Any) -> None:
        for path, tensor in _tensor_leaves(inputs):
            self._sequence += 1
            if not tensor_is_finite(tensor):
                name = f"module.{module_name or '<model>'}.input{path}"
                self._mark_nonfinite_activation(name, tensor_stats(tensor))

    def _observe_module_output(self, module_name: str, output: Any) -> None:
        for path, tensor in _tensor_leaves(output):
            name = f"module.{module_name or '<model>'}.output{path}"
            self._sequence += 1
            if not tensor_is_finite(tensor):
                self._mark_nonfinite_activation(name, tensor_stats(tensor))
            if tensor.requires_grad:
                def record_gradient(gradient: torch.Tensor, *, label: str = name):
                    self._sequence += 1
                    if not tensor_is_finite(gradient):
                        self._mark_nonfinite_gradient(label, tensor_stats(gradient))
                    return gradient

                self._handles.append(tensor.register_hook(record_gradient))

    def attach(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            self._handles.append(
                module.register_forward_pre_hook(
                    lambda current, inputs, module_name=name: self._observe_module_input(
                        module_name, inputs
                    )
                )
            )
            self._handles.append(
                module.register_forward_hook(
                    lambda current, _inputs, output, module_name=name: self._observe_module_output(
                        module_name, output
                    )
                )
            )
            if module.__class__.__name__ == "NMF2D" and hasattr(module, "diagnostic_observer"):
                previous = module.diagnostic_observer
                self._observer_restore.append((module, previous))
                module.diagnostic_observer = (
                    lambda stage, tensor, module_name=name: self.observe_nmf(module_name, stage, tensor)
                )

    def snapshot(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "inputs": self.input_stats,
            "nmf_tensors": self.nmf_tensors,
            "first_nonfinite_activation": self.first_nonfinite_activation,
            "first_nonfinite_gradient": self.first_nonfinite_gradient,
            "nonfinite_activation_count": self.nonfinite_activation_count,
            "nonfinite_gradient_count": self.nonfinite_gradient_count,
        }

    def close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        for module, observer in reversed(self._observer_restore):
            module.diagnostic_observer = observer
        self._observer_restore.clear()


def _tensor_leaves(value: Any, prefix: str = ""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _tensor_leaves(nested, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            yield from _tensor_leaves(nested, f"{prefix}[{index}]")
