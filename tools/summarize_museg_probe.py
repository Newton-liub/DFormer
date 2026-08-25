#!/usr/bin/env python3
"""Convert one bounded MUSeg qualification probe log into structured JSON."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from tools.museg_protocol import write_json
except ModuleNotFoundError:
    from museg_protocol import write_json

_TELEMETRY = re.compile(
    r"Iter\s+(?P<step>\d+)/\d+:.*?loss=(?P<loss>[-+0-9.eE]+).*?"
    r"step=(?P<seconds>[-+0-9.eE]+)s\s+throughput=(?P<throughput>[-+0-9.eE]+).*?"
    r"allocated=(?P<allocated>[-+0-9.eE]+)\s+MiB\s+reserved=(?P<reserved>[-+0-9.eE]+)\s+MiB.*?"
    r"free=(?P<free>[-+0-9.eE]+)/(?P<total>[-+0-9.eE]+)\s+MiB.*?"
    r"free_ratio=(?P<ratio>[-+0-9.eE]+)\s+amp_scale=(?P<scale>[-+0-9.eE]+)"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()
    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    matches = list(_TELEMETRY.finditer(text))
    last = matches[-1].groupdict() if matches else None
    lower = text.lower()
    oom = "out of memory" in lower or "cuda error: out of memory" in lower
    anomaly = None
    if args.exit_code:
        anomaly = "oom" if oom else "non_oom_failure"
    payload = {
        "schema_version": "museg-4090-probe-result-v1",
        "batch_size": args.batch_size,
        "exit_code": args.exit_code,
        "completed_steps": int(last["step"]) if last else 0,
        "stable_throughput_images_per_second": float(last["throughput"]) if last else None,
        "step_time_seconds": float(last["seconds"]) if last else None,
        "allocated_mib": float(last["allocated"]) if last else None,
        "reserved_mib": float(last["reserved"]) if last else None,
        "free_mib": float(last["free"]) if last else None,
        "free_ratio": float(last["ratio"]) if last else None,
        "loss": float(last["loss"]) if last else None,
        "amp_scale": float(last["scale"]) if last else None,
        "oom": oom,
        "anomaly": anomaly,
    }
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())