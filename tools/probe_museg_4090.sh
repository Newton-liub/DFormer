#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROTOCOL_MANIFEST="${1:-${DFORMER_PROTOCOL_MANIFEST:-}}"
if [[ -z "${PROTOCOL_MANIFEST}" ]]; then
    printf '%s\n' 'usage: probe_museg_4090.sh <qualification-protocol-manifest.json>' >&2
    exit 2
fi
PROTOCOL_MANIFEST="$(cd -- "$(dirname -- "${PROTOCOL_MANIFEST}")" && pwd)/$(basename -- "${PROTOCOL_MANIFEST}")"
PROBE_OUTPUT_ROOT="${DFORMER_PROBE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/museg_4090_probe}"
PROBE_STEPS="${DFORMER_PROBE_STEPS:-60}"
PROBE_SEED="${DFORMER_PROBE_SEED:-}"
mkdir -p "${PROBE_OUTPUT_ROOT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -z "${PROBE_SEED}" ]]; then
    PROBE_SEED="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["seeds"][0])' "${PROTOCOL_MANIFEST}")"
fi

"${PYTHON_BIN}" tools/preflight_train.py \
    --protocol-manifest "${PROTOCOL_MANIFEST}" \
    --report "${PROBE_OUTPUT_ROOT}/preflight.json"

completed=()
overall_status=0
for batch_size in 4 8 12 16; do
    batch_output="${PROBE_OUTPUT_ROOT}/batch-${batch_size}"
    if [[ -e "${batch_output}" ]]; then
        printf 'probe output already exists: %s\n' "${batch_output}" >&2
        exit 2
    fi
    set +e
    "${PYTHON_BIN}" tools/run_museg_seed.py \
        --protocol-manifest "${PROTOCOL_MANIFEST}" \
        --seed "${PROBE_SEED}" \
        --batch-size "${batch_size}" \
        --max-train-iters "${PROBE_STEPS}" \
        --min-free-vram-gib "${DFORMER_PROBE_MIN_FREE_GIB:-2}" \
        --min-free-vram-ratio "${DFORMER_PROBE_MIN_FREE_RATIO:-0.10}" \
        --output-dir "${batch_output}"
    status=$?
    set -e

    "${PYTHON_BIN}" tools/summarize_museg_probe.py \
        --log "${batch_output}/launcher.log" \
        --output "${batch_output}/probe-result.json" \
        --batch-size "${batch_size}" \
        --exit-code "${status}"

    if (( status != 0 )); then
        anomaly="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("anomaly") or "unknown")' "${batch_output}/probe-result.json")"
        overall_status="${status}"
        if [[ "${anomaly}" == "oom" ]]; then
            printf 'batch %s reached OOM; larger probes were not started\n' "${batch_size}" >&2
        else
            printf 'batch %s failed with %s; all probing stopped\n' "${batch_size}" "${anomaly}" >&2
        fi
        break
    fi
    completed+=("${batch_size}")
done

"${PYTHON_BIN}" - "${PROBE_OUTPUT_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("batch-*/probe-result.json"))]
safe = [
    row for row in rows
    if row["exit_code"] == 0
    and row["free_mib"] is not None
    and row["free_mib"] >= 2048
    and row["free_ratio"] is not None
    and row["free_ratio"] >= 0.10
    and row["stable_throughput_images_per_second"] is not None
]
best = max(safe, key=lambda row: row["stable_throughput_images_per_second"]) if safe else None
summary = {
    "schema_version": "museg-4090-probe-summary-v1",
    "runs": rows,
    "recommended_batch_by_stable_throughput": best["batch_size"] if best else None,
    "requires_user_confirmation": True,
    "all_requested_batches_completed": len(rows) == 4 and all(row["exit_code"] == 0 for row in rows),
}
(root / "probe-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if (( overall_status != 0 )); then
    exit "${overall_status}"
fi
printf '%s\n' 'Probe records are complete. Stop here until the user confirms the batch size.'
