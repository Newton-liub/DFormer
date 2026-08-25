#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROTOCOL_MANIFEST="${1:-${DFORMER_PROTOCOL_MANIFEST:-}}"
if [[ -z "${PROTOCOL_MANIFEST}" ]]; then
    printf '%s\n' 'usage: train_museg_4090.sh <protocol-manifest.json> [seed ...]' >&2
    exit 2
fi
PROTOCOL_MANIFEST="$(cd -- "$(dirname -- "${PROTOCOL_MANIFEST}")" && pwd)/$(basename -- "${PROTOCOL_MANIFEST}")"
shift || true
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PREFLIGHT_REPORT="${DFORMER_PREFLIGHT_REPORT:-}"
PREFLIGHT_ARGS=(--protocol-manifest "${PROTOCOL_MANIFEST}")
if [[ -n "${PREFLIGHT_REPORT}" ]]; then
    PREFLIGHT_ARGS+=(--report "${PREFLIGHT_REPORT}")
fi
"${PYTHON_BIN}" tools/preflight_train.py "${PREFLIGHT_ARGS[@]}"

SEED_ARGS=()
if (( $# > 0 )); then
    SEED_ARGS=(--seeds "$@")
fi
"${PYTHON_BIN}" tools/run_museg_3seed.py \
    --protocol-manifest "${PROTOCOL_MANIFEST}" \
    --python "${PYTHON_BIN}" \
    "${SEED_ARGS[@]}"
