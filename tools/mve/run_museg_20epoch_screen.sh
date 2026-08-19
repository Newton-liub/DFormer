#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-museg20}"
REPO_ROOT="/root/DFormer"
OUTPUT_ROOT="/root/rivermind-data/mve_outputs/museg_20epoch"
PYTHON_BIN="/usr/local/miniconda3/envs/py310/bin/python"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SCREEN_LOG="${OUTPUT_ROOT}/screen-train.log"
EXIT_CODE_PATH="${OUTPUT_ROOT}/train.exit_code"

run_training() {
    mkdir -p "${OUTPUT_ROOT}"
    rm -f "${EXIT_CODE_PATH}"
    cd "${REPO_ROOT}"

    set +e
    CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 "${PYTHON_BIN}" utils/train.py \
        --config local_configs.MUSeg.DFormerv2_S_20Epoch \
        --gpus 1 \
        --no-syncbn \
        --no-sliding \
        --no-compile \
        --no-mst \
        --amp \
        --val_amp \
        --no-pad_SUNRGBD \
        --use_seed \
        2>&1 | tee -a "${SCREEN_LOG}"
    status=${PIPESTATUS[0]}
    printf '%s\n' "${status}" > "${EXIT_CODE_PATH}"
    exit "${status}"
}

if [[ "${1:-}" == "--inside-screen" ]]; then
    run_training
fi

if [[ ! "${SESSION_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf 'invalid screen session name: %s\n' "${SESSION_NAME}" >&2
    exit 2
fi
if ! command -v screen >/dev/null 2>&1; then
    printf 'screen is not installed\n' >&2
    exit 3
fi
if screen -ls | grep -F ".${SESSION_NAME}" >/dev/null 2>&1; then
    printf 'screen session already exists: %s\n' "${SESSION_NAME}" >&2
    exit 4
fi

mkdir -p "${OUTPUT_ROOT}"
rm -f "${EXIT_CODE_PATH}"
screen -DmS "${SESSION_NAME}" bash "${SCRIPT_PATH}" --inside-screen
sleep 1
if ! screen -ls | grep -F ".${SESSION_NAME}" >/dev/null 2>&1; then
    printf 'screen session failed to start: %s\n' "${SESSION_NAME}" >&2
    exit 5
fi
printf 'started screen session %s\nlog: %s\nexit code: %s\n' "${SESSION_NAME}" "${SCREEN_LOG}" "${EXIT_CODE_PATH}"