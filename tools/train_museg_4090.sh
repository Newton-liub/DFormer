#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${HOME}/rivermind-data/DFormer#"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/miniconda3/envs/py310/bin/python}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DFORMER_DATA_ROOT="${DFORMER_DATA_ROOT:-${HOME}/rivermind-data/dataset}"
export DFORMER_PRETRAINED="${DFORMER_PRETRAINED:-${HOME}/rivermind-data/pretrained/DFormerv2_Small_pretrained.pth}"
export DFORMER_OUTPUT_ROOT="${DFORMER_OUTPUT_ROOT:-${HOME}/rivermind-data/dformer_outputs}"

BATCH_SIZE="${DFORMER_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${DFORMER_VAL_BATCH_SIZE:-1}"
WORKERS="${DFORMER_WORKERS:-8}"
EPOCHS="${DFORMER_EPOCHS:-20}"
RUN_NAME_ARGS=()
if [[ -n "${SWANLAB_RUN_NAME:-}" ]]; then
    RUN_NAME_ARGS=(--swanlab-run-name "${SWANLAB_RUN_NAME}")
fi
MST_ARGS=(--no-mst)
if [[ "${DFORMER_MST:-0}" == "1" ]]; then
    MST_ARGS=(--mst)
fi

"${PYTHON_BIN}" tools/preflight_train.py --swanlab-mode online

"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc-per-node=1 utils/train.py \
    --config local_configs.MUSeg.DFormerv2_S_4090 \
    --gpus 1 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --val-batch-size "${VAL_BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --checkpoint-dir "${DFORMER_OUTPUT_ROOT}/museg_dformerv2_s_4090/checkpoint" \
    --log-interval "${DFORMER_LOG_INTERVAL:-20}" \
    --no-syncbn --no-compile --no-sliding --no-use_seed \
    --amp --val_amp "${MST_ARGS[@]}" \
    --swanlab-mode online \
    --swanlab-project "${SWANLAB_PROJECT:-DFormer-liu}" \
    --swanlab-workspace "${SWANLAB_WORKSPACE:-Newton_liub}" \
    "${RUN_NAME_ARGS[@]}"
