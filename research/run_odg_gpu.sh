#!/usr/bin/env bash
# Manual launch only. This script NEVER starts a cloud instance or schedules training.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DFORMER_DATASET_ROOT:?Set the parent directory containing SUNRGBD}"
: "${MICRO_BATCH:?Choose after GPU authorization; use the same value for both models}"
: "${ACCUM_STEPS:?MICRO_BATCH times ACCUM_STEPS must equal 16}"
: "${ODG_GPU_AUTHORIZED:?Set to 1 only after explicit user authorization}"
: "${ODG_PLATFORM_STOP_CONFIRMED:?Set to 1 only after reading back the platform stop scheduler}"
[[ "$ODG_GPU_AUTHORIZED" == 1 && "$ODG_PLATFORM_STOP_CONFIRMED" == 1 ]]
PYTHON="${PYTHON:-/usr/local/miniconda3/envs/py310/bin/python}"
common=(--config local_configs.research.ODG_SUNRGBD --gpus 1
        --micro-batch "$MICRO_BATCH" --accum-steps "$ACCUM_STEPS"
        --schedule-epochs 300 --warmup-epochs 10 --pad_SUNRGBD)
case "${1:-}" in
  smoke)
    # Real batches for each mode, no dev/test inference; no smoke resume.
    for mode in original odg; do
      DFORMER_EXPERIMENT_NAME="sun-odg-smoke-$mode" "$PYTHON" -m research.train_odg \
        "${common[@]}" --geometry-mode "$mode" --smoke-steps 200 --num-workers 4
    done
    ;;
  odg|baseline)
    mode=odg; [[ "$1" != baseline ]] || mode=original
    DFORMER_EXPERIMENT_NAME="sun-dev-$mode-seed12345" "$PYTHON" -m research.train_odg \
      "${common[@]}" --geometry-mode "$mode" --stop-after-epoch 30 --save-predictions 3
    ;;
  resume)
    : "${2:?Specify the experiment last.pth or stage-epoch checkpoint}"
    : "${GEOMETRY_MODE:?Set the checkpoint's original or odg mode}"
    "$PYTHON" -m research.train_odg "${common[@]}" --geometry-mode "$GEOMETRY_MODE" \
      --resume "$2" --stop-after-epoch 100 --save-predictions 3
    ;;
  *) echo "Usage: bash research/run_odg_gpu.sh {smoke|odg|baseline|resume CHECKPOINT}" >&2; exit 2 ;;
esac
