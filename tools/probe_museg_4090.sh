#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/miniconda3/envs/py310/bin/python}"
PROBE_OUTPUT_ROOT="${DFORMER_PROBE_OUTPUT_ROOT:-${REPO_ROOT}/outputs/museg_4090_probe}"
PROBE_STEPS="${DFORMER_PROBE_STEPS:-60}"
mkdir -p "${PROBE_OUTPUT_ROOT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"${PYTHON_BIN}" tools/preflight_train.py --swanlab-mode disabled

completed=()
for batch_size in 4 8 12 16; do
    batch_output="${PROBE_OUTPUT_ROOT}/batch-${batch_size}"
    log_path="${batch_output}/probe.log"
    mkdir -p "${batch_output}"
    printf '\n=== probing global batch %s for %s steps ===\n' "${batch_size}" "${PROBE_STEPS}"

    set +e
    DFORMER_OUTPUT_ROOT="${batch_output}" "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone --nproc-per-node=1 utils/train.py \
        --config local_configs.MUSeg.DFormerv2_S_4090 \
        --gpus 1 \
        --batch-size "${batch_size}" \
        --val-batch-size 1 \
        --workers "${DFORMER_WORKERS:-8}" \
        --max-train-iters "${PROBE_STEPS}" \
        --log-interval 10 \
        --min-free-vram-gib "${DFORMER_PROBE_MIN_FREE_GIB:-2}" \
        --min-free-vram-ratio "${DFORMER_PROBE_MIN_FREE_RATIO:-0.10}" \
        --checkpoint-dir "${batch_output}/checkpoints-disabled" \
        --no-syncbn --no-compile --no-mst --no-sliding --no-use_seed \
        --amp --val_amp --swanlab-mode disabled \
        2>&1 | tee "${log_path}"
    status=${PIPESTATUS[0]}
    set -e

    if (( status != 0 )); then
        printf '\nBatch %s failed with exit code %s; stopping larger probes.\n' "${batch_size}" "${status}" >&2
        printf 'Completed batches: %s\n' "${completed[*]:-none}" >&2
        printf 'Manual choice: select the largest completed batch only after checking stable loss, throughput, '\
'maximum memory, and at least 10%% free VRAM in each probe log.\n' >&2
        exit "${status}"
    fi

    completed+=("${batch_size}")
    printf '%s\n' "Batch ${batch_size} completed. Final sampled metrics:"
    grep -E 'throughput=.*reserved=.*free_ratio=.*amp_scale=' "${log_path}" | tail -n 1 || true
done

printf '\nAll probes completed: %s\n' "${completed[*]}"
printf '%s\n' 'Manual choice: compare median steady-state throughput after warm-up, peak reported memory, loss/AMP-scale stability, and retain at least 10% VRAM headroom. Do not select a batch automatically.'
