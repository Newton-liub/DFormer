#!/usr/bin/env bash
# Durable screen entry for the explicitly authorized, single formal Round-1.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "usage: $0 <python> <formal-run-SHA> <remote-output-root> <original-C0>" >&2
    exit 2
fi
PYTHON="$1"
EXPECTED_SHA="$2"
ROOT="$3"
C0="$4"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]]
command -v screen >/dev/null
[[ -f "$C0" ]]
# Refuse replacing an existing session or an earlier formal receipt.
if screen -ls 2>/dev/null | grep -q '[.]natural-missing-round1-formal'; then
    echo 'Existing formal screen session; refusing duplicate launch' >&2
    exit 2
fi
[[ ! -e "$ROOT/monitor/formal-run-receipt.json" ]]
mkdir -p "$ROOT/monitor"
export SOURCE_COMMIT="$EXPECTED_SHA"
export DFORMER_DATA_ROOT=/root/rivermind-data/dataset
export MUSEG_DEPTH16_ROOT=/root/rivermind-data/dataset/MUSeg_DFormer/Depth16
export DFORMER_OUTPUT_ROOT="$ROOT/checkpoints"
export PYTHONUNBUFFERED=1
# Capital -D -m keeps this job waiting for screen; SSH disconnects do not end it.
screen -D -m -L -Logfile "$ROOT/monitor/screen.log" \
    -S natural-missing-round1-formal \
    "$PYTHON" -u -m tools.mmfr.natural_missing_formal \
    --c0-checkpoint "$C0" --output-dir "$ROOT/monitor" \
    --dataset-root /root/rivermind-data/dataset/MUSeg_DFormer --tracking-mode online
# screen may not preserve the application's exit code: use the durable receipt.
"$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("FORMAL_TERMINAL",r.get("status"),flush=True); sys.exit(0 if r.get("status")=="SUCCEEDED" else 1)' \
    "$ROOT/monitor/formal-run-receipt.json"
