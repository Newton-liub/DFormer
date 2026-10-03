#!/usr/bin/env bash
# Natural-only bounded numerical diagnosis; cannot launch Grid/Replay/S1.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "usage: $0 <python> <source-SHA> <remote-output-root> <original-C0>" >&2
    exit 2
fi
PYTHON="$1"
EXPECTED_SHA="$2"
ROOT="$3"
C0="$4"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
command -v screen >/dev/null
[[ -f "$C0" ]]
if screen -ls 2>/dev/null | grep -q '[.]natural-missing-numerical'; then
    echo 'Existing numerical screen session; refusing duplicate launch' >&2
    exit 2
fi
[[ ! -e "$ROOT/monitor/numerical-run-receipt.json" ]]
mkdir -p "$ROOT/monitor"
export SOURCE_COMMIT="$EXPECTED_SHA"
export DFORMER_DATA_ROOT=/root/rivermind-data/dataset
export MUSEG_DEPTH16_ROOT=/root/rivermind-data/dataset/MUSeg_DFormer/Depth16
export DFORMER_OUTPUT_ROOT="$ROOT/diagnosis-only"
export PYTHONUNBUFFERED=1
screen -D -m -L -Logfile "$ROOT/monitor/screen.log" \
    -S natural-missing-numerical \
    "$PYTHON" -u -m tools.mmfr.natural_missing_numerical \
    --c0-checkpoint "$C0" --output-dir "$ROOT/monitor" \
    --successful-updates 1664 --diagnostic-start-attempt 1536
# screen's status is not the training status; consult the durable receipt.
"$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("NUMERICAL_TERMINAL",r.get("status"),flush=True); sys.exit(int(r.get("child_exit_code",1)))' \
    "$ROOT/monitor/numerical-run-receipt.json"
