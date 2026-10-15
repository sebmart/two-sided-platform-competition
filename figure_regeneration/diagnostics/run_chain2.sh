#!/bin/bash
# Round 2: jointly refined grids (CLAUDE_CODE_TASK_2.md). Resumable.
cd "$(dirname "$0")/.."
export DIAG_P="${DIAG_P:-$(sysctl -n hw.perflevel0.physicalcpu)}"
step() { echo; echo "##### $(date) :: $*"; caffeinate -i ./run_diagnostics.sh "$@"; }
echo "chain2 start $(date) P=$DIAG_P"
for v in rate01 step05r step025r; do DIAG_SEEDS="$(seq 1 32)" step sensitivity $v; done
step summarize
echo "chain2 done $(date)"
