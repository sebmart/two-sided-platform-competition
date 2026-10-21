#!/bin/bash
# Task 4 step 1: cycle recovery on the four sensitivity variants used in Table 9, then summarize. Resumable.
cd "$(dirname "$0")/.."
export DIAG_P="${DIAG_P:-$(sysctl -n hw.perflevel0.physicalcpu)}"
export DIAG_TIMEOUT="${DIAG_TIMEOUT:-43200}"
step() { echo; echo "##### $(date) :: $*"; caffeinate -i ./run_diagnostics.sh "$@"; }
echo "chain4 start $(date) P=$DIAG_P timeout=$DIAG_TIMEOUT"
for v in cap7 step05r step025r cap10; do step cycles sens-$v; done
step summarize
echo "chain4 done $(date)"
