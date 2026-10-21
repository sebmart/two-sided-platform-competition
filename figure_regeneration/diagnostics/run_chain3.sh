#!/bin/bash
# Task 3 step 1: cycle recovery for the two remaining experiments, then summarize. Resumable.
cd "$(dirname "$0")/.."
export DIAG_P="${DIAG_P:-$(sysctl -n hw.perflevel0.physicalcpu)}"
step() { echo; echo "##### $(date) :: $*"; caffeinate -i ./run_diagnostics.sh "$@"; }
echo "chain3 start $(date) P=$DIAG_P"
step cycles dedicated_demand
step cycles dedicated_supply
step summarize
echo "chain3 done $(date)"
