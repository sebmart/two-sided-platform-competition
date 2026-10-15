#!/bin/bash
# Full diagnostic chain for Referee 5 Comment 3(a): cheap modes first, then the long sweeps. Resumable (rerun = resume).
cd "$(dirname "$0")/.."
export DIAG_P="${DIAG_P:-$(sysctl -n hw.perflevel0.physicalcpu)}"
export DIAG_SEEDS_SENS="${DIAG_SEEDS_SENS:-$(seq 1 32)}"
step() { echo; echo "##### $(date) :: $*"; caffeinate -i ./run_diagnostics.sh "$@"; }
echo "chain start $(date) P=$DIAG_P"
step cycles order_of_events
DIAG_SEEDS="$DIAG_SEEDS_SENS" step sensitivity step05
DIAG_SEEDS="$DIAG_SEEDS_SENS" step sensitivity step025
DIAG_SEEDS="$DIAG_SEEDS_SENS" step sensitivity cap7
step summarize
step regret order_of_events
step summarize
DIAG_SEEDS="$DIAG_SEEDS_SENS" step sensitivity cap10
step summarize
step cycles supply_sensitivity
step regret supply_sensitivity
step summarize
echo "chain done $(date)"
