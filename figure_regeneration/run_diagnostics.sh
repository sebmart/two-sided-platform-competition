#!/bin/bash
# Parallel driver for diagnostics.jl (one (experiment|variant, seed) per Julia process, like regenerate_all.sh).
#   ./run_diagnostics.sh regret order_of_events        # best-response residuals at every reported equilibrium
#   ./run_diagnostics.sh cycles order_of_events        # terminal cycles of the price / pay / simultaneous stages
#   ./run_diagnostics.sh sensitivity step05            # step05 | step025 | cap7 | cap10  (order_of_events, sequential)
#   ./run_diagnostics.sh summarize                     # merge everything into diagnostics/summary.txt
# Tunables: DIAG_P (parallel processes, default 10), DIAG_SEEDS (default: for regret/cycles the seeds the figures
# actually use, for sensitivity "1 .. 44"), DIAG_TIMEOUT (seconds per unit before SIGKILL, default 14400),
# DIAG_WINDOW (10), DIAG_STRIDE (5), DIAG_PMAX_FACTOR (1.5), JULIA (path to a Julia 1.12 binary).
# Resumable: existing unit files in diagnostics/units/ are skipped. A unit that times out or fails leaves no unit
# file (the write is temp+rename), so relaunching retries it.
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 VECLIB_MAXIMUM_THREADS=1
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$HERE/../freeriding_simulations"
if [ -z "${JULIA:-}" ]; then
  if [ -x "$HOME/.juliaup/bin/julia" ]; then JULIA="$HOME/.juliaup/bin/julia"; else JULIA="$(command -v julia)"; fi
fi
export JULIA
P="${DIAG_P:-10}"
TIMEOUT="${DIAG_TIMEOUT:-14400}"
MODE="${1:?usage: run_diagnostics.sh <regret|cycles|sensitivity|summarize> [experiment|variant]}"
WHAT="${2:-}"
mkdir -p "$HERE/diagnostics/units" "$HERE/diagnostics/logs"
if [ "$MODE" = "summarize" ]; then
  exec "$JULIA" --project="$PROJ" --startup-file=no "$HERE/diagnostics.jl" summarize
fi
[ -n "$WHAT" ] || { echo "need an experiment (regret/cycles) or a variant (sensitivity)"; exit 1; }
if [ -n "${DIAG_SEEDS:-}" ]; then
  SEEDS="$DIAG_SEEDS"
elif [ "$MODE" = "sensitivity" ]; then
  SEEDS="$(seq 1 44)"
else
  SEEDS="$("$JULIA" --project="$PROJ" --startup-file=no "$HERE/diagnostics.jl" needed_seeds "$WHAT" | tail -1)"
fi
echo "=== diagnostics $MODE $WHAT :: P=$P timeout=${TIMEOUT}s julia=$("$JULIA" --version) :: seeds: $(echo $SEEDS | tr '\n' ' ') :: $(date) ==="
# one unit: run Julia under a wall-clock timeout (a native wedge can only be killed from outside), log to a file
run_unit() {
  julia="$1"; proj="$2"; here="$3"; mode="$4"; what="$5"; seed="$6"; timeout="$7"
  out="$here/diagnostics/units/${mode}_${what}_seed${seed}.jld2"
  [ -f "$out" ] && exit 0
  log="$here/diagnostics/logs/${mode}_${what}_seed${seed}.log"
  "$julia" --project="$proj" --startup-file=no "$here/diagnostics.jl" "$mode" "$what" "$seed" "$out" > "$log" 2>&1 &
  pid=$!
  ( sleep "$timeout"; kill -9 "$pid" 2>/dev/null && echo "TIMEOUT after ${timeout}s" >> "$log" ) &
  killer=$!
  wait "$pid"; rc=$?
  kill "$killer" 2>/dev/null; wait "$killer" 2>/dev/null
  if [ "$rc" -ne 0 ] || [ ! -f "$out" ]; then echo "FAILED $mode $what seed $seed (rc=$rc, see $log)"; else echo "done $mode $what seed $seed :: $(tail -1 "$log")"; fi
}
export -f run_unit
printf "%s\n" $SEEDS | xargs -P "$P" -I{} bash -c 'run_unit "$@"' _ "$JULIA" "$PROJ" "$HERE" "$MODE" "$WHAT" {} "$TIMEOUT"
echo "=== done $MODE $WHAT :: $(date) ==="
