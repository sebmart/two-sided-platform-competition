#!/bin/bash
# One-command, robust regeneration of all four paper figures from scratch.
#   ./regenerate_all.sh                # all four experiments, then build figures
#   ./regenerate_all.sh order_of_events dedicated_supply   # subset
# Resumable: re-running picks up existing per-seed unit files in results/units/.
#
# Native thread pools are pinned to 1 BEFORE Julia starts (the in-Julia BLAS pin is belt-and-
# suspenders). Each (experiment, seed) runs as an isolated subprocess; the supervisor SIGKILLs any
# that wedge and replaces the seed. See README.md.
set -u
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1
export VECLIB_MAXIMUM_THREADS=1

# Tunables (override by exporting before calling):
export REGEN_P="${REGEN_P:-10}"            # concurrent units
export REGEN_TIMEOUT="${REGEN_TIMEOUT:-3600}"  # seconds per unit before SIGKILL
export REGEN_TARGET="${REGEN_TARGET:-32}"  # converged seeds wanted per config
export REGEN_SEEDCAP="${REGEN_SEEDCAP:-44}"    # max seeds attempted per experiment

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$HERE/../freeriding_simulations"
JULIA="$(command -v julia)"

echo "=== REGENERATE ALL :: $(date) ==="
echo "P=$REGEN_P timeout=${REGEN_TIMEOUT}s target=$REGEN_TARGET seedcap=$REGEN_SEEDCAP  exps=${*:-<all>}"

"$JULIA" --project="$PROJ" --startup-file=no "$HERE/supervise.jl" "$@" || { echo "SUPERVISE FAILED"; exit 1; }
echo "=== building figures :: $(date) ==="
"$JULIA" --project="$PROJ" --startup-file=no "$HERE/build_figures.jl" "$@" || { echo "BUILD FAILED"; exit 1; }
echo "=== REGENERATE ALL DONE :: $(date) ==="
