# Figure regeneration — robust, from-scratch (2026-06-07)

Regenerates the paper's four kept figures from scratch with the **fixed engine** (Jun-5 symmetry
fix + Jun-6 perf optimization), robustly enough to survive the rare **native deadlock** that hung an
earlier single-process run. One command does everything; everything is saved so figures can be
re-plotted or re-averaged without re-simulating.

## What it produces
Four figures (the `quality` / no-service-utility figure is **retired**):

| figure | what changed vs paper |
|---|---|
| `order_of_events` (Fig 4) | **symmetry**: only firm-1 share ≤50% simulated; right half reflected with firm-swap (engine is exactly symmetric). No more mirror-color artifact. |
| `dedicated_demand` | unchanged (regenerated with fixed engine) |
| `supply_sensitivity` | unchanged (regenerated with fixed engine) |
| `dedicated_supply` | **N = total_supply = 1.5** (paper used 4.0), per Sébastien |

## Run it
```bash
./regenerate_all.sh                       # all four, then build figures
./regenerate_all.sh order_of_events       # one experiment
```
Outputs:
- `results/units/<exp>_seed<N>.jld2` — one file per (experiment, seed); **every raw equilibrium**.
- `results/<exp>_results.jld2` — merged per experiment (all seeds, `converged` flag).
- `figures/<exp>.pdf` — the figure, built from the merged results.

Re-plot only (no simulating): `julia --project=$PROJ build_figures.jl [exp ...]`.

## Why it's robust (the deadlock story)
A prior all-in-one-process run **deadlocked**: one worker wedged in a native library call
(`__psynch_cvwait`) at an extreme degenerate config — uninterruptible from inside Julia, so the whole
sweep hung at 0% CPU. You cannot kill a native wedge from within the process; you *can* `kill -9` it
from outside. So:

- **Isolation**: each (experiment, seed) runs as its own subprocess (`run_unit.jl`), single-threaded.
- **External supervision** (`supervise.jl`): keeps ≤ `REGEN_P` units running, enforces a per-unit
  wall-clock timeout, and **SIGKILLs** any that wedge — then draws a fresh replacement seed.
- **Replacement to target**: keeps drawing seeds until every config has `REGEN_TARGET` (32) converged
  seeds, up to `REGEN_SEEDCAP` (44) attempts. Configs short of target are reported, not silently dropped.
- **Thread pins**: `OPENBLAS_NUM_THREADS=1` etc. set in the environment before Julia starts (belt-and-
  suspenders over the in-engine `BLAS.set_num_threads(1)`), to avoid the native threading that deadlocks.
- **Atomic unit writes** (temp+rename) so a SIGKILL never leaves a half-written file mistaken for done.
- **Resumable**: existing unit files are tallied and skipped on restart.

## Tunables (env vars)
`REGEN_P` (concurrency, default 10), `REGEN_TIMEOUT` (sec/unit, 3600), `REGEN_TARGET` (32),
`REGEN_SEEDCAP` (44).

## Files
- `experiments.jl` — single source of truth: base params, the four config sweeps, config-key. The two
  changes (Fig 4 symmetry, dedicated_supply N=1.5) live here.
- `run_unit.jl` — one (experiment, seed) → one jld2.
- `supervise.jl` — the robust orchestrator (spawn / timeout-kill / replace / merge).
- `build_figures.jl` — merged results → the four PDFs (converged-only, capped at 32/config; Fig 4 reflect).
- `regenerate_all.sh` — top-level entry (env pins + supervise + build).

Engine + Julia env are referenced from the repo root (`..`), the single source of truth for
the model. The original `results_cap5_32seeds.jld2` and `figures/` there are left untouched.
