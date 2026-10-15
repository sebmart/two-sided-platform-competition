# Task: run the simulation diagnostics for Referee 5's Comment 3(a)

You are working on the revision of *Labor Cost Free-Riding in the Gig Economy* (Management Science, MS-RMA-2023-02138.R2). Referee 5 objected to how the paper describes its numerical procedure (Section 5 and the simulation appendix) and asked for three things: (1) best-response residuals at the reported outcomes, or a relabeling of what the procedure returns; (2) convergence-versus-cycling statistics, with cycles summarized by a neutral statistic (time average or full range) instead of the "most favorable state"; (3) sensitivity to the price step and to the queue cap. The relabeling and the text rewrite are already done. **Your job is to produce the numbers**, by running three diagnostics on top of the coauthor's Julia simulation code, and to hand back a summary file plus a short report. You do not edit the paper or the response letter.

## Where things are

- Project folder: `~/Documents/Claude/Projects/Free-riding revision/`
- The three diagnostic files, drafted but **never executed** (written without a Julia installation): `minor revision/simulation_diagnostics/diagnostics.jl`, `run_diagnostics.sh`, `DIAGNOSTICS_README.md`. Read the README first; it explains each mode.
- Simulation code: public repo `https://github.com/sebmart/two-sided-platform-competition` (coauthor Sébastien Martin). Clone it into a new folder `code/` inside the project folder, i.e. `~/Documents/Claude/Projects/Free-riding revision/code/two-sided-platform-competition/`. Its README maps paper parameters to code fields; `figure_regeneration/README.md` explains the harness.
- Requires Julia 1.12 (the `Manifest.toml` is resolved for 1.12.6). `~/.julia` exists on this Mac, so check `juliaup status` / `julia --version` before installing; if Julia 1.12 is not there, install via `curl -fsSL https://install.julialang.org | sh` then `juliaup add 1.12` and `juliaup default 1.12`.

## Hard constraints

1. Do not touch `latex/paper/` (Overleaf-synced) and do not edit anything under `minor revision/` except to add files inside `minor revision/simulation_diagnostics/`. No git commands in the paper's repository.
2. In the cloned code repo: do not modify the engine (`simulation_utils_logit.jl`), `experiments.jl`, `run_unit.jl`, `supervise.jl`, `build_figures.jl`, or anything under `figure_regeneration/results/` or `figure_regeneration/figures/`. Do not commit or push to the coauthor's repo. The only files you edit there are the three diagnostic files you copied in (and new files under `figure_regeneration/diagnostics/`).
3. Do not regenerate the paper's figures (`regenerate_all.sh`). The saved 32-seed results in `figure_regeneration/results/<experiment>_results.jld2` are the inputs; two of the three diagnostics read them and must not rewrite them.
4. If a diagnostic produces an uncomfortable number (large regrets, long cycles, results that move under a finer step), report it as is. Nothing is to be tuned or filtered to look better.
5. Run long jobs under `caffeinate -i` and with `nohup`/a log file so they survive a closed terminal; set `DIAG_P` to the number of performance cores (`sysctl -n hw.perflevel0.physicalcpu`).

## Setup

```bash
cd ~/Documents/Claude/Projects/Free-riding\ revision
mkdir -p code && cd code
git clone https://github.com/sebmart/two-sided-platform-competition
cd two-sided-platform-competition/figure_regeneration
julia --project=../freeriding_simulations -e 'using Pkg; Pkg.instantiate()'
cp "../../../minor revision/simulation_diagnostics/"{diagnostics.jl,run_diagnostics.sh,DIAGNOSTICS_README.md} .
chmod +x run_diagnostics.sh
```

## Step 1: make the scripts actually work (expect to fix things)

`diagnostics.jl` was written by reading the engine, not by running it. Run one seed of each mode first and fix whatever fails:

```bash
DIAG_P=1 DIAG_SEEDS=1 ./run_diagnostics.sh cycles order_of_events
DIAG_P=1 DIAG_SEEDS=1 ./run_diagnostics.sh regret order_of_events
DIAG_P=1 DIAG_SEEDS=1 ./run_diagnostics.sh sensitivity cap7
./run_diagnostics.sh summarize
```

Logs go to `diagnostics/logs/`, unit outputs to `diagnostics/units/<mode>_<what>_seed<k>.jld2`. When fixing, preserve the *intent* of each mode, stated below, rather than the draft's implementation details.

**What the engine does** (verified by reading the code; use this to check the draft). `sequentialNE` nests price → demand → pay → supply: the price stage calls `sequentialNE_demand`, which re-solves demand with `sequentialNE_pay` (pay re-solving supply via `sequentialNE_supply`) underneath. At the pay and price stages, each firm compares its profit one grid step below and one step above its current decision and moves to the better one every iteration; it never compares with staying put, so a stage ends only when a state repeats (a cycle, typically a two-cycle bracketing the best response) and the engine returns the "most favorable" state on that cycle (largest min-profit across firms, ties by max profit; demand/supply stages: largest total rate). Decisions live on an integer grid: prices and pay in units of `price_step` ($1), rates in units of `rate_step` (0.02/hr). `simultaneousNE` compares staying put against four diagonal (price, pay) moves, with `simultaneousNE_supplydemand` underneath. `max_queue` (5) caps the three queues; the stationary distribution is exact for the capped chain. Exported API: `ProblemParameters, FirmDecisions, WorkerDecisions, DemandDecisions, initial_guess, simulation, result_metrics, sequentialNE, sequentialNE_demand, sequentialNE_pay, sequentialNE_supply, simultaneousNE, simultaneousNE_supplydemand`.

**Results schema.** `results/<exp>_results.jld2` holds one `DataFrame` (via `JLD2.load_object`) with columns `seed, sequential, params, firms, workers, demand, iters, converged`, one row per (configuration, seed). `build_figures.jl` keeps, per configuration, the converged rows sorted by seed and truncated to the first 32 (`get_results`). Replicate that selection exactly in `summarize`.

**Mode intents.**

- `regret`: at every converged reported row, for each firm, (a) sequential timing: sweep unilateral *price* deviations on the grid with demand, pay and supply re-solved underneath (`sequentialNE_demand` from the reported state with the deviating firm's price changed), and unilateral *pay* deviations with supply re-solved underneath and demand held fixed (`sequentialNE_supply`), exactly the continuations the main solve uses; (b) simultaneous timing: sweep *joint* (price, pay) deviations with `simultaneousNE_supplydemand` underneath. Record base profit, best deviation profit, the regret (best minus base, floored at 0), and the arg-best deviation. Sweep window: ±`DIAG_WINDOW` (10) steps around the reported decision plus the whole grid at stride `DIAG_STRIDE` (5); prices above `service_value` get no demand and need not be swept. Sanity check: the deviation equal to the reported decision must reproduce the base profit (regret contribution 0). If a continuation solve fails or hits the budget, count it in `n_failed` and keep going.
- `cycles` (sequential configurations only): restart the engine's *price-stage* update map from the reported state and iterate until a state repeats; the reported state must lie on the recovered cycle (if it doesn't, the map in the script does not match the engine's; fix the map by copying the engine's stage logic, do not change the engine). Record cycle length, the time average over the cycle of each firm's profit, price and pay, the most-favorable state per the engine's rule (which should equal the reported state), and the min/max over the cycle. Do the same for the pay stage at the reported prices and demand.
- `sensitivity`: re-solve the ten sequential `order_of_events` configurations (`configs_order_of_events()`, `sequential == true`) for each seed with `price_step` 0.5 (`step05`) or 0.25 (`step025`), or `max_queue` 7 (`cap7`) or 10 (`cap10`), everything else as in `experiments.jl` (same `initial_guess` with `RANDOMNESS`, same `MAX_SIMULATIONS` and `max_iter=1000`, scaled ×2 / ×4 for the finer grids so the budget is not binding by construction). Record the same columns as `run_unit.jl` plus the variant name.
- `summarize`: merge `diagnostics/units/*.jld2`; apply the first-32-converged-seeds rule; write `diagnostics/summary.txt` with (i) regret quantiles (50/90/max, absolute, and relative where base profit > $1/hr) per experiment × timing × decision kind, and the share of outcomes with regret exactly 0; (ii) cycle-length distribution per stage, and the gap between the time-average and the most-favorable-state summaries of pay and profit (per configuration and overall); (iii) the sensitivity table: for each demand share, base vs each variant, mean pay and profit of each firm, with the 25–75 percentile band, in the same form as Figure 4's data. Save the merged DataFrames next to it.

## Step 2: full runs (after Step 1 passes)

```bash
export DIAG_P=$(sysctl -n hw.perflevel0.physicalcpu)
caffeinate -i ./run_diagnostics.sh cycles order_of_events          # minutes
caffeinate -i ./run_diagnostics.sh regret order_of_events          # ~30 min at 10 processes
for v in step05 step025 cap7 cap10; do caffeinate -i ./run_diagnostics.sh sensitivity $v; done   # the slow part
caffeinate -i ./run_diagnostics.sh regret supply_sensitivity       # then dedicated_demand, dedicated_supply if time allows
./run_diagnostics.sh summarize
```

Seeds default to 1–44 (the harness drew replacement seeds up to 44); `summarize` keeps 32 per configuration. The runner is resumable (existing unit files are skipped), so an interrupted run can simply be relaunched.

## Deliverables

Copy into `minor revision/simulation_diagnostics/results/`:

1. `summary.txt` from `diagnostics/`.
2. `REPORT.md`: what you changed in the three scripts and why (a diff summary is fine); the exact commands run, Julia version, wall times; the headline numbers for each of the referee's three bullets; anything surprising or any configuration that failed; and a note on whether the time-average summaries would visibly change Figure 4 relative to the most-favorable-state summaries.
3. The fixed `diagnostics.jl` and `run_diagnostics.sh` (overwrite the drafts in `minor revision/simulation_diagnostics/`).

Leave the raw `diagnostics/units/*.jld2` and merged DataFrames in the code folder; do not copy them into `minor revision/`.

## If you get stuck

Ask Zhen rather than guessing when: the engine's stage logic is ambiguous for the cycle map; a mode would take more than a few hours even at full parallelism (say so with an estimate and ask whether to reduce seeds or drop the `step025` variant); or `Pkg.instantiate()` fails for the plotting packages (the diagnostics do not plot; a project without `Plots`, `PGFPlotsX` and `Weave` is acceptable if the engine loads without them).
