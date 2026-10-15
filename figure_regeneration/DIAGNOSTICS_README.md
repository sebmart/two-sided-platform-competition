# Diagnostics for Referee 5's Comment 3 (numerical procedure)

Two files that live in `figure_regeneration/` next to `experiments.jl`:

- `diagnostics.jl` — one (experiment | variant, seed) per Julia process, same pattern as `run_unit.jl`.
- `run_diagnostics.sh` — parallel driver (`DIAG_P` processes, default 10), resumable, with a per-unit wall-clock
  timeout, plus a `summarize` step. `diagnostics/run_chain.sh` runs the whole set in a sensible order.

Everything uses the engine's own stage solvers (`simulation_utils_logit.jl` is not modified), so deviation payoffs are
computed exactly as the main solve computes continuation payoffs. The saved results in `results/` are read, never
written. Requires Julia 1.12 (the runner prefers `~/.juliaup/bin/julia`; override with `JULIA=...`).

## What each mode computes

All modes work on the *reported* outcomes only: converged rows, first `REGEN_TARGET` (32) seeds per configuration,
exactly the rows `build_figures.jl` plots. `./run_diagnostics.sh` derives the seed list from the results file
(`diagnostics.jl needed_seeds <experiment>`) unless `DIAG_SEEDS` is set.

**regret** (`./run_diagnostics.sh regret <experiment>`). At every reported equilibrium:
- sequential timing: for each firm, sweep unilateral *price* deviations with `sequentialNE_demand` re-solving
  demand → pay → supply underneath (the deviating firm's pay is left as reported and re-solved by the pay stage, exactly
  as the engine's price step does), and unilateral *pay* deviations (≤ price) with `sequentialNE_supply` re-solving
  supply underneath (demand held fixed, as in the pay stage);
- simultaneous timing: for each firm, sweep *joint* (price, pay) deviations with `simultaneousNE_supplydemand`
  underneath (a profitable joint move can exist when neither coordinate alone helps; pay ≤ price without loss);
- the deviation *to the reported decision itself* is also re-solved (`self_profit`). The lower stages are
  path-dependent cycle dynamics, so this generally does not reproduce the reported profit. Two regrets are recorded:
  `regret` = best deviation − reported profit, and `regret_self` = best deviation − self-continuation profit; both
  floored at 0, the raw gain is kept too.
The sweep covers ±`DIAG_WINDOW` (10) grid steps around the reported decision plus the whole grid at stride
`DIAG_STRIDE` (5); the price grid runs to `DIAG_PMAX_FACTOR` (1.5) × service value, because reported prices above the
service value do occur. Cost with the defaults: about 1 h per `order_of_events` seed (20 outcomes), dominated by the
price deviations (one demand → pay → supply solve each, 0.3–3 s) and the joint sweep (~2,000 supply+demand solves per
firm-outcome at ~0.03 s).

**cycles** (`./run_diagnostics.sh cycles <experiment>`). Restarts the engine's *price-stage* update map (copied from
`sequentialNE`) from the reported state and iterates until a (firms, workers, demand) state repeats; records the cycle
length, whether the reported state lies on the cycle, the time average over the cycle of profits, prices and pay, the
most-favorable state under the engine's rule (largest min profit, ties by max profit; should be the reported state),
the min/max over the cycle, and how many cycle states carry stale engine profits (`sequentialNE_demand` returns the
simulation of its last iterate, not of the demand state it selects). Same for the pay stage (map copied from
`sequentialNE_pay`, at the reported prices and demand) and, on the simultaneous configurations, for the simultaneous
firm map (copied from `simultaneousNE`: stay vs four diagonal moves). Cheap: ~20 s per `order_of_events` seed.

**sensitivity** (`./run_diagnostics.sh sensitivity <variant>`). Re-solves the ten sequential `order_of_events`
configurations with one or two parameters changed, same `initial_guess`, `RANDOMNESS`, `MAX_SIMULATIONS` and
`max_iter = 1000` as `experiments.jl` / `run_unit.jl`, with the iteration and solve budgets scaled per variant
(`DIAG_SENS_SCALE` overrides, for budget trials):

| variant | `price_step` | `rate_step` | `max_queue` | budget scale | cost per seed |
|---|---|---|---|---|---|
| `step05` | 0.5 | 0.02 | 5 | ×4 | seconds per configuration |
| `step025` | 0.25 | 0.02 | 5 | ×16 (×4 was binding) | seconds per configuration |
| `cap7` | 1 | 0.02 | 7 | ×1 | ~1 min per configuration |
| `cap10` | 1 | 0.02 | 10 | ×1 | ~15 min per configuration in parallel (76 CPU-hours for 32 seeds) |
| `rate01` | 1 | 0.01 | 5 | ×16 | ~8 min per seed |
| `step05r` | 0.5 | 0.01 | 5 | ×16 | ~3 min per seed |
| `step025r` | 0.25 | 0.005 | 5 | ×64 | ~4 min per seed |

The budgets for the round-2 variants (`rate01`, `step05r`, `step025r`) were set from one-seed trials so that they do not
bind (15×–35× headroom over the largest observed solve count). `initial_guess` scales its random starts with
`price_step` and `rate_step`, so the starts are valid states of the refined grids with the same dollar / hourly values.
The summary reports, per variant, how many converged rows needed more solves than the ×1 / ×4 / ×16 budgets.

`regret` and `cycles` can also be run on a sensitivity variant's outcomes: `./run_diagnostics.sh regret sens-step05r`
reads the merged `sensitivity_step05r_seed*.jld2` units instead of `results/<exp>_results.jld2` and applies the same
first-32-converged rule. `DIAG_REGRET_KINDS=price` restricts the sweep to price deviations; set `DIAG_WINDOW` /
`DIAG_STRIDE` in grid steps (e.g. 20 / 10 on a $0.5 grid to cover the same dollar range as 10 / 5 on the $1 grid).

**summarize** (`./run_diagnostics.sh summarize`). Merges `diagnostics/units/*.jld2`, applies the first-32-converged-
seeds rule, and writes `diagnostics/summary.txt`: regret quantiles (both definitions, absolute and relative) per
experiment / timing / decision kind and per configuration; cycle-length distributions, on-cycle / most-favorable
checks and the gap between time-average and reported-state summaries, overall and per configuration in Figure 4's
form; the sensitivity table (base vs each variant, mean and p25–p75 of pay and profit per firm, n used / attempted).
Merged DataFrames are saved alongside.

**Figures from time averages.** `build_figures_timeavg.jl` (next to the unchanged `build_figures.jl`) rebuilds the four
figures with each reported row's profit and pay replaced by the time average over its terminal cycle, read from
`diagnostics/cycles_<exp>.jld2` (so run `cycles <exp>` and `summarize` first). It writes `figures_timeavg/<exp>.pdf`,
`figures_timeavg/compare_<exp>.pdf` (most favorable state dashed vs time average solid) and `figures_timeavg/CHANGES.txt`
(largest change per plotted series). The repo's `figures/` is not touched.

## Running

```
./run_diagnostics.sh cycles order_of_events          # ~2 min with 8 processes
./run_diagnostics.sh sensitivity step05 ; ./run_diagnostics.sh sensitivity step025
./run_diagnostics.sh sensitivity cap7   ; ./run_diagnostics.sh sensitivity cap10
./run_diagnostics.sh regret order_of_events          # ~4 h with 8 processes
./run_diagnostics.sh regret supply_sensitivity       # then the other two experiments if time allows
./run_diagnostics.sh summarize
```
or `nohup diagnostics/run_chain.sh > diagnostics/logs/chain.log &` for all of it. Tunables: `DIAG_P`, `DIAG_SEEDS`,
`DIAG_TIMEOUT` (s per unit, default 14400), `DIAG_WINDOW`, `DIAG_STRIDE`, `DIAG_PMAX_FACTOR`, `JULIA`. Logs in
`diagnostics/logs/`, unit outputs in `diagnostics/units/<mode>_<what>_seed<k>.jld2` (temp + rename; a killed unit
leaves no file and is retried on relaunch).

## Things worth knowing about the engine (no action needed)

- At the pay and price stages each firm compares one step down with one step up and always moves to the better of the
  two; it never compares with staying put. So these stages end only when a state repeats, typically in a two-cycle that
  brackets the best response, and the "converged" flag means "finished within budget", not "reached a fixed point".
  The stage order is price → demand → pay → supply.
- `sequentialNE_demand` returns the simulation of its last iterate together with the decisions of the demand state it
  selects from the terminal cycle (the one with the largest total demand); when those differ, the profits the price
  stage records for a state are not the profits of that state. The cycles mode reports how often this happens.
- The lower-stage solves are not fixed-point maps: re-solving them from a reported state at the reported decisions
  usually moves demand / supply by one grid step. The regret mode reports this as the self-vs-reported gap.
