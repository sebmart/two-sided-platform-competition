# Task 4: time averages over the terminal cycle for the sensitivity variants (Table 9)

Continues `CLAUDE_CODE_TASK.md`, `_2.md`, `_3.md` (same folder; results in `results/`). Same constraints: engine,
`experiments.jl`, harness, `results/` and the repo's `figures/` are read-only; no commits or pushes in the coauthor's
repo; nothing under `latex/paper/`; numbers reported as they come out.

## Why

After Task 3 the paper's figures summarize each terminal price-stage cycle by its time average. The sensitivity table in
the paper (appendix F.4, "Table 9": base vs `cap7`, `cap10`, `step05r` ("grid /2"), `step025r` ("grid /4")) was built
from the `sensitivity` section of `summary.txt`, which still summarizes every row by the engine's reported state (most
favorable state), because the variants were run before the switch. The table must use the same summary as the figures.
The engine's reported state and the cycle recovery already exist for the variants' outcomes (`cycles sens-<variant>`
reads `diagnostics/units/sensitivity_<variant>_seed*.jld2`), so this is cycle recovery plus a summary change; no re-solve.

## Step 1: cycle recovery on the four variants used in the table

```bash
cd ~/Documents/Claude/Projects/Free-riding\ revision/code/two-sided-platform-competition/figure_regeneration
export DIAG_P=$(sysctl -n hw.perflevel0.physicalcpu)
for v in cap7 step05r step025r cap10; do caffeinate -i ./run_diagnostics.sh cycles sens-$v; done
./run_diagnostics.sh summarize
```

Before launching, check in `cycles_unit` / `price_step` that the price-stage update map takes the grid steps and the
queue cap from each row's own `params` (the variant's `price_step`, `rate_step`, `max_queue`), not from `BASE_P` or
from the base experiment's parameters; if it does not, fix that first (the variants' rows carry their own `params`).
`cap10` is the slow one (each price-stage iteration re-solves demand, pay and supply with a queue cap of 10); set
`DIAG_TIMEOUT` generously and let it run in the background. Only the sequential `order_of_events` configurations exist
in these variants. Confirm, as in Tasks 1 and 3, that the reported state lies on the recovered cycle for every reported
row of every variant, and report any exception. The 6 non-converged `cap10` rows at the 5% share are not reported rows
and need no cycle.

## Step 2: time-average sensitivity table in `summarize`

Extend the sensitivity section of `summarize` so that it prints a second table with the same rows and columns
(firm1_demand | variant | n/attempted | pay1 | pay2 | profit1 | profit2; mean [p25, p75] over the first 32 converged
seeds), in which every row, **including `base`**, summarizes each outcome by the time average over its terminal
price-stage cycle: profit = time average of each firm's profit over the cycle states (recomputed from each state's
decisions with `simulation`, as the cycles mode does), pay = time average of the firm's pay over the cycle, and a firm's
pay enters the average over seeds only when its time-averaged profit is positive (the rule of
`build_figures_timeavg.jl`). Read the cycles from `diagnostics/cycles_order_of_events.jld2` for `base` and
`diagnostics/cycles_sens-<variant>.jld2` for the variants, matched to rows by (params, sequential, seed). If a reported
row has no cycle record, stop and say which, rather than falling back to the reported state. Keep the existing
reported-state table, clearly labeled, so the two can be compared, and print under the new table, per variant, the
largest absolute difference between the two summaries in any cell (pay and profit separately).

The base row of the new table must reproduce the per-configuration time-average means already printed in the
`cycles: order_of_events` section of `summary.txt` (the "figure summary ... vs time-average summary" block); say so, or
say why not.

## Deliverables

1. `results/summary.txt` regenerated: the four new cycle sections (`cycles: sens-cap7`, ...) and the two sensitivity
   tables (reported state, time average), with the largest-difference lines.
2. A Section 3.6 in `results/REPORT.md`: cycle-length distribution and on-cycle check per variant; wall times; the new
   time-average table reproduced verbatim; the largest cell differences; and one sentence per variant saying whether
   the qualitative statements of appendix F.4 (smaller firm pays less at every share below 50%; pay levels at shares
   of 15% and above within about $2 of the base case, with the largest difference stated; pay and profit at the 5% and
   10% shares more sensitive; cap 10 lowering the smaller firm's profit by $1–5 per hour between 10% and 35%) still
   hold under the time-average summary, with the numbers that would replace them if not.
3. The modified `diagnostics.jl` copied into `minor revision/simulation_diagnostics/` (keep the previous copy as
   `diagnostics_task3.jl`).
