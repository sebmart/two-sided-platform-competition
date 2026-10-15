# Follow-up task: joint grid refinement for the price-step sensitivity check

This continues `CLAUDE_CODE_TASK.md` in the same folder. The first round is done and the results are in
`results/summary.txt` and `results/REPORT.md`; read Section 3.3 and item 3 of Section 4 of the report before starting.
The same constraints apply: engine, `experiments.jl`, harness and saved results are read-only; nothing under
`latex/paper/` or `minor revision/*.tex` is touched; no commits or pushes in the coauthor's repo; report numbers as they
come out.

## Why

The step-size variants (`step05`, `step025`) refined the price/pay grid while leaving the rate grid at its base value
(`rate_step` = 0.02 per hour). On the finer price grids a single pay step often cannot move the discretised supply at
all, so the pay stage sees a flat profit and the ±1 dynamics can drift (the small firm's pay falling to $32, below the
$40 reservation wage, at shares up to 20% under `step025` looks like exactly that). Before anything is written into the
paper we need to know whether the non-robustness in Section 3.3 survives when both grids are refined together.

## What to run

Add three variants to the `sensitivity` mode of `diagnostics.jl`, everything else as in `experiments.jl` (same
`configs_order_of_events()` sequential configurations, same `initial_guess` with `RANDOMNESS`, seeds 1–32, first-32-converged rule):

| variant | `price_step` | `rate_step` | purpose |
|---|---|---|---|
| `rate01` | 1 (base) | 0.01 | rate refinement alone: isolates the rate-grid effect |
| `step05r` | 0.5 | 0.01 | joint refinement, factor 2 |
| `step025r` | 0.25 | 0.005 | joint refinement, factor 4 |

Budgets: do not assume a scaling. Run one seed of each variant first at a generous budget (start from ×16 for `rate01`
and `step05r`, ×64 for `step025r`, on both `max_iter` and `max_simulations`), look at the queueing-solve counts and
wall time in the unit log, and set the budget so that it does not bind for the 32-seed run; keep the existing
bookkeeping that reports how many converged rows exceeded the smaller budgets. Check that `initial_guess` puts the
random starts on the refined grids (rates in units of the new `rate_step`); if it rounds to the base grid, say so in the
report and handle it the way the engine would (the start must be a valid grid state for the new parameters).

Then:

```bash
export DIAG_P=$(sysctl -n hw.perflevel0.physicalcpu)
for v in rate01 step05r step025r; do caffeinate -i ./run_diagnostics.sh sensitivity $v; done
./run_diagnostics.sh summarize
```

Expected cost: the earlier `step05` and `step025` runs took 4 and 3 minutes for 32 seeds at cap 5; the refined rate grid
makes the supply and demand stages take more moves, so expect minutes to an hour per variant, not the cap-10 scale. If a
variant looks like it will exceed two hours, stop and say so with an estimate.

## Optional, if time allows after the above

Regret on the jointly refined outcomes, to see whether the finer-grid outcomes are closer to or further from grid best
responses than the base ones: run the `regret` mode on the `step05r` sensitivity results (price deviations only is
enough), with the window and stride doubled in grid steps (±20, stride 10) so the sweep covers the same dollar range as
the base run. If `regret` currently reads only `results/<exp>_results.jld2`, add a way to point it at a sensitivity
variant's merged results. Report the same quantities as Section 3.1 of the report, side by side with the base numbers.

## Deliverables

1. `results/summary.txt` regenerated (the sensitivity table now has base, step05, step025, cap7, cap10, rate01, step05r,
   step025r at all ten shares).
2. A new Section 3.4 in `results/REPORT.md` with: the budgets used and solve counts; a table at all ten shares with base,
   `rate01`, `step05r`, `step025r` (mean and p25–p75 of pay and profit for both firms, as in Figure 4); and a plain
   verdict in one paragraph on each of three questions: (a) does refining the rate grid alone change the base figure?
   (b) does the jointly refined grid reproduce the base figure's qualitative pattern (the smaller firm pays less, the pay
   gap closes toward 50/50, and the profit ordering by share), and where does it not? (c) with the joint refinement, do
   the `step05`/`step025` anomalies (pay equalising at about $65 from 30% share up; the small firm's exit at $32 pay at
   low shares) go away, persist, or change? If the optional regret run was done, a short paragraph comparing residuals
   on the two grids.
3. The updated `diagnostics.jl` and `DIAGNOSTICS_README.md` copied over the versions in `minor revision/simulation_diagnostics/`.

Put the verdict paragraph at the very top of the new section so it can be read first.
