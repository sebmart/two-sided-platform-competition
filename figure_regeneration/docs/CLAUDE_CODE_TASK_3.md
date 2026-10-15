# Task 3: re-plot the four simulation figures from time averages over the terminal cycle

Continues `CLAUDE_CODE_TASK.md` and `CLAUDE_CODE_TASK_2.md` (same folder; results in `results/`). Same constraints:
engine, `experiments.jl`, harness, `results/` and the existing `figures/` in the code repo are read-only; no commits or
pushes in the coauthor's repo; nothing under `latex/paper/`; numbers reported as they come out. The optional `step05r`
regret paragraph from Task 2 can finish in the background; it is not needed here.

## Why

Referee 5 asked that cycles be summarized by a neutral statistic rather than the most favorable state. The cycle
diagnostics show the two summaries agree to within about $1, so the authors have decided to re-plot all four figures from
the time average over each terminal cycle. The paper text will say exactly that, so the figures must be produced that way.

## Step 1: cycle recovery for the two remaining experiments

```bash
cd ~/Documents/Claude/Projects/Free-riding\ revision/code/two-sided-platform-competition/figure_regeneration
export DIAG_P=$(sysctl -n hw.perflevel0.physicalcpu)
caffeinate -i ./run_diagnostics.sh cycles dedicated_demand
caffeinate -i ./run_diagnostics.sh cycles dedicated_supply
./run_diagnostics.sh summarize
```

Both experiments are sequential only. Some of their configurations have an inactive firm (no demand pool or no
workers); the cycle map must treat it the way the engine does (it never updates). Confirm, as before, that the reported
state lies on the recovered cycle for every reported row, and report any exception.

## Step 2: `build_figures_timeavg.jl`

Write a new script next to `build_figures.jl` (do not modify the original) that reproduces its four figures exactly in
layout, fonts, colors, axes, the Figure-4 symmetry reflection, the pay-reported-only-when-profit-positive convention and
the first-32-converged-seeds rule, with one change: each reported row's profit and pay for each firm are the **time
averages over the row's terminal cycle**, taken from `diagnostics/cycles_<exp>.jld2` (the price-stage cycle for
sequential rows, the simultaneous firm map's cycle for simultaneous rows; profits recomputed from each cycle state's
decisions with `simulation`, as the cycle mode already does), instead of the metrics of the single reported state. Match
cycle records to result rows by (params, sequential, seed). If a reported row has no cycle record, stop and say so
rather than falling back silently. The convention is then: a firm's pay enters the average over seeds when its
time-averaged profit is positive.

Two cosmetic changes requested by the authors, if the strings live in the plotting script: panel titles or legend entries
that say "Sequential Equilibrium" / "Simultaneous Equilibrium" become "Sequential timing" / "Simultaneous timing", and
any title containing "equilibrium" for the other three figures drops the word. Change nothing else visually.

Write the output to `figures_timeavg/<exp>.pdf` (new folder). Also write, for each experiment, a comparison PDF
`figures_timeavg/compare_<exp>.pdf` that overlays the current figure's series (most favorable state) and the new ones
(time average) in the same panels, dashed vs solid, so the authors can eyeball the difference, and print to the console
and to `figures_timeavg/CHANGES.txt` the largest absolute change in any plotted mean and any band edge, per experiment
and series.

## Step 3: put the new figures where the paper reads them

The paper's copies are in `~/Documents/Claude/Projects/Free-riding revision/minor revision/Figures/simulations/`
(`order_of_events.pdf`, `supply_sensitivity.pdf`, `dedicated_demand.pdf`, `dedicated_supply.pdf`; ignore `quality.pdf`).
First copy those four current files into a new subfolder `Figures/simulations/most_favorable_state/` (keep the originals;
nothing is deleted in this project), then copy the four new PDFs over the originals' paths. Copy the four `compare_*.pdf`
and `CHANGES.txt` into `minor revision/simulation_diagnostics/results/figures_timeavg/`.

## Deliverables

1. The four re-plotted figures in place (Step 3), originals preserved in `most_favorable_state/`.
2. `results/summary.txt` regenerated with the two new cycle sections.
3. A Section 3.5 in `results/REPORT.md`: cycle-length distribution and on-cycle check for the two new experiments, the
   largest changes per figure from `CHANGES.txt`, and one sentence per figure saying whether anything visible changed.
4. `build_figures_timeavg.jl` copied into `minor revision/simulation_diagnostics/`.
