# Simulation diagnostics for Referee 5, Comment 3(a): run report

Run on 2026-10-06/07 (round 1) and 2026-10-07 (round 2, Section 3.4; round 3, Section 3.5) on Zhen's Mac (Apple Silicon, 8 performance cores, macOS 26.7.1) with Julia 1.12.7
(installed through juliaup; the project `Manifest.toml` resolved and precompiled without changes, plotting packages
included). Engine: `simulation_utils_logit.jl` at the HEAD of `sebmart/two-sided-platform-competition`, unmodified.
Inputs: the saved 32-seed results `figure_regeneration/results/<experiment>_results.jld2`, unmodified.

## 1. What was changed in the three draft scripts, and why

The drafts had never been executed. The fixes below preserve the intent of each mode; where the engine's actual behaviour
differs from what the task sheet assumed, the script follows the engine and the difference is noted.

### `diagnostics.jl`

**Reported-row selection (all modes).** The draft took every converged row of a seed. The figures use only the first 32
converged seeds per configuration (`build_figures.get_results`). A `reported_mask` function now replicates that rule
exactly, both in the units (so no work is spent on rows the figures do not use) and in `summarize`. A `needed_seeds`
mode prints the seeds that the figures actually use, so the runner does not launch empty units (for `order_of_events`
these are seeds 1–31 and 33; seed 32 is absent from the saved file).

**Regret mode.**
- *Price sweep upper end.* The draft stopped at `service_value / price_step` (= 100) on the grounds that higher prices
  attract no demand. The logit demand model gives positive demand at negative utility, and the saved results contain
  reported prices up to 117 (`dedicated_demand`) and 131 (`dedicated_supply`). The sweep now runs to
  `DIAG_PMAX_FACTOR × service_value / price_step` (default 1.5, i.e. 150) and always covers the ±window around the
  reported price.
- *Price deviations keep the deviating firm's pay as reported.* The draft capped pay at the new price
  (`min(pay, v)`). The engine's price stage passes the current pay unchanged into `sequentialNE_demand` and lets the pay
  stage underneath re-solve it; the script now does the same, so a price deviation is evaluated exactly as the engine's
  own price step evaluates its ±1 candidates.
- *Self-continuation recorded.* The task sheet's sanity check ("the deviation equal to the reported decision must
  reproduce the base profit") fails by construction: the demand, pay and supply stages are path-dependent cycle
  dynamics with a "most favorable state" selection, so re-solving them from the reported state at the reported
  decision generally moves demand and supply by a grid step and changes profits (see Section 3.1 for the numbers). The
  script therefore records the re-solved profit at the reported decision (`self_profit`) and reports regret against
  both bases: `regret` = best deviation − reported profit (what the task asked for), and `regret_self` = best
  deviation − self-continuation profit (the within-procedure best-response residual: the gain over "not deviating",
  with the same continuation applied to both).
- *Inactive firms skipped.* A firm with no supply or no demand pool never updates in the engine (its price/pay are
  frozen), so it is not swept (relevant for `dedicated_demand` / `dedicated_supply`, not for `order_of_events`).
- *Bookkeeping.* Raw gain (unfloored), reported and arg-best decisions, the re-solved pay at the arg-best price
  deviation, per-row wall time; the engine's per-process stationary-distribution cache is reset per reported row to
  bound memory (the engine itself resets it per equilibrium solve).

**Cycles mode.**
- *Pay-stage activity condition fixed.* The draft let a firm update its pay when `firm1_demand + firm12_demand > 0`
  (a parameter). The engine's pay stage uses the *current demand rates* (`demand.rate1 + demand.rate12_1 +
  demand.rate12_12 > 0`). Copied from the engine.
- *Cycle detection on (firms, workers, demand).* The engine's price-stage state tuple also carries the two profits
  returned by `sequentialNE_demand`. That function does not recompute the simulation for the demand state it selects
  from a demand cycle, so the profits in the engine's state can be stale relative to the decisions in the same tuple
  (the draft, by seeding the start state with a fresh simulation, could never have matched such a tuple). The script
  detects the cycle on the decisions alone, attaches to each cycle state the profits the engine would have recorded
  (from its predecessor on the cycle) for the most-favorable-state rule, and separately the true profits
  (`simulation` at that state, which is what the figures plot) for the time averages. It counts how many cycle
  states have stale engine profits.
- *Diagnostic columns added:* whether the reported state lies on the recovered cycle, whether the engine's rule
  applied to the recovered cycle picks the reported state, the reported values, and min/max of prices as well as pay
  and profit.
- *Simultaneous stage added.* The task restricted cycles to sequential configurations. Since the simultaneous solver
  does compare with staying put and can end at a fixed point, a third stage `simul` (map copied from `simultaneousNE`:
  stay vs the four diagonal moves, strict improvement, engine order LH, HH, LL, HL) is run on the simultaneous
  configurations. It is cheap and answers the referee's "convergence versus cycling" question for the right-hand
  panels of Figure 4 too.

**Sensitivity mode.**
- *Budget scaling.* The task sheet proposed scaling `max_iter` and `max_simulations` ×2 for `step05` and ×4 for
  `step025` "so the budget is not binding by construction". It is binding: `step025`, seed 1, firm-1 demand 0.2,
  exhausted the ×4 budget of 2,000,000 queueing solves in under 90 s. Both the price and the pay stage take
  proportionally more one-step moves on a finer grid, so the number of solves grows roughly with the square of the
  refinement. The script uses ×4 (`step05`) and ×16 (`step025`); the summary reports, per variant, how many converged
  rows needed more solves than the ×2 / ×4 budgets would have allowed, so the effect of that choice stays visible.
- Per-row wall time recorded; the solve error message is included in the non-convergence warning.

**Summarize.**
- Regret: both regret definitions, zero-regret shares, 50/90/max quantiles, relative regret where the reported profit
  exceeds $1/hr, the self-vs-reported gap, the share of arg-best deviations more than one grid step away, failed
  solves; overall per experiment × timing × kind, and per configuration.
- Cycles: cycle-length distribution, fixed-point share, on-cycle and most-favorable checks, stale-profit share,
  |time average − reported| and cycle range for pay, price and profit (in $), and per configuration the Figure-4
  summary (mean, p25–p75) under the reported-state rule next to the same summary under the time-average rule, with the
  figure's convention that a firm's pay enters only when its profit is positive.
- Sensitivity: per variant the non-convergence count, budget usage and solve times; per demand share the base and
  each variant with mean and p25–p75 of pay and profit for both firms (Figure 4's quantities), with n used / attempted.
- Merged DataFrames saved to `diagnostics/regret_<exp>.jld2`, `cycles_<exp>.jld2`, `sensitivity.jld2`.

### `run_diagnostics.sh`

- Prefers the juliaup Julia 1.12 binary (`~/.juliaup/bin/julia`, overridable with `JULIA`); on this Mac
  `command -v julia` finds Homebrew's Julia 1.10.2 first.
- Default seeds for `regret` / `cycles` come from `diagnostics.jl needed_seeds`; `sensitivity` keeps 1–44.
- Each unit runs under a wall-clock timeout (`DIAG_TIMEOUT`, default 4 h) and is SIGKILLed if it wedges (the harness
  README documents a native deadlock that can only be killed from outside); a killed or failed unit leaves no unit
  file, so a relaunch retries it.
- The seed is passed as a positional argument instead of substituting `{}` inside the quoted script.
- Logs the Julia version, seeds and timing; prints each unit's last log line when it finishes.

### `DIAGNOSTICS_README.md`

Updated to describe the modes as implemented (price sweep range, self-continuation, simultaneous cycle stage, budget
scaling, seeds, timeout) and the observed costs.


### Round 2 additions (2026-10-07, `CLAUDE_CODE_TASK_2.md`)

- `diagnostics.jl`: three sensitivity variants `rate01` (`rate_step` 0.01), `step05r` (`price_step` 0.5, `rate_step`
  0.01), `step025r` (`price_step` 0.25, `rate_step` 0.005), with budget scales ×16 / ×16 / ×64 set from one-seed trials;
  `DIAG_SENS_SCALE` overrides the scale for trials. The per-variant summary line now reports how many converged rows
  needed more solves than the ×1 / ×4 / ×16 budgets. `regret` and `cycles` accept `sens-<variant>` as the experiment
  name and then read the merged `sensitivity_<variant>_seed*.jld2` units (same first-32-converged rule); the regret
  header prints the window and stride in dollars for the grid in use. `DIAG_REGRET_KINDS` restricts the sweep to a
  subset of `price,pay,joint`.
- `DIAGNOSTICS_README.md`: variant table with budgets and costs, the `sens-<variant>` usage.
- `run_diagnostics.sh`: unchanged.


### Round 3 additions (2026-10-07, `CLAUDE_CODE_TASK_3.md`)

- `build_figures_timeavg.jl` (new, next to the unchanged `build_figures.jl`): the four figures from the time average
  over each reported row's terminal cycle, plus comparison overlays and a change table (Section 3.5).
- `diagnostics.jl`, `run_diagnostics.sh`: unchanged in round 3 (the `cycles` mode already handled inactive firms the
  way the engine does).

### Round 4 additions (2026-10-14, `CLAUDE_CODE_TASK_4.md`)

- `diagnostics.jl`: the cycle map runs the lower stages with the variant's own iteration budget on
  `cycles sens-<variant>` (`MAXIT_REF`, 1000 × `variant_scale`); `summarize` adds `cycles: sens-<variant>` sections and a
  second sensitivity table summarizing every row by the time average over its terminal price-stage cycle, with the
  largest cell differences and a base check (Section 3.6). Previous version kept as `diagnostics_task3.jl`.
- `run_diagnostics.sh`, `build_figures_timeavg.jl`, `DIAGNOSTICS_README.md`: unchanged in round 4.

## 2. Commands run, versions, wall times

```
# setup (2026-10-06, ~22:00 EDT)
curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12     # Julia 1.12.7 via juliaup
cd ~/Documents/Claude/Projects/Free-riding\ revision && mkdir -p code && cd code
git clone https://github.com/sebmart/two-sided-platform-competition
cd two-sided-platform-competition/figure_regeneration
julia --project=../freeriding_simulations -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'   # 85 s, all packages incl. Plots
cp "../../../minor revision/simulation_diagnostics/"{diagnostics.jl,run_diagnostics.sh,DIAGNOSTICS_README.md} .

# step 1: one seed of each mode, outputs to a scratch folder (not diagnostics/units), then fixes
julia --project=../freeriding_simulations diagnostics.jl cycles order_of_events 1 <scratch>/smoke_cycles_seed1.jld2   # 20 s
DIAG_WINDOW=2 DIAG_STRIDE=50 julia --project=... diagnostics.jl regret order_of_events 1 <scratch>/smoke_regret_seed1.jld2   # 199 s
julia --project=../freeriding_simulations diagnostics.jl sensitivity cap7 1 <scratch>/smoke_sens_cap7_seed1.jld2       # ~8 min
# single-configuration timings of the other variants (seed 1, firm-1 demand 0.2): step05 8 s, step025 (x4 budget) FAILED
# "Timeout: Simulation exceeded 2000000 simulations" in < 90 s, step025 (x16 budget) 10 s, cap10 537 s

# step 2: full runs, DIAG_P=8 (performance cores), each step under caffeinate -i, driven by diagnostics/run_chain.sh
nohup diagnostics/run_chain.sh > diagnostics/logs/chain.log &
#   cycles order_of_events        22:11 -> 22:13   (2 min)
#   sensitivity step05  (32 seeds) 22:13 -> 22:17   (4 min)
#   sensitivity step025 (32 seeds) 22:17 -> 22:19   (3 min)
#   sensitivity cap7    (32 seeds) 22:19 -> 22:49   (30 min)
#   regret order_of_events        22:49 -> 23:51   (62 min wall; 7.7 CPU-hours: price sweeps 6.2 h, joint 1.5 h, pay 1 min)
#   sensitivity cap10   (32 seeds) 23:51 -> 08:48   (9 h wall incl. side runs; 76 CPU-hours)
#   cycles supply_sensitivity      00:40 -> 00:49   (side run, 4 processes)
#   regret supply_sensitivity      00:55 -> 02:56   (side run, 4 processes; 8.1 CPU-hours)
./run_diagnostics.sh summarize

# round 2 (2026-10-07): one-seed budget trials to a scratch folder, then the 32-seed runs through diagnostics/run_chain2.sh
DIAG_SENS_SCALE=16 julia --project=... diagnostics.jl sensitivity rate01   1 <scratch>/trial_rate01_seed1.jld2    # 492 s
DIAG_SENS_SCALE=16 julia --project=... diagnostics.jl sensitivity step05r  1 <scratch>/trial_step05r_seed1.jld2   # 168 s
DIAG_SENS_SCALE=64 julia --project=... diagnostics.jl sensitivity step025r 1 <scratch>/trial_step025r_seed1.jld2  # 221 s
nohup diagnostics/run_chain2.sh > diagnostics/logs/chain2.log &        # DIAG_P=8, seeds 1-32
#   sensitivity rate01    11:18 -> 11:47   (28 min)
#   sensitivity step05r   11:47 -> 11:59   (12 min)
#   sensitivity step025r  11:59 -> 12:19   (20 min)
#   summarize
DIAG_WINDOW=20 DIAG_STRIDE=10 DIAG_REGRET_KINDS=price caffeinate -i ./run_diagnostics.sh regret sens-step05r   # optional regret, see 3.4

# round 3 (2026-10-07): cycle recovery for the two remaining experiments, then the time-average figures
nohup diagnostics/run_chain3.sh > diagnostics/logs/chain3.log &      # cycles dedicated_demand 15:02 -> 15:06, cycles dedicated_supply 15:06 -> 15:08, summarize
julia --project=../freeriding_simulations build_figures_timeavg.jl    # all four figures + compare_*.pdf + CHANGES.txt, ~25 s per experiment

# round 4 (2026-10-14): cycle recovery on the four Table-9 variants, then summarize (diagnostics/run_chain4.sh, DIAG_P=8, DIAG_TIMEOUT=43200)
julia --project=../freeriding_simulations diagnostics.jl cycles sens-cap7 1 <scratch>/smoke_cycles_cap7_seed1.jld2   # 65 s smoke test
nohup diagnostics/run_chain4.sh > diagnostics/logs/chain4.log &
#   cycles sens-cap7      11:01 -> 11:08   (7 min)
#   cycles sens-step05r   11:08 -> 11:10   (2 min)
#   cycles sens-step025r  11:10 -> 11:12   (2.5 min)
#   cycles sens-cap10     11:12 -> 12:56   (103 min; 13.1 CPU-hours)
#   summarize             12:56 -> 12:57
```

Julia 1.12.7 (juliaup channel 1.12), macOS 26.7.1, 8 performance cores used. Sensitivity seeds: 1–32 for every variant
(all 320 solves per variant converged, so no replacement seeds were needed; the figures' own seed set for
`order_of_events` is 1–31 and 33).

## 3. Headline numbers (Figure 4's experiment, `order_of_events`, 10 configurations × 32 seeds)

### 3.1 Referee bullet (1): best-response residuals at the reported outcomes

Sweep: every grid value within ±10 steps of the reported decision plus the whole grid at stride 5; prices 0–150
($1 steps), pay 0–price. 1,920 firm-outcome rows, 0 failed continuation solves.

| timing / deviation | n | zero-regret share | median | p90 | max | relative median (base > $1/hr) | arg-best more than 1 step away |
|---|---|---|---|---|---|---|---|
| sequential, price (demand → pay → supply re-solved) | 640 | 12.7% | $0.79/hr | $2.39 | $6.10 | 5.3% | 82% |
| sequential, pay (supply re-solved) | 640 | 16.9% | $0.27/hr | $0.57 | $3.02 | 1.8% | 47% |
| simultaneous, joint (price, pay) (supply + demand re-solved) | 640 | 13.4% | $0.54/hr | $2.04 | $8.11 | 3.1% | 83% |

Regret against the self-continuation profit instead of the reported profit (same sweep): price median $0.82 (p90 $2.77,
max $6.54, zero 8.4%); pay median $0.18 (p90 $0.56, zero 27%); joint median $0.57 (p90 $1.98, zero 12%).

Reading these numbers:
- The reported outcomes are **not grid best responses**. Roughly 60% of price and joint rows have a regret above $0.50/hr
  and about 40% above $1/hr, against reported profits of $10–$20/hr for the typical firm. Pay regrets are smaller:
  87% below $0.50/hr.
- The best price deviation is usually several steps away (median 4 steps for rows with positive regret, 92% within the
  ±10 window, 75% are price *cuts*), and the pay re-solved after it moves by a median of 2 steps. So this is not a
  one-step rounding residual of the two-cycle: the firm's profit along the engine's own continuation is not
  single-peaked around the reported price, or the ±1 comparison that drives the price stage does not see the
  improvement.
- Prices above the service value ($100) are the best deviation in 4.5% of positive-regret price rows (4.3% hit the
  sweep's upper end of 150): these are small-firm cases with near-zero base profit, where a very high price with the
  trickle of demand the logit model still gives beats the reported state by cents. They do not affect the medians.
- The largest residuals: sequential price, firm 2 at firm-1 demand 1.6 (seed 6): reported 97/76 earns $10.03/hr, the
  continuation after cutting to 87 (pay re-solved to 62) earns $16.13/hr. Pay: firm 1 at 5% share (seeds 11, 13, 15):
  reported pay 31 earns $0.21/hr; pay 50 at the same price and demand earns $3.23/hr (supply re-solved). Joint: firm 1
  at 10% share (seeds 11, 18, 23): reported 92/35 earns $0.33/hr; 93/45 earns $8.44/hr. In the simultaneous solver,
  (92/35) → (93/45) is not reachable by any sequence of the four diagonal ±1 moves that improve at every step, so the
  local search stops there.
- Firm 1 (the smaller firm on the simulated half) has larger relative residuals than firm 2 (price: median 7.8% vs
  3.8% of profit; pay: 2.2% vs 1.3%).


**Second experiment (`supply_sensitivity`, 22 sequential configurations × 32 seeds, 2,816 firm-outcome rows, 0 failed
solves).** Same magnitudes: price regret median $0.56/hr (p90 $2.20, max $5.76, zero in 17%, 83% of arg-best prices
more than one step away; relative median 3.7%); pay regret median $0.24/hr (p90 $0.60, zero in 18%, 86% below $0.50).
Regret is larger in the 50/50 configurations than at 25/75 for most α, and shows no trend in α. One group of outliers:
at α = 0.25 and 50/50 demand, 17 pay rows have regrets above $2/hr, up to $23.9/hr (seeds 15, 16, 20, 31; e.g. reported
91/55 earning $14.1/hr, pay 0 at the same price and demand earning $38.1/hr). With a supply sensitivity that low the
workers' logit response to the wage is almost flat, so the flexible workers stay when pay is cut to zero; the
corner is 55 steps away and invisible to the ±1 comparison. The per-configuration table is in `summary.txt`.

**The sanity check fails by construction, and the failure is itself a finding.** Re-solving the lower stages from the
reported state at the reported decision (the "self" deviation) reproduces the reported profit exactly in only 1.2% of
price rows (median |self − reported| = $0.49/hr, max $4.73), in 46% of pay rows (median $0.11, max $0.80) and in 65%
of joint rows (max $0.87). The demand, pay and supply stages are one-step dynamics ending in cycles from which a
"most favorable" state is selected; restarted from that selected state they typically land one grid step away
(e.g. flexible-worker rate 52 → 53 and demand 63/67 → 61/65 at the 50/50 configuration, seed 1). The self-continuation
gap for price deviations ($0.49/hr median) is of the same order as the median regret, so part of the measured residual
is the noise floor of the continuation itself; the `regret_self` column removes that component by comparing deviations
with "not deviating" under the same continuation, and its median is essentially the same ($0.82 vs $0.79). The
engine's comparisons at the price and pay stages never evaluate "stay", so this gap never enters the procedure.

### 3.2 Referee bullet (2): convergence versus cycling, and neutral cycle summaries

Recovered by restarting the engine's own stage maps from each reported state (320 sequential and 320 simultaneous
outcomes; the reported state lies on the recovered cycle in 100% of cases).

| stage | fixed point | L = 2 | L = 4 | L = 6 | L = 8 | L ≥ 10 |
|---|---|---|---|---|---|---|
| price stage (sequential) | 0.0% | 90.6% | 6.6% | 1.6% | 0.9% | 0.3% (one L = 10) |
| pay stage (sequential, at reported prices and demand) | 0.0% | 71.6% | 15.6% | 9.4% | 2.5% | 0.9% (L = 10, 20) |
| simultaneous firm map | 63.1% | 11.9% | 4.1% | 8.1% | 3.1% | 9.7% (up to L = 48) |

- As the revised text says, the sequential stages never end at a fixed point: by construction a firm always moves one
  step, so the terminal object is a cycle, in 91% of price-stage cases the two-cycle bracketing the best response. The
  simultaneous solver, which does compare with staying put, reaches a fixed point in 63% of outcomes and otherwise
  cycles, sometimes for a long time (14% of outcomes have cycles of 8 or more states, up to 48).
- The engine's most-favorable-state rule applied to the recovered cycle returns the reported state in 99.1% of price
  cycles and 100% of pay and simultaneous cycles. The 3 exceptions are exact ties at the symmetric 50/50 configuration,
  where the cycle contains the firm-swapped mirror of the reported state (same min and max profit); which one is
  returned depends on where the walk starts.
- Stale profits: in 95.6% of price-stage cycles at least one state's profits as recorded by the engine differ from
  the profits of that state's decisions, because `sequentialNE_demand` returns the simulation of its last iterate
  rather than of the demand state it selects. The selection rule therefore ranks cycle states on slightly wrong
  numbers; it still picked the reported state (the figures' numbers themselves are recomputed from the decisions, so
  they are unaffected).
- **Time average versus most-favorable state.** Over the price-stage cycle: |time average − reported| has median $0.50
  for pay and price (half a grid step, as expected for a two-cycle), p90 $2.00 for pay, max $11.25 for pay (seed 14,
  firm-1 demand 0.2: pay cycles between 28 and 43 along a 4-cycle), and median $0.40/hr, p90 $1.02, max $2.63 for
  profit. The full cycle range (max − min) has median $1 for pay and price and $0.85/hr for profit (p90 $2.10, max
  $5.83). Pay-stage cycles are tighter (pay range median $1, max $7; profit range median $0.58/hr, max $2.94).
- **Would Figure 4 change?** No visibly. The per-configuration means over 32 seeds under the time-average rule are
  within $1.2 of the reported-state means for pay (largest: pay 1 at 5% share, 32.8 → 31.6; at 40% share 55.2 → 54.2)
  and within $0.5/hr for profit, with one exception at 5% share where firm 1's mean profit is $0.83 reported vs $0.43
  time-averaged (a mean driven by 2–3 seeds with a $3–5 profit on part of the cycle; the medians are $0.2 in both). The
  p25–p75 bands shift by similar amounts and overlap almost entirely. The qualitative pattern (the smaller firm pays
  less and the pay gap closes toward 50/50) is unchanged under both rules; the same holds for the simultaneous panel,
  where 63% of states are fixed points and the time average coincides with the reported state. The configuration table
  is in `summary.txt` (section "cycles", "by configuration").


**Second experiment (`supply_sensitivity`, 22 sequential configurations × 32 seeds, Section 5.3's figure).** Same
picture: price-stage cycles are two-cycles in 86% of outcomes (L = 4: 9.5%, up to L = 10), pay-stage cycles in 89%,
no fixed points; the reported state lies on the recovered cycle in 703 of 704 price-stage outcomes and in all pay-stage
outcomes. The one exception (seed 5, alpha = 0.1, 50/50 demand) recovers a two-cycle one price step below the reported
state with profits about $1/hr lower; a near-tie resolved differently, most likely a last-bit floating-point difference
between the library versions that produced the saved results (Julia 1.12.6) and this run (1.12.7). The six cases where
the engine's rule does not return the reported state are all firm-swap ties in the symmetric configurations.
|Time average − reported| for pay: median $0.50, p90 $1.50, max $4.20; for profit: median $0.34/hr, p90 $0.70, max
$3.25. Per-configuration means move by well under $1 (table in `summary.txt`).

### 3.3 Referee bullet (3): sensitivity to the price step and to the queue cap

Ten sequential configurations re-solved from the same random starts (seeds 1–32, `randomness` 0.1) with one
parameter changed. All 320 solves converged for every variant (within ×4 of the base budget for `step05`, ×16 for
`step025`; 2 of the `step05` and 13 of the `step025` solves needed more queueing solves than the task sheet's ×2 / ×4
budgets would have allowed and would have been replaced by other seeds under that rule). Mean [p25, p75] over 32
seeds, the same quantities Figure 4 plots:

| firm-1 share | variant | pay 1 | pay 2 | profit 1 | profit 2 |
|---|---|---|---|---|---|
| 10% | base ($1, cap 5) | 41.6 [40, 43] | 68.1 [67, 69] | 5.2 [4.6, 5.8] | 27.7 [27.3, 28.3] |
| | cap 7 | 42.8 [42, 44.3] | 67.7 [67, 69] | 4.5 [4.2, 4.9] | 27.2 [27.0, 27.4] |
| | cap 10 | 41.2 [40, 42] | 68.3 [68, 69] | 4.1 [3.8, 4.6] | 27.0 [26.9, 27.3] |
| | step 0.5 | 35.2 [31.5, 39.5] | 69.1 [68, 69.5] | 1.6 [0.4, 2.2] | 29.3 [29.7, 30.5] |
| | step 0.25 | 32.0 [32, 32] | 70.5 [70.8, 71.8] | 0.0 [0, 0] | 29.3 [29.1, 29.3] |
| 25% | base | 46.2 [46, 47] | 70.4 [69, 71] | 13.1 [12.9, 13.6] | 18.6 [18.7, 19.4] |
| | cap 7 | 45.2 [45, 46] | 70.3 [68, 72] | 11.6 [11.0, 12.7] | 17.5 [17.1, 18.3] |
| | cap 10 | 45.0 [44, 46] | 70.2 [69, 71] | 10.7 [10.6, 11.1] | 17.9 [17.8, 18.2] |
| | step 0.5 | 45.7 [42, 45] | 71.3 [70.4, 72.6] | 10.0 [9.6, 10.8] | 16.8 [15.5, 17.5] |
| | step 0.25 | 38.2 [37.3, 39.5] | 74.1 [73.5, 74.8] | 6.3 [2.1, 9.1] | 16.3 [15.9, 18.0] |
| 35% | base | 50.7 [49, 51.3] | 72.1 [71, 73] | 13.4 [12.8, 13.7] | 12.3 [11.9, 12.7] |
| | cap 7 | 51.6 [50, 52] | 70.7 [69, 72] | 12.5 [12.3, 12.9] | 12.0 [11.8, 12.3] |
| | cap 10 | 50.3 [49, 53] | 70.6 [70, 71.3] | 11.2 [10.8, 12.7] | 12.3 [11.8, 12.6] |
| | step 0.5 | 65.5 [64, 68.5] | 65.9 [63.5, 67] | 10.0 [9.0, 10.6] | 18.0 [17.8, 19.3] |
| | step 0.25 | 42.4 [40, 44.5] | 75.7 [74.8, 77.3] | 13.5 [10.9, 17.6] | 11.5 [11.0, 12.6] |
| 45% | base | 59.6 [57.8, 62] | 67.8 [65, 70] | 13.5 [13.2, 14.4] | 12.5 [11.0, 13.9] |
| | cap 7 | 59.1 [56, 62] | 67.5 [66, 69] | 12.5 [11.4, 13.6] | 11.3 [10.2, 11.4] |
| | cap 10 | 58.8 [56.8, 60.3] | 67.6 [65.8, 69] | 12.7 [11.3, 13.9] | 11.3 [10.3, 12.1] |
| | step 0.5 | 65.9 [64.4, 68] | 64.6 [62.9, 67.1] | 12.6 [12.0, 13.3] | 15.8 [15.2, 16.3] |
| | step 0.25 | 45.7 [41.6, 48.1] | 76.9 [75.8, 78.8] | 13.5 [11.7, 17.2] | 7.3 [5.7, 7.4] |
| 50% | base | 64.4 [64, 65.3] | 64.3 [63, 65.3] | 13.4 [13.1, 14.3] | 13.3 [12.7, 14.4] |
| | cap 7 | 63.0 [61.8, 64] | 64.3 [63, 65] | 13.0 [11.8, 14.1] | 12.4 [10.4, 13.9] |
| | cap 10 | 63.5 [63, 65] | 64.2 [63, 66] | 12.6 [10.8, 13.8] | 12.2 [10.7, 13.5] |
| | step 0.5 | 66.1 [64.4, 68.6] | 64.6 [63, 65.5] | 13.9 [13.2, 14.5] | 14.6 [14.0, 15.2] |
| | step 0.25 | 57.3 [47.3, 64.9] | 66.9 [60.4, 77.3] | 10.5 [4.3, 15.5] | 7.8 [4.2, 10.6] |

(All ten shares are in `summary.txt`.)

- **Queue cap.** Cap 7 reproduces the base figure: pay within about $1 and profit within about $1.5/hr at every share,
  same ordering of the two firms everywhere, overlapping bands. Cap 10 reproduces the pay pattern (within $2 of the
  base means at every share from 10% up) and the profit pattern at shares of 35% and above (within $1.3/hr); between
  10% and 30% share firm 1's profit is lower than on the base grid by $1–2.4/hr at most shares and by $4.7/hr at 30%
  (10.0 vs 14.6), while firm 2's is within $1. (The only visible pay difference, pay 1 at 5% share, 32.8 base vs 16.4
  cap 7 vs 9.4 cap 10, is a firm earning $0.2–0.8/hr either way.) 6 of the 32 cap-10 solves at the 5% share exhausted
  the 500,000-solve budget (the harness would have drawn replacement seeds); that row uses the 26 converged seeds.
- **Price step.** The results are **not robust to the price step**, and the two finer grids move in different
  directions:
  - with step $0.50, at shares of 30% and above both firms' pay converges to about $64–66 and the larger firm earns
    more than the smaller one (e.g. 35% share: profits 10.0 vs 18.0, against 13.4 vs 12.3 on the base grid); below 30%
    the small firm's pay and profit are lower than on the base grid;
  - with step $0.25, the small firm is out of the market at shares up to 20% (pay $32 for every seed, profit ≈ 0:
    with a $32 per-job pay the hourly wage is below the $40 reservation wage, so it attracts no workers), while at
    35–45% the free-riding asymmetry is *stronger* than on the base grid (pay 42–46 vs 76–77, the small firm earning
    13.5 vs 7.3–11.5), and at 50/50 the outcomes are dispersed across seeds (p25–p75 of pay 2: 60–77).
  Because every stage moves one grid step per iteration, the price step is not only a resolution parameter but also
  the step size of the best-response dynamics; both the path and the terminal cycle change with it, and the reported
  equilibria on the three grids are different selections from what is evidently a large set of candidate outcomes.
  One caveat for interpreting the two step variants: the rate grid (0.02 per hour) was left unchanged, so refining the
  price grid alone also changes the relative resolution of the price/pay and supply/demand grids (a $0.25 pay change
  often does not move the discretised supply at all). A joint refinement (price step and rate step together) would be
  the cleaner robustness check; it was not part of the task and was not run.

### 3.4 Joint grid refinement (follow-up task of 2026-10-07): price step and rate step refined together

**Verdict.** (a) *Refining the rate grid alone does not change the base figure*: with `rate_step` 0.01 and the $1 price
grid, pay is within $1.8 of the base means at every share and profits are within $1.7/hr (mostly $0.2–1.1/hr higher
for both firms), with the same ordering of the two firms everywhere; the only visible difference is at the 5% share,
where firm 1's mean profit is $2.3/hr instead of $0.8 because a few seeds end with firm 1 earning $3–4/hr
(p25–p75: $0.1–3.8). (b) *The jointly refined grids reproduce the base figure's qualitative pattern*: on both `step05r`
(price $0.50, rate 0.01) and `step025r` (price $0.25, rate 0.005) the smaller firm pays less at every share below 50%,
its pay rises from the low 20s–40s toward the larger firm's as the share approaches 50/50 while the larger firm's pay
stays at $66–73 and meets it at $64–65, and the profit ordering by share is the base one: firm 1 earns less than firm 2
up to the 30% share, about the same at 35%, and more at 40–45%. Pay means are within $2 of the base at every share
from 15% up and profits within $1.6/hr (the refined grids give slightly lower profits for both firms, by $0.3–1.5/hr).
Where it does not reproduce the base: at the two lowest shares. At 5% the small firm pays $25 (`step05r`, every seed
within $25–25.5) or $23 (`step025r`) instead of $33 and earns $0.05–0.10/hr instead of $0.2–0.8/hr; at 10% it pays $38
(`step05r`, p25–p75 $31.5–42.5) or $31 (`step025r`) instead of $42 and earns $3.3 or $1.3/hr instead of $5.2/hr. The
smaller firm is thus pushed further toward the margin at 5–10% share on the finer grids, with wide seed-to-seed
dispersion at 10% under `step05r`. (c) *The round-1 anomalies do not survive the joint refinement.* The equalisation of
both firms' pay at about $65 from the 30% share up under `step05` is gone: at 30–35% the pays are $48–52 vs $71–72, as
on the base grid. The small firm's exit at a constant $32 pay at shares up to 20% under `step025` is gone too: at 15%
and 20% the refined grids give pay $44.7–46.7 and profits of $7.7–10.4/hr, matching the base. What remains is a milder,
share-limited version of the exit: at 5% (both refined grids) and, partly, at 10% (`step025r`, and some `step05r`
seeds) the small firm's per-job pay is low enough that its hourly wage falls under the $40 reservation wage and it
retains almost no workers; the base grid shows the same firm at $0.2/hr median profit at 5%, so this is a sharper
version of a feature the base case already has rather than a new regime. The round-1 non-robustness to the price step
is therefore attributable to refining the price grid while leaving the rate grid coarse, as the follow-up task
hypothesised.

**Budgets and solve counts.** One-seed trials (seed 1) at ×16 (`rate01`, `step05r`) and ×64 (`step025r`) on both
`max_iter` and `max_simulations`: all 30 solves converged, with at most 524k, 232k and 1.43M queueing solves and
8.2, 2.8 and 3.7 min per seed. Those budgets were kept for the 32-seed runs (15–35× headroom). Full runs: all 960
solves converged; largest solve counts 1.04M (`rate01`), 372k (`step05r`), 1.52M (`step025r`) against budgets of 8M,
8M and 32M; converged rows that needed more than the base budget of 500k solves: 10, 0 and 17 (none needed more than
2M, i.e. the ×4 budget). Wall time on 8 cores: 28 min, 12 min, 20 min (longest single seed 11.1, 3.7, 8.1 min).
`initial_guess` scales its random starts with both `price_step` and `rate_step` (prices $98–99 and rates 3.63/hr, 0.59/hr
etc. on every grid, in units of the new steps), so the starts are valid states of the refined grids; no handling was
needed.

**Table (all ten shares; mean [p25, p75] over the first 32 converged seeds, Figure 4's quantities).**

| firm-1 share | variant | pay 1 | pay 2 | profit 1 | profit 2 |
|---|---|---|---|---|---|
| 5% | base | 32.8 [29.0, 32.8] | 67.3 [66.8, 68.0] | 0.83 [0.21, 0.29] | 34.4 [34.7, 34.8] |
| | rate01 | 34.8 [28.0, 38.0] | 67.7 [67.0, 69.0] | 2.34 [0.14, 3.78] | 33.6 [33.0, 34.9] |
| | step05r | 25.3 [25.0, 25.5] | 66.5 [66.5, 66.5] | 0.10 [0.09, 0.10] | 34.5 [34.4, 34.6] |
| | step025r | 22.7 [21.3, 23.5] | 68.4 [67.8, 69.3] | 0.05 [0.05, 0.05] | 33.0 [32.3, 34.0] |
| 10% | base | 41.6 [40.0, 43.0] | 68.1 [67.0, 69.0] | 5.16 [4.56, 5.75] | 27.7 [27.3, 28.3] |
| | rate01 | 43.0 [42.0, 44.0] | 68.2 [67.8, 69.0] | 6.60 [6.31, 7.00] | 29.4 [29.2, 29.9] |
| | step05r | 38.4 [31.5, 42.5] | 68.2 [67.5, 69.0] | 3.29 [0.34, 5.14] | 28.0 [27.0, 29.5] |
| | step025r | 30.6 [26.3, 31.3] | 69.8 [68.5, 70.3] | 1.28 [0.09, 0.95] | 27.2 [26.3, 28.5] |
| 15% | base | 44.8 [44.0, 45.0] | 68.6 [68.0, 69.0] | 8.69 [8.40, 8.94] | 25.1 [24.7, 25.7] |
| | rate01 | 44.5 [44.0, 45.0] | 68.7 [68.0, 70.0] | 9.32 [9.00, 9.45] | 25.7 [25.4, 26.1] |
| | step05r | 44.9 [44.5, 45.6] | 69.3 [68.5, 70.0] | 7.89 [7.62, 8.50] | 23.8 [23.1, 24.6] |
| | step025r | 44.7 [43.8, 45.3] | 69.8 [68.9, 70.3] | 7.70 [7.00, 8.20] | 23.3 [22.1, 24.1] |
| 20% | base | 45.1 [44.0, 46.0] | 69.4 [68.0, 70.0] | 11.0 [10.4, 11.8] | 21.7 [21.2, 22.3] |
| | rate01 | 45.5 [45.0, 46.0] | 68.7 [68.0, 69.0] | 11.6 [10.9, 12.2] | 22.4 [22.3, 22.7] |
| | step05r | 46.3 [45.9, 47.0] | 70.3 [69.4, 71.1] | 10.4 [10.2, 11.1] | 20.6 [19.5, 21.6] |
| | step025r | 46.7 [45.8, 47.6] | 71.1 [69.9, 71.8] | 10.2 [9.9, 10.7] | 20.2 [19.8, 21.2] |
| 25% | base | 46.2 [46.0, 47.0] | 70.4 [69.0, 71.0] | 13.1 [12.9, 13.6] | 18.6 [18.7, 19.4] |
| | rate01 | 45.4 [45.0, 46.0] | 69.3 [68.0, 70.0] | 13.3 [13.2, 13.4] | 18.7 [18.4, 19.0] |
| | step05r | 46.3 [46.0, 46.5] | 71.1 [70.0, 72.1] | 12.3 [12.0, 12.9] | 17.7 [17.2, 18.3] |
| | step025r | 47.2 [46.5, 47.6] | 71.7 [70.7, 72.6] | 11.8 [11.5, 12.4] | 17.0 [16.2, 18.0] |
| 30% | base | 46.3 [45.0, 47.0] | 71.5 [71.0, 72.0] | 14.6 [14.1, 15.7] | 15.4 [15.0, 15.9] |
| | rate01 | 46.4 [45.0, 47.3] | 69.7 [68.0, 71.0] | 14.8 [14.3, 15.5] | 15.6 [15.1, 15.9] |
| | step05r | 47.6 [46.0, 47.1] | 72.1 [71.0, 72.6] | 14.0 [13.6, 15.1] | 14.9 [14.7, 15.3] |
| | step025r | 49.3 [46.9, 51.3] | 72.6 [71.5, 73.3] | 12.8 [11.5, 14.1] | 14.3 [13.8, 15.0] |
| 35% | base | 50.7 [49.0, 51.3] | 72.1 [71.0, 73.0] | 13.4 [12.8, 13.7] | 12.3 [11.9, 12.7] |
| | rate01 | 51.7 [51.0, 53.0] | 71.0 [69.0, 72.0] | 13.8 [13.3, 14.6] | 13.3 [12.5, 14.3] |
| | step05r | 51.9 [50.0, 53.5] | 71.2 [70.0, 72.1] | 12.6 [12.1, 13.0] | 12.3 [12.0, 12.7] |
| | step025r | 52.1 [49.1, 53.4] | 71.3 [69.7, 72.1] | 12.4 [12.0, 12.8] | 12.3 [11.8, 12.7] |
| 40% | base | 55.2 [53.0, 59.0] | 69.7 [68.0, 71.3] | 13.3 [12.5, 13.8] | 12.2 [11.3, 13.0] |
| | rate01 | 56.9 [56.0, 58.0] | 69.3 [69.0, 69.3] | 13.6 [13.1, 14.1] | 13.2 [12.5, 14.1] |
| | step05r | 54.6 [52.9, 55.1] | 70.3 [69.0, 71.1] | 12.9 [12.9, 13.5] | 11.3 [11.1, 11.5] |
| | step025r | 55.2 [52.5, 55.8] | 70.6 [69.4, 71.5] | 12.5 [12.7, 13.4] | 11.1 [10.8, 11.4] |
| 45% | base | 59.6 [57.8, 62.0] | 67.8 [65.0, 70.0] | 13.5 [13.2, 14.4] | 12.5 [11.0, 13.9] |
| | rate01 | 60.9 [60.0, 63.0] | 67.3 [66.0, 68.0] | 13.4 [12.2, 14.3] | 13.0 [12.9, 13.6] |
| | step05r | 59.6 [56.5, 62.0] | 68.1 [65.9, 70.0] | 13.0 [12.9, 13.8] | 11.9 [10.3, 14.8] |
| | step025r | 60.1 [56.7, 63.3] | 68.4 [66.8, 70.6] | 12.7 [11.8, 13.8] | 11.7 [10.1, 14.1] |
| 50% | base | 64.4 [64.0, 65.3] | 64.3 [63.0, 65.3] | 13.4 [13.1, 14.3] | 13.3 [12.7, 14.4] |
| | rate01 | 64.4 [63.0, 65.0] | 64.2 [63.8, 65.0] | 13.5 [12.6, 14.3] | 13.3 [12.7, 14.0] |
| | step05r | 64.3 [63.5, 66.0] | 65.1 [63.5, 66.1] | 12.9 [10.6, 14.3] | 12.5 [10.4, 14.2] |
| | step025r | 64.5 [63.2, 66.3] | 65.4 [64.2, 66.8] | 12.5 [10.6, 14.1] | 12.0 [10.2, 14.2] |

(The `step05`, `step025`, `cap7` and `cap10` rows from round 1 remain in `summary.txt` for comparison.)

**Reading the table.** Against the base grid, `step05r` and `step025r` move the figure by amounts comparable to the
seed-to-seed band at every share from 15% up; `step025r` is a little further from the base than `step05r` (as expected
for a dynamics whose step size has changed by more), and its 30% row shows the only visible drift in the middle of the
figure (firm 1's pay $49 vs $46, profit $12.8 vs $14.6/hr, with a wider band). The rate-only refinement (`rate01`) sits
on top of the base at every share. The low-share rows (5–10%) are where the discretisation genuinely matters: the small
firm's outcome there is a near-exit on every grid and its exact pay/profit depends on the grid.

**Optional regret check on the jointly refined grid.** Price-deviation regret (Section 3.1's first row) was recomputed
on the 320 `step05r` outcomes with the window and stride doubled in grid steps (±20 steps = ±$10, stride 10 steps =
$5, i.e. the same dollar range as the base sweep; 67 candidates per firm-outcome instead of 47; 16.9 CPU-hours, 0
failed solves). Side by side:

| price-deviation regret, sequential | base grid ($1 / 0.02) | `step05r` ($0.50 / 0.01) |
|---|---|---|
| zero-regret share | 12.7% | 7.2% |
| median / p90 / max ($/hr) | 0.79 / 2.39 / 6.10 | 0.76 / 3.33 / 6.44 |
| relative median / p90 (base profit > $1/hr) | 5.3% / 21% | 5.5% / 29% |
| share with regret ≤ $0.50 / ≤ $1 / > $2 | 41% / 59% / 15% | 38% / 59% / 20% |
| arg-best price deviation: median / p90 distance ($) | 4 / 10 | 5 / 13 |
| share of arg-best more than $1 away / price cuts | 79% / 75% | 85% / 81% |
| self-continuation gap, median \|self − reported\| ($/hr) | 0.49 | 0.23 |

The finer-grid outcomes are not closer to grid best responses than the base ones: the median residual is the same
($0.76 vs $0.79/hr, about 5% of profit for both), the upper tail is somewhat heavier (p90 $3.3 vs $2.4/hr, 20% vs 15%
of rows above $2/hr), and the best deviation sits a little further away in dollars (median $5 vs $4). What does halve is
the self-continuation gap (median $0.23 vs $0.49/hr), as expected when the lower-stage dynamics take half-size steps.
So the residuals of Section 3.1 are a property of the local ±1 search with cycle selection, not of the coarseness of the
$1 grid; refining the grid changes which non-equilibrium point the procedure stops at, not how far from a best response
it stops. (`regret_sens-step05r` section of `summary.txt`.)

**One caveat for the paper.** The refined-grid runs use larger iteration budgets than the base run (×16 / ×64 on
`max_iter` and `max_simulations`); 10 (`rate01`) and 17 (`step025r`) of the 320 converged solves needed more than the
base budget of 500,000 queueing solves (none more than 2,000,000). Under the base harness rule those seeds would have
been replaced by other seeds rather than solved to completion; the effect on the means is likely small (the affected
rows are concentrated at the 5–10% shares) but was not quantified.

### 3.5 Figures re-plotted from the time average over the terminal cycle (Task 3, 2026-10-07)

**What changed visibly.** Nothing that a reader would notice. Figure 4 (`order_of_events`): the largest change in any
plotted mean is $1.19 in firm 1's pay at the 5% share (32.8 → 31.6) and $1.12/hr in the simultaneous-timing profits at
the 40/60 shares (13.8 → 12.7); all other means move by less than $1, and the p25–p75 band edges by at most $2.9 (the
simultaneous pay band at 40% share). The two timing panels keep their shapes and crossings. `supply_sensitivity`: means
move by at most $0.85 (firm 1's pay at α = 0.16, 25/75 panel) and $0.38/hr in profit; band edges by at most $1.5.
`dedicated_demand`: means by at most $1.06 (firm 1's pay at 10% dedicated demand, 25/75 panel) and $0.74/hr in profit;
band edges by at most $2.0. `dedicated_supply`: means by at most $0.50 (pay) and $0.16/hr (profit); band edges by at
most $1.5. In the comparison overlays (`results/figures_timeavg/compare_<exp>.pdf`, dashed = most favorable state,
solid = time average) the two versions of every series lie within the line width of each other almost everywhere; the
only places where the dashed line separates visibly are the 35–65% shares of the simultaneous Figure-4 panels and the
10% point of the 25/75 `dedicated_demand` panel, each by about $1. The full table of changes per series (mean and both
band edges, with the x position of the largest change) is `results/figures_timeavg/CHANGES.txt`.

**Cycle recovery for the two remaining experiments** (`cycles dedicated_demand`: 31 configurations × 32 seeds = 992
sequential outcomes, 43 seeds used because of the replacement seeds; `cycles dedicated_supply`: 37 × 32 = 1,184; both in
under 7 minutes on 8 cores):

| | price-stage cycle length | fixed points | reported state on recovered cycle | engine's rule picks the reported state |
|---|---|---|---|---|
| `dedicated_demand` | L = 2: 94.0%, L = 4: 4.1%, L = 6: 1.3%, L = 8–10: 0.6% | 0% | 926 / 992 (93.3%) | 90.0% |
| `dedicated_supply` | L = 2: 92.8%, L = 4: 6.2%, L = 6–8: 0.3% | 0.8% (9 rows) | 1,182 / 1,184 (99.8%) | 99.1% |

Pay-stage cycles: `dedicated_demand` L = 2 in 82%, L = 4 in 11%, up to L = 28; `dedicated_supply` L = 2 in 99%; the
reported state is on the pay cycle in 100% of rows in both. The 9 fixed points are configurations where firm 1 has
neither a dedicated nor a shared worker pool and firm 2's price happens not to move; the 32 rows per experiment with an
inactive firm are otherwise ordinary two-cycles of the active firm (the map leaves the inactive firm's price and pay
untouched, as the engine does).

**Exception: 68 rows whose reported state is not on the recovered cycle** (66 in `dedicated_demand`, 6.7% of its
price-stage rows, in 15 of 31 configurations and 28 seeds, 30 of them in firm-symmetric configurations; 2 in
`dedicated_supply`, both at 85% dedicated supply split 50/50). In every such case the recovered cycle is a nearby
two-cycle (one L = 4, two L = 6) that the reported state leads into after one step, one price step away: |time
average − reported| has median $1.50 and max $6.50 for price, median $1.50 and max $7 for pay, median $1.28/hr and max
$5.69/hr for profit, against medians of $0.50, $0.50 and $0.57/hr for the on-cycle rows. These are not near-ties: the
smallest profit gap in any ±1 comparison along the diverging paths is $0.05/hr. The cause is reproducibility, not the
cycle map: re-solving the same configuration from the same seed's `initial_guess` with the engine's own `sequentialNE`
on this machine does **not** reproduce the saved outcome for these rows (e.g. seed 10 at 20/20/60 demand ends at prices
96/98 instead of the saved 93/93; seed 1 at 20/20/360 ends with the saved prices and pays but a different flexible-worker
rate, 67 vs 69), while it reproduces an on-cycle row bit for bit. The saved results were produced on another Mac with
Julia 1.12.6; configurations with a competitive demand pool (and, for `dedicated_supply`, dedicated worker pools) use the
engine's sparse-LU path for the stationary distribution, whose last-bit floating-point differences across machines are
enough to resolve exact or near-exact ties in the engine's selection rules (mirror states in the symmetric
configurations, for instance) differently and send the path to a neighbouring cycle. For the three other experiments the
same test found 1 such row in 1,024. The time-average figures use the recovered cycle for these 68 rows; their aggregate
effect is contained in the `dedicated_demand` changes above (the largest change in that figure, $1.06 in pay at 10%
dedicated demand, 25/75 panel, is where several of them sit). Worth telling Sébastien: the saved `dedicated_demand`
results are not bit-reproducible on a different machine for about 7% of rows.

**Procedure.** `build_figures_timeavg.jl` (next to `build_figures.jl`, which is unchanged) rebuilds the four figures
with the same selection (converged rows, first 32 seeds per configuration), the Figure-4 symmetry reflection with firm
swap, the same layout, fonts, colors and axis rules, and one change of metric: each reported row's profit and pay per
firm are the time averages over the row's terminal cycle from `diagnostics/cycles_<exp>.jld2` (price-stage cycle for
sequential rows, simultaneous firm-map cycle for simultaneous rows; profits recomputed from each cycle state's decisions
with `simulation`). Records are matched by (params, sequential, seed) and a missing record is an error (none was
missing: 640 + 704 + 992 + 1,184 records for as many reported rows). A firm's pay enters the seed average when its
time-averaged profit is positive. Figure 4's panel titles read "Sequential timing" / "Simultaneous timing"; no other
title contained "equilibrium". Output in `figure_regeneration/figures_timeavg/` (new folder; the repo's `figures/` is
untouched). The paper's copies `minor revision/Figures/simulations/{order_of_events, supply_sensitivity,
dedicated_demand, dedicated_supply}.pdf` were first copied to `Figures/simulations/most_favorable_state/` (verified
identical to the repo's `figures/`) and then replaced by the new PDFs; `quality.pdf` was left alone.

### 3.6 Time averages over the terminal cycle for the sensitivity variants, Table 9 (Task 4, 2026-10-14)

**Verdict, one sentence per variant (appendix F.4's statements, re-checked on the time-average summary).**
- `cap7`: every statement holds; the smaller firm pays less at every share below 50%, its pay at shares of 15% and
  above is within $1.7 of the base (largest: $1.68 at 40%, 54.2 → 55.9), and the low-share cells move the way the
  paragraph says (5%: pay $31.6 → 16.8, profit $0.43 → 0.27/hr).
- `cap10`: every statement holds; pay at 15% and above within $1.6 of the base (largest: $1.59 in firm 2's pay at 35%,
  71.8 → 70.2), and the cap lowers the smaller firm's profit by $1.1–4.6/hr at the shares from 10% to 35% (10%: 1.08,
  15%: 1.54, 20%: 1.96, 25%: 2.55, 30%: 4.58, 35%: 2.17), so "$1–5 per hour" remains correct (a tighter phrasing would be
  "$1–4.6").
- `step05r` ("grid /2"): every statement holds; pay at 15% and above within $1.3 of the base (largest: $1.29 in firm 1's
  pay at 30%, 46.5 → 47.8).
- `step025r` ("grid /4"): every statement holds, including the parenthetical: the single cell more than $2 from the base
  at shares of 15% and above is firm 1's pay at 30% (46.5 → 49.5, a difference of $3.03), so "the largest difference is
  $3" stays exactly as written; all other cells at 15% and above are within $1.9.

Nothing in the paragraph needs a different number. The entries of Table 9 that visibly change are all at the 5%
share: `cap 10`'s firm-1 pay (9.4 → 11.1) and the base row's firm-1 cells (pay $32.8 → 31.6, profit $0.8 → 0.4/hr, the
same change the time-average figure already shows, Section 3.5); every other entry moves by at most 1.0. Under the new
summary the base row coincides with Figure 4, so the table note's sentence saying that the base row "differs slightly"
from the figure because it uses the most favorable state would no longer apply (not edited here; nothing under
`latex/paper/` or `minor revision/*.tex` was touched).

**Cycle recovery on the four variants** (`cycles sens-cap7`, `sens-step05r`, `sens-step025r`, `sens-cap10`: the
engine's price-stage map restarted from each reported state, the lower stages run with the variant's own iteration
budget, see Procedure). Only sequential configurations exist in these variants; the 6 non-converged `cap10` rows at the
5% share are not reported rows and got no cycle.

| variant | reported rows | price-stage cycle length | reported state on recovered cycle | engine's rule picks the reported state | pay-stage cycle length | wall time (8 processes) |
|---|---|---|---|---|---|---|
| `cap7` | 320 | L = 2: 84.7%, L = 4: 11.6%, L = 6: 2.8%, L = 8: 0.9% | 320 / 320 (100%) | 100% | L = 2: 77.8%, L = 4: 18.1%, up to L = 14 | 7 min (66–111 s per seed, 48 CPU-min) |
| `step05r` | 320 | L = 2: 87.5%, L = 4: 11.2%, L = 6: 0.9%, L = 8: 0.3% | 320 / 320 (100%) | 100% | L = 2: 71.9%, L = 4: 10.6%, L = 6–12: 16.6%, up to L = 30 | 2 min (20–28 s per seed) |
| `step025r` | 320 | L = 2: 85.9%, L = 4: 8.8%, L = 6: 4.1%, L = 8–10: 1.2% | 320 / 320 (100%) | 100% | L = 2: 70.3%, L = 4: 13.8%, L = 6–10: 13.8%, up to L = 20 | 2.5 min (18–42 s per seed) |
| `cap10` | 314 | L = 2: 85.7%, L = 4: 10.8%, L = 6: 1.3%, L = 8: 1.9%, L = 10: 0.3% | 314 / 314 (100%) | 100% | L = 2: 75.8%, L = 4: 17.2%, L = 6–10: 6.7%, one L = 20 | 103 min (14–36 min per seed, 13.1 CPU-hours) |

No exception: the reported state lies on the recovered cycle for every reported row of every variant, the engine's
most-favorable rule returns the reported state in every row (in the base experiment it did so in 99.1%), no fixed
points, no failed recovery, no unit timed out (`DIAG_TIMEOUT` was 43,200 s). As in the base experiment, most
price-stage cycles (95.9–97.2% depending on the variant) carry stale engine profits (Section 3.2), which does not affect
the time averages (profits are recomputed with `simulation` at every cycle state). Per row, |time average − reported|
in pay has median $0.50 (`cap7`, `step05r`), $0.25 (`step025r`) and $0.75 (`cap10`), with maxima of $4.75, $4.00, $3.00
and $4.50; in profit the medians are $0.13–0.39/hr and the maxima $1.6–2.7/hr. The cycle ranges (max − min over the
cycle) have medians of $0.5–1.5 in pay and $0.3–0.8/hr in profit. Full numbers in the four `cycles: sens-<variant>`
sections of `summary.txt`.

**The time-average sensitivity table, verbatim from `summary.txt`** (TABLE 2 of the sensitivity section; the
reported-state table it sits next to is TABLE 1 and is unchanged from Section 3.3 / 3.4 apart from its label). Only the
four variants used in Table 9 have cycle records, so the table has exactly Table 9's rows; the three other variants of
Section 3.4 (`step05`, `step025`, `rate01`) are listed as omitted.

```
   -- TABLE 2, time average: each outcome (base included) summarized by the time average over its terminal price-stage cycle (profit = time average of the profit recomputed with `simulation` at each cycle state; pay = time average of the firm's pay); a firm's pay enters only when its time-averaged profit is positive (build_figures_timeavg.jl rule) --
   cycle records: diagnostics/cycles_order_of_events.jld2 (base), diagnostics/cycles_sens-<variant>.jld2 (variants), matched by (params, sequential, seed); a reported row without a record is an error
   variants without cycle records, omitted from this table (run `./run_diagnostics.sh cycles sens-<variant>` and summarize): step05, step025, rate01
   firm1_demand | variant | n/attempted | pay1 | pay2 | profit1 | profit2
    0.20 | base    | 32/32 |  31.62 [ 30.50, 31.56] |  67.75 [ 66.38, 68.50] |   0.43 [  0.21,  0.25] |  34.24 [ 34.46, 34.60]
    0.20 | cap7    | 32/32 |  16.77 [ 15.00, 17.00] |  66.11 [ 66.00, 66.00] |   0.27 [  0.22,  0.23] |  32.07 [ 31.99, 32.19]
    0.20 | cap10   | 26/32 |  11.08 [ 10.50, 10.50] |  65.27 [ 64.50, 64.50] |   0.23 [  0.22,  0.22] |  30.78 [ 31.02, 31.02]
    0.20 | step05r | 32/32 |  25.43 [ 25.00, 25.75] |  66.57 [ 66.50, 66.75] |   0.10 [  0.10,  0.10] |  34.42 [ 34.14, 34.63]
    0.20 | step025r | 32/32 |  22.86 [ 21.88, 23.75] |  68.43 [ 67.88, 69.12] |   0.05 [  0.05,  0.06] |  33.02 [ 32.20, 33.88]
    0.40 | base    | 32/32 |  41.62 [ 40.88, 42.50] |  67.97 [ 67.00, 69.00] |   5.08 [  4.62,  5.47] |  27.52 [ 27.10, 27.92]
    0.40 | cap7    | 32/32 |  42.36 [ 41.50, 43.50] |  67.59 [ 67.00, 68.50] |   4.33 [  4.04,  4.57] |  27.07 [ 26.80, 27.20]
    0.40 | cap10   | 32/32 |  40.88 [ 39.50, 42.12] |  68.14 [ 67.50, 69.00] |   4.00 [  3.79,  4.26] |  26.85 [ 26.77, 27.03]
    0.40 | step05r | 32/32 |  38.07 [ 31.25, 41.88] |  68.41 [ 67.88, 69.25] |   3.13 [  0.26,  4.81] |  27.81 [ 26.89, 28.90]
    0.40 | step025r | 32/32 |  30.54 [ 26.50, 31.25] |  69.75 [ 68.75, 70.00] |   1.22 [  0.09,  0.95] |  27.20 [ 26.33, 28.70]
    0.60 | base    | 32/32 |  44.05 [ 43.50, 45.00] |  68.68 [ 68.50, 69.00] |   8.41 [  8.30,  8.93] |  24.73 [ 24.28, 25.12]
    0.60 | cap7    | 32/32 |  45.42 [ 44.38, 46.50] |  67.98 [ 67.50, 68.50] |   7.20 [  6.88,  7.56] |  24.01 [ 23.91, 24.18]
    0.60 | cap10   | 32/32 |  44.60 [ 44.38, 45.00] |  68.48 [ 68.00, 68.81] |   6.87 [  6.65,  7.12] |  23.81 [ 23.64, 24.15]
    0.60 | step05r | 32/32 |  44.50 [ 43.94, 45.38] |  69.14 [ 68.25, 69.81] |   7.57 [  6.88,  8.42] |  23.63 [ 22.65, 24.54]
    0.60 | step025r | 32/32 |  44.40 [ 43.12, 45.28] |  69.70 [ 69.06, 70.16] |   7.55 [  6.93,  8.07] |  23.13 [ 21.98, 24.03]
    0.80 | base    | 32/32 |  45.22 [ 44.50, 45.50] |  69.22 [ 68.50, 69.50] |  10.78 [ 10.61, 11.39] |  21.50 [ 21.54, 21.90]
    0.80 | cap7    | 32/32 |  45.84 [ 45.00, 47.00] |  69.25 [ 68.25, 70.00] |   9.44 [  9.05,  9.86] |  20.54 [ 20.32, 21.08]
    0.80 | cap10   | 32/32 |  45.78 [ 45.00, 47.12] |  68.72 [ 67.50, 69.81] |   8.82 [  8.56,  9.00] |  20.84 [ 20.55, 21.15]
    0.80 | step05r | 32/32 |  46.17 [ 45.50, 46.53] |  70.04 [ 69.09, 70.81] |  10.16 [  9.51, 10.74] |  20.50 [ 19.50, 21.47]
    0.80 | step025r | 32/32 |  46.38 [ 45.59, 47.19] |  70.93 [ 69.88, 71.44] |   9.94 [  9.78, 10.37] |  19.91 [ 19.14, 21.11]
    1.00 | base    | 32/32 |  45.97 [ 45.88, 46.00] |  70.31 [ 69.00, 71.00] |  13.10 [ 12.94, 13.79] |  18.44 [ 18.47, 19.10]
    1.00 | cap7    | 32/32 |  45.06 [ 44.94, 45.25] |  69.80 [ 68.00, 72.00] |  11.01 [ 10.44, 11.72] |  17.18 [ 16.92, 17.93]
    1.00 | cap10   | 32/32 |  44.74 [ 44.00, 45.50] |  69.92 [ 69.00, 70.62] |  10.55 [ 10.27, 11.04] |  17.66 [ 17.50, 18.05]
    1.00 | step05r | 32/32 |  46.41 [ 46.00, 46.75] |  70.84 [ 69.84, 71.75] |  12.09 [ 11.76, 12.62] |  17.61 [ 17.20, 18.07]
    1.00 | step025r | 32/32 |  47.27 [ 46.62, 47.62] |  71.59 [ 70.70, 72.16] |  11.69 [ 11.32, 12.18] |  16.93 [ 16.26, 17.77]
    1.20 | base    | 32/32 |  46.52 [ 45.50, 47.00] |  70.88 [ 70.00, 71.50] |  14.23 [ 14.05, 14.81] |  15.16 [ 14.86, 15.67]
    1.20 | cap7    | 32/32 |  48.08 [ 47.00, 48.00] |  71.22 [ 69.75, 73.00] |  12.80 [ 12.29, 13.39] |  14.17 [ 13.98, 14.14]
    1.20 | cap10   | 32/32 |  47.13 [ 43.66, 52.00] |  70.10 [ 68.50, 71.50] |   9.65 [  8.79, 11.23] |  14.33 [ 13.45, 15.10]
    1.20 | step05r | 32/32 |  47.81 [ 46.00, 47.25] |  71.78 [ 70.75, 72.75] |  13.74 [ 13.55, 14.69] |  14.75 [ 14.65, 15.22]
    1.20 | step025r | 32/32 |  49.55 [ 46.88, 51.12] |  72.20 [ 71.47, 72.88] |  12.53 [ 11.45, 13.71] |  14.23 [ 13.66, 14.93]
    1.40 | base    | 32/32 |  50.88 [ 49.00, 52.50] |  71.78 [ 70.88, 73.00] |  13.13 [ 12.45, 13.67] |  12.22 [ 11.60, 12.75]
    1.40 | cap7    | 32/32 |  51.69 [ 49.50, 53.00] |  70.59 [ 69.50, 71.50] |  12.30 [ 11.99, 12.89] |  11.97 [ 11.76, 12.42]
    1.40 | cap10   | 32/32 |  50.25 [ 47.25, 52.62] |  70.19 [ 68.75, 71.00] |  10.96 [ 10.93, 12.02] |  12.44 [ 11.68, 12.96]
    1.40 | step05r | 32/32 |  52.08 [ 50.22, 53.50] |  71.15 [ 70.00, 71.50] |  12.50 [ 11.89, 12.93] |  12.23 [ 12.03, 12.76]
    1.40 | step025r | 32/32 |  52.25 [ 49.28, 53.44] |  71.22 [ 69.59, 72.03] |  12.39 [ 11.91, 12.85] |  12.27 [ 11.72, 12.78]
    1.60 | base    | 32/32 |  54.17 [ 51.56, 56.50] |  70.17 [ 69.00, 72.00] |  13.48 [ 13.05, 14.26] |  11.72 [ 11.15, 12.11]
    1.60 | cap7    | 32/32 |  55.85 [ 53.00, 56.94] |  69.39 [ 68.00, 71.06] |  12.36 [ 12.18, 13.23] |  11.46 [ 10.88, 11.43]
    1.60 | cap10   | 32/32 |  53.39 [ 52.00, 56.00] |  70.63 [ 69.88, 71.75] |  12.40 [ 12.22, 13.39] |  11.34 [ 10.48, 11.18]
    1.60 | step05r | 32/32 |  54.46 [ 52.00, 55.31] |  70.36 [ 69.09, 71.31] |  12.86 [ 12.88, 13.59] |  11.18 [ 10.84, 11.48]
    1.60 | step025r | 32/32 |  55.04 [ 52.17, 55.53] |  70.67 [ 69.59, 71.62] |  12.55 [ 12.67, 13.36] |  11.02 [ 10.65, 11.40]
    1.80 | base    | 32/32 |  59.09 [ 56.69, 61.31] |  67.88 [ 66.50, 69.06] |  13.42 [ 13.34, 14.01] |  12.12 [ 10.65, 14.34]
    1.80 | cap7    | 32/32 |  58.73 [ 55.50, 60.62] |  67.75 [ 66.25, 69.00] |  12.62 [ 11.91, 13.74] |  11.14 [ 10.21, 11.42]
    1.80 | cap10   | 32/32 |  58.43 [ 56.25, 60.25] |  67.67 [ 66.50, 69.50] |  12.51 [ 11.56, 13.52] |  10.99 [ 10.15, 11.00]
    1.80 | step05r | 32/32 |  59.75 [ 56.25, 62.50] |  68.08 [ 65.75, 70.06] |  12.90 [ 12.85, 13.72] |  11.86 [ 10.26, 14.94]
    1.80 | step025r | 32/32 |  60.05 [ 56.59, 63.44] |  68.31 [ 66.66, 70.44] |  12.65 [ 11.80, 13.77] |  11.66 [ 10.12, 14.09]
    2.00 | base    | 32/32 |  64.15 [ 63.50, 65.00] |  64.34 [ 63.50, 65.62] |  13.30 [ 13.12, 14.20] |  13.09 [ 11.59, 14.28]
    2.00 | cap7    | 32/32 |  62.84 [ 61.50, 64.00] |  64.30 [ 63.00, 65.56] |  12.83 [ 11.91, 13.90] |  12.19 [ 10.09, 13.90]
    2.00 | cap10   | 32/32 |  63.23 [ 62.00, 65.00] |  64.12 [ 63.00, 65.50] |  12.37 [ 10.99, 13.63] |  11.90 [ 10.87, 13.39]
    2.00 | step05r | 32/32 |  64.31 [ 63.25, 65.88] |  64.93 [ 63.25, 66.19] |  12.88 [ 10.82, 14.22] |  12.57 [ 10.48, 14.29]
    2.00 | step025r | 32/32 |  64.43 [ 63.16, 66.16] |  65.29 [ 64.06, 66.47] |  12.42 [ 10.48, 14.08] |  11.93 [ 10.22, 14.24]

   -- largest absolute difference between TABLE 2 (time average) and TABLE 1 (reported state), per variant, over the ten shares; 'any cell' = mean, p25 or p75 of any column, 'means' = the mean columns only (the figure's values) --
   base    : pay    any cell 2.500 (share 1.60, pay1 p75: 59.00 -> 56.50)         means 1.188 (share 0.20, pay1 mean: 32.81 -> 31.62)
           : profit any cell 1.132 (share 2.00, profit2 p25: 12.73 -> 11.59)      means 0.426 (share 1.60, profit2 mean: 12.15 -> 11.72)
   cap7    : pay    any cell 1.375 (share 1.80, pay1 p75: 62.00 -> 60.62)         means 0.516 (share 1.00, pay2 mean: 70.31 -> 69.80)
           : profit any cell 1.156 (share 1.20, profit1 p25: 13.44 -> 12.29)      means 0.621 (share 1.00, profit1 mean: 11.63 -> 11.01)
   cap10   : pay    any cell 2.500 (share 0.20, pay1 p25: 8.00 -> 10.50)          means 1.692 (share 0.20, pay1 mean: 9.38 -> 11.08)
           : profit any cell 1.129 (share 1.80, profit2 p75: 12.12 -> 11.00)      means 0.350 (share 0.80, profit2 mean: 21.19 -> 20.84)
   step05r : pay    any cell 0.875 (share 1.60, pay1 p25: 52.88 -> 52.00)         means 0.441 (share 0.60, pay1 mean: 44.94 -> 44.50)
           : profit any cell 0.734 (share 0.60, profit1 p25: 7.62 -> 6.88)        means 0.316 (share 0.60, profit1 mean: 7.89 -> 7.57)
   step025r: pay    any cell 0.625 (share 0.20, pay1 p25: 21.25 -> 21.88)         means 0.346 (share 1.20, pay2 mean: 72.55 -> 72.20)
           : profit any cell 0.694 (share 0.80, profit2 p25: 19.84 -> 19.14)      means 0.251 (share 1.20, profit1 mean: 12.78 -> 12.53)
   base check: the base rows of TABLE 2 reproduce the per-configuration time-average means of the `cycles: order_of_events` section (price stage) in all 40 cells (max |difference| 0)
```

The base rows reproduce the per-configuration time-average means printed in the `cycles: order_of_events` section
(the "figure summary ... vs time-average summary" block, price stage) in all 40 cells, as the last line says; the
check is computed in `summarize`. (The two could have differed in pay cells, because that section filters a firm's
time-averaged pay by the sign of its *reported-state* profit while the new table, like `build_figures_timeavg.jl`,
filters by the sign of its *time-averaged* profit; no seed of the base experiment has profits of different signs
under the two summaries, so they agree exactly.) The base means also equal the "TA" values of
`results/figures_timeavg/CHANGES.txt` (e.g. firm 1's pay at 5%, 31.625).

**Largest cell differences between the two summaries** (the lines under TABLE 2): over the mean, p25 and p75 of any
column at any share, pay moves by at most $2.50 (`base`, firm 1's pay p75 at 40%: 59.00 → 56.50; `cap10`, firm 1's pay
p25 at 5%: 8.00 → 10.50), $1.38 (`cap7`), $0.88 (`step05r`), $0.63 (`step025r`); profit by at most $1.13–1.16/hr
(`base`, `cap7`, `cap10`, all in a band edge) and $0.69–0.73/hr (`step05r`, `step025r`). Restricted to the means, which
are what Table 9 prints: pay by at most $1.19 (`base`, 5%), $1.69 (`cap10`, firm 1's pay at 5%: 9.38 → 11.08), $0.52
(`cap7`), $0.44 (`step05r`), $0.35 (`step025r`); profit by at most $0.43/hr (`base`), $0.62/hr (`cap7`, firm 1 at 25%),
$0.35/hr (`cap10`), $0.32/hr (`step05r`), $0.25/hr (`step025r`). The refined grids move least, as their cycles are
narrower in dollars (the price step is $0.50 / $0.25); the two cap variants and the base move by about one price step
in the low-share pay cells.

**Table 9 recomputed from the time averages** (means of TABLE 2 rounded to one decimal, in the layout of
`tab:sim-sensitivity`; the `cap 10` entry at 5% uses 26 starts, as before):

| | | 5% | 10% | 15% | 20% | 25% | 30% | 35% | 40% | 45% | 50% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pay, firm 1 | base | 31.6 | 41.6 | 44.0 | 45.2 | 46.0 | 46.5 | 50.9 | 54.2 | 59.1 | 64.2 |
| | cap 7 | 16.8 | 42.4 | 45.4 | 45.8 | 45.1 | 48.1 | 51.7 | 55.9 | 58.7 | 62.8 |
| | cap 10 | 11.1 | 40.9 | 44.6 | 45.8 | 44.7 | 47.1 | 50.2 | 53.4 | 58.4 | 63.2 |
| | grid /2 | 25.4 | 38.1 | 44.5 | 46.2 | 46.4 | 47.8 | 52.1 | 54.5 | 59.8 | 64.3 |
| | grid /4 | 22.9 | 30.5 | 44.4 | 46.4 | 47.3 | 49.5 | 52.2 | 55.0 | 60.0 | 64.4 |
| Pay, firm 2 | base | 67.8 | 68.0 | 68.7 | 69.2 | 70.3 | 70.9 | 71.8 | 70.2 | 67.9 | 64.3 |
| | cap 7 | 66.1 | 67.6 | 68.0 | 69.2 | 69.8 | 71.2 | 70.6 | 69.4 | 67.8 | 64.3 |
| | cap 10 | 65.3 | 68.1 | 68.5 | 68.7 | 69.9 | 70.1 | 70.2 | 70.6 | 67.7 | 64.1 |
| | grid /2 | 66.6 | 68.4 | 69.1 | 70.0 | 70.8 | 71.8 | 71.2 | 70.4 | 68.1 | 64.9 |
| | grid /4 | 68.4 | 69.8 | 69.7 | 70.9 | 71.6 | 72.2 | 71.2 | 70.7 | 68.3 | 65.3 |
| Profit, firm 1 | base | 0.4 | 5.1 | 8.4 | 10.8 | 13.1 | 14.2 | 13.1 | 13.5 | 13.4 | 13.3 |
| | cap 7 | 0.3 | 4.3 | 7.2 | 9.4 | 11.0 | 12.8 | 12.3 | 12.4 | 12.6 | 12.8 |
| | cap 10 | 0.2 | 4.0 | 6.9 | 8.8 | 10.6 | 9.7 | 11.0 | 12.4 | 12.5 | 12.4 |
| | grid /2 | 0.1 | 3.1 | 7.6 | 10.2 | 12.1 | 13.7 | 12.5 | 12.9 | 12.9 | 12.9 |
| | grid /4 | 0.1 | 1.2 | 7.5 | 9.9 | 11.7 | 12.5 | 12.4 | 12.6 | 12.7 | 12.4 |
| Profit, firm 2 | base | 34.2 | 27.5 | 24.7 | 21.5 | 18.4 | 15.2 | 12.2 | 11.7 | 12.1 | 13.1 |
| | cap 7 | 32.1 | 27.1 | 24.0 | 20.5 | 17.2 | 14.2 | 12.0 | 11.5 | 11.1 | 12.2 |
| | cap 10 | 30.8 | 26.9 | 23.8 | 20.8 | 17.7 | 14.3 | 12.4 | 11.3 | 11.0 | 11.9 |
| | grid /2 | 34.4 | 27.8 | 23.6 | 20.5 | 17.6 | 14.8 | 12.2 | 11.2 | 11.9 | 12.6 |
| | grid /4 | 33.0 | 27.2 | 23.1 | 19.9 | 16.9 | 14.2 | 12.3 | 11.0 | 11.7 | 11.9 |

Against the current Table 9 (most favorable state), 171 of the 200 entries change by at most 0.3, 190 by at most 0.5,
and only two by more than 1.0: `cap 10`, firm 1's pay at 5% (9.4 → 11.1) and the base row's firm-1 pay at 5%
(32.8 → 31.6); next come the base row's firm-1 pay at 40% (55.2 → 54.2) and at 15% (44.8 → 44.0), then changes of 0.7
or less (among them the base firm-1 profit at 5%, 0.8 → 0.4, and firm 2's profit at 5% under `cap 7`, 32.7 → 32.1).

**Procedure and code changes** (`diagnostics.jl`; the previous version is kept as
`minor revision/simulation_diagnostics/diagnostics_task3.jl`).
- Pre-launch check requested by the task: the cycle map (`price_step`, `pay_step`) takes `p = r.params` from each
  reported row, so a variant's `price_step`, `rate_step` and `max_queue` flow into every lower-stage solve, and the
  ±1 moves are in grid units; nothing read `BASE_P`. What did not match the engine was the lower-stage iteration cap:
  the engine passes its own `max_iter` down to `sequentialNE_demand` / `_pay` / `_supply`, and the refined-grid
  variants were solved with `max_iter` 16,000 (`step05r`) and 64,000 (`step025r`) rather than the 1,000 the cycle map
  used. The map now uses `1000 × variant_scale(variant)` on `cycles sens-<variant>` (`MAXIT_REF`), i.e. the budget the
  sensitivity solve used, unless `DIAG_MAXITER` is set; on the base experiments nothing changes (×1). A cap that is
  never hit gives identical dynamics, so this only matters for rows where a lower stage needs more than 1,000
  iterations; none of the recoveries failed, before or after.
- `summarize`: the cycles loop also picks up `cycles_sens-<variant>_seed*.jld2` units (first-32-converged rule applied
  against the variant's own merged results) and writes `diagnostics/cycles_sens-<variant>.jld2`; the sensitivity
  section prints TABLE 1 (reported state, as before) and TABLE 2 (time average, every row including `base`, cycle
  records matched by (params, sequential, seed); a reported row without a record stops `summarize` with the row's
  identity; a variant with no cycle file at all is listed as omitted), then per variant the largest absolute
  difference between the two tables in any cell and in the means only, and the base check against the
  `cycles: order_of_events` section.
- Runs: `diagnostics/run_chain4.sh` (cap7 11:01 → 11:08, step05r → 11:10, step025r → 11:12, cap10 → 12:56, summarize
  → 12:57 on 2026-10-14; `DIAG_P=8`, `DIAG_TIMEOUT=43200`, Julia 1.12.7, under `caffeinate -i`). The smoke test
  (one `cap7` seed and a `summarize`) went to the scratch folder, not to `diagnostics/units/`.

## 4. Surprises and things to double-check

1. The lower-stage continuations are not reproducible from their own output (Section 3.1). This makes the "regret at
   the reported state" a two-number object. The response letter should say which base it quotes.
2. The regrets are large relative to the "two-cycle bracketing the best response" picture: the best price deviation is
   typically 4 steps away and 92% of them lie inside the ±10 window the engine's ±1 comparisons could in principle
   walk to. Either the continuation profit is not single-peaked in the firm's own price (so a ±1 walk gets stuck), or
   the cycle-selection rules underneath change the landscape between iterations. Worth a look at a few of the top
   cases listed above before deciding how to describe the procedure.
3. Price-step sensitivity (Section 3.3) looked like the result most likely to draw a follow-up from the referee. The
   round-2 joint refinement (Section 3.4) shows it was an artefact of refining the price grid while leaving the rate
   grid coarse: with both grids refined the base pattern is reproduced from the 15% share up, and only the small firm's
   near-exit at 5–10% share depends on the grid. Cap sensitivity is benign for pay; cap 10 lowers firm 1's profit by
   $1–5/hr at shares of 10–30%.
4. The saved `dedicated_demand` results are not bit-reproducible on this machine for about 7% of rows (Section 3.5):
   re-solving from the same seed lands on a neighbouring cycle. Last-bit floating-point differences in the sparse
   stationary-distribution solve across machines are the likely cause. The figures are unaffected beyond the ~$1 changes
   reported, but it means a reader regenerating the results on other hardware will not get the identical per-seed
   outcomes.
5. `sequentialNE_demand` returns a stale `sim` for the state it selects (Section 3.2). It affects only the ranking
   inside the price stage's most-favorable rule, not the reported numbers, and the rule still returns the reported
   state in all non-tied cases. Flagging it for Sébastien; not fixed (engine is read-only for this task).
6. The simultaneous solver's else-branch for an inactive firm assigns `bestfirm1`/`bestfirm2` (typo for `best_firm1`),
   which would throw `UndefVarError` if a firm ever had no market under simultaneous timing. Not reachable in the
   four experiments (only `order_of_events` has simultaneous configurations and both firms are active). Flagging only.
7. Budget scaling for `step025` had to be ×16 rather than ×4 (Section 1); with ×4, 13 of the 320 solves would have been
   recorded as non-converged.
8. No unit timed out and no continuation solve threw. The only non-converged solves are 6 of 32 cap-10 solves at the
   5% share (budget of 500,000 queueing solves exhausted, the same cap the harness uses); all other variants converged
   on every seed.

## 5. Run status: everything launched has finished

- `sensitivity cap10`: 2026-10-06 23:51 → 2026-10-07 08:48 (seeds 1–25 and 28 through the chain on the 8 performance
  cores, seeds 26–27 and 29–32 through two side runs on the 4 efficiency cores; ~15 min per configuration with 8–12
  solves in parallel, ~2.3 h per seed, 76 CPU-hours in total). The 1,331-state chain makes every queueing solve much
  heavier than at cap 5. 6 of 320 solves non-converged (Section 3.3).
- `cycles supply_sensitivity`: 00:40 → 00:49 on the efficiency cores. `regret supply_sensitivity`: 00:55 → 02:56 on
  the efficiency cores (8.1 CPU-hours).
- `dedicated_demand` and `dedicated_supply` regret and cycles sweeps were not launched (each regret sweep would be
  another ~4–8 CPU-hours; some of their configurations have an inactive firm, which the script skips). They can be run
  with `./run_diagnostics.sh cycles dedicated_demand` etc.; `summarize` picks them up automatically.
- At the time of writing the chain script was still finishing one duplicate cap-10 seed (it had started seed 26
  before the side run's unit file existed; the result is identical and overwrites the same file), after which it
  re-runs `summarize`. The summary in `results/` was produced after all 32 cap-10 units were on disk and will not
  change.
- Round 2 (`rate01`, `step05r`, `step025r`, 32 seeds each): done 2026-10-07 11:18 → 12:19 through
  `diagnostics/run_chain2.sh`; all 960 solves converged. The optional price-deviation regret sweep on the `step05r`
  outcomes (`regret sens-step05r`, window ±20 steps = ±$10, stride 10 steps = $5) is reported in Section 3.4.
- Round 3 (`cycles dedicated_demand`, `cycles dedicated_supply`, time-average figures): done 2026-10-07 15:02 → 15:11.
- Round 4 (`cycles sens-cap7`, `sens-step05r`, `sens-step025r`, `sens-cap10`, 32 seeds each; 1,274 reported rows,
  all recovered, all on-cycle): done 2026-10-14 11:01 → 12:57 through `diagnostics/run_chain4.sh`.
- Refresh the numbers at any time with `./run_diagnostics.sh summarize` from `figure_regeneration/`.

## 6. Files

- `minor revision/simulation_diagnostics/results/summary.txt` — copy of `figure_regeneration/diagnostics/summary.txt`
  (regret for `order_of_events`, `supply_sensitivity` and `sens-step05r`, cycles for all four experiments, the sensitivity table with eight variants).
- `minor revision/simulation_diagnostics/results/REPORT.md` — this file.
- `minor revision/simulation_diagnostics/{diagnostics.jl, run_diagnostics.sh, DIAGNOSTICS_README.md}` — the fixed
  scripts (overwriting the drafts); identical to the copies in `code/two-sided-platform-competition/figure_regeneration/`.
- `minor revision/simulation_diagnostics/build_figures_timeavg.jl` — the time-average figure builder (Section 3.5).
- `minor revision/simulation_diagnostics/diagnostics_task3.jl` — the round-3 version of `diagnostics.jl`, kept when the
  round-4 version (Section 3.6) overwrote the copy above.
- `figure_regeneration/diagnostics/cycles_sens-{cap7,cap10,step05r,step025r}.jld2` — merged cycle records of the four
  variants (round 4), left in the code folder with the other merged DataFrames; `summary.txt` now also has the four
  `cycles: sens-<variant>` sections and the two sensitivity tables.
- `minor revision/simulation_diagnostics/results/figures_timeavg/{compare_<exp>.pdf, CHANGES.txt}` — overlays of the
  most-favorable-state and time-average series, and the table of largest changes.
- `minor revision/Figures/simulations/{order_of_events, supply_sensitivity, dedicated_demand, dedicated_supply}.pdf` —
  now the time-average versions; the previous versions are in `Figures/simulations/most_favorable_state/`.
- Raw per-seed outputs `figure_regeneration/diagnostics/units/*.jld2` (32 + 32 + 43 + 32 cycles, 32 + 32 regret, 7 × 32
  sensitivity, 32 regret on `sens-step05r`), merged DataFrames `figure_regeneration/diagnostics/{regret_order_of_events, regret_supply_sensitivity,
  cycles_order_of_events, cycles_supply_sensitivity, sensitivity}.jld2`, logs in `figure_regeneration/diagnostics/logs/`
  — left in the code folder as instructed.
- Nothing under `latex/paper/`, `results/`, the repo's `figures/` or the engine was modified; no git commands were run in either
  repository (the clone is at the public HEAD, untouched except for the three diagnostic files and `diagnostics/`).
