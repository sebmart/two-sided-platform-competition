# Labor Cost Free-Riding in the Gig Economy: simulation code

Simulation code and results for *Labor Cost Free-Riding in the Gig Economy* by Zhen Lian, Sébastien Martin, and Garrett van Ryzin. The code, written in Julia, simulates price, pay, demand, and supply competition between two firms that draw on a shared pool of multi-apping workers (Section 5 and the simulation appendix of the paper).

## What is in this repository

| Path | What it is |
|---|---|
| `simulation_utils_logit.jl` | The engine (`module FreeridingSimulation`): the queueing model, the metrics, and the nested best-response solvers (`sequentialNE`, `simultaneousNE`). |
| `freeriding_simulations/` | The Julia project (`Project.toml` and `Manifest.toml`). |
| `figure_regeneration/` | **The current pipeline (June 2026 revision).** `experiments.jl` holds every parameter and grid behind the four figures in the paper, `regenerate_all.sh` reproduces them from scratch, `results/` holds the merged 32-seed results each figure is built from, and `figures/` holds the four PDFs that appear in the paper. |
| `paper_simulations.jmd`, `results_cap5_32seeds.jld2`, `figures/` | The original November 2024 pipeline and its results (first revision). Kept for the record; superseded by `figure_regeneration/`. |

## Parameters

All parameters live in `figure_regeneration/experiments.jl` (`BASE_P` and the four `configs_*` functions). The base case is the parameter table of the simulation appendix:

| Paper | Code field | Base value |
|---|---|---|
| T1, T2: average task time | `time1`, `time2` | 1 hour each |
| w0: workers' outside wage | `supply_reservation_wage` | $40 per hour |
| N: multi-apping worker pool | `firm12_supply` | 4 workers |
| N1, N2: truly dedicated worker pools | `firm1_supply`, `firm2_supply` | 0 |
| alpha: supply logit sensitivity | `supply_sensitivity` | 5 |
| U_service: service utility | `service_value` | $100 |
| D1, D2: each firm's own customer pool | `firm1_demand`, `firm2_demand` | 2 per hour each |
| Dc: competitive (multi-apping) customer pool | `firm12_demand` | 0 |
| alpha_demand: demand logit sensitivity | `demand_sensitivity` | 0.2 |
| Cap of the three queues | `max_queue` | 5 |
| Cost of not being served (not in the table) | `no_service_cost` | 0 |

The numerical procedure uses the same settings in every experiment:

| Setting | Code | Value |
|---|---|---|
| Step for price and pay | `price_step` | $1 |
| Step for arrival rates (demand and supply stages) | `rate_step` | 0.02 per hour |
| Random starts per configuration | `initial_guess(p, seed; randomness)`, seeds 1, 2, ... | 32 converged seeds, `randomness = 0.1` |
| Iteration cap per stage | `max_iter` | 1000 |
| Cap on queueing solves per equilibrium | `max_simulations` | 500,000 |

The engine works on an integer grid: prices and pay are integers in units of `price_step`, arrival rates in units of `rate_step`. Each stage moves one decision by one step at a time and stops when it returns to a state it has already visited, which is either a fixed point or a cycle. As the paper describes, a cycle is resolved to its most favorable state. The saved results record whether each solve finished within the budget (the `converged` column) but not whether it ended in a fixed point or in a cycle.

## The four experiments

| Experiment | Figure in the paper | What varies |
|---|---|---|
| `order_of_events` | Figure 4 (Section 5.2) | Firm 1's share of a fixed total demand of 4 per hour, from 5% to 95% in 19 points, under sequential and simultaneous timing. Only shares at or below 50% are simulated; the right half is the mirror image with the two firms swapped, since the model is exactly symmetric in the firms. |
| `supply_sensitivity` | Section 5.3 figure | alpha on a log grid from 0.1 to 10 (11 points), at a 25/75 and a 50/50 demand split. |
| `dedicated_supply` | Appendix, dedicated worker pools | Share of truly dedicated workers from 0% to 100% in steps of 10%, plus 85% and 95%, split 0/100, 25/75, or 50/50 between the firms. **The total workforce N + N1 + N2 is 1.5 in this experiment** (not 4), so that the three regimes are visible. |
| `dedicated_demand` | Appendix, competitive demand | Share of demand that is firm-specific, from 0% to 100% in steps of 10%, split 0/100, 25/75, or 50/50; the rest is the competitive pool Dc. |

Every configuration has 32 converged random starts; the harness draws replacement seeds until it does. In the June 2026 run, the only configuration that needed more than 32 attempts was `dedicated_demand` at 90% dedicated demand, all of it on firm 2, where about a third of the starts hit the solve cap.

## Reproducing the figures

Requires Julia 1.12 (the `Manifest.toml` is resolved for 1.12.6).

```bash
cd figure_regeneration
julia --project=../freeriding_simulations -e 'using Pkg; Pkg.instantiate()'
./regenerate_all.sh                   # all four experiments, then the figures (about 30 min with 10 workers on an Apple M4 Max)
./regenerate_all.sh order_of_events   # a single experiment
julia --project=../freeriding_simulations build_figures.jl   # re-plot from results/ without simulating
```

Each (experiment, seed) pair runs as its own single-threaded subprocess; `supervise.jl` kills and replaces any that hang, and `build_figures.jl` plots from the merged `results/<experiment>_results.jld2`. See `figure_regeneration/README.md` for the details.

Random starts depend on Julia's random number stream, which changed between Julia 1.11 and 1.12, so a fresh run reproduces the figures but not every individual equilibrium bit for bit. The November 2024 results were produced with Julia 1.11.1; its lock file is available with `git show 92d1419:freeriding_simulations/Manifest.toml`.

## What changed in the June 2026 revision

- **Engine fix.** When a queue has zero arrivals the Markov chain is reducible and the stationary-distribution system is rank-deficient. The sparse least-squares solve then returned an arbitrary, firm-asymmetric solution at extreme demand splits (for instance 95/5). Those cases now use a dense minimum-norm solve. Mid-range results are unchanged.
- **Speed.** The stationary distribution is memoized within one equilibrium solve, the full-rank case uses a square sparse LU solve, and BLAS is pinned to one thread. Validated bit-identical to the previous engine on every tested configuration.
- **Figure 4.** The earlier version mirrored the simulated left half without swapping the two firms' series, so the right half showed the wrong firm. The mirror now swaps the firms.
- **Dedicated-pool figure.** Total workforce set to 1.5 instead of 4.
- **Retired.** The "quality" (cost of no service) experiment is no longer in the paper.
- **Harness.** All four figures regenerated from scratch, with 32 converged seeds per configuration, through `figure_regeneration/`.
