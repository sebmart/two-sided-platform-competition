# Shared definitions for the robust figure regeneration: base parameters, the config sweeps for
# every experiment, and a canonical config-key for convergence tracking. SINGLE SOURCE OF TRUTH —
# run_unit.jl, supervise.jl, and build_figures.jl all include this so the configs can never drift.
#
# Two deliberate changes vs the published paper_simulations.jmd:
#   * order_of_events uses SYMMETRY: only firm-1 share <= 50% is simulated (10 of 19 points); the
#     right half is reflected with a firm-swap at plot time (the engine is exactly firm-symmetric).
#   * dedicated_supply uses N = total_supply = 1.5 (paper used 4.0), per Sebastien (2026-06-07).
#
# Everything else (base_p, demand, grids, randomness, max_simulations) matches the paper exactly.

import Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))   # repo root: engine + Julia project
const PROJ = joinpath(ROOT, "freeriding_simulations")
const ENGINE = joinpath(ROOT, "simulation_utils_logit.jl")   # has the Jun-5 symmetry fix + Jun-6 perf opt
Pkg.activate(PROJ)
include(ENGINE)
using .FreeridingSimulation, Accessors, DataFrames

# ---- knobs ----
const RANDOMNESS      = 0.1
const MAX_SIMULATIONS = 500_000
const DEDICATED_SUPPLY_N = 1.5          # <-- N for the private-labor-pools figure (paper was 4.0)
const TOTAL_DEMAND    = 4.0

const BASE_P = ProblemParameters(
    time1=1.0, time2=1.0, firm1_demand=2.0, firm2_demand=2.0, firm12_demand=0.0,
    service_value=100.0, no_service_cost=0.0, demand_sensitivity=0.2,
    firm1_supply=0.0, firm2_supply=0.0, firm12_supply=4.0,
    supply_reservation_wage=40.0, supply_sensitivity=5.0,
    rate_step=0.02, price_step=1.0, max_queue=5,
)

# ---- the four experiments: each returns Vector{Tuple{ProblemParameters,Bool}} (config, sequential) ----

# order_of_events (Fig 4): symmetric base demand, vary firm-1 share. SYMMETRY: only <=50% simulated.
const ORDER_FIRM1_FULL = collect((0.0:0.2:4.0)[2:end-1])           # 0.2..3.8 (19 pts) -- for the plot axis
const ORDER_FIRM1_HALF = ORDER_FIRM1_FULL[1:ceil(Int, length(ORDER_FIRM1_FULL)/2)]  # 0.2..2.0 (10 pts)
function configs_order_of_events()
    out = Tuple{ProblemParameters,Bool}[]
    for s in ORDER_FIRM1_HALF, seq in (true, false)
        p = @set BASE_P.firm1_demand = s
        p = @set p.firm2_demand = 4.0 - s
        push!(out, (p, seq))
    end
    return out
end

# dedicated_demand ("impact of competition"): vary dedicated demand fraction and its firm-1 split.
const DED_FRACTIONS      = collect(0.0:0.1:1.0)
const DED_FRACTIONS_FIRM1 = [0.0, 0.25, 0.5]
# dedicated_supply uses a FINER grid near the top (extra 85% & 95% points) where the crossover lives.
const DED_SUPPLY_FRACTIONS = sort(unique([collect(0.0:0.1:1.0); 0.85; 0.95]))
function configs_dedicated_demand()
    out = Tuple{ProblemParameters,Bool}[]
    for fd in DED_FRACTIONS, f1 in DED_FRACTIONS_FIRM1
        p = @set BASE_P.firm1_demand  = TOTAL_DEMAND * fd * f1
        p = @set p.firm2_demand       = TOTAL_DEMAND * fd * (1 - f1)
        p = @set p.firm12_demand      = TOTAL_DEMAND * (1 - fd)
        push!(out, (p, true))
    end
    return out
end

# supply_sensitivity: vary supply sensitivity (log grid) at two demand splits.
const SUPPLY_SENSITIVITIES = collect(10 .^ range(-1, 1, length=11))
const SS_DEMAND_RATIOS     = [1/4, 1/2]
function configs_supply_sensitivity()
    out = Tuple{ProblemParameters,Bool}[]
    for ss in SUPPLY_SENSITIVITIES, ratio in SS_DEMAND_RATIOS
        p = @set BASE_P.supply_sensitivity = ss
        p = @set p.firm1_demand = TOTAL_DEMAND * ratio
        p = @set p.firm2_demand = TOTAL_DEMAND * (1 - ratio)
        push!(out, (p, true))
    end
    return out
end

# dedicated_supply ("private labor pools"): vary dedicated SUPPLY fraction and its firm-1 split.
# N = total_supply = DEDICATED_SUPPLY_N (1.5). Demand stays symmetric at base (2/2).
function configs_dedicated_supply()
    out = Tuple{ProblemParameters,Bool}[]
    for fd in DED_SUPPLY_FRACTIONS, f1 in DED_FRACTIONS_FIRM1
        p = @set BASE_P.firm1_supply  = DEDICATED_SUPPLY_N * fd * f1
        p = @set p.firm2_supply       = DEDICATED_SUPPLY_N * fd * (1 - f1)
        p = @set p.firm12_supply      = DEDICATED_SUPPLY_N * (1 - fd)
        push!(out, (p, true))
    end
    return out
end

const EXPERIMENTS = ["order_of_events", "dedicated_demand", "supply_sensitivity", "dedicated_supply"]

function configs_for(exp::AbstractString)
    raw = exp == "order_of_events"    ? configs_order_of_events()    :
          exp == "dedicated_demand"   ? configs_dedicated_demand()   :
          exp == "supply_sensitivity" ? configs_supply_sensitivity() :
          exp == "dedicated_supply"   ? configs_dedicated_supply()   :
          error("unknown experiment $exp")
    return unique(raw)   # dedup (e.g. fraction_dedicated=0 collapses the firm-1 split panels)
end

# Canonical hashable key for a config -- only the fields that ever vary across experiments.
config_key(p::ProblemParameters, seq::Bool) = (
    p.firm1_demand, p.firm2_demand, p.firm12_demand,
    p.firm1_supply, p.firm2_supply, p.firm12_supply,
    p.supply_sensitivity, p.no_service_cost, seq,
)
