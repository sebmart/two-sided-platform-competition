# diagnostics.jl -- diagnostics of the numerical procedure.
#
# Lives in figure_regeneration/ next to experiments.jl. One (experiment | variant, seed) per process, like
# run_unit.jl, so it can be parallelised with run_diagnostics.sh. Everything is computed with the engine's own
# stage solvers (simulation_utils_logit.jl is not modified), so deviation payoffs are evaluated exactly the way
# the main solve evaluates continuation payoffs (same grid, same cycle-selection rules, same iteration cap).
#
#   julia --project=PROJ diagnostics.jl regret       <experiment> <seed> <outfile>
#   julia --project=PROJ diagnostics.jl cycles       <experiment> <seed> <outfile>
#   julia --project=PROJ diagnostics.jl sensitivity  <variant>    <seed> <outfile>   variant in step05 step025 cap7 cap10
#   julia --project=PROJ diagnostics.jl needed_seeds <experiment>                     -> prints the seeds the figures use
#   julia --project=PROJ diagnostics.jl summarize                                     -> diagnostics/summary.txt
#
# regret      At every reported equilibrium of this seed (converged rows, first REGEN_TARGET=32 seeds per
#             configuration, i.e. exactly the rows build_figures.jl plots), sweep each firm's unilateral deviations on
#             the decision grid and record the largest profit gain (the grid-restricted best-response residual).
#             Sequential timing: a price deviation re-solves demand -> pay -> supply (sequentialNE_demand, pay of the
#             deviating firm left as reported, exactly as the engine's price stage does), a pay deviation re-solves
#             supply (sequentialNE_supply, demand held fixed, as the engine's pay stage does).
#             Simultaneous timing: joint (price, pay) deviations re-solve supply+demand (simultaneousNE_supplydemand).
#             The "self" deviation (deviating to the reported decision itself) is also re-solved and recorded: the
#             lower stages are path-dependent cycle dynamics, so it need not reproduce the reported profit. Regret is
#             reported against both the reported profit (`regret`) and the self-continuation profit (`regret_self`).
#             Sweep: +-DIAG_WINDOW steps around the reported decision plus the whole grid at stride DIAG_STRIDE; the
#             price grid runs to DIAG_PMAX_FACTOR x service_value / price_step (1.5 x 100 = 150 by default; reported
#             prices above the service value do occur in the saved results).
# cycles      Restart the engine's price-stage update map (copied from sequentialNE) from the reported state and
#             iterate until a state repeats; record the cycle length, whether the reported state lies on the cycle,
#             the time average over the cycle of profits / prices / pay, the most-favorable state under the engine's
#             rule (largest min profit, ties by max profit; should be the reported state) and the min/max over the
#             cycle. Same for the pay stage (map copied from sequentialNE_pay) at the reported prices and demand, and
#             for the simultaneous firm map (copied from simultaneousNE) on the simultaneous configurations.
#             On a sensitivity variant's outcomes (`cycles sens-<variant>`) the lower stages run with the variant's
#             own iteration budget (1000 x variant_scale, as the sensitivity solve did) unless DIAG_MAXITER is set.
# sensitivity Re-solve the ten sequential order_of_events configurations for this seed with a finer price step
#             (step05 = 0.5, step025 = 0.25) or a larger queue cap (cap7, cap10), everything else as experiments.jl.
# summarize   Merge diagnostics/units/*.jld2 into tables (text file + a DataFrame per mode). The sensitivity section
#             prints two tables: the reported-state summary (the engine's most favorable state on the terminal cycle)
#             and the time-average summary (every row, base included, summarized by the time average over its
#             terminal price-stage cycle, from diagnostics/cycles_order_of_events.jld2 and cycles_sens-<variant>.jld2).

const HERE = @__DIR__
include(joinpath(HERE, "experiments.jl"))
using JLD2, Statistics, DataFrames, Printf, Dates
const FS = FreeridingSimulation          # for the (unexported) per-process stationary-distribution cache reset

const DIAG    = joinpath(HERE, "diagnostics")
const DUNITS  = joinpath(DIAG, "units")
const RESULTS = joinpath(HERE, "results")
mkpath(DUNITS)

const WINDOW = parse(Int, get(ENV, "DIAG_WINDOW", "10"))          # +- steps around the reported decision
const STRIDE = parse(Int, get(ENV, "DIAG_STRIDE", "5"))           # coarse sweep of the whole grid
const MAXIT  = parse(Int, get(ENV, "DIAG_MAXITER", "1000"))       # per-stage iteration cap, as in run_unit.jl
const MAXIT_SET = haskey(ENV, "DIAG_MAXITER")
const MAXIT_REF = Ref(MAXIT)     # cap used by the cycle maps: MAXIT, or the variant's own budget on `cycles sens-<variant>`
const TARGET = parse(Int, get(ENV, "REGEN_TARGET", "32"))         # seeds per configuration kept by build_figures.jl
const PMAX_FACTOR = parse(Float64, get(ENV, "DIAG_PMAX_FACTOR", "1.5"))   # price sweep upper end, x service_value
const SENS_VARIANTS = ("step05", "step025", "cap7", "cap10", "rate01", "step05r", "step025r")
const REGRET_KINDS = split(get(ENV, "DIAG_REGRET_KINDS", "price,pay,joint"), ",")   # subset of the deviation kinds to sweep

pricemax(p::ProblemParameters) = round(Int, PMAX_FACTOR * p.service_value / p.price_step)

"Candidate grid values: window around x plus the whole range at the stride (x itself included; callers skip it)."
function candidates(x::Int, lo::Int, hi::Int)
    s = Set{Int}()
    for v in (x-WINDOW):(x+WINDOW); lo <= v <= hi && push!(s, v); end
    for v in lo:STRIDE:hi; push!(s, v); end
    lo <= hi && push!(s, hi)
    return sort!(collect(s))
end

# ---------------------------------------------------------------- reported-row selection ---------------------
"Mask of the rows the figures use: converged, first TARGET seeds per (params, sequential), as build_figures.get_results."
function reported_mask(data::DataFrame)
    idx = Dict{Any,Vector{Int}}()
    for (i, r) in enumerate(eachrow(data))
        r.converged || continue
        push!(get!(idx, (r.params, r.sequential), Int[]), i)
    end
    keep = falses(nrow(data))
    for (_, is) in idx
        sort!(is, by = i -> data.seed[i])
        for i in is[1:min(TARGET, length(is))]; keep[i] = true; end
    end
    return keep
end

"Merged rows of an experiment. `exp` is one of EXPERIMENTS (read from results/<exp>_results.jld2) or `sens-<variant>`
 (the sensitivity units diagnostics/units/sensitivity_<variant>_seed*.jld2 merged), so regret / cycles can be run on a
 sensitivity variant's outcomes."
function load_results(exp::AbstractString)
    if startswith(exp, "sens-")
        variant = exp[6:end]
        df = load_units("sensitivity_$(variant)_")
        df === nothing && error("no sensitivity units for variant $variant")
        return df
    end
    f = joinpath(RESULTS, "$(exp)_results.jld2"); isfile(f) || error("no results file $f"); return load_object(f)
end

function reported_rows(exp::AbstractString, seed::Int)
    data = load_results(exp)
    data = data[reported_mask(data), :]
    return data[data.seed .== seed, :]
end

needed_seeds(exp::AbstractString) = (data = load_results(exp); sort(unique(data.seed[reported_mask(data)])))

trueprofits(p, firms, workers, demand) = (s = simulation(p, firms, workers, demand); (s.profit1, s.profit2))
firm_active(p, firm) = firm == 1 ? (p.firm1_supply + p.firm12_supply > 0 && p.firm1_demand + p.firm12_demand > 0) :
                                   (p.firm2_supply + p.firm12_supply > 0 && p.firm2_demand + p.firm12_demand > 0)

# ---------------------------------------------------------------- regret --------------------------------------
function regret_unit(exp::AbstractString, seed::Int)
    rows = reported_rows(exp, seed)
    out = DataFrame(seed=Int[], sequential=Bool[], params=ProblemParameters[], firm=Int[], kind=String[],
                    base_profit=Float64[], self_profit=Float64[], best_dev_profit=Float64[],
                    gain=Float64[], regret=Float64[], regret_self=Float64[],
                    n_dev=Int[], n_failed=Int[], rep_price=Int[], rep_pay=Int[],
                    argbest_price=Int[], argbest_pay=Int[], seconds=Float64[])
    fl(x) = isfinite(x) ? max(0.0, x) : NaN
    for r in eachrow(rows)
        FS._clear_stationary_cache!()          # bound per-process memory, as the engine does per equilibrium solve
        p, firms, workers, demand = r.params, r.firms, r.workers, r.demand
        base1, base2 = trueprofits(p, firms, workers, demand)
        pmax = max(pricemax(p), firms.price1 + WINDOW, firms.price2 + WINDOW)
        for firm in (1, 2)
            firm_active(p, firm) || continue     # a firm without a market never moves in the engine; nothing to test
            base  = firm == 1 ? base1 : base2
            price = firm == 1 ? firms.price1 : firms.price2
            pay   = firm == 1 ? firms.pay1   : firms.pay2
            setfirm(v, q) = firm == 1 ? FirmDecisions(v, firms.price2, q, firms.pay2) :
                                        FirmDecisions(firms.price1, v, firms.pay1, q)
            getp(sim) = firm == 1 ? sim.profit1 : sim.profit2
            if r.sequential
                # (a) price deviations: re-solve demand -> pay -> supply from the reported state, pay left as reported
                #     (the engine's price stage passes the current pay unchanged into sequentialNE_demand)
                "price" in REGRET_KINDS && begin
                t0 = time()
                function evalprice(v)
                    f2, _, _, sim2, _ = sequentialNE_demand(p, setfirm(v, pay), workers, demand; max_iter=MAXIT)
                    return getp(sim2), (firm == 1 ? f2.pay1 : f2.pay2)
                end
                selfp = try evalprice(price)[1] catch; NaN end
                best = -Inf; nd = 0; nf = 0; argv = price; argq = pay
                for v in candidates(price, 0, pmax)
                    v == price && continue
                    nd += 1
                    pr, q2 = try evalprice(v) catch; (nf += 1; continue) end
                    if pr > best; best = pr; argv = v; argq = q2; end
                end
                push!(out, (seed, true, p, firm, "price", base, selfp, best, best - base, fl(best - base), fl(best - selfp),
                            nd, nf, price, pay, argv, argq, time() - t0))
                end
                # (b) pay deviations: re-solve supply only, demand held fixed (the engine's pay stage); pay <= price
                "pay" in REGRET_KINDS && begin
                t0 = time()
                evalpay(q) = getp(sequentialNE_supply(p, setfirm(price, q), workers, demand; max_iter=MAXIT)[2])
                selfp = try evalpay(pay) catch; NaN end
                best = -Inf; nd = 0; nf = 0; argq = pay
                for q in candidates(pay, 0, price)
                    q == pay && continue
                    nd += 1
                    pr = try evalpay(q) catch; (nf += 1; continue) end
                    if pr > best; best = pr; argq = q; end
                end
                push!(out, (seed, true, p, firm, "pay", base, selfp, best, best - base, fl(best - base), fl(best - selfp),
                            nd, nf, price, pay, price, argq, time() - t0))
                end
            elseif "joint" in REGRET_KINDS
                # simultaneous timing: joint (price, pay) deviations, re-solving supply and demand together.
                # pay > price gives profit <= 0 = profit at (v, v), so restricting to pay <= price loses nothing.
                t0 = time()
                evaljoint(v, q) = getp(simultaneousNE_supplydemand(p, setfirm(v, q), workers, demand; max_iter=MAXIT)[3])
                selfp = try evaljoint(price, pay) catch; NaN end
                best = -Inf; nd = 0; nf = 0; argv = price; argq = pay
                for v in candidates(price, 0, pmax), q in candidates(pay, 0, v)
                    (v == price && q == pay) && continue
                    nd += 1
                    pr = try evaljoint(v, q) catch; (nf += 1; continue) end
                    if pr > best; best = pr; argv = v; argq = q; end
                end
                push!(out, (seed, false, p, firm, "joint", base, selfp, best, best - base, fl(best - base), fl(best - selfp),
                            nd, nf, price, pay, argv, argq, time() - t0))
            end
        end
        println("regret: $exp seed=$seed firm1_demand=$(p.firm1_demand) seq=$(r.sequential) done ($(round(sum(out.seconds), digits=1))s so far)"); flush(stdout)
    end
    return out
end

# ---------------------------------------------------------------- cycles --------------------------------------
# Each step function is a copy of one engine stage's update, mapping (firms, workers, demand) to the next state and
# returning (firms, workers, demand, profit1, profit2) with the profits *as the engine records them in its state
# tuple* (for the price stage that is the sim returned by sequentialNE_demand, which the engine does not recompute
# for the state it selects from a demand cycle, so it can be stale relative to the returned decisions).

"One step of the price-stage map (sequentialNE): each firm compares one step down vs one step up, ties -> lower."
function price_step(p, firms, workers, demand)
    firm1price = firms.price1
    if firm_active(p, 1)
        lo = FirmDecisions(max(0, firms.price1 - 1), firms.price2, firms.pay1, firms.pay2)
        _, _, _, simlo, _ = sequentialNE_demand(p, lo, workers, demand; max_iter=MAXIT_REF[])
        hi = FirmDecisions(firms.price1 + 1, firms.price2, firms.pay1, firms.pay2)
        _, _, _, simhi, _ = sequentialNE_demand(p, hi, workers, demand; max_iter=MAXIT_REF[])
        firm1price = simlo.profit1 >= simhi.profit1 ? lo.price1 : hi.price1
    end
    firm2price = firms.price2
    if firm_active(p, 2)
        lo = FirmDecisions(firms.price1, max(0, firms.price2 - 1), firms.pay1, firms.pay2)
        _, _, _, simlo, _ = sequentialNE_demand(p, lo, workers, demand; max_iter=MAXIT_REF[])
        hi = FirmDecisions(firms.price1, firms.price2 + 1, firms.pay1, firms.pay2)
        _, _, _, simhi, _ = sequentialNE_demand(p, hi, workers, demand; max_iter=MAXIT_REF[])
        firm2price = simlo.profit2 >= simhi.profit2 ? lo.price2 : hi.price2
    end
    f = FirmDecisions(firm1price, firm2price, firms.pay1, firms.pay2)
    f, w, d, sim, _ = sequentialNE_demand(p, f, workers, demand; max_iter=MAXIT_REF[])
    return (f, w, d, sim.profit1, sim.profit2)
end

"One step of the pay-stage map (sequentialNE_pay): one step down vs one step up (capped at price), ties -> higher.
 A firm updates only if it has supply and *currently positive demand rates* (the engine's condition at this stage)."
function pay_step(p, firms, workers, demand)
    firm1pay = firms.pay1
    if p.firm1_supply + p.firm12_supply > 0 && demand.rate1 + demand.rate12_1 + demand.rate12_12 > 0
        lo = FirmDecisions(firms.price1, firms.price2, max(0, firms.pay1 - 1), firms.pay2)
        _, simlo, _ = sequentialNE_supply(p, lo, workers, demand; max_iter=MAXIT_REF[])
        hi = FirmDecisions(firms.price1, firms.price2, min(firms.price1, firms.pay1 + 1), firms.pay2)
        _, simhi, _ = sequentialNE_supply(p, hi, workers, demand; max_iter=MAXIT_REF[])
        firm1pay = simlo.profit1 > simhi.profit1 ? lo.pay1 : hi.pay1
    end
    firm2pay = firms.pay2
    if p.firm2_supply + p.firm12_supply > 0 && demand.rate2 + demand.rate12_2 + demand.rate12_12 > 0
        lo = FirmDecisions(firms.price1, firms.price2, firms.pay1, max(0, firms.pay2 - 1))
        _, simlo, _ = sequentialNE_supply(p, lo, workers, demand; max_iter=MAXIT_REF[])
        hi = FirmDecisions(firms.price1, firms.price2, firms.pay1, min(firms.price2, firms.pay2 + 1))
        _, simhi, _ = sequentialNE_supply(p, hi, workers, demand; max_iter=MAXIT_REF[])
        firm2pay = simlo.profit2 > simhi.profit2 ? lo.pay2 : hi.pay2
    end
    f = FirmDecisions(firms.price1, firms.price2, firm1pay, firm2pay)
    w, sim, _ = sequentialNE_supply(p, f, workers, demand; max_iter=MAXIT_REF[])
    return (f, w, demand, sim.profit1, sim.profit2)
end

"One step of the simultaneous firm map (simultaneousNE): each firm compares staying put with four diagonal
 (price, pay) moves, strict improvement required, evaluated in the engine's order LH, HH, LL, HL."
function simul_step(p, firms, workers, demand)
    sim = simulation(p, firms, workers, demand)      # the engine's `sim` for the current state (recomputed by the stage below)
    best1 = firms
    if firm_active(p, 1)
        bestp = sim.profit1
        for (dv, dq) in ((-1, 1), (1, 1), (-1, -1), (1, -1))
            f2 = FirmDecisions(max(0, firms.price1 + dv), firms.price2, max(0, firms.pay1 + dq), firms.pay2)
            _, _, s2, _ = simultaneousNE_supplydemand(p, f2, workers, demand; max_iter=MAXIT_REF[])
            if s2.profit1 > bestp; bestp = s2.profit1; best1 = f2; end
        end
    end
    best2 = firms
    if firm_active(p, 2)
        bestp = sim.profit2
        for (dv, dq) in ((-1, 1), (1, 1), (-1, -1), (1, -1))
            f2 = FirmDecisions(firms.price1, max(0, firms.price2 + dv), firms.pay1, max(0, firms.pay2 + dq))
            _, _, s2, _ = simultaneousNE_supplydemand(p, f2, workers, demand; max_iter=MAXIT_REF[])
            if s2.profit2 > bestp; bestp = s2.profit2; best2 = f2; end
        end
    end
    f = FirmDecisions(best1.price1, best2.price2, best1.pay1, best2.pay2)
    w, d, sim, _ = simultaneousNE_supplydemand(p, f, workers, demand; max_iter=MAXIT_REF[])
    return (f, w, d, sim.profit1, sim.profit2)
end

"Iterate `stepfn` from the reported state until a (firms, workers, demand) state repeats. Returns the states of the
 terminal cycle as (firms, workers, demand, engine_profit1, engine_profit2) and whether the start lies on it. The
 engine profits attached to the first cycle state come from the closing transition, i.e. from its predecessor on the
 cycle, exactly as the engine would have recorded them."
function terminal_cycle(stepfn, p, firms, workers, demand)
    states = Any[(firms, workers, demand, NaN, NaN)]
    seen = Dict{Any,Int}((firms, workers, demand) => 1)
    while length(states) <= MAXIT_REF[]
        f, w, d, e1, e2 = stepfn(p, states[end][1], states[end][2], states[end][3])
        k = (f, w, d)
        if haskey(seen, k)
            j = seen[k]
            cyc = states[j:end]
            cyc[1] = (cyc[1][1], cyc[1][2], cyc[1][3], e1, e2)
            return cyc, j == 1
        end
        push!(states, (f, w, d, e1, e2)); seen[k] = length(states)
    end
    error("no cycle found within $(MAXIT_REF[]) iterations")
end

function cycle_stats(p, cyc, reported)
    L = length(cyc)
    tp = [trueprofits(p, s[1], s[2], s[3]) for s in cyc]          # what the figures plot for a state
    pr1 = first.(tp); pr2 = last.(tp)
    e1 = [s[4] for s in cyc]; e2 = [s[5] for s in cyc]            # what the engine's selection rule sees
    price1 = [s[1].price1 for s in cyc]; price2 = [s[1].price2 for s in cyc]
    pay1 = [s[1].pay1 for s in cyc]; pay2 = [s[1].pay2 for s in cyc]
    # the engine's rule: largest min profit, ties broken by largest max profit (first in walk order on exact ties)
    ifav = argmax([(min(e1[i], e2[i]), max(e1[i], e2[i])) for i in 1:L])
    favstate = (cyc[ifav][1], cyc[ifav][2], cyc[ifav][3])
    n_stale = count(i -> abs(e1[i] - pr1[i]) > 1e-9 || abs(e2[i] - pr2[i]) > 1e-9, 1:L)
    return (len=L, fav_is_reported=(favstate == reported), n_stale=n_stale,
            avg_profit1=mean(pr1), avg_profit2=mean(pr2), avg_price1=mean(price1), avg_price2=mean(price2),
            avg_pay1=mean(pay1), avg_pay2=mean(pay2),
            fav_profit1=pr1[ifav], fav_profit2=pr2[ifav], fav_price1=price1[ifav], fav_price2=price2[ifav],
            fav_pay1=pay1[ifav], fav_pay2=pay2[ifav],
            min_profit1=minimum(pr1), max_profit1=maximum(pr1), min_profit2=minimum(pr2), max_profit2=maximum(pr2),
            min_price1=minimum(price1), max_price1=maximum(price1), min_price2=minimum(price2), max_price2=maximum(price2),
            min_pay1=minimum(pay1), max_pay1=maximum(pay1), min_pay2=minimum(pay2), max_pay2=maximum(pay2))
end

function cycles_unit(exp::AbstractString, seed::Int)
    if startswith(exp, "sens-") && !MAXIT_SET
        MAXIT_REF[] = 1000 * variant_scale(exp[6:end])     # the lower-stage cap the sensitivity solve used
        println("cycles: $exp uses the variant's lower-stage iteration cap max_iter=$(MAXIT_REF[])")
    end
    rows = reported_rows(exp, seed)
    out = DataFrame(seed=Int[], sequential=Bool[], params=ProblemParameters[], stage=String[], len=Int[],
        reported_on_cycle=Bool[], fav_is_reported=Bool[], n_stale=Int[],
        rep_profit1=Float64[], rep_profit2=Float64[], rep_price1=Int[], rep_price2=Int[], rep_pay1=Int[], rep_pay2=Int[],
        avg_profit1=Float64[], avg_profit2=Float64[], avg_price1=Float64[], avg_price2=Float64[],
        avg_pay1=Float64[], avg_pay2=Float64[], fav_profit1=Float64[], fav_profit2=Float64[],
        fav_price1=Int[], fav_price2=Int[], fav_pay1=Int[], fav_pay2=Int[],
        min_profit1=Float64[], max_profit1=Float64[], min_profit2=Float64[], max_profit2=Float64[],
        min_price1=Int[], max_price1=Int[], min_price2=Int[], max_price2=Int[],
        min_pay1=Int[], max_pay1=Int[], min_pay2=Int[], max_pay2=Int[], seconds=Float64[])
    for r in eachrow(rows)
        FS._clear_stationary_cache!()
        p, firms, workers, demand = r.params, r.firms, r.workers, r.demand
        rp1, rp2 = trueprofits(p, firms, workers, demand)
        stages = r.sequential ? (("price", price_step), ("pay", pay_step)) : (("simul", simul_step),)
        for (stage, fn) in stages
            t0 = time()
            try
                cyc, onit = terminal_cycle(fn, p, firms, workers, demand)
                c = cycle_stats(p, cyc, (firms, workers, demand))
                push!(out, (seed, r.sequential, p, stage, c.len, onit, c.fav_is_reported, c.n_stale,
                    rp1, rp2, firms.price1, firms.price2, firms.pay1, firms.pay2,
                    c.avg_profit1, c.avg_profit2, c.avg_price1, c.avg_price2,
                    c.avg_pay1, c.avg_pay2, c.fav_profit1, c.fav_profit2, c.fav_price1, c.fav_price2, c.fav_pay1, c.fav_pay2,
                    c.min_profit1, c.max_profit1, c.min_profit2, c.max_profit2,
                    c.min_price1, c.max_price1, c.min_price2, c.max_price2,
                    c.min_pay1, c.max_pay1, c.min_pay2, c.max_pay2, time() - t0))
                onit || @warn "reported state NOT on the recovered cycle" exp seed stage firm1_demand=p.firm1_demand len=c.len
            catch e
                @warn "cycle recovery failed" exp seed stage firm1_demand=p.firm1_demand exception=e
            end
        end
        println("cycles: $exp seed=$seed firm1_demand=$(p.firm1_demand) seq=$(r.sequential) done"); flush(stdout)
    end
    return out
end

# ---------------------------------------------------------------- sensitivity ---------------------------------
function variant_params(p::ProblemParameters, variant::AbstractString)
    variant == "step05"  && return @set p.price_step = 0.5
    variant == "step025" && return @set p.price_step = 0.25
    variant == "cap7"    && return @set p.max_queue = 7
    variant == "cap10"   && return @set p.max_queue = 10
    variant == "rate01"  && return @set p.rate_step = 0.01                       # rate refinement alone
    variant == "step05r" && return (p = @set p.price_step = 0.5;  @set p.rate_step = 0.01)    # joint refinement x2
    variant == "step025r" && return (p = @set p.price_step = 0.25; @set p.rate_step = 0.005)  # joint refinement x4
    error("unknown variant $variant ($(join(SENS_VARIANTS, " | ")))")
end
# Budget scaling (applied to both max_iter and max_simulations). For the finer price grids the price AND pay stages
# both take proportionally more one-step moves, so the number of queueing solves grows roughly with the square of the
# refinement: x4 for step05 and x16 for step025. (The task sheet proposed x2 / x4; with x4, step025 seed 1 at
# firm1_demand 0.2 exhausted the 2,000,000-solve budget in under 90 s.) The jointly refined grids (round 2) were sized
# from one-seed trials at x16 / x16 / x64. DIAG_SENS_SCALE overrides the scale for trials.
# The summary reports how many converged rows exceeded the x1 / x4 / x16 budgets, so these choices remain visible.
const SCALE_OVERRIDE = get(ENV, "DIAG_SENS_SCALE", "")
function variant_scale(variant)
    isempty(SCALE_OVERRIDE) || return parse(Int, SCALE_OVERRIDE)
    variant == "step05"   && return 4
    variant == "step025"  && return 16
    variant == "rate01"   && return 16
    variant == "step05r"  && return 16
    variant == "step025r" && return 64
    return 1
end

function sensitivity_unit(variant::AbstractString, seed::Int)
    out = DataFrame(seed=Int[], variant=String[], sequential=Bool[], params=ProblemParameters[],
                    firms=FirmDecisions[], workers=WorkerDecisions[], demand=DemandDecisions[], iters=Int[], converged=Bool[],
                    seconds=Float64[])
    scale = variant_scale(variant)
    for (p0, seq) in configs_order_of_events()
        seq || continue
        p = variant_params(p0, variant)
        t0 = time()
        local firms, workers, demand, iters, conv
        try
            firms, workers, demand = initial_guess(p, seed, randomness=RANDOMNESS)
            firms, workers, demand, sim, iters = sequentialNE(p, firms, workers, demand;
                max_iter=1000 * scale, max_simulations=MAX_SIMULATIONS * scale)
            conv = true
        catch e
            @warn "solve failed (recorded non-converged)" variant seed firm1_demand=p.firm1_demand exception=sprint(showerror, e)
            firms, workers, demand = initial_guess(p, seed, randomness=RANDOMNESS)
            iters = -1; conv = false
        end
        push!(out, (seed, variant, true, p, firms, workers, demand, iters, conv, time() - t0))
        println("sensitivity: $variant seed=$seed firm1_demand=$(p.firm1_demand) converged=$conv iters=$iters ($(round(time()-t0, digits=1))s)"); flush(stdout)
    end
    return out
end

# ---------------------------------------------------------------- summarize -----------------------------------
function load_units(prefix::AbstractString)
    files = filter(f -> startswith(basename(f), prefix) && endswith(f, ".jld2"), readdir(DUNITS; join=true))
    isempty(files) && return nothing
    dfs = [load_object(f) for f in files]
    dfs = filter(d -> nrow(d) > 0, dfs)
    isempty(dfs) && return nothing
    return reduce(vcat, dfs)
end

"Keep the diagnostic rows whose (params, sequential, seed) is one of the figures' reported rows."
function keep_reported(df, exp)
    base = load_results(exp)
    base = base[reported_mask(base), :]
    rep = Set((r.params, r.sequential, r.seed) for r in eachrow(base))
    return df[[(r.params, r.sequential, r.seed) in rep for r in eachrow(df)], :]
end

q(v, x) = isempty(v) ? NaN : quantile(v, x)
me(v) = isempty(v) ? NaN : mean(v)
mx(v) = isempty(v) ? NaN : maximum(v)
cfglabel(p::ProblemParameters) = @sprintf("d1=%.2f d2=%.2f d12=%.2f s1=%.2f s2=%.2f s12=%.2f alpha=%.3g",
    p.firm1_demand, p.firm2_demand, p.firm12_demand, p.firm1_supply, p.firm2_supply, p.firm12_supply, p.supply_sensitivity)

function regret_line(pr, label, s)
    s = s[.!isnan.(s.regret), :]
    nrow(s) == 0 && return
    rel = [r.base_profit > 1.0 ? r.regret / r.base_profit : NaN for r in eachrow(s)]
    rel = filter(!isnan, rel)
    selfgap = [r.self_profit - r.base_profit for r in eachrow(s) if isfinite(r.self_profit)]
    rs = filter(!isnan, s.regret_self)
    far = count(r -> abs(r.argbest_price - r.rep_price) > 1 || abs(r.argbest_pay - r.rep_pay) > 1, eachrow(s))
    pr(@sprintf("  %-34s n=%4d | regret vs reported (\$/hr): zero %5.1f%%  median %.3f  p90 %.3f  max %.3f | rel (base>\$1): median %.3f  p90 %.3f  max %.3f",
        label, nrow(s), 100 * mean(s.regret .<= 1e-9), median(s.regret), q(s.regret, 0.9), maximum(s.regret), q(rel, 0.5), q(rel, 0.9), mx(rel)))
    pr(@sprintf("  %-34s        | regret vs self-continuation:  zero %5.1f%%  median %.3f  p90 %.3f  max %.3f | self-base: |median| %.3f  |max| %.3f  exact %5.1f%% | argbest >1 step away: %5.1f%% | failed solves: %d",
        "", isempty(rs) ? NaN : 100 * mean(rs .<= 1e-9), q(rs, 0.5), q(rs, 0.9), mx(rs),
        isempty(selfgap) ? NaN : median(abs.(selfgap)), isempty(selfgap) ? NaN : maximum(abs.(selfgap)),
        isempty(selfgap) ? NaN : 100 * mean(abs.(selfgap) .<= 1e-9), 100 * far / nrow(s), sum(s.n_failed)))
end

function summarize()
    io = open(joinpath(DIAG, "summary.txt"), "w")
    pr(args...) = (println(io, args...); println(args...))
    pr("DIAGNOSTICS SUMMARY  (window=$WINDOW steps, stride=$STRIDE, price sweep to $(PMAX_FACTOR)x service value, target=$TARGET seeds/config)")
    pr("generated $(Dates.now()) with Julia $VERSION")
    # ---------------- regret ----------------
    sens_exps = sort(unique([m.captures[1] for m in filter(!isnothing, match.(r"^regret_(sens-[A-Za-z0-9]+)_seed", readdir(DUNITS)))]))
    for exp in [EXPERIMENTS; sens_exps]
        df = load_units("regret_$(exp)_")
        df === nothing && continue
        df = keep_reported(df, exp)
        ps = df.params[1].price_step
        pr("\n== regret: $exp  ($(nrow(df)) firm-outcome rows; $(length(unique(df.seed))) seeds; total sweep time $(round(sum(df.seconds)/3600, digits=2)) h; price_step=\$$ps; window and stride are in grid steps as set by DIAG_WINDOW / DIAG_STRIDE when the units were run) ==")
        pr("   regret = best deviation profit - reported profit, floored at 0; regret_self uses the re-solved profit at the reported decision as the base")
        for seq in (true, false), kind in ("price", "pay", "joint")
            s = df[(df.kind .== kind) .& (df.sequential .== seq), :]
            nrow(s) == 0 && continue
            regret_line(pr, "$(seq ? "sequential" : "simultaneous") / $kind / all configs", s)
        end
        pr("  -- by configuration (over firms and seeds) --")
        for seq in (true, false), kind in ("price", "pay", "joint")
            s = df[(df.kind .== kind) .& (df.sequential .== seq), :]
            nrow(s) == 0 && continue
            for p in unique(s.params)
                t = s[s.params .== p, :]
                t = t[.!isnan.(t.regret), :]
                nrow(t) == 0 && continue
                rs = filter(!isnan, t.regret_self)
                pr(@sprintf("    %-12s %-5s %-60s n=%3d  regret: zero %5.1f%% median %.3f max %.3f | regret_self: zero %5.1f%% median %.3f max %.3f | base profit mean %.2f",
                    seq ? "sequential" : "simultaneous", kind, cfglabel(p), nrow(t), 100 * mean(t.regret .<= 1e-9), median(t.regret), maximum(t.regret),
                    isempty(rs) ? NaN : 100 * mean(rs .<= 1e-9), q(rs, 0.5), mx(rs), mean(t.base_profit)))
            end
        end
        save_object(joinpath(DIAG, "regret_$(exp).jld2"), df)
    end
    # ---------------- cycles ----------------
    cyc_sens_exps = sort(unique([m.captures[1] for m in filter(!isnothing, match.(r"^cycles_(sens-[A-Za-z0-9]+)_seed", readdir(DUNITS)))]))
    for exp in [EXPERIMENTS; cyc_sens_exps]
        df = load_units("cycles_$(exp)_")
        df === nothing && continue
        df = keep_reported(df, exp)
        pr("\n== cycles: $exp  ($(nrow(df)) stage-outcome rows; $(length(unique(df.seed))) seeds) ==")
        for stage in ("price", "pay", "simul")
            s = df[df.stage .== stage, :]
            nrow(s) == 0 && continue
            ps = s.params[1].price_step
            lens = sort(unique(s.len))
            dist = join(["L=$L: $(round(100*mean(s.len .== L), digits=1))%" for L in lens], ", ")
            pr("  $stage stage ($(nrow(s)) outcomes): cycle-length distribution: $dist")
            pr(@sprintf("    fixed point (L=1): %5.1f%%   reported state on recovered cycle: %5.1f%%   engine's most-favorable state == reported: %5.1f%%   cycles with stale engine profits: %5.1f%%",
                100 * mean(s.len .== 1), 100 * mean(s.reported_on_cycle), 100 * mean(s.fav_is_reported), 100 * mean(s.n_stale .> 0)))
            dpay = [ps * max(abs(r.avg_pay1 - r.rep_pay1), abs(r.avg_pay2 - r.rep_pay2)) for r in eachrow(s)]
            dprice = [ps * max(abs(r.avg_price1 - r.rep_price1), abs(r.avg_price2 - r.rep_price2)) for r in eachrow(s)]
            dprof = [max(abs(r.avg_profit1 - r.rep_profit1), abs(r.avg_profit2 - r.rep_profit2)) for r in eachrow(s)]
            rpay = [ps * max(r.max_pay1 - r.min_pay1, r.max_pay2 - r.min_pay2) for r in eachrow(s)]
            rprice = [ps * max(r.max_price1 - r.min_price1, r.max_price2 - r.min_price2) for r in eachrow(s)]
            rprof = [max(r.max_profit1 - r.min_profit1, r.max_profit2 - r.min_profit2) for r in eachrow(s)]
            pr(@sprintf("    |time average - reported|: pay (\$) median %.3f p90 %.3f max %.3f | price (\$) median %.3f p90 %.3f max %.3f | profit (\$/hr) median %.3f p90 %.3f max %.3f",
                median(dpay), q(dpay, .9), maximum(dpay), median(dprice), q(dprice, .9), maximum(dprice), median(dprof), q(dprof, .9), maximum(dprof)))
            pr(@sprintf("    cycle range (max-min):     pay (\$) median %.3f p90 %.3f max %.3f | price (\$) median %.3f p90 %.3f max %.3f | profit (\$/hr) median %.3f p90 %.3f max %.3f",
                median(rpay), q(rpay, .9), maximum(rpay), median(rprice), q(rprice, .9), maximum(rprice), median(rprof), q(rprof, .9), maximum(rprof)))
            pr("    -- by configuration: figure summary (reported / most-favorable state) vs time-average summary, means over seeds [p25, p75] --")
            pr("       config | n | pay1 rep | pay1 avg | pay2 rep | pay2 avg | profit1 rep | profit1 avg | profit2 rep | profit2 avg")
            for p in unique(s.params)
                t = s[s.params .== p, :]
                f(v) = @sprintf("%6.2f [%6.2f,%6.2f]", me(v), q(v, .25), q(v, .75))
                # as in build_figures.get_results, pay enters the figure only when that firm's profit is positive
                pay1r = [ps * r.rep_pay1 for r in eachrow(t) if r.rep_profit1 > 0]; pay1a = [ps * r.avg_pay1 for r in eachrow(t) if r.rep_profit1 > 0]
                pay2r = [ps * r.rep_pay2 for r in eachrow(t) if r.rep_profit2 > 0]; pay2a = [ps * r.avg_pay2 for r in eachrow(t) if r.rep_profit2 > 0]
                pr("       $(cfglabel(p)) | n=$(nrow(t)) | $(f(pay1r)) | $(f(pay1a)) | $(f(pay2r)) | $(f(pay2a)) | $(f(t.rep_profit1)) | $(f(t.avg_profit1)) | $(f(t.rep_profit2)) | $(f(t.avg_profit2))")
            end
        end
        save_object(joinpath(DIAG, "cycles_$(exp).jld2"), df)
    end
    # ---------------- sensitivity ----------------
    sens = load_units("sensitivity_")
    if sens !== nothing
        base = load_results("order_of_events")
        pr("\n== sensitivity (order_of_events, sequential): first $TARGET converged seeds per configuration, mean [p25, p75] as in Figure 4 ==")
        pr("   pay of a firm enters only when its profit is positive (build_figures rule); n = converged seeds used / attempted")
        for variant in SENS_VARIANTS
            v = sens[sens.variant .== variant, :]
            nrow(v) == 0 && continue
            vc = v[v.converged, :]
            pr(@sprintf("   %-8s: %d seeds attempted, %d non-converged rows (budget x%d: max_iter %d, max_simulations %d), mean solve time %.1f min (max %.1f min), queueing solves median %d max %d, converged rows above the x1 / x4 / x16 budgets (%d / %d / %d solves): %d / %d / %d",
                variant, length(unique(v.seed)), sum(.!v.converged), variant_scale(variant), 1000 * variant_scale(variant), MAX_SIMULATIONS * variant_scale(variant),
                mean(v.seconds)/60, maximum(v.seconds)/60,
                nrow(vc) == 0 ? 0 : round(Int, median(vc.iters)), nrow(vc) == 0 ? 0 : maximum(vc.iters),
                MAX_SIMULATIONS, 4MAX_SIMULATIONS, 16MAX_SIMULATIONS,
                count(vc.iters .> MAX_SIMULATIONS), count(vc.iters .> 4MAX_SIMULATIONS), count(vc.iters .> 16MAX_SIMULATIONS)))
        end
        # the rows of one (share, variant): base from results/order_of_events_results.jld2, variants from the units;
        # converged rows only, first TARGET seeds (build_figures.get_results rule); returns (rows used, rows attempted)
        function sens_rows(p0, label)
            v = label == "base" ? base[(base.params .== p0) .& base.sequential, :] :
                                  sens[(sens.variant .== label) .& (sens.params .== variant_params(p0, label)), :]
            vc = v[v.converged, :]; sort!(vc, :seed); nrow(vc) > TARGET && (vc = vc[1:TARGET, :])
            return vc, nrow(v)
        end
        fcell(v) = @sprintf("%6.2f [%6.2f,%6.2f]", me(v), q(v, .25), q(v, .75))
        cells_rep = Dict{Tuple{Float64,String},Any}()     # (share, variant) => (pay1, pay2, profit1, profit2) samples
        cells_ta  = Dict{Tuple{Float64,String},Any}()
        # ---- table 1: reported state (the engine's most favorable state on the terminal cycle) ----
        pr("   -- TABLE 1, reported state: each outcome summarized by the engine's reported state (its most favorable state on the terminal price-stage cycle); profit recomputed with `simulation` --")
        pr("   firm1_demand | variant | n/attempted | pay1 | pay2 | profit1 | profit2")
        for s in ORDER_FIRM1_HALF
            p0 = @set BASE_P.firm1_demand = s; p0 = @set p0.firm2_demand = 4.0 - s
            function stats(rows, label, attempted)
                pay1 = Float64[]; pay2 = Float64[]; pr1 = Float64[]; pr2 = Float64[]
                for r in eachrow(rows)
                    sim = simulation(r.params, r.firms, r.workers, r.demand)
                    push!(pr1, sim.profit1); push!(pr2, sim.profit2)
                    sim.profit1 > 0 && push!(pay1, r.firms.pay1 * r.params.price_step)
                    sim.profit2 > 0 && push!(pay2, r.firms.pay2 * r.params.price_step)
                end
                cells_rep[(s, label)] = (pay1, pay2, pr1, pr2)
                pr(@sprintf("   %5.2f | %-7s | %2d/%2d | %s | %s | %s | %s", s, label, nrow(rows), attempted, fcell(pay1), fcell(pay2), fcell(pr1), fcell(pr2)))
            end
            for label in ["base"; collect(SENS_VARIANTS)]
                label == "base" || nrow(sens[sens.variant .== label, :]) > 0 || continue
                vc, attempted = sens_rows(p0, label)
                stats(vc, label, attempted)
            end
        end
        # ---- table 2: time average over the terminal price-stage cycle (every row, base included) ----
        # cycle records: price-stage rows of diagnostics/cycles_order_of_events.jld2 (base) and cycles_sens-<variant>.jld2
        # (variants), written by the cycles section above; matched to result rows by (params, sequential, seed).
        function load_cycle_dict(file)
            isfile(file) || return nothing
            d = Dict{Any,Any}()
            for r in eachrow(load_object(file))
                r.stage == "price" || continue
                k = (r.params, r.sequential, r.seed)
                haskey(d, k) && error("duplicate price-stage cycle record in $file: seed=$(r.seed) firm1_demand=$(r.params.firm1_demand)")
                d[k] = r
            end
            return d
        end
        cyc = Dict{String,Any}("base" => load_cycle_dict(joinpath(DIAG, "cycles_order_of_events.jld2")))
        for variant in SENS_VARIANTS; cyc[variant] = load_cycle_dict(joinpath(DIAG, "cycles_sens-$(variant).jld2")); end
        cyc["base"] === nothing && error("time-average table: no diagnostics/cycles_order_of_events.jld2 (run `cycles order_of_events`, then summarize)")
        ta_labels = ["base"; [v for v in SENS_VARIANTS if nrow(sens[sens.variant .== v, :]) > 0 && cyc[v] !== nothing]]
        no_cyc = [v for v in SENS_VARIANTS if nrow(sens[sens.variant .== v, :]) > 0 && cyc[v] === nothing]
        pr("\n   -- TABLE 2, time average: each outcome (base included) summarized by the time average over its terminal price-stage cycle (profit = time average of the profit recomputed with `simulation` at each cycle state; pay = time average of the firm's pay); a firm's pay enters only when its time-averaged profit is positive (build_figures_timeavg.jl rule) --")
        pr("   cycle records: diagnostics/cycles_order_of_events.jld2 (base), diagnostics/cycles_sens-<variant>.jld2 (variants), matched by (params, sequential, seed); a reported row without a record is an error")
        isempty(no_cyc) || pr("   variants without cycle records, omitted from this table (run `./run_diagnostics.sh cycles sens-<variant>` and summarize): $(join(no_cyc, ", "))")
        pr("   firm1_demand | variant | n/attempted | pay1 | pay2 | profit1 | profit2")
        for s in ORDER_FIRM1_HALF
            p0 = @set BASE_P.firm1_demand = s; p0 = @set p0.firm2_demand = 4.0 - s
            for label in ta_labels
                vc, attempted = sens_rows(p0, label)
                d = cyc[label]
                pay1 = Float64[]; pay2 = Float64[]; pr1 = Float64[]; pr2 = Float64[]
                for r in eachrow(vc)
                    k = (r.params, r.sequential, r.seed)
                    haskey(d, k) || error("time-average table: reported row has no cycle record: variant=$label seed=$(r.seed) firm1_demand=$(r.params.firm1_demand) sequential=$(r.sequential) -- run `./run_diagnostics.sh cycles $(label == "base" ? "order_of_events" : "sens-" * label)` and summarize")
                    c = d[k]; ps = r.params.price_step
                    push!(pr1, c.avg_profit1); push!(pr2, c.avg_profit2)
                    c.avg_profit1 > 0 && push!(pay1, c.avg_pay1 * ps)
                    c.avg_profit2 > 0 && push!(pay2, c.avg_pay2 * ps)
                end
                cells_ta[(s, label)] = (pay1, pay2, pr1, pr2)
                pr(@sprintf("   %5.2f | %-7s | %2d/%2d | %s | %s | %s | %s", s, label, nrow(vc), attempted, fcell(pay1), fcell(pay2), fcell(pr1), fcell(pr2)))
            end
        end
        # ---- largest cell differences between the two tables, per variant ----
        pr("\n   -- largest absolute difference between TABLE 2 (time average) and TABLE 1 (reported state), per variant, over the ten shares; 'any cell' = mean, p25 or p75 of any column, 'means' = the mean columns only (the figure's values) --")
        colnames = ("pay1", "pay2", "profit1", "profit2"); statnames = ("mean", "p25", "p75")
        stat3(v) = (me(v), q(v, .25), q(v, .75))
        for label in ta_labels
            best = Dict{String,Any}()      # "pay/any", "pay/means", "profit/any", "profit/means" => (|d|, share, col, stat, rep, ta)
            for s in ORDER_FIRM1_HALF
                (haskey(cells_rep, (s, label)) && haskey(cells_ta, (s, label))) || continue
                for ci in 1:4
                    r3 = stat3(cells_rep[(s, label)][ci]); t3 = stat3(cells_ta[(s, label)][ci])
                    for si in 1:3
                        dd = abs(t3[si] - r3[si]); isnan(dd) && continue
                        kind = ci <= 2 ? "pay" : "profit"
                        for sel in ("any", si == 1 ? "means" : "")
                            isempty(sel) && continue
                            key = "$kind/$sel"
                            if !haskey(best, key) || dd > best[key][1]
                                best[key] = (dd, s, colnames[ci], statnames[si], r3[si], t3[si])
                            end
                        end
                    end
                end
            end
            g(key) = haskey(best, key) ? @sprintf("%.3f (share %.2f, %s %s: %.2f -> %.2f)", best[key]...) : "n/a"
            pr(@sprintf("   %-8s: pay    any cell %-52s means %s", label, g("pay/any"), g("pay/means")))
            pr(@sprintf("   %-8s: profit any cell %-52s means %s", "", g("profit/any"), g("profit/means")))
        end
        # ---- check: the base rows of TABLE 2 against the time-average means of the `cycles: order_of_events` section ----
        # That section filters a firm's time-averaged pay by the sign of its *reported-state* profit (rep_profit > 0); TABLE 2
        # filters by the sign of its *time-averaged* profit. The profit means are identical by construction; the pay means
        # can differ only in cells where some seed's reported-state and time-averaged profits have different signs.
        ndiff = 0; maxd = 0.0
        for s in ORDER_FIRM1_HALF
            p0 = @set BASE_P.firm1_demand = s; p0 = @set p0.firm2_demand = 4.0 - s
            vc, _ = sens_rows(p0, "base"); d = cyc["base"]
            recs = [d[(r.params, r.sequential, r.seed)] for r in eachrow(vc)]
            ps = p0.price_step
            cycsec = (me([c.avg_pay1 * ps for c in recs if c.rep_profit1 > 0]), me([c.avg_pay2 * ps for c in recs if c.rep_profit2 > 0]),
                      me([c.avg_profit1 for c in recs]), me([c.avg_profit2 for c in recs]))
            tab2 = Tuple(me(cells_ta[(s, "base")][ci]) for ci in 1:4)
            for ci in 1:4
                dd = abs(cycsec[ci] - tab2[ci]); (isnan(dd) ? false : dd > 1e-9) || continue
                ndiff += 1; maxd = max(maxd, dd)
                pr(@sprintf("   base check: share %.2f %s mean differs from the `cycles: order_of_events` section: %.3f there (pay filtered by reported-state profit > 0) vs %.3f in TABLE 2 (filtered by time-averaged profit > 0)", s, colnames[ci], cycsec[ci], tab2[ci]))
            end
        end
        pr(ndiff == 0 ? "   base check: the base rows of TABLE 2 reproduce the per-configuration time-average means of the `cycles: order_of_events` section (price stage) in all 40 cells (max |difference| 0)" :
                        @sprintf("   base check: %d of 40 base cells differ from the `cycles: order_of_events` section (max |difference| %.3f), all because that section filters time-averaged pay by the sign of the reported-state profit while TABLE 2 uses the sign of the time-averaged profit; the other cells agree exactly", ndiff, maxd))
        save_object(joinpath(DIAG, "sensitivity.jld2"), sens)
    end
    close(io)
    println("SUMMARY_DONE -> $(joinpath(DIAG, "summary.txt"))")
end

# ---------------------------------------------------------------- main ----------------------------------------
length(ARGS) >= 1 || error("usage: diagnostics.jl <regret|cycles|sensitivity|needed_seeds|summarize> ...")
const MODE = ARGS[1]
if MODE == "summarize"
    summarize()
elseif MODE == "needed_seeds"
    length(ARGS) >= 2 || error("usage: diagnostics.jl needed_seeds <experiment>")
    println(join(needed_seeds(ARGS[2]), " "))
else
    length(ARGS) >= 4 || error("usage: diagnostics.jl $MODE <experiment|variant> <seed> <outfile>")
    WHAT = ARGS[2]; SEED = parse(Int, ARGS[3]); OUT = ARGS[4]
    t0 = time()
    df = MODE == "regret"      ? regret_unit(WHAT, SEED) :
         MODE == "cycles"      ? cycles_unit(WHAT, SEED) :
         MODE == "sensitivity" ? sensitivity_unit(WHAT, SEED) :
         error("unknown mode $MODE")
    tmp = OUT * ".tmp"
    save_object(tmp, df)
    mv(tmp, OUT; force=true)
    println("DIAG_DONE mode=$MODE what=$WHAT seed=$SEED rows=$(nrow(df)) seconds=$(round(time() - t0, digits=1))")
end
