# Re-plot the four figures from the TIME AVERAGE over each reported row's terminal cycle instead of the single
# "most favorable" state the engine returns (Referee 5, Comment 3: cycles summarized by a neutral statistic).
#
# Same data selection and styling as build_figures.jl (converged seeds only, first REGEN_TARGET per configuration,
# Figure-4 symmetry reflection with firm swap, pay enters the seed average only when that firm's profit is positive),
# with these changes:
#   * each reported row's profit and pay per firm are the time averages over the row's terminal cycle, taken from
#     diagnostics/cycles_<exp>.jld2 (price-stage cycle for sequential rows, simultaneous firm-map cycle for simultaneous
#     rows; profits recomputed from each cycle state's decisions with `simulation`, as diagnostics.jl does). Records
#     are matched to result rows by (params, sequential, seed); a reported row without a record is an error.
#   * the convention becomes: a firm's pay enters the average over seeds when its time-averaged profit is positive.
#   * Figure 4 panel titles: "Sequential timing" / "Simultaneous timing" (were "... Equilibrium").
# Output: figures_timeavg/<exp>.pdf, plus figures_timeavg/compare_<exp>.pdf overlaying the current series (most
# favorable state, dashed) and the new ones (time average, solid), and figures_timeavg/CHANGES.txt with the largest
# absolute change in every plotted mean and band edge, per experiment and series.
#
# Usage: julia --project=PROJ build_figures_timeavg.jl [experiment ...]   (default: all with a cycles file)
# Requires: diagnostics/cycles_<exp>.jld2 (written by `./run_diagnostics.sh summarize` after `cycles <exp>`).

const HERE = @__DIR__
include(joinpath(HERE, "experiments.jl"))
using JLD2, Statistics, Plots, Printf, Dates
pgfplotsx()
now_string() = string(Dates.now())

const RESULTS = joinpath(HERE, "results")
const CYCLES  = joinpath(HERE, "diagnostics")
const FIGS    = joinpath(HERE, "figures_timeavg")
const TARGET  = parse(Int, get(ENV, "REGEN_TARGET", "32"))
const DFONT = 18; const TFONT = 18
mkpath(FIGS)

# ---------------------------------------------------------------- data ---------------------------------------
"Cycle records of an experiment keyed by (params, sequential, seed): the price-stage cycle for sequential rows, the
 simultaneous firm-map cycle for simultaneous rows."
function load_cycles(exp)
    f = joinpath(CYCLES, "cycles_$(exp).jld2")
    isfile(f) || error("no cycle records $f -- run `./run_diagnostics.sh cycles $exp` and `summarize` first")
    cyc = load_object(f)
    d = Dict{Any,Any}()
    for r in eachrow(cyc)
        stage = r.sequential ? "price" : "simul"
        r.stage == stage || continue
        k = (r.params, r.sequential, r.seed)
        haskey(d, k) && error("duplicate cycle record for $k")
        d[k] = r
    end
    return d
end

"Reported rows of one configuration: converged seeds only, capped at TARGET, sorted by seed (as build_figures.jl)."
function reported(data, p; sequential=true)
    mask = (data.params .== p) .& (data.sequential .== sequential) .& (data.converged)
    sub = data[mask, :]
    sort!(sub, :seed)
    nrow(sub) > TARGET && (sub = sub[1:TARGET, :])
    return sub
end

"Per-row (profit1, profit2, pay1, pay2) in dollars from the single reported state (the current figures)."
function metrics_mfs(r)
    sim = simulation(r.params, r.firms, r.workers, r.demand)
    m = result_metrics(r.params, r.firms, r.workers, r.demand, sim)
    return (m["Firm Profit 1"], m["Firm Profit 2"], m["Firm Pay 1"], m["Firm Pay 2"])
end

"Per-row (profit1, profit2, pay1, pay2) as time averages over the row's terminal cycle."
function metrics_timeavg(r, cycles)
    k = (r.params, r.sequential, r.seed)
    haskey(cycles, k) || error("reported row has no cycle record: seed=$(r.seed) sequential=$(r.sequential) firm1_demand=$(r.params.firm1_demand) firm12_demand=$(r.params.firm12_demand) firm1_supply=$(r.params.firm1_supply) alpha=$(r.params.supply_sensitivity)")
    c = cycles[k]
    ps = r.params.price_step
    return (c.avg_profit1, c.avg_profit2, c.avg_pay1 * ps, c.avg_pay2 * ps)
end

"Seed statistics of one configuration under a per-row metric function (mean, p25, p75; pay only where profit > 0)."
function summarize_config(sub, metric)
    pay1 = Float64[]; pay2 = Float64[]; profit1 = Float64[]; profit2 = Float64[]
    for r in eachrow(sub)
        pr1, pr2, py1, py2 = metric(r)
        push!(profit1, pr1); push!(profit2, pr2)
        pr1 > 0.0 && push!(pay1, py1)
        pr2 > 0.0 && push!(pay2, py2)
    end
    q(v, x) = length(v) > 0 ? quantile(v, x) : NaN
    me(v)   = length(v) > 0 ? mean(v) : NaN
    return (n=nrow(sub),
        avg_pay1=me(pay1), avg_pay2=me(pay2), avg_profit1=me(profit1), avg_profit2=me(profit2),
        max_pay1=q(pay1,.75), max_pay2=q(pay2,.75), max_profit1=q(profit1,.75), max_profit2=q(profit2,.75),
        min_pay1=q(pay1,.25), min_pay2=q(pay2,.25), min_profit1=q(profit1,.25), min_profit2=q(profit2,.25))
end

swapfirms(r) = (n=r.n,
    avg_pay1=r.avg_pay2, avg_pay2=r.avg_pay1, avg_profit1=r.avg_profit2, avg_profit2=r.avg_profit1,
    max_pay1=r.max_pay2, max_pay2=r.max_pay1, max_profit1=r.max_profit2, max_profit2=r.max_profit1,
    min_pay1=r.min_pay2, min_pay2=r.min_pay1, min_profit1=r.min_profit2, min_profit2=r.min_profit1)

rib(results, avg, lo, hi) = ([getfield(r,avg)-getfield(r,lo) for r in results],
                             [getfield(r,hi)-getfield(r,avg) for r in results])

const FIELDS = (:avg_pay1, :avg_pay2, :avg_profit1, :avg_profit2, :max_pay1, :max_pay2, :max_profit1, :max_profit2,
                :min_pay1, :min_pay2, :min_profit1, :min_profit2)

# ---------------------------------------------------------------- series per figure ---------------------------
# Each builder returns the plotted series as a vector of (panel label, x, results) so that the same code draws the new
# figure, the comparison overlay and the change table.

function series_order_of_events(data, metric)
    x = 100 .* ORDER_FIRM1_FULL ./ 4.0
    setp(s) = (p = @set BASE_P.firm1_demand = s; @set p.firm2_demand = 4.0 - s)
    halves(seq) = [summarize_config(reported(data, setp(s); sequential=seq), metric) for s in ORDER_FIRM1_HALF]
    full(seq) = (h = halves(seq); [h; swapfirms.(reverse(h)[2:end])])
    return [("sequential", x, full(true)), ("simultaneous", x, full(false))]
end

function series_dedicated(data, metric, kind)
    N = kind == :supply ? DEDICATED_SUPPLY_N : TOTAL_DEMAND
    setp(fd, f1) = kind == :supply ?
        (p = @set BASE_P.firm1_supply = N*fd*f1; p = @set p.firm2_supply = N*fd*(1-f1); @set p.firm12_supply = N*(1-fd)) :
        (p = @set BASE_P.firm1_demand = N*fd*f1; p = @set p.firm2_demand = N*fd*(1-f1); @set p.firm12_demand = N*(1-fd))
    fracs = kind == :supply ? DED_SUPPLY_FRACTIONS : DED_FRACTIONS
    x = 100 .* fracs
    return [("$(Int(100*f1))/$(Int(100*(1-f1)))", x, [summarize_config(reported(data, setp(fd, f1)), metric) for fd in fracs]) for f1 in DED_FRACTIONS_FIRM1]
end

function series_supply_sensitivity(data, metric)
    setp(ss, ratio) = (p = @set BASE_P.supply_sensitivity = ss; p = @set p.firm1_demand = TOTAL_DEMAND*ratio; @set p.firm2_demand = TOTAL_DEMAND*(1-ratio))
    return [("$(Int(100*ratio))/$(Int(100*(1-ratio)))", SUPPLY_SENSITIVITIES, [summarize_config(reported(data, setp(ss, ratio)), metric) for ss in SUPPLY_SENSITIVITIES]) for ratio in SS_DEMAND_RATIOS]
end

# ---------------------------------------------------------------- drawing -----------------------------------
# `new` = time-average series (solid, drawn as in build_figures.jl); `old` = most-favorable-state series (drawn only in
# the comparison figure: dashed means, lighter ribbons). Axis limits are computed from the drawn series, as the original.

allres(sets...) = [r for s in sets for (_, _, res) in s for r in res]
function limits(sets...)
    rs = allres(sets...)
    pmax = maximum(max(r.max_profit1, r.max_profit2) for r in rs)
    ymax = maximum(maximum(filter(!isnan, [r.max_pay1, r.max_pay2])) for r in rs)
    return pmax, ymax
end

function draw_pair!(pp, x, res, field_avg, field_lo, field_hi, color, label; old=nothing)
    plot!(pp, x, [getfield(r, field_avg) for r in res], ribbon=rib(res, field_avg, field_lo, field_hi), fillalpha=.2, label=label, color=color, lw=2)
    if old !== nothing
        plot!(pp, x, [getfield(r, field_avg) for r in old], ribbon=rib(old, field_avg, field_lo, field_hi), fillalpha=.08, label="", color=color, lw=2, ls=:dash)
    end
end

# ---------- Figure 4: order_of_events (2x2), symmetry half+reflect ----------
function build_order_of_events(new, path; old=nothing)
    pmax, ymax = old === nothing ? limits(new) : limits(new, old)
    prange = [0.0, 1.05pmax]; yrange = [0.0, 1.05ymax]
    plots = []
    for (i, (name, x, res)) in enumerate(new)
        seq = name == "sequential"
        o = old === nothing ? nothing : old[i][3]
        pp = plot(title=seq ? "Sequential timing" : "Simultaneous timing",
            ylabel=(seq ? "Profit (\$/hr)" : ""), xlabel="", legend=(seq ? :top : false),
            ylims=prange, xlims=(0,100), titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        draw_pair!(pp, x, res, :avg_profit1, :min_profit1, :max_profit1, :blue, (seq ? "Firm 1" : ""); old=o)
        draw_pair!(pp, x, res, :avg_profit2, :min_profit2, :max_profit2, :red,  (seq ? "Firm 2" : ""); old=o)
        push!(plots, pp)
    end
    for (i, (name, x, res)) in enumerate(new)
        seq = name == "sequential"
        o = old === nothing ? nothing : old[i][3]
        pp = plot(ylabel=(seq ? "Worker Pay (\$/hr)" : ""), xlabel="Firm 1 Market Share (%)",
            legend=false, ylims=yrange, xlims=(0,100), titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        draw_pair!(pp, x, res, :avg_pay1, :min_pay1, :max_pay1, :blue, ""; old=o)
        draw_pair!(pp, x, res, :avg_pay2, :min_pay2, :max_pay2, :red,  ""; old=o)
        push!(plots, pp)
    end
    savefig(plot(plots..., layout=(2,2), size=(1200,600)), path)
end

# ---------- dedicated-share figures (2x3) ----------
function build_dedicated(new, kind, path; old=nothing)
    xlab = kind == :supply ? "Dedicated Supply (%)" : "Dedicated Demand (%)"
    titw = kind == :supply ? "Dedicated Supply Share" : "Dedicated Demand Share"
    pmax, ymax = old === nothing ? limits(new) : limits(new, old)
    prange = [0.0, 1.05pmax]
    yrange = kind == :supply ? [0.0, 1.05ymax] : [0.0, ymax + 0.05*(ymax-40.0)]
    profit_plots = []; pay_plots = []
    for (i, (name, x, res)) in enumerate(new)
        o = old === nothing ? nothing : old[i][3]
        f1 = DED_FRACTIONS_FIRM1[i]
        pp = plot(title="$titw: $(Int(100*f1))% / $(Int(100*(1-f1)))%",
            ylabel=(i==1 ? "Profit (\$/hr)" : ""), xlabel="", legend=(i==1 ? :topright : false),
            ylims=prange, xlims=(0,100), titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        draw_pair!(pp, x, res, :avg_profit1, :min_profit1, :max_profit1, :blue, "Firm 1"; old=o)
        draw_pair!(pp, x, res, :avg_profit2, :min_profit2, :max_profit2, :red,  "Firm 2"; old=o)
        push!(profit_plots, pp)
        qq = plot(ylabel=(i==1 ? "Worker Pay (\$/hr)" : ""), xlabel=xlab, legend=false,
            ylims=yrange, xlims=(0,100), titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        draw_pair!(qq, x, res, :avg_pay1, :min_pay1, :max_pay1, :blue, ""; old=o)
        draw_pair!(qq, x, res, :avg_pay2, :min_pay2, :max_pay2, :red,  ""; old=o)
        push!(pay_plots, qq)
    end
    savefig(plot(profit_plots..., pay_plots..., layout=(2,3), size=(1200,600)), path)
end

# ---------- supply_sensitivity (2x2, log x) ----------
function build_supply_sensitivity(new, path; old=nothing)
    pmax, ymax = old === nothing ? limits(new) : limits(new, old)
    prange = [0.0, 1.05pmax]; yrange = [0.0, 1.05ymax]
    profit_plots = []; pay_plots = []
    for (i, (name, x, res)) in enumerate(new)
        o = old === nothing ? nothing : old[i][3]
        ratio = SS_DEMAND_RATIOS[i]
        pp = plot(title="Demand $(Int(100*ratio))%/$(Int(100*(1-ratio)))%",
            ylabel=(i==1 ? "Profit (\$/hr)" : ""), xlabel="", legend=(i==1 ? :topright : false),
            ylims=prange, xscale=:log10, titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        draw_pair!(pp, x, res, :avg_profit1, :min_profit1, :max_profit1, :blue, (i==1 ? "Firm 1" : ""); old=o)
        draw_pair!(pp, x, res, :avg_profit2, :min_profit2, :max_profit2, :red,  (i==1 ? "Firm 2" : ""); old=o)
        push!(profit_plots, pp)
        qq = plot(ylabel=(i==1 ? "Worker Pay (\$/hr)" : ""), xlabel="Supply Sensitivity", legend=false,
            ylims=yrange, xscale=:log10, titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        draw_pair!(qq, x, res, :avg_pay1, :min_pay1, :max_pay1, :blue, ""; old=o)
        draw_pair!(qq, x, res, :avg_pay2, :min_pay2, :max_pay2, :red,  ""; old=o)
        push!(pay_plots, qq)
    end
    savefig(plot(profit_plots..., pay_plots..., layout=(2,2), size=(1200,600)), path)
end

# ---------------------------------------------------------------- change table ------------------------------
"Largest |time average - most favorable| over the x grid for every plotted mean and band edge, per panel series."
function change_lines(exp, new, old)
    lines = String[]
    push!(lines, "== $exp: largest absolute change (time average vs most favorable state) over the x grid, \$ (pay) or \$/hr (profit) ==")
    for ((name, x, res), (_, _, ores)) in zip(new, old)
        for f in FIELDS
            d = [abs(getfield(r, f) - getfield(o, f)) for (r, o) in zip(res, ores)]
            d = filter(!isnan, d)
            isempty(d) && continue
            i = argmax([isnan(getfield(r, f) - getfield(o, f)) ? -Inf : abs(getfield(r, f) - getfield(o, f)) for (r, o) in zip(res, ores)])
            push!(lines, @sprintf("  %-14s %-12s max |change| = %6.3f  at x = %-8s (MFS %7.3f -> TA %7.3f)   mean |change| = %.3f",
                name, String(f), maximum(d), string(round(x[i], digits=3)), getfield(ores[i], f), getfield(res[i], f), mean(d)))
        end
    end
    return lines
end

# ---------------------------------------------------------------- main ---------------------------------------
exps = isempty(ARGS) ? filter(e -> isfile(joinpath(CYCLES, "cycles_$(e).jld2")), EXPERIMENTS) : ARGS
for exp in exps
    f = joinpath(RESULTS, "$(exp)_results.jld2")
    isfile(f) || error("no results file $f")
    data = load_object(f)
    cycles = load_cycles(exp)
    println("building $exp from $(nrow(data)) rows ($(sum(.!data.converged)) non-converged) and $(length(cycles)) cycle records...")
    mta(r) = metrics_timeavg(r, cycles)
    if exp == "order_of_events"
        new = series_order_of_events(data, mta); old = series_order_of_events(data, metrics_mfs)
        build_order_of_events(new, joinpath(FIGS, "$(exp).pdf"))
        build_order_of_events(new, joinpath(FIGS, "compare_$(exp).pdf"); old=old)
    elseif exp == "dedicated_demand" || exp == "dedicated_supply"
        kind = exp == "dedicated_supply" ? :supply : :demand
        new = series_dedicated(data, mta, kind); old = series_dedicated(data, metrics_mfs, kind)
        build_dedicated(new, kind, joinpath(FIGS, "$(exp).pdf"))
        build_dedicated(new, kind, joinpath(FIGS, "compare_$(exp).pdf"); old=old)
    elseif exp == "supply_sensitivity"
        new = series_supply_sensitivity(data, mta); old = series_supply_sensitivity(data, metrics_mfs)
        build_supply_sensitivity(new, joinpath(FIGS, "$(exp).pdf"))
        build_supply_sensitivity(new, joinpath(FIGS, "compare_$(exp).pdf"); old=old)
    else
        error("unknown experiment $exp")
    end
    lines = change_lines(exp, new, old)
    foreach(println, lines)
    open(joinpath(FIGS, "changes_$(exp).txt"), "w") do io; foreach(l -> println(io, l), lines); end
    println("  saved figures_timeavg/$(exp).pdf and compare_$(exp).pdf")
end
# merge the per-experiment change tables into CHANGES.txt (all experiments built so far)
open(joinpath(FIGS, "CHANGES.txt"), "w") do io
    println(io, "Largest absolute change in each plotted series when the figures are built from the time average over the terminal cycle (TA) instead of the most favorable state (MFS). Means and p25 (min_*) / p75 (max_*) band edges, over the first $TARGET converged seeds per configuration. Generated $(now_string()).")
    for exp in EXPERIMENTS
        g = joinpath(FIGS, "changes_$(exp).txt")
        isfile(g) && (println(io); print(io, read(g, String)))
    end
end
println("BUILD_FIGURES_TIMEAVG_DONE")
