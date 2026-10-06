# Build the four regenerated figures from the merged results files. Reads results/<exp>_results.jld2,
# uses ONLY converged seeds (capped at TARGET per config), and reproduces the published plot styling.
# order_of_events uses SYMMETRY: it plots the simulated firm-1<=50% half and reflects the right half
# with a firm-swap (legitimate because the engine is exactly firm-symmetric).
#
# Usage: julia --project=PROJ build_figures.jl [experiment ...]   (default: all available)

const HERE = @__DIR__
include(joinpath(HERE, "experiments.jl"))
using JLD2, Statistics, Plots
pgfplotsx()

const RESULTS = joinpath(HERE, "results")
const FIGS    = joinpath(HERE, "figures")
const TARGET  = parse(Int, get(ENV, "REGEN_TARGET", "32"))
const DFONT = 18; const TFONT = 18
mkpath(FIGS)

# metrics for one config: converged seeds only, capped at TARGET, sorted by seed for determinism.
function get_results(data, p; sequential=true)
    mask = (data.params .== p) .& (data.sequential .== sequential) .& (data.converged)
    sub = data[mask, :]
    sort!(sub, :seed)
    nrow(sub) > TARGET && (sub = sub[1:TARGET, :])
    pay1 = Float64[]; pay2 = Float64[]; profit1 = Float64[]; profit2 = Float64[]
    for r in eachrow(sub)
        sim = simulation(r.params, r.firms, r.workers, r.demand)
        m = result_metrics(r.params, r.firms, r.workers, r.demand, sim)
        push!(profit1, m["Firm Profit 1"]); push!(profit2, m["Firm Profit 2"])
        profit1[end] > 0.0 && push!(pay1, m["Firm Pay 1"])
        profit2[end] > 0.0 && push!(pay2, m["Firm Pay 2"])
    end
    q(v, x) = length(v) > 0 ? quantile(v, x) : NaN
    me(v)   = length(v) > 0 ? mean(v) : NaN
    return (n=nrow(sub),
        avg_pay1=me(pay1), avg_pay2=me(pay2), avg_profit1=me(profit1), avg_profit2=me(profit2),
        max_pay1=q(pay1,.75), max_pay2=q(pay2,.75), max_profit1=q(profit1,.75), max_profit2=q(profit2,.75),
        min_pay1=q(pay1,.25), min_pay2=q(pay2,.25), min_profit1=q(profit1,.25), min_profit2=q(profit2,.25))
end

# swap firm-1 and firm-2 fields (for the symmetry reflection of order_of_events)
swapfirms(r) = (n=r.n,
    avg_pay1=r.avg_pay2, avg_pay2=r.avg_pay1, avg_profit1=r.avg_profit2, avg_profit2=r.avg_profit1,
    max_pay1=r.max_pay2, max_pay2=r.max_pay1, max_profit1=r.max_profit2, max_profit2=r.max_profit1,
    min_pay1=r.min_pay2, min_pay2=r.min_pay1, min_profit1=r.min_profit2, min_profit2=r.min_profit1)

rib(results, avg, lo, hi) = ([getfield(r,avg)-getfield(r,lo) for r in results],
                             [getfield(r,hi)-getfield(r,avg) for r in results])

# ---------- Figure 1: order_of_events (Fig 4), symmetry half+reflect ----------
function build_order_of_events(data)
    x = 100 .* ORDER_FIRM1_FULL ./ 4.0
    # collect simulated half, reflect with firm-swap -> full 19-point series
    setp(s) = (p = @set BASE_P.firm1_demand = s; @set p.firm2_demand = 4.0 - s)
    halves(seq) = [get_results(data, setp(s); sequential=seq) for s in ORDER_FIRM1_HALF]
    full(seq) = (h = halves(seq); [h; swapfirms.(reverse(h)[2:end])])
    seqres = full(true); simres = full(false)
    pmax = maximum(max(r.max_profit1, r.max_profit2) for r in [seqres; simres])
    ymax = maximum(maximum(filter(!isnan, [r.max_pay1, r.max_pay2])) for r in [seqres; simres])
    prange = [0.0, 1.05pmax]; yrange = [0.0, 1.05ymax]
    plots = []
    for (seq, res) in ((true, seqres), (false, simres))
        pp = plot(title=seq ? "Sequential Equilibrium" : "Simultaneous Equilibrium",
            ylabel=(seq ? "Profit (\$/hr)" : ""), xlabel="", legend=(seq ? :top : false),
            ylims=prange, xlims=(0,100), titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        plot!(pp, x, [r.avg_profit1 for r in res], ribbon=rib(res,:avg_profit1,:min_profit1,:max_profit1), fillalpha=.2, label=(seq ? "Firm 1" : ""), color=:blue, lw=2)
        plot!(pp, x, [r.avg_profit2 for r in res], ribbon=rib(res,:avg_profit2,:min_profit2,:max_profit2), fillalpha=.2, label=(seq ? "Firm 2" : ""), color=:red, lw=2)
        push!(plots, pp)
    end
    for (seq, res) in ((true, seqres), (false, simres))
        pp = plot(ylabel=(seq ? "Worker Pay (\$/hr)" : ""), xlabel="Firm 1 Market Share (%)",
            legend=false, ylims=yrange, xlims=(0,100), titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        plot!(pp, x, [r.avg_pay1 for r in res], ribbon=rib(res,:avg_pay1,:min_pay1,:max_pay1), fillalpha=.2, label="", color=:blue, lw=2)
        plot!(pp, x, [r.avg_pay2 for r in res], ribbon=rib(res,:avg_pay2,:min_pay2,:max_pay2), fillalpha=.2, label="", color=:red, lw=2)
        push!(plots, pp)
    end
    savefig(plot(plots..., layout=(2,2), size=(1200,600)), joinpath(FIGS, "order_of_events.pdf"))
end

# ---------- generic dedicated-share figure (2x3): used by dedicated_demand & dedicated_supply ----------
function build_dedicated(data, kind)  # kind = :demand or :supply
    N = kind == :supply ? DEDICATED_SUPPLY_N : TOTAL_DEMAND
    setp(fd, f1) = kind == :supply ?
        (p = @set BASE_P.firm1_supply = N*fd*f1; p = @set p.firm2_supply = N*fd*(1-f1); @set p.firm12_supply = N*(1-fd)) :
        (p = @set BASE_P.firm1_demand = N*fd*f1; p = @set p.firm2_demand = N*fd*(1-f1); @set p.firm12_demand = N*(1-fd))
    xlab = kind == :supply ? "Dedicated Supply (%)" : "Dedicated Demand (%)"
    titw = kind == :supply ? "Dedicated Supply Share" : "Dedicated Demand Share"
    fracs = kind == :supply ? DED_SUPPLY_FRACTIONS : DED_FRACTIONS   # supply has extra 85%/95% points
    x = 100 .* fracs
    series(f1) = [get_results(data, setp(fd, f1)) for fd in fracs]
    cols = [series(f1) for f1 in DED_FRACTIONS_FIRM1]
    pmax = maximum(max(r.max_profit1, r.max_profit2) for c in cols for r in c)
    ymax = maximum(maximum(filter(!isnan, [r.max_pay1, r.max_pay2])) for c in cols for r in c)
    prange = [0.0, 1.05pmax]
    yrange = kind == :supply ? [0.0, 1.05ymax] : [0.0, ymax + 0.05*(ymax-40.0)]
    profit_plots = []; pay_plots = []
    for (i, f1) in enumerate(DED_FRACTIONS_FIRM1)
        res = cols[i]
        pp = plot(title="$titw: $(Int(100*f1))% / $(Int(100*(1-f1)))%",
            ylabel=(i==1 ? "Profit (\$/hr)" : ""), xlabel="", legend=(i==1 ? :topright : false),
            ylims=prange, xlims=(0,100), titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        plot!(pp, x, [r.avg_profit1 for r in res], ribbon=rib(res,:avg_profit1,:min_profit1,:max_profit1), fillalpha=.2, label="Firm 1", color=:blue, lw=2)
        plot!(pp, x, [r.avg_profit2 for r in res], ribbon=rib(res,:avg_profit2,:min_profit2,:max_profit2), fillalpha=.2, label="Firm 2", color=:red, lw=2)
        push!(profit_plots, pp)
        qq = plot(ylabel=(i==1 ? "Worker Pay (\$/hr)" : ""), xlabel=xlab, legend=false,
            ylims=yrange, xlims=(0,100), titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        plot!(qq, x, [r.avg_pay1 for r in res], ribbon=rib(res,:avg_pay1,:min_pay1,:max_pay1), fillalpha=.2, label="", color=:blue, lw=2)
        plot!(qq, x, [r.avg_pay2 for r in res], ribbon=rib(res,:avg_pay2,:min_pay2,:max_pay2), fillalpha=.2, label="", color=:red, lw=2)
        push!(pay_plots, qq)
    end
    name = kind == :supply ? "dedicated_supply.pdf" : "dedicated_demand.pdf"
    savefig(plot(profit_plots..., pay_plots..., layout=(2,3), size=(1200,600)), joinpath(FIGS, name))
end

# ---------- Figure: supply_sensitivity (2x2, log x) ----------
function build_supply_sensitivity(data)
    setp(ss, ratio) = (p = @set BASE_P.supply_sensitivity = ss; p = @set p.firm1_demand = TOTAL_DEMAND*ratio; @set p.firm2_demand = TOTAL_DEMAND*(1-ratio))
    series(ratio) = [get_results(data, setp(ss, ratio)) for ss in SUPPLY_SENSITIVITIES]
    cols = [series(r) for r in SS_DEMAND_RATIOS]
    pmax = maximum(max(r.max_profit1, r.max_profit2) for c in cols for r in c)
    ymax = maximum(maximum(filter(!isnan, [r.max_pay1, r.max_pay2])) for c in cols for r in c)
    prange = [0.0, 1.05pmax]; yrange = [0.0, 1.05ymax]
    profit_plots = []; pay_plots = []
    for (i, ratio) in enumerate(SS_DEMAND_RATIOS)
        res = cols[i]
        pp = plot(title="Demand $(Int(100*ratio))%/$(Int(100*(1-ratio)))%",
            ylabel=(i==1 ? "Profit (\$/hr)" : ""), xlabel="", legend=(i==1 ? :topright : false),
            ylims=prange, xscale=:log10, titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        plot!(pp, SUPPLY_SENSITIVITIES, [r.avg_profit1 for r in res], ribbon=rib(res,:avg_profit1,:min_profit1,:max_profit1), fillalpha=.2, label=(i==1 ? "Firm 1" : ""), color=:blue, lw=2)
        plot!(pp, SUPPLY_SENSITIVITIES, [r.avg_profit2 for r in res], ribbon=rib(res,:avg_profit2,:min_profit2,:max_profit2), fillalpha=.2, label=(i==1 ? "Firm 2" : ""), color=:red, lw=2)
        push!(profit_plots, pp)
        qq = plot(ylabel=(i==1 ? "Worker Pay (\$/hr)" : ""), xlabel="Supply Sensitivity", legend=false,
            ylims=yrange, xscale=:log10, titlefont=TFONT, guidefont=DFONT, tickfont=DFONT, legendfont=DFONT)
        plot!(qq, SUPPLY_SENSITIVITIES, [r.avg_pay1 for r in res], ribbon=rib(res,:avg_pay1,:min_pay1,:max_pay1), fillalpha=.2, label="", color=:blue, lw=2)
        plot!(qq, SUPPLY_SENSITIVITIES, [r.avg_pay2 for r in res], ribbon=rib(res,:avg_pay2,:min_pay2,:max_pay2), fillalpha=.2, label="", color=:red, lw=2)
        push!(pay_plots, qq)
    end
    savefig(plot(profit_plots..., pay_plots..., layout=(2,2), size=(1200,600)), joinpath(FIGS, "supply_sensitivity.pdf"))
end

exps = isempty(ARGS) ? EXPERIMENTS : ARGS
for exp in exps
    f = joinpath(RESULTS, "$(exp)_results.jld2")
    if !isfile(f); println("skip $exp (no results file)"); continue; end
    data = load_object(f)
    println("building $exp from $(nrow(data)) rows ($(sum(.!data.converged)) non-converged)...")
    exp == "order_of_events"    && build_order_of_events(data)
    exp == "dedicated_demand"   && build_dedicated(data, :demand)
    exp == "supply_sensitivity" && build_supply_sensitivity(data)
    exp == "dedicated_supply"   && build_dedicated(data, :supply)
    println("  saved figures/$(exp).pdf")
end
println("BUILD_FIGURES_DONE")
