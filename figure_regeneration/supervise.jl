# Robust supervisor for the figure regeneration. For each experiment it keeps up to P isolated
# (experiment, seed) subprocesses running, enforces a wall-clock timeout per unit and SIGKILLs any
# that wedge (the native-deadlock case — the only thing that survives an in-Julia interrupt), and
# keeps drawing fresh seeds until every config has TARGET converged seeds (or a seed cap is hit).
# Then it merges the per-seed unit files into one results file per experiment. Resumable: existing
# unit files are tallied and skipped on restart.
#
# Usage: julia --project=PROJ supervise.jl [experiment ...]    (default: all four)
# Env knobs: REGEN_P (concurrency), REGEN_TIMEOUT (sec/unit), REGEN_TARGET, REGEN_SEEDCAP

const HERE = @__DIR__
include(joinpath(HERE, "experiments.jl"))
using JLD2

const P        = parse(Int, get(ENV, "REGEN_P",       "10"))
const TIMEOUT  = parse(Int, get(ENV, "REGEN_TIMEOUT", "3600"))   # 60 min/unit; only true wedges hit it
const TARGET   = parse(Int, get(ENV, "REGEN_TARGET",  "32"))
const SEED_CAP = parse(Int, get(ENV, "REGEN_SEEDCAP", "44"))     # ~37% buffer for replacements
const UNITS    = joinpath(HERE, "results", "units")
const RESULTS  = joinpath(HERE, "results")
const LOGS     = joinpath(HERE, "logs")
const JULIA    = joinpath(Sys.BINDIR, "julia")
mkpath(UNITS); mkpath(RESULTS); mkpath(LOGS)

exps = isempty(ARGS) ? EXPERIMENTS : ARGS
println("SUPERVISE start: experiments=$(exps)  P=$P  timeout=$(TIMEOUT)s  target=$TARGET  seed_cap=$SEED_CAP")
flush(stdout)

unitfile(exp, seed) = joinpath(UNITS, "$(exp)_seed$(seed).jld2")

function spawn_unit(exp, seed)
    out = unitfile(exp, seed)
    log = joinpath(LOGS, "$(exp)_seed$(seed).log")
    cmd = `$JULIA --project=$PROJ --startup-file=no $(joinpath(HERE, "run_unit.jl")) $exp $seed $out`
    cmd = addenv(cmd, "OPENBLAS_NUM_THREADS"=>"1", "OMP_NUM_THREADS"=>"1", "OPENBLAS_MAIN_FREE"=>"1")
    io = open(log, "w")
    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    return (proc=proc, seed=seed, t0=time(), out=out, io=io)
end

# tally converged seeds per config-key from a finished unit file; returns true if the file was usable
function tally!(conv, out)
    isfile(out) || return false
    local df
    try
        df = load_object(out)
    catch e
        @warn "could not load unit file (ignored)" out
        return false
    end
    for r in eachrow(df)
        k = config_key(r.params, r.sequential)
        if haskey(conv, k) && r.converged
            conv[k] += 1
        end
    end
    return true
end

function run_experiment(exp)
    configs = configs_for(exp)
    keys = [config_key(p, s) for (p, s) in configs]
    conv = Dict(k => 0 for k in keys)
    println("\n===== $exp : $(length(configs)) unique configs =====")

    # resume: tally any unit files already on disk; remember which seeds are done
    done_seeds = Set{Int}()
    for seed in 1:SEED_CAP
        if isfile(unitfile(exp, seed))
            tally!(conv, unitfile(exp, seed)) && push!(done_seeds, seed)
        end
    end
    !isempty(done_seeds) && println("  resumed: $(length(done_seeds)) existing unit files tallied")

    active = Any[]
    next_seed = 1
    mincount() = isempty(conv) ? 0 : minimum(values(conv))
    last_report = 0.0

    while true
        # satisfied? (all configs have TARGET) -> stop launching, let in-flight drain
        satisfied = mincount() >= TARGET
        # launch up to P
        while length(active) < P && next_seed <= SEED_CAP && !satisfied
            if next_seed in done_seeds || isfile(unitfile(exp, next_seed))
                next_seed += 1; continue
            end
            push!(active, spawn_unit(exp, next_seed))
            next_seed += 1
            satisfied = mincount() >= TARGET
        end

        # target reached: kill any surplus in-flight units (incl. a possible wedge) and stop now,
        # rather than waiting out the per-unit timeout. Their files (if already written) still merge.
        if satisfied
            for a in active
                try; kill(a.proc, Base.SIGKILL); catch; end
                try; close(a.io); catch; end
            end
            empty!(active)
            break
        end

        isempty(active) && next_seed > SEED_CAP && break

        sleep(5)

        keep = Any[]
        for a in active
            if process_exited(a.proc)
                close(a.io)
                tally!(conv, a.out)
            elseif time() - a.t0 > TIMEOUT
                println("  WEDGE: killing $exp seed=$(a.seed) after $(round(Int, time()-a.t0))s (native deadlock?) -> replacing")
                kill(a.proc, Base.SIGKILL)
                try; wait(a.proc); catch; end
                close(a.io)
                # outfile absent/partial -> not tallied; a fresh seed replaces it
            else
                push!(keep, a)
            end
        end
        active = keep

        if time() - last_report > 30
            println("  [$exp] min converged/config=$(mincount())/$TARGET  running=$(length(active))  next_seed=$next_seed")
            flush(stdout)
            last_report = time()
        end
    end

    # report + merge
    short = [(k, c) for (k, c) in conv if c < TARGET]
    println("  $exp DONE: min=$(mincount())/$TARGET  configs_short=$(length(short))")
    for (k, c) in sort(short, by=x->x[2])
        println("    short: count=$c  key=$k")
    end

    merged = DataFrame()
    nfiles = 0
    for seed in 1:SEED_CAP
        f = unitfile(exp, seed)
        if isfile(f)
            try
                merged = nrow(merged) == 0 ? load_object(f) : vcat(merged, load_object(f))
                nfiles += 1
            catch; end
        end
    end
    outpath = joinpath(RESULTS, "$(exp)_results.jld2")
    save_object(outpath, merged)
    println("  merged $nfiles unit files -> $outpath ($(nrow(merged)) rows, $(sum(.!merged.converged)) non-converged)")
    flush(stdout)
    return conv
end

for exp in exps
    run_experiment(exp)
end
println("\nSUPERVISE_ALL_DONE")
