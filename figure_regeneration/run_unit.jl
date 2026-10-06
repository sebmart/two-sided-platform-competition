# Run ONE (experiment, seed) unit: solve every config of the experiment for this seed, single
# process, single thread. Writes a DataFrame (one row per config, with a `converged` flag) to the
# given outfile via temp+rename so a SIGKILL mid-run never leaves a half-written file the supervisor
# would mistake for complete. This is the isolated, kill-safe unit of work in the robust harness.
#
# Usage: julia --project=PROJ run_unit.jl <experiment> <seed> <outfile>

const HERE = @__DIR__
include(joinpath(HERE, "experiments.jl"))
using JLD2

length(ARGS) >= 3 || error("usage: run_unit.jl <experiment> <seed> <outfile>")
const EXP  = ARGS[1]
const SEED = parse(Int, ARGS[2])
const OUT  = ARGS[3]

configs = configs_for(EXP)

rows = DataFrame(seed=Int[], sequential=Bool[], params=ProblemParameters[],
                 firms=FirmDecisions[], workers=WorkerDecisions[], demand=DemandDecisions[],
                 iters=Int[], converged=Bool[])

for (p, sequential) in configs
    local firms, workers, demand, iters, conv
    try
        firms, workers, demand = initial_guess(p, SEED, randomness=RANDOMNESS)
        if sequential
            firms, workers, demand, sim, iters = sequentialNE(p, firms, workers, demand;
                max_iter=1000, max_simulations=MAX_SIMULATIONS)
        else
            firms, workers, demand, sim, iters = simultaneousNE(p, firms, workers, demand;
                max_iter=1000, max_simulations=MAX_SIMULATIONS)
        end
        conv = true
    catch e
        @warn "solve failed (recorded non-converged)" exp=EXP seed=SEED firm1_demand=p.firm1_demand sequential=sequential
        firms, workers, demand = initial_guess(p, SEED, randomness=RANDOMNESS)
        iters = -1; conv = false
    end
    push!(rows, (SEED, sequential, p, firms, workers, demand, iters, conv))
end

tmp = OUT * ".tmp"
save_object(tmp, rows)
mv(tmp, OUT; force=true)          # atomic: OUT exists only when fully written
println("UNIT_DONE exp=$EXP seed=$SEED configs=$(nrow(rows)) nonconv=$(sum(.!rows.converged))")
