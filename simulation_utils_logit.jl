module FreeridingSimulation

export ProblemParameters, WorkerDecisions, DemandDecisions, FirmDecisions
export initial_guess, result_metrics, simulation, update_params, print_results
export sequentialNE, sequentialNE_demand, sequentialNE_pay, sequentialNE_supply
export simultaneousNE, simultaneousNE_supplydemand

using Random, Distributions, Printf
using LinearAlgebra: norm, Tridiagonal
using SparseArrays
using OffsetArrays: Origin # for 1-based indexing

import Base: broadcastable

"""Problem Parameters"""
@kwdef struct ProblemParameters
    # Firm parameters
    ### Job duration for each firm
    time1::Float64
    time2::Float64
    
    # Demand parameters (logit model)
    firm1_demand::Float64 # the maximum rate of customers dedicated to firm 1
    firm2_demand::Float64 # the maximum rate of customers dedicated to firm 2
    firm12_demand::Float64 # the maximum rate of customers that are flexible
    service_value::Float64 # value of being served by a firm
    no_service_cost::Float64 # cost of not being served
    demand_sensitivity::Float64 # sensitivity of the customers to the utility in the logit model
    
    # Worker parameters (also logit model)
    firm1_supply::Float64 # the maximum rate of workers dedicated to firm 1
    firm2_supply::Float64 # the maximum rate of workers dedicated to firm 2
    firm12_supply::Float64 # the maximum rate of workers that are flexible
    supply_sensitivity::Float64 # sensitivity of the workers to the wage in the logit model
    supply_reservation_wage::Float64 # the equilibrium wage when the supply of workers is infinitely elastic (supply_sensitivity -> Inf)

    ### Simulation parameters
    "Discretized step for the arrival rates"
    rate_step::Float64
    "Discretized step for the prices"
    price_step::Float64
    "the cap of the queue in the simulation (higher = closer to full queuing setting accurate but slower)"
    max_queue::Int
end
broadcastable(p::ProblemParameters) = Ref(p)


"""Decisions of the workers"""
@kwdef struct WorkerDecisions
    "Arrival rate of workers dedicated to firm 1"
    rate1::Int
    "Arrival rate of workers dedicated to firm 2"
    rate2::Int
    "Arrival rate of flexible workers that only work for firm 1"
    rate12_1::Int
    "Arrival rate of flexible workers that only work for firm 2"
    rate12_2::Int
    "Arrival rate of flexible workers that work for both firms"
    rate12_12::Int
end

function Base.show(io::IO, w::WorkerDecisions)
    print(io, "Workers Rates     -- Dedi 1: $(w.rate1), Dedi 2 : $(w.rate2), Flex: 1=$(w.rate12_1) 2=$(w.rate12_2) 12=$(w.rate12_12)")
end

"""Decisions of the customers"""
@kwdef struct DemandDecisions
    "Arrival rate of dedicated customers to firm 1"
    rate1::Int
    "Arrival rate of dedicated customers to firm 2"
    rate2::Int
    "Arrival rate of flexible customers that only go to firm 1"
    rate12_1::Int
    "Arrival rate of flexible customers that only go to firm 2"
    rate12_2::Int
    "Arrival rate of flexible customers that go to both firms"
    rate12_12::Int
end

function Base.show(io::IO, d::DemandDecisions)
    print(io, "Customer Rates    -- Dedi 1: $(d.rate1), Dedi 2 : $(d.rate2), Flex: 1=$(d.rate12_1) 2=$(d.rate12_2) 12=$(d.rate12_12)")
end

"""Decisions of the firms"""
@kwdef struct FirmDecisions
    "Price charged by Firm 1 to its customers for one request"
    price1::Int
    "Price charged by Firm 2 to its customers for one request"
    price2::Int
    "Wage paid by Firm 1 to its workers for one request"
    pay1::Int
    "Wage paid by Firm 2 to its workers for one request"
    pay2::Int
end

function Base.show(io::IO, f::FirmDecisions)
    print(io, @sprintf("""Firm Price/Pay    -- 1: %d/%d 2: %d/%d""", f.price1, f.pay1, f.price2, f.pay2))
end

"""The discrete problem corresponding to the queuing model"""
struct DiscreteMarkovChain
    proba_arrival1::Float64
    proba_arrival2::Float64
    proba_arrival12::Float64
    proba_service1::Float64
    proba_service2::Float64
    proba_service12::Float64
    max_queue::Int
end


"""Simulation results"""
@kwdef struct SimulationResults
    stationary_distribution::Any = nothing

    # number of workers of each type (as given by Little's Law)
    workers1::Float64 = 0.
    workers2::Float64 = 0.
    workers12::Float64 = 0.
    
    # wages of each type of workers
    wage1::Float64 = 0.
    wage2::Float64 = 0.
    wage12::Float64 = 0.
    
    # worker utilization for each type of workers
    utilization1::Float64 = 0.
    utilization2::Float64 = 0.
    utilization12::Float64 = 0.

    # service level of each type of customers
    service1::Float64 = 0.
    service2::Float64 = 0.
    service12::Float64 = 0.

    # utility of each type of customers
    demand1_utility::Float64 = 0.
    demand2_utility::Float64 = 0.
    demand12_utility::Float64 = 0.

    # profit of each firm
    profit1::Float64 = 0.
    profit2::Float64 = 0.
end
function Base.show(io::IO, s::SimulationResults)
    print(io, 
            @sprintf("""
Firm Profit       -- Firm 1: %8.2f, Firm 2: %8.2f
Number of Workers -- Dedi 1: %8.2f, Dedi 2: %8.2f, Flex: %8.2f
Worker Utilization-- Dedi 1: %8.2f, Dedi 2: %8.2f, Flex: %8.2f
Worker Wage       -- Dedi 1: %8.2f, Dedi 2: %8.2f, Flex: %8.2f
Service Level     -- Dedi 1: %8.2f, Dedi 2: %8.2f, Flex: %8.2f
Demand Utility    -- Dedi 1: %8.2f, Dedi 2: %8.2f, Flex: %8.2f
""", 
            s.profit1, s.profit2, 
            s.workers1, s.workers2, s.workers12, 
            s.utilization1, s.utilization2, s.utilization12,
            s.wage1, s.wage2, s.wage12,
            s.service1, s.service2, s.service12,
            s.demand1_utility, s.demand2_utility, s.demand12_utility))
end

function solve_stationary_distribution(mc::DiscreteMarkovChain)
    state_id(i,j,k) = i + (mc.max_queue+1) * (j + (mc.max_queue+1) * k) + 1
    I = Int[] # the origin state
    J = Int[] # the destination state
    V = Float64[]
    function transition!(i1,j1,k1,i2,j2,k2,v) # note that the matrix is transposed
        push!(J, state_id(i1,j1,k1))
        push!(I, state_id(i2,j2,k2))
        push!(V, v)
    end

    # service or arrival that changes a queue's size
    for i in 1:mc.max_queue, j in 0:mc.max_queue, k in 0:mc.max_queue
        # Server 1   
        transition!(i-1,j,k,i,j,k, mc.proba_arrival1)
        transition!(i,k,j,i-1,k,j, mc.proba_service1 * i / (i + j))
        transition!(j,k,i,j,k,i-1, mc.proba_service1 * i / (i + j))

        # Server 2
        transition!(j,i-1,k,j,i,k, mc.proba_arrival2)
        transition!(k,i,j,k,i-1,j, mc.proba_service2 * i / (i + j))
        transition!(k,j,i,k,j,i-1, mc.proba_service2 * i / (i + j))

        # Server 12
        transition!(j,k,i-1,j,k,i, mc.proba_arrival12)
        transition!(i,j,k,i-1,j,k, mc.proba_service12 * i / (i + j + k))
        transition!(j,i,k,j,i-1,k, mc.proba_service12 * i / (i + j + k))
        transition!(j,k,i,j,k,i-1, mc.proba_service12 * i / (i + j + k))
    end

    # we now need to deal with edge cases
    # Case 1: server is available but cannot serve (all queues that it can serve are empty)
    for i in 0:mc.max_queue
        transition!(0,i,0,0,i,0, mc.proba_service1) # server 1 : queues 1 and 12 are empty
        transition!(i,0,0,i,0,0, mc.proba_service2) # server 2 : queue 2 and 12 are empty
    end
    transition!(0,0,0,0,0,0, mc.proba_service12) # server 12 : all queues are empty

    # Case 2: arrival when queue is full
    for i in 0:mc.max_queue, j in 0:mc.max_queue
        transition!(mc.max_queue,i,j,mc.max_queue,i,j, mc.proba_arrival1) # queue 1 is full
        transition!(i,mc.max_queue,j,i,mc.max_queue,j, mc.proba_arrival2) # queue 2 is full
        transition!(i,j,mc.max_queue,i,j,mc.max_queue, mc.proba_arrival12) # queue 12 is full
    end 

    # removes the identity matrix and add normalization constraint
    n_rows = (mc.max_queue+1)^3 + 1
    for i in 1:(mc.max_queue+1)^3
        push!(I, i)
        push!(J, i)
        push!(V, -1.)

        push!(I, n_rows)
        push!(J, i)
        push!(V, 1.)
    end

    # if the arrival rate of a queue is zero, we add the constraint that the queue is empty
    if mc.proba_arrival1 == 0
        for i in 1:mc.max_queue, j in 0:mc.max_queue, k in 0:mc.max_queue
            push!(I, state_id(i,j,k))
            push!(J, state_id(i,j,k))
            push!(V, 1.) # this cancels the identity matrix entry and effectively sets the state probability to zero
        end
    end
    if mc.proba_arrival2 == 0
        for i in 0:mc.max_queue, j in 1:mc.max_queue, k in 0:mc.max_queue
            push!(I, state_id(i,j,k))
            push!(J, state_id(i,j,k))
            push!(V, 1.) # this cancels the identity matrix entry and effectively sets the state probability to zero
        end
    end
    if mc.proba_arrival12 == 0
        for i in 0:mc.max_queue, j in 0:mc.max_queue, k in 1:mc.max_queue
            push!(I, state_id(i,j,k))
            push!(J, state_id(i,j,k))
            push!(V, 1.) # this cancels the identity matrix entry and effectively sets the state probability to zero
        end
    end

    # build the sparse matrix
    A = sparse(I,J,V,(mc.max_queue+1)^3+1, (mc.max_queue+1)^3)
    # build the right hand side
    b = zeros((mc.max_queue+1)^3+1)
    b[(mc.max_queue+1)^3+1] = 1.
    # solve the linear system
    x = Origin(0)(reshape(A\b, mc.max_queue+1, mc.max_queue+1, mc.max_queue+1))
    # clean up the solution 
    if mc.proba_arrival1 == 0
        x[1:end,:,:] .= 0.
    end
    if mc.proba_arrival2 == 0
        x[:,1:end,:] .= 0.
    end
    if mc.proba_arrival12 == 0
        x[:,:,1:end] .= 0.
    end
    x ./= sum(x) # for numerical stability
    return x
end

"simulates the queueing problem, returns average earnings for each worker type and probability of service for each customer type"
function simulation(p::ProblemParameters, firms::FirmDecisions, w::WorkerDecisions, d::DemandDecisions)
    # build the underlying discrete markov chain
    total_rate = w.rate1 + w.rate2 + w.rate12_1 + w.rate12_2 + w.rate12_12 + d.rate1 + d.rate2 + d.rate12_1 + d.rate12_2 + d.rate12_12
    mc = DiscreteMarkovChain(
        (w.rate1+w.rate12_1)/total_rate, (w.rate2+w.rate12_2)/total_rate, w.rate12_12/total_rate,
        (d.rate1+d.rate12_1)/total_rate, (d.rate2+d.rate12_2)/total_rate, d.rate12_12/total_rate,
        p.max_queue
    )
    # Solve the linear system
    state_proba = solve_stationary_distribution(mc)


    ## Now use it to compute all metrics
    worker1_wait = 0. # average number of workers waiting
    worker2_wait = 0.
    worker12_wait = 0.
    
    served_demand = zeros(3,3) # served_demand[i,j] is the average of customers of type i served by workers of type j on a given MC time step
    for i in 0:p.max_queue, j in 0:p.max_queue, k in 0:p.max_queue # iterate through all states to compute the metrics
        pr = state_proba[i,j,k]
        worker1_wait  += i * pr 
        worker2_wait  += j * pr 
        worker12_wait += k * pr

        if i > 0 # if there is at least one customer in queue 1
            served_demand[1,1] += pr * mc.proba_service1 * i/(i+k)
            served_demand[3,1] += pr * mc.proba_service12 * i/(i+j+k)
        end
        if j > 0 # if there is at least one customer in queue 2
            served_demand[2,2] += pr * mc.proba_service2 * j/(j+k)
            served_demand[3,2] += pr * mc.proba_service12 * j/(i+j+k)
        end
        if k > 0 # if there is at least one customer in queue 12
            served_demand[1,3] += pr * mc.proba_service1 * k/(i+k)
            served_demand[2,3] += pr * mc.proba_service2 * k/(j+k)
            served_demand[3,3] += pr * mc.proba_service12 * k/(i+j+k)
        end
    end
    time_per_step = 1/(total_rate*p.rate_step)
    workers1 =  worker1_wait + (served_demand[1,1]+served_demand[3,1])*p.time1/time_per_step
    workers2 =  worker2_wait + (served_demand[2,2]+served_demand[3,2])*p.time2/time_per_step
    workers12 = worker12_wait + (served_demand[1,3]*p.time1 + served_demand[2,3]*p.time2 + served_demand[3,3]*(p.time1+p.time2)/2)/time_per_step

    if workers1 > 0
        utilization1 = 1-worker1_wait/workers1
        wage1 = firms.pay1*p.price_step*(served_demand[1,1]+served_demand[3,1])/(workers1*time_per_step)
    else
        utilization1 = NaN
        wage1 = NaN
    end
    if workers2 > 0
        utilization2 = 1-worker2_wait/workers2
        wage2 = firms.pay2*p.price_step*(served_demand[2,2]+served_demand[3,2])/(workers2*time_per_step)
    else
        utilization2 = NaN
        wage2 = NaN
    end
    if workers12 > 0
        utilization12 = 1-worker12_wait/workers12
        wage12 = p.price_step*(served_demand[1,3]*firms.pay1+served_demand[2,3]*firms.pay2+served_demand[3,3]*(firms.pay1+firms.pay2)/2)/(workers12*time_per_step)
    else
        utilization12 = NaN
        wage12 = NaN
    end

    if mc.proba_service1 > 0
        service1 = (served_demand[1,1]+served_demand[1,3])/mc.proba_service1
    else
        service1 = NaN
    end
    if mc.proba_service2 > 0
        service2 = (served_demand[2,2]+served_demand[2,3])/mc.proba_service2
    else
        service2 = NaN
    end
    if mc.proba_service12 > 0
        service12 = (served_demand[3,1]+served_demand[3,2]+served_demand[3,3])/mc.proba_service12
    else
        service12 = NaN
    end

    # utility per customer
    utility_1 = service1 * (p.service_value - p.price_step*firms.price1) + (1-service1) * (-p.no_service_cost)
    utility_2 = service2 * (p.service_value - p.price_step*firms.price2) + (1-service2) * (-p.no_service_cost)
    utility_12 = (served_demand[3,1]+served_demand[3,3]/2)/mc.proba_service12 * (p.service_value - p.price_step*firms.price1) +
        (served_demand[3,2]+served_demand[3,3]/2)/mc.proba_service12 * (p.service_value - p.price_step*firms.price2) +
        (1-service12) * (-p.no_service_cost)

    # profit per hour (not per time step)
    profit1 = p.price_step*(firms.price1-firms.pay1)*(served_demand[1,1]+served_demand[1,3]+served_demand[3,1]+served_demand[3,3]/2)/time_per_step
    profit2 = p.price_step*(firms.price2-firms.pay2)*(served_demand[2,2]+served_demand[2,3]+served_demand[3,2]+served_demand[3,3]/2)/time_per_step
    isnan(profit1) && (profit1 = 0.0) # very rare special case that happens when demand and supply are zero
    isnan(profit2) && (profit2 = 0.0)
    return SimulationResults(
        stationary_distribution=state_proba,
        workers1=workers1, workers2=workers2, workers12=workers12,
        wage1=wage1, wage2=wage2, wage12=wage12,
        utilization1=utilization1, utilization2=utilization2, utilization12=utilization12,
        demand1_utility=utility_1, demand2_utility=utility_2, demand12_utility=utility_12,
        service1=service1, service2=service2, service12=service12,
        profit1=profit1, profit2=profit2)

end

"computes the four-stage equilibrium (first stage: price, second stage: demand, third stage: pay, fourth stage: supply)"
function sequentialNE(p::ProblemParameters, firms::FirmDecisions, workers::WorkerDecisions, demand::DemandDecisions; iter_print=1, max_iter=200, verbose=false, show_intermediate=false, max_simulations=300)
    next_state = Dict{Any,Any}()
    stage_iter = 1
    firms, workers, demand, sim, total_iter = sequentialNE_demand(p, firms, workers, demand, max_iter=max_iter)
    previous_state = (firms, workers, demand, sim.profit1, sim.profit2)
    if verbose
        println("================== Iteration 0 ($total_iter simulations) ==================")
        print_results(p, firms, workers, demand, sim)
    end
    while true
        if total_iter >= max_simulations
            error("Timeout: Simulation exceeded $max_simulations simulations")
        end
        iter_this_stage = 0
        # updating firm 1 price
        if p.firm1_supply + p.firm12_supply > 0 && p.firm1_demand + p.firm12_demand > 0 # if firm 1 has the possibility to have a market
            lower_price = FirmDecisions(max(0,firms.price1-1), firms.price2, firms.pay1, firms.pay2)
            firmslower, workerslower, demandlower, simlower, iterlower = sequentialNE_demand(p, lower_price, workers, demand, max_iter=max_iter)
            iter_this_stage += iterlower
            higher_price = FirmDecisions(firms.price1+1, firms.price2, firms.pay1, firms.pay2)
            firmshigher, workershigher, demandhigher, simhigher, iterhigher = sequentialNE_demand(p, higher_price, workers, demand, max_iter=max_iter)
            iter_this_stage += iterhigher
            if simlower.profit1 >= simhigher.profit1
                firm1price = lower_price.price1
            else
                firm1price = higher_price.price1
            end
            if verbose && show_intermediate
                println("===Firm 1 price choice: $(lower_price.price1*p.price_step):\$$(simlower.profit1) ($iterlower simulations) vs $(higher_price.price1*p.price_step):\$$(simhigher.profit1) ($iterhigher simulations)")
            end
        else
            firm1price = firms.price1
        end

        # updating firm 2 price
        if p.firm2_supply + p.firm12_supply > 0 && p.firm2_demand + p.firm12_demand > 0 # if firm 2 has the possibility to have a market
            lower_price = FirmDecisions(firms.price1, max(0,firms.price2-1), firms.pay1, firms.pay2)
            firmslower, workerslower, demandlower, simlower, iterlower = sequentialNE_demand(p, lower_price, workers, demand, max_iter=max_iter)
            iter_this_stage += iterlower
            higher_price = FirmDecisions(firms.price1, firms.price2+1, firms.pay1, firms.pay2)
            firmshigher, workershigher, demandhigher, simhigher, iterhigher = sequentialNE_demand(p, higher_price, workers, demand, max_iter=max_iter)
            iter_this_stage += iterhigher
            if simlower.profit2 >= simhigher.profit2
                firm2price = lower_price.price2
            else
                firm2price = higher_price.price2
            end 
            if verbose && show_intermediate
                println("===Firm 2 price choice: $(lower_price.price2*p.price_step):\$$(simlower.profit2) ($iterlower simulations) vs $(higher_price.price2*p.price_step):\$$(simhigher.profit2) ($iterhigher simulations)")
            end
        else
            firm2price = firms.price2
        end

        # Compute best response solution
        firms = FirmDecisions(firm1price, firm2price, firms.pay1, firms.pay2)
        firms, workers, demand, sim, iter = sequentialNE_demand(p, firms, workers, demand, max_iter=max_iter)
        iter_this_stage += iter

        if verbose && stage_iter % iter_print == 0
            println("================== Iteration $stage_iter ($iter_this_stage simulations) ==================")
            print_results(p, firms, workers, demand, sim)
        end
        state = (firms,workers,demand,sim.profit1,sim.profit2)
        if haskey(next_state, previous_state)
            @assert next_state[previous_state] == state
            verbose && println("Firm prices converged at iteration $stage_iter ($total_iter total simulations)")
            break
        end
        if stage_iter >= max_iter
            display(p)
            print_results(p, firms, workers, demand, sim)
            error("Firm prices did not converge after $max_iter iterations ($total_iter simulations)")
        end
        next_state[previous_state] = state
        previous_state = state
        total_iter += iter_this_stage
        stage_iter+=1
    end

    # Go through all states in the equilibrium loop, choose the one with maximum minimum profit over the two firms (use max_profit to break ties).
    starting_state = previous_state
    opt_state = starting_state
    current_state = starting_state
    loop_count = 0
    opt_min_profit = -Inf
    opt_max_profit = -Inf
    while true
        loop_count += 1
        profit1, profit2 = current_state[4], current_state[5]
        min_profit = min(profit1, profit2)
        max_profit = max(profit1, profit2)
        if min_profit > opt_min_profit || (min_profit == opt_min_profit && max_profit > opt_max_profit)
            opt_min_profit = min_profit
            opt_max_profit = max_profit
            opt_state = current_state
        end
        current_state = next_state[current_state]
        if current_state == starting_state
            break
        end
    end
    if verbose && loop_count > 10
        @warn "Firm price equilibrium loop is $loop_count"
    end
    firms, workers, demand = opt_state
    sim = simulation(p, firms, workers, demand)
    total_iter += 1
    return firms, workers, demand, sim, total_iter
end

"second stage of the full equilibrium: the demand chooses its rate given fixed prices"
function sequentialNE_demand(p::ProblemParameters, firms::FirmDecisions, workers::WorkerDecisions, demand::DemandDecisions; iter_print=1, max_iter=200, verbose=false, show_intermediate=false)
    next_state = Dict{Any,Any}()
    stage_iter = 1

    # computes the lower-stage equilibrium (firm pay/worker supply)
    firms, workers, sim, iter_this_stage = sequentialNE_pay(p, firms, workers, demand, max_iter=max_iter)
    total_iter = iter_this_stage
    previous_state = (firms, workers, demand)
    if verbose
        println("================== Iteration 0 ($iter_this_stage simulations) ==================")
        print_results(p, firms, workers, demand, sim)
    end
    while true
        # now we update the demand - we first use the demand choice model to compute the theoretical demand and then we update the demand rate towards it
        demand1 = demand.rate1
        demand2 = demand.rate2
        demand12_1 = demand.rate12_1
        demand12_2 = demand.rate12_2
        demand12_12 = demand.rate12_12

        # first, we need to deal with edge-cases for utility computation
        if demand1 + demand12_1 > 0
            utility1 = sim.demand1_utility
        else
            utility1 = 0.0 # if there are no dedicated customers, we set the utility to 0 to encourage some customers to enter (zero utility is the reservation utility)
        end
        if demand2 + demand12_2 > 0
            utility2 = sim.demand2_utility
        else
            utility2 = 0.0 # if there are no dedicated customers, we set the utility to 0 to encourage some customers to enter (zero utility is the reservation utility)
        end
        if demand12_12 > 0
            utility12 = sim.demand12_utility
        else
            utility12 = 0.0
        end

        # Evaluate the theoretical demand of each type of customer
        theoretical_demand1 = p.firm1_demand * 1/(1+exp(-p.demand_sensitivity * utility1))
        theoretical_demand2 = p.firm2_demand * 1/(1+exp(-p.demand_sensitivity * utility2))
        max_utility = max(utility1, utility2, utility12) # we need to shift the utilities by the max utility to avoid numerical issues
        theoretical_demand12_1 = p.firm12_demand * exp(p.demand_sensitivity * (utility1-max_utility))/
            (1+exp(p.demand_sensitivity * (utility1-max_utility)) + exp(p.demand_sensitivity * (utility2-max_utility)) + exp(p.demand_sensitivity * (utility12-max_utility)))
        theoretical_demand12_2 = p.firm12_demand * exp(p.demand_sensitivity * (utility2-max_utility))/
            (1+exp(p.demand_sensitivity * (utility1-max_utility)) + exp(p.demand_sensitivity * (utility2-max_utility)) + exp(p.demand_sensitivity * (utility12-max_utility)))
        theoretical_demand12_12 = p.firm12_demand * exp(p.demand_sensitivity * (utility12-max_utility))/
            (1+exp(p.demand_sensitivity * (utility1-max_utility)) + exp(p.demand_sensitivity * (utility2-max_utility)) + exp(p.demand_sensitivity * (utility12-max_utility)))
        
        if verbose && show_intermediate && (stage_iter % iter_print == 0)
            println("===Demand 1     rate choice: theoretical=$(@sprintf("%6.3f", theoretical_demand1)) actual=$(@sprintf("%6.3f", demand1*p.rate_step))")
            println("===Demand 2     rate choice: theoretical=$(@sprintf("%6.3f", theoretical_demand2)) actual=$(@sprintf("%6.3f", demand2*p.rate_step))")
            println("===Demand 12_1  rate choice: theoretical=$(@sprintf("%6.3f", theoretical_demand12_1)) actual=$(@sprintf("%6.3f", demand12_1*p.rate_step))")
            println("===Demand 12_2  rate choice: theoretical=$(@sprintf("%6.3f", theoretical_demand12_2)) actual=$(@sprintf("%6.3f", demand12_2*p.rate_step))")
            println("===Demand 12_12 rate choice: theoretical=$(@sprintf("%6.3f", theoretical_demand12_12)) actual=$(@sprintf("%6.3f", demand12_12*p.rate_step))")
        end
        # Update the demand of each type of customer
        if demand1*p.rate_step <= theoretical_demand1 && p.firm1_demand > 0
            demand1 += 1
        else
            demand1 = max(0, demand1 - 1)
        end
        if demand2*p.rate_step <= theoretical_demand2 && p.firm2_demand > 0
            demand2 += 1
        else
            demand2 = max(0, demand2 - 1)
        end
        if demand12_1*p.rate_step <= theoretical_demand12_1 && p.firm12_demand > 0
            demand12_1 += 1
        else
            demand12_1 = max(0, demand12_1 - 1)
        end
        if demand12_2*p.rate_step <= theoretical_demand12_2 && p.firm12_demand > 0
            demand12_2 += 1
        else
            demand12_2 = max(0, demand12_2 - 1)
        end
        if demand12_12*p.rate_step <= theoretical_demand12_12 && p.firm12_demand > 0
            demand12_12 += 1
        else
            demand12_12 = max(0, demand12_12 - 1)
        end
        demand = DemandDecisions(demand1, demand2, demand12_1, demand12_2, demand12_12)
        # computes the new lower-stage equilibrium (firm pay/worker supply)
        firms, workers, sim, iter_this_stage = sequentialNE_pay(p, firms, workers, demand, max_iter=max_iter)
        total_iter += iter_this_stage

        if verbose && stage_iter % iter_print == 0
            println("================== Iteration $stage_iter ($iter_this_stage simulations) ==================")
            print_results(p, firms, workers, demand, sim)
        end
        state = (firms, workers, demand)
        if haskey(next_state, previous_state)
            @assert next_state[previous_state] == state
            verbose && println("Demand rates converged at iteration $stage_iter")
            break
        end
        if stage_iter >= max_iter
            display(p)
            print_results(p, firms, workers, demand, sim)
            error("Demand rates did not converge after $max_iter iterations")
        end
        next_state[previous_state] = state
        previous_state = state
        stage_iter+=1
    end

    # Go through all states in the equilibrium loop, choose the one with maximum total rate of demand
    starting_state = previous_state
    max_state = starting_state
    current_state = starting_state
    loop_count = 0
    max_demand_count = 0
    while true
        loop_count += 1
        demand_count = current_state[3].rate1 + current_state[3].rate2 + current_state[3].rate12_1 + current_state[3].rate12_2 + current_state[3].rate12_12
        if demand_count > max_demand_count
            max_demand_count = demand_count
            max_state = current_state
        end
        current_state = next_state[current_state]
        if current_state == starting_state
            break
        end
    end
    if verbose && loop_count > 10
        @warn "Demand equilibrium loop is $loop_count"
    end
    firms, workers, demand = max_state

    # Check if any demand rates are 1 and set them to 0: this stabilizes the simulation.
    if demand.rate1 == 1 || demand.rate2 == 1 || demand.rate12_1 == 1 || demand.rate12_2 == 1 || demand.rate12_12 == 1
        demand = DemandDecisions(
            demand.rate1 == 1 ? 0 : demand.rate1,
            demand.rate2 == 1 ? 0 : demand.rate2,
            demand.rate12_1 == 1 ? 0 : demand.rate12_1,
            demand.rate12_2 == 1 ? 0 : demand.rate12_2,
            demand.rate12_12 == 1 ? 0 : demand.rate12_12
        )
        firms, workers, sim, iter = sequentialNE_pay(p, firms, workers, demand, max_iter=max_iter)
        total_iter += iter
    end
    return firms, workers, demand, sim, total_iter
end

"third stage of the full equilibrium: the firms choose the worker pay given fixed demand and prices"
function sequentialNE_pay(p::ProblemParameters, firms::FirmDecisions, workers::WorkerDecisions, demand::DemandDecisions; iter_print=1, max_iter=200, verbose=false, show_intermediate=false)
    next_state = Dict{Any,Any}()
    stage_iter = 1
    workers, sim, total_iter = sequentialNE_supply(p, firms, workers, demand, max_iter=max_iter)
    previous_state = (firms, workers, sim.profit1, sim.profit2)
    if verbose
        println("================== Iteration 0 ==================")
        print_results(p, firms, workers, demand, sim)
    end
    while true
        iter_this_stage = 0
        # updating firm 1 pay
        if p.firm1_supply + p.firm12_supply > 0 && demand.rate1 + demand.rate12_1 + demand.rate12_12 > 0 # if firm 1 has the possibility to have a market
            lower_pay = FirmDecisions(firms.price1, firms.price2, max(0,firms.pay1-1), firms.pay2)
            workers_lower, simlower, iterlower = sequentialNE_supply(p, lower_pay, workers, demand, max_iter=max_iter)
            iter_this_stage += iterlower
            higher_pay = FirmDecisions(firms.price1, firms.price2, min(firms.price1,firms.pay1+1), firms.pay2) # pay cannot exceed price (stabilizes the simulation without loss of generality)
            workers_higher, simhigher, iterhigher = sequentialNE_supply(p, higher_pay, workers, demand, max_iter=max_iter)
            iter_this_stage += iterhigher
            if simlower.profit1 > simhigher.profit1
                firm1pay = lower_pay.pay1
            else
                firm1pay = higher_pay.pay1
            end
            if verbose && (stage_iter % iter_print == 0) && show_intermediate
                println("===Firm 1 pay choice: $(lower_pay.pay1*p.price_step):\$$(simlower.profit1) ($iterlower simulations) vs $(higher_pay.pay1*p.price_step):\$$(simhigher.profit1) ($iterhigher simulations)")
            end
        else
            firm1pay = firms.pay1
        end

        # updating firm 2 pay
        if p.firm2_supply + p.firm12_supply > 0 && demand.rate2 + demand.rate12_2 + demand.rate12_12 > 0 # if firm 2 has the possibility to have a market
            lower_pay = FirmDecisions(firms.price1, firms.price2, firms.pay1, max(0,firms.pay2-1))
            workers_lower, simlower, iterlower = sequentialNE_supply(p, lower_pay, workers, demand, max_iter=max_iter)
            iter_this_stage += iterlower
            higher_pay = FirmDecisions(firms.price1, firms.price2, firms.pay1, min(firms.price2,firms.pay2+1))
            workers_higher, simhigher, iterhigher = sequentialNE_supply(p, higher_pay, workers, demand, max_iter=max_iter)
            iter_this_stage += iterhigher
            if simlower.profit2 > simhigher.profit2
                firm2pay = lower_pay.pay2
            else
                firm2pay = higher_pay.pay2
            end
            if verbose && (stage_iter % iter_print == 0) && show_intermediate
                println("===Firm 2 pay choice: $(lower_pay.pay2*p.price_step):\$$(simlower.profit2) ($iterlower simulations) vs $(higher_pay.pay2*p.price_step):\$$(simhigher.profit2) ($iterhigher simulations)")
            end
        else
            firm2pay = firms.pay2
        end

        # Compute best response solution
        firms = FirmDecisions(firms.price1, firms.price2, firm1pay, firm2pay)
        workers, sim, iter = sequentialNE_supply(p, firms, workers, demand, max_iter=max_iter)
        iter_this_stage += iter

        if verbose && stage_iter % iter_print == 0
            println("================== Iteration $stage_iter ($iter_this_stage simulations) ==================")
            print_results(p, firms, workers, demand, sim)
        end
        state = (firms,workers,sim.profit1,sim.profit2)
        if haskey(next_state, previous_state)
            @assert next_state[previous_state] == state
            verbose && println("Firm pay converged at iteration $stage_iter ($total_iter total simulations)")
            break
        end
        if stage_iter >= max_iter
            display(p)
            print_results(p, firms, workers, demand, sim)
            error("Firm pay did not converge after $max_iter iterations ($total_iter simulations)")
        end
        next_state[previous_state] = state
        previous_state = state
        total_iter += iter_this_stage
        stage_iter+=1
    end
    # Go through all states in the equilibrium loop, choose the one with maximum minimum profit over the two firms (use max_profit to break ties).
    starting_state = previous_state
    opt_state = starting_state
    current_state = starting_state
    loop_count = 0
    opt_min_profit = -Inf
    opt_max_profit = -Inf
    while true
        loop_count += 1
        profit1, profit2 = current_state[3], current_state[4]
        min_profit = min(profit1, profit2)
        max_profit = max(profit1, profit2)
        if min_profit > opt_min_profit || (min_profit == opt_min_profit && max_profit > opt_max_profit)
            opt_min_profit = min_profit
            opt_max_profit = max_profit
            opt_state = current_state
        end
        current_state = next_state[current_state]
        if current_state == starting_state
            break
        end
    end
    if verbose && loop_count > 10
        @warn "Firm pay equilibrium loop is $loop_count"
    end
    firms, workers = opt_state
    sim = simulation(p, firms, workers, demand)
    total_iter += 1
    return firms, workers, sim, total_iter
end

"fourth stage of the full equilibrium: the workers choose their supply given fixed demand and pay"
function sequentialNE_supply(p::ProblemParameters, firms::FirmDecisions, workers::WorkerDecisions, demand::DemandDecisions; iter_print=1, max_iter=200, verbose=false, show_intermediate=false)
    next_state = Dict{Any,Any}()
    stage_iter = 1 
    total_iter = 0
    previous_state = workers

    # starting simulation
    sim = simulation(p, firms, workers, demand)
    total_iter = 1
    if verbose
        println("================== Iteration 0 ==================")
        print_results(p, firms, workers, demand, sim)
    end
    while true
        # We update the worker supply rate towards a theoretical target defined by the logit model
        # Step 1:calculate the number of workers of each type (remember that the sim does not differentiate between a dedicated worker for firm 1 and a flexible worker that only works for firm 1)
        if sim.workers1 > 0
            workers1 = sim.workers1 * workers.rate1 / (workers.rate1 + workers.rate12_1)
            workers12_1 = sim.workers1 * workers.rate12_1 / (workers.rate1 + workers.rate12_1)
        else
            workers1 = 0.
            workers12_1 = 0.
        end
        if sim.workers2 > 0
            workers2 = sim.workers2 * workers.rate2 / (workers.rate2 + workers.rate12_2)
            workers12_2 = sim.workers2 * workers.rate12_2 / (workers.rate2 + workers.rate12_2)
        else
            workers2 = 0.
            workers12_2 = 0.
        end
        if sim.workers12 > 0
            workers12_12 = sim.workers12
        else
            workers12_12 = 0.
        end

        # Step 2 adjust the wage of each type of worker to handle the case where the wage is NaN (which happens when the supply is 0)
        wage1 = isnan(sim.wage1) ? -Inf : sim.wage1
        wage2 = isnan(sim.wage2) ? -Inf : sim.wage2
        wage12 = isnan(sim.wage12) ? -Inf : sim.wage12

        # Step 3: Logit model to determine the optimal supply of workers given the wage of the different types
        theoretical_supply1 = p.firm1_supply * 1/(1+exp(-p.supply_sensitivity * (wage1-p.supply_reservation_wage))) # equivalent to p.firm1_supply * exp(p.supply_sensitivity * wage1) / (exp(p.supply_sensitivity * p.supply_reservation_wage) + exp(p.supply_sensitivity * wage1))
        theoretical_supply2 = p.firm2_supply * 1/(1+exp(-p.supply_sensitivity * (wage2-p.supply_reservation_wage)))
        max_wage = max(wage1, wage2, wage12, p.supply_reservation_wage) # we need to shift the wages by the max wage to avoid numerical issues
        theoretical_supply12_1 = p.firm12_supply * exp(p.supply_sensitivity * (wage1-max_wage))/
            (exp(p.supply_sensitivity * (wage1-max_wage)) + exp(p.supply_sensitivity * (wage2-max_wage)) + exp(p.supply_sensitivity * (wage12-max_wage)) + exp(p.supply_sensitivity * (p.supply_reservation_wage-max_wage)))
        theoretical_supply12_2 = p.firm12_supply * exp(p.supply_sensitivity * (wage2-max_wage)  )/
            (exp(p.supply_sensitivity * (wage1-max_wage)) + exp(p.supply_sensitivity * (wage2-max_wage)) + exp(p.supply_sensitivity * (wage12-max_wage)) + exp(p.supply_sensitivity * (p.supply_reservation_wage-max_wage)))
        theoretical_supply12_12 = p.firm12_supply * exp(p.supply_sensitivity * (wage12-max_wage))/
            (exp(p.supply_sensitivity * (wage1-max_wage)) + exp(p.supply_sensitivity * (wage2-max_wage)) + exp(p.supply_sensitivity * (wage12-max_wage)) + exp(p.supply_sensitivity * (p.supply_reservation_wage-max_wage)))
            # equivalent to p.firm12_supply * exp(p.supply_sensitivity * wage12) / (exp(p.supply_sensitivity * p.supply_reservation_wage) + exp(p.supply_sensitivity * wage1) + exp(p.supply_sensitivity * wage2) + exp(p.supply_sensitivity * wage12))
        # Step 4: update the worker supply rate towards the theoretical target
        if p.firm1_supply > 0 && workers1 <= theoretical_supply1
            rate1 = workers.rate1 + 1
        else
            rate1 = max(0, workers.rate1 - 1)
        end
        if p.firm2_supply > 0 && workers2 <= theoretical_supply2
            rate2 = workers.rate2 + 1
        else
            rate2 = max(0, workers.rate2 - 1)
        end

        if p.firm12_supply > 0 && workers12_1 <= theoretical_supply12_1
            rate12_1 = workers.rate12_1 + 1
        else
            rate12_1 = max(0, workers.rate12_1 - 1)
        end

        if p.firm12_supply > 0 && workers12_2 <= theoretical_supply12_2
            rate12_2 = workers.rate12_2 + 1
        else
            rate12_2 = max(0, workers.rate12_2 - 1)
        end

        if p.firm12_supply > 0 && workers12_12 <= theoretical_supply12_12
            rate12_12 = workers.rate12_12 + 1
        else
            rate12_12 = max(0, workers.rate12_12 - 1)
        end

        # Now compute the new simulation with the updated worker supply
        workers = WorkerDecisions(rate1, rate2, rate12_1, rate12_2, rate12_12)
        sim = simulation(p, firms, workers, demand)
        total_iter += 1

        if verbose && show_intermediate && (stage_iter % iter_print == 0)
            if sim.workers1 > 0
                new_workers1 = sim.workers1 * workers.rate1 / (workers.rate1 + workers.rate12_1)
                new_workers12_1 = sim.workers1 * workers.rate12_1 / (workers.rate1 + workers.rate12_1)
            else
                new_workers1 = 0.
                new_workers12_1 = 0.
            end
            if sim.workers2 > 0
                new_workers2 = sim.workers2 * workers.rate2 / (workers.rate2 + workers.rate12_2)
                new_workers12_2 = sim.workers2 * workers.rate12_2 / (workers.rate2 + workers.rate12_2)
            else
                new_workers2 = 0.
                new_workers12_2 = 0.
            end
            if sim.workers12 > 0
                new_workers12_12 = sim.workers12
            else
                new_workers12_12 = 0.
            end 
            println("=== Workers 1     : theoretical=$(@sprintf("%5.3f", theoretical_supply1)) actual=$(@sprintf("%5.3f", workers1)) -> new=$(@sprintf("%5.3f", new_workers1))")
            println("=== Workers 2     : theoretical=$(@sprintf("%5.3f", theoretical_supply2)) actual=$(@sprintf("%5.3f", workers2)) -> new=$(@sprintf("%5.3f", new_workers2))")
            println("=== Workers 12_1  : theoretical=$(@sprintf("%5.3f", theoretical_supply12_1)) actual=$(@sprintf("%5.3f", workers12_1)) -> new=$(@sprintf("%5.3f", new_workers12_1))")
            println("=== Workers 12_2  : theoretical=$(@sprintf("%5.3f", theoretical_supply12_2)) actual=$(@sprintf("%5.3f", workers12_2)) -> new=$(@sprintf("%5.3f", new_workers12_2))")
            println("=== Workers 12_12 : theoretical=$(@sprintf("%5.3f", theoretical_supply12_12)) actual=$(@sprintf("%5.3f", workers12_12)) -> new=$(@sprintf("%5.3f", new_workers12_12))")
        end
        if verbose && stage_iter % iter_print == 0
            println("================== Iteration $stage_iter ==================")
            print_results(p, firms, workers, demand, sim)
        end
        state = workers
        if haskey(next_state, previous_state)
            @assert next_state[previous_state] == state
            verbose && println("Worker supply converged at iteration $stage_iter")
            break
        end
        if stage_iter >= max_iter
            display(p)
            print_results(p, firms, workers, demand, sim)
            error("Worker supply did not converge after $max_iter iterations")
        end
        next_state[previous_state] = state
        previous_state = state
        stage_iter+=1
    end

    # Go through all states in the equilibrium loop, choose the one with maximum total rate of workers.
    starting_state = previous_state
    max_state = starting_state
    current_state = starting_state
    loop_count = 0
    max_worker_count = 0
    while true
        loop_count += 1
        worker_count = current_state.rate1 + current_state.rate2 + current_state.rate12_1 + current_state.rate12_2 + current_state.rate12_12
        if worker_count > max_worker_count
            max_worker_count = worker_count
            max_state = current_state
        end
        current_state = next_state[current_state]
        if current_state == starting_state
            break
        end
    end
    if verbose && loop_count > 10
        @warn "Worker supply equilibrium loop is $loop_count"
    end
    workers = max_state
    # Check if any worker rates are 1 and set them to 0: this stabilizes the simulation.
    workers = WorkerDecisions(
        workers.rate1 == 1 ? 0 : workers.rate1,
        workers.rate2 == 1 ? 0 : workers.rate2, 
        workers.rate12_1 == 1 ? 0 : workers.rate12_1,
        workers.rate12_2 == 1 ? 0 : workers.rate12_2,
        workers.rate12_12 == 1 ? 0 : workers.rate12_12
    )
    sim = simulation(p, firms, workers, demand)
    total_iter += 1

    return workers, sim, total_iter
end

"simultaneous equilibrium between firms (price and pay), the first stage of the simultaneous equilibrium"
function simultaneousNE(p::ProblemParameters, firms::FirmDecisions, workers::WorkerDecisions, demand::DemandDecisions; iter_print=1, max_iter=200, verbose=false, show_intermediate=false, max_simulations=300)
    next_state = Dict{Any,Any}()
    stage_iter = 1
    workers, demand, sim, total_iter = simultaneousNE_supplydemand(p, firms, workers, demand, max_iter=max_iter)
    previous_state = (firms, workers, demand, sim.profit1, sim.profit2)
    if verbose
        println("================== Iteration 0 ($total_iter simulations) ==================")
        result_metrics(p, firms, workers, demand, sim)
    end
    while true
        if total_iter >= max_simulations
            error("Timeout: Simulation exceeded $max_simulations simulations")
        end
        iter_this_stage = 0
        # ===============================
        # === Firm 1 Price/Pay update ===
        # ==============================
        if p.firm1_supply + p.firm12_supply > 0 && p.firm1_demand + p.firm12_demand > 0 # if firm 1 has the possibility to have a market
            # option 0: no change
            best_firm1 = firms
            best_profit1 = sim.profit1

            # option 1: firm 1 lowers price and raises pay
            firmLH = FirmDecisions(max(0,firms.price1-1), firms.price2, firms.pay1+1, firms.pay2)
            _, _, simLH, iterLH = simultaneousNE_supplydemand(p, firmLH, workers, demand, max_iter=max_iter)
            iter_this_stage += iterLH
            if simLH.profit1 > best_profit1
                best_firm1 = firmLH
                best_profit1 = simLH.profit1
            end
            if verbose && show_intermediate && (stage_iter % iter_print == 0)
                println("===Firm 1 LH: $(firmLH.price1*p.price_step)/$(firmLH.pay1*p.price_step):\$$(simLH.profit1) ($iterLH simulations)")
            end

            # option 2: firm 1 raises price and raises pay
            firmHH = FirmDecisions(firms.price1+1, firms.price2, firms.pay1+1, firms.pay2)
            _, _, simHH, iterHH = simultaneousNE_supplydemand(p, firmHH, workers, demand, max_iter=max_iter)
            iter_this_stage += iterHH
            if simHH.profit1 > best_profit1
                best_firm1 = firmHH
                best_profit1 = simHH.profit1
            end
            if verbose && show_intermediate && (stage_iter % iter_print == 0)
                println("===Firm 1 HH: $(firmHH.price1*p.price_step)/$(firmHH.pay1*p.price_step):\$$(simHH.profit1) ($iterHH simulations)")
            end

            # option 3: firm 1 lowers price and lowers pay
            firmLL = FirmDecisions(max(0,firms.price1-1), firms.price2, max(0,firms.pay1-1), firms.pay2)
            _, _, simLL, iterLL = simultaneousNE_supplydemand(p, firmLL, workers, demand, max_iter=max_iter)
            iter_this_stage += iterLL
            if simLL.profit1 > best_profit1
                best_firm1 = firmLL
                best_profit1 = simLL.profit1
            end
            if verbose && show_intermediate && (stage_iter % iter_print == 0)
                println("===Firm 1 LL: $(firmLL.price1*p.price_step)/$(firmLL.pay1*p.price_step):\$$(simLL.profit1) ($iterLL simulations)")
            end
            # option 4: firm 1 raises price and lowers pay
            firmHL = FirmDecisions(firms.price1+1, firms.price2, max(0,firms.pay1-1), firms.pay2)
            _, _, simHL, iterHL = simultaneousNE_supplydemand(p, firmHL, workers, demand, max_iter=max_iter)
            iter_this_stage += iterHL
            if simHL.profit1 > best_profit1
                best_firm1 = firmHL
                best_profit1 = simHL.profit1
            end
            if verbose && show_intermediate && (stage_iter % iter_print == 0)
                println("===Firm 1 HL: $(firmHL.price1*p.price_step)/$(firmHL.pay1*p.price_step):\$$(simHL.profit1) ($iterHL simulations)")
            end
        else
            bestfirm1 = firms
        end

        # ===============================
        # === Firm 2 Price/Pay update ===
        # ===============================
        if p.firm2_supply + p.firm12_supply > 0 && p.firm2_demand + p.firm12_demand > 0 # if firm 2 has the possibility to have a market
            # option 0: no change
            best_firm2 = firms
            best_profit2 = sim.profit2

            # option 1: firm 2 lowers price and raises pay
            firmLH = FirmDecisions(firms.price1, max(0,firms.price2-1), firms.pay1, firms.pay2+1)
            _, _, simLH, iterLH = simultaneousNE_supplydemand(p, firmLH, workers, demand, max_iter=max_iter)
            iter_this_stage += iterLH
            if simLH.profit2 > best_profit2
                best_firm2 = firmLH
                best_profit2 = simLH.profit2
            end
            if verbose && show_intermediate && (stage_iter % iter_print == 0)
                println("===Firm 2 LH: $(firmLH.price2*p.price_step)/$(firmLH.pay2*p.price_step):\$$(simLH.profit2) ($iterLH simulations)")
            end

            # option 2: firm 2 raises price and raises pay
            firmHH = FirmDecisions(firms.price1, firms.price2+1, firms.pay1, firms.pay2+1)
            _, _, simHH, iterHH = simultaneousNE_supplydemand(p, firmHH, workers, demand, max_iter=max_iter)
            iter_this_stage += iterHH
            if simHH.profit2 > best_profit2
                best_firm2 = firmHH
                best_profit2 = simHH.profit2
            end
            if verbose && show_intermediate && (stage_iter % iter_print == 0)
                println("===Firm 2 HH: $(firmHH.price2*p.price_step)/$(firmHH.pay2*p.price_step):\$$(simHH.profit2) ($iterHH simulations)")
            end

            # option 3: firm 2 lowers price and lowers pay
            firmLL = FirmDecisions(firms.price1, max(0,firms.price2-1), firms.pay1, max(0,firms.pay2-1))
            _, _, simLL, iterLL = simultaneousNE_supplydemand(p, firmLL, workers, demand, max_iter=max_iter)
            iter_this_stage += iterLL
            if simLL.profit2 > best_profit2
                best_firm2 = firmLL
                best_profit2 = simLL.profit2
            end
            if verbose && show_intermediate && (stage_iter % iter_print == 0)
                println("===Firm 2 LL: $(firmLL.price2*p.price_step)/$(firmLL.pay2*p.price_step):\$$(simLL.profit2) ($iterLL simulations)")
            end

            # option 4: firm 2 raises price and lowers pay
            firmHL = FirmDecisions(firms.price1, firms.price2+1, firms.pay1, max(0,firms.pay2-1))
            _, _, simHL, iterHL = simultaneousNE_supplydemand(p, firmHL, workers, demand, max_iter=max_iter)
            iter_this_stage += iterHL
            if simHL.profit2 > best_profit2
                best_firm2 = firmHL
                best_profit2 = simHL.profit2
            end
            if verbose && show_intermediate && (stage_iter % iter_print == 0)
                println("===Firm 2 HL: $(firmHL.price2*p.price_step)/$(firmHL.pay2*p.price_step):\$$(simHL.profit2) ($iterHL simulations)")
            end
        else
            bestfirm2 = firms
        end

        # ===========================
        # === Simultaneous update ===
        # ===========================
        firms = FirmDecisions(best_firm1.price1, best_firm2.price2, best_firm1.pay1, best_firm2.pay2)
        workers, demand, sim, iter = simultaneousNE_supplydemand(p, firms, workers, demand, max_iter=max_iter)
        iter_this_stage += iter
        total_iter += iter_this_stage
        
        if verbose && stage_iter % iter_print == 0
            println("================== Iteration $stage_iter ($iter_this_stage simulations) ==================")
            result_metrics(p, firms, workers, demand, sim)
        end
        state = (firms, workers, demand, sim.profit1, sim.profit2)
        if haskey(next_state, previous_state)
            @assert next_state[previous_state] == state
            verbose && println("Firm decisions converged at iteration $stage_iter ($total_iter total simulations)")
            break
        end
        if stage_iter >= max_iter
            error("Firm prices did not converge after $max_iter iterations ($total_iter simulations)")
        end
        next_state[previous_state] = state
        previous_state = state

        stage_iter+=1
    end
     # Go through all states in the equilibrium loop, choose the one with maximum smaller profit over the two firms, and use the higher profit to break ties.
     starting_state = previous_state
     opt_state = starting_state
     current_state = starting_state
     loop_count = 0
     opt_min_profit = -Inf
     opt_max_profit = -Inf
     while true
        loop_count += 1
        profit1, profit2 = current_state[4], current_state[5]
        min_profit = min(profit1, profit2)
        max_profit = max(profit1, profit2)
        if min_profit > opt_min_profit || (min_profit == opt_min_profit && max_profit > opt_max_profit)
            opt_min_profit = min_profit
            opt_max_profit = max_profit
            opt_state = current_state
        end
        current_state = next_state[current_state]
        if current_state == starting_state
            break
        end
     end
     if verbose && loop_count > 10
         @warn "Firm price equilibrium loop is $loop_count"
     end
     firms, workers, demand = opt_state
     sim = simulation(p, firms, workers, demand)
     total_iter += 1
     return firms, workers, demand, sim, total_iter
end

"simultaneous equilibrium between workers and demand, the second stage of the simultaneous equilibrium"
function simultaneousNE_supplydemand(p::ProblemParameters, firms::FirmDecisions, workers::WorkerDecisions, demand::DemandDecisions; iter_print=1, max_iter=200, verbose=false, show_intermediate=false)
    next_state = Dict{Any,Any}()
    stage_iter = 1 
    total_iter = 0
    previous_state = (workers, demand)

    # starting simulation
    sim = simulation(p, firms, workers, demand)
    total_iter = 1
    if verbose
        println("================== Iteration 0 ==================")
        result_metrics(p, firms, workers, demand, sim)
    end
    while true
        # =====================
        # === WORKER UPDATE === We update the worker supply rate towards a theoretical target defined by the logit model
        # =====================
        # Step 1:calculate the number of workers of each type (remember that the sim does not differentiate between a dedicated worker for firm 1 and a flexible worker that only works for firm 1)
        if sim.workers1 > 0
            workers1 = sim.workers1 * workers.rate1 / (workers.rate1 + workers.rate12_1)
            workers12_1 = sim.workers1 * workers.rate12_1 / (workers.rate1 + workers.rate12_1)
        else
            workers1 = 0.
            workers12_1 = 0.
        end
        if sim.workers2 > 0
            workers2 = sim.workers2 * workers.rate2 / (workers.rate2 + workers.rate12_2)
            workers12_2 = sim.workers2 * workers.rate12_2 / (workers.rate2 + workers.rate12_2)
        else
            workers2 = 0.
            workers12_2 = 0.
        end
        if sim.workers12 > 0
            workers12_12 = sim.workers12
        else
            workers12_12 = 0.
        end

        # Step 2 adjust the wage of each type of worker to handle the case where the wage is NaN (which happens when the supply is 0)
        wage1 = isnan(sim.wage1) ? -Inf : sim.wage1
        wage2 = isnan(sim.wage2) ? -Inf : sim.wage2
        wage12 = isnan(sim.wage12) ? -Inf : sim.wage12

        # Step 3: Logit model to determine the optimal supply of workers given the wage of the different types
        theoretical_supply1 = p.firm1_supply * 1/(1+exp(-p.supply_sensitivity * (wage1-p.supply_reservation_wage))) # equivalent to p.firm1_supply * exp(p.supply_sensitivity * wage1) / (exp(p.supply_sensitivity * p.supply_reservation_wage) + exp(p.supply_sensitivity * wage1))
        theoretical_supply2 = p.firm2_supply * 1/(1+exp(-p.supply_sensitivity * (wage2-p.supply_reservation_wage)))
        max_wage = max(wage1, wage2, wage12, p.supply_reservation_wage) # we need to shift the wages by the max wage to avoid numerical issues
        theoretical_supply12_1 = p.firm12_supply * exp(p.supply_sensitivity * (wage1-max_wage))/
            (exp(p.supply_sensitivity * (wage1-max_wage)) + exp(p.supply_sensitivity * (wage2-max_wage)) + exp(p.supply_sensitivity * (wage12-max_wage)) + exp(p.supply_sensitivity * (p.supply_reservation_wage-max_wage)))
        theoretical_supply12_2 = p.firm12_supply * exp(p.supply_sensitivity * (wage2-max_wage)  )/
            (exp(p.supply_sensitivity * (wage1-max_wage)) + exp(p.supply_sensitivity * (wage2-max_wage)) + exp(p.supply_sensitivity * (wage12-max_wage)) + exp(p.supply_sensitivity * (p.supply_reservation_wage-max_wage)))
        theoretical_supply12_12 = p.firm12_supply * exp(p.supply_sensitivity * (wage12-max_wage))/
            (exp(p.supply_sensitivity * (wage1-max_wage)) + exp(p.supply_sensitivity * (wage2-max_wage)) + exp(p.supply_sensitivity * (wage12-max_wage)) + exp(p.supply_sensitivity * (p.supply_reservation_wage-max_wage)))
            # equivalent to p.firm12_supply * exp(p.supply_sensitivity * wage12) / (exp(p.supply_sensitivity * p.supply_reservation_wage) + exp(p.supply_sensitivity * wage1) + exp(p.supply_sensitivity * wage2) + exp(p.supply_sensitivity * wage12))
        # Step 4: update the worker supply rate towards the theoretical target
        if p.firm1_supply > 0 && workers1 <= theoretical_supply1
            rate1 = workers.rate1 + 1
        else
            rate1 = max(0, workers.rate1 - 1)
        end
        if p.firm2_supply > 0 && workers2 <= theoretical_supply2
            rate2 = workers.rate2 + 1
        else
            rate2 = max(0, workers.rate2 - 1)
        end

        if p.firm12_supply > 0 && workers12_1 <= theoretical_supply12_1
            rate12_1 = workers.rate12_1 + 1
        else
            rate12_1 = max(0, workers.rate12_1 - 1)
        end

        if p.firm12_supply > 0 && workers12_2 <= theoretical_supply12_2
            rate12_2 = workers.rate12_2 + 1
        else
            rate12_2 = max(0, workers.rate12_2 - 1)
        end

        if p.firm12_supply > 0 && workers12_12 <= theoretical_supply12_12
            rate12_12 = workers.rate12_12 + 1
        else
            rate12_12 = max(0, workers.rate12_12 - 1)
        end

        # Now compute the new simulation with the updated worker supply
        workers = WorkerDecisions(rate1, rate2, rate12_1, rate12_2, rate12_12)

        # =====================
        # === DEMAND UPDATE === We update the demand rate towards a theoretical target defined by the logit model
        # =====================
        # now we update the demand - we first use the demand choice model to compute the theoretical demand and then we update the demand rate towards it
        demand1 = demand.rate1
        demand2 = demand.rate2
        demand12_1 = demand.rate12_1
        demand12_2 = demand.rate12_2
        demand12_12 = demand.rate12_12

        # first, we need to deal with edge-cases for utility computation
        if demand1 + demand12_1 > 0
            utility1 = sim.demand1_utility
        else
            utility1 = 0.0 # if there are no dedicated customers, we set the utility to 0 to encourage some customers to enter (zero utility is the reservation utility)
        end
        if demand2 + demand12_2 > 0
            utility2 = sim.demand2_utility
        else
            utility2 = 0.0 # if there are no dedicated customers, we set the utility to 0 to encourage some customers to enter (zero utility is the reservation utility)
        end
        if demand12_12 > 0
            utility12 = sim.demand12_utility
        else
            utility12 = 0.0
        end

        # Evaluate the theoretical demand of each type of customer
        theoretical_demand1 = p.firm1_demand * 1/(1+exp(-p.demand_sensitivity * utility1))
        theoretical_demand2 = p.firm2_demand * 1/(1+exp(-p.demand_sensitivity * utility2))
        max_utility = max(utility1, utility2, utility12) # we need to shift the utilities by the max utility to avoid numerical issues
        theoretical_demand12_1 = p.firm12_demand * exp(p.demand_sensitivity * (utility1-max_utility))/
            (1+exp(p.demand_sensitivity * (utility1-max_utility)) + exp(p.demand_sensitivity * (utility2-max_utility)) + exp(p.demand_sensitivity * (utility12-max_utility)))
        theoretical_demand12_2 = p.firm12_demand * exp(p.demand_sensitivity * (utility2-max_utility))/
            (1+exp(p.demand_sensitivity * (utility1-max_utility)) + exp(p.demand_sensitivity * (utility2-max_utility)) + exp(p.demand_sensitivity * (utility12-max_utility)))
        theoretical_demand12_12 = p.firm12_demand * exp(p.demand_sensitivity * (utility12-max_utility))/
            (1+exp(p.demand_sensitivity * (utility1-max_utility)) + exp(p.demand_sensitivity * (utility2-max_utility)) + exp(p.demand_sensitivity * (utility12-max_utility)))

        if verbose && show_intermediate && (stage_iter % iter_print == 0)

        end
        # Update the demand of each type of customer
        if demand1*p.rate_step <= theoretical_demand1 && p.firm1_demand > 0
            demand1 += 1
        else
            demand1 = max(0, demand1 - 1)
        end
        if demand2*p.rate_step <= theoretical_demand2 && p.firm2_demand > 0
            demand2 += 1
        else
            demand2 = max(0, demand2 - 1)
        end
        if demand12_1*p.rate_step <= theoretical_demand12_1 && p.firm12_demand > 0
            demand12_1 += 1
        else
            demand12_1 = max(0, demand12_1 - 1)
        end
        if demand12_2*p.rate_step <= theoretical_demand12_2 && p.firm12_demand > 0
            demand12_2 += 1
        else
            demand12_2 = max(0, demand12_2 - 1)
        end
        if demand12_12*p.rate_step <= theoretical_demand12_12 && p.firm12_demand > 0
            demand12_12 += 1
        else
            demand12_12 = max(0, demand12_12 - 1)
        end
        demand = DemandDecisions(demand1, demand2, demand12_1, demand12_2, demand12_12)


        # =====================
        # === SIMULATION UPDATE === We update the simulation with the new worker and demand rates
        # =====================

        sim = simulation(p, firms, workers, demand)
        total_iter += 1

        # print worker and demand decisions if requested
        if verbose && show_intermediate && (stage_iter % iter_print == 0)
            if sim.workers1 > 0
                new_workers1 = sim.workers1 * workers.rate1 / (workers.rate1 + workers.rate12_1)
                new_workers12_1 = sim.workers1 * workers.rate12_1 / (workers.rate1 + workers.rate12_1)
            else
                new_workers1 = 0.
                new_workers12_1 = 0.
            end
            if sim.workers2 > 0
                new_workers2 = sim.workers2 * workers.rate2 / (workers.rate2 + workers.rate12_2)
                new_workers12_2 = sim.workers2 * workers.rate12_2 / (workers.rate2 + workers.rate12_2)
            else
                new_workers2 = 0.
                new_workers12_2 = 0.
            end
            if sim.workers12 > 0
                new_workers12_12 = sim.workers12
            else
                new_workers12_12 = 0.
            end 
            println("=== Workers 1     : theoretical=$(@sprintf("%5.3f", theoretical_supply1)) actual=$(@sprintf("%5.3f", workers1)) -> new=$(@sprintf("%5.3f", new_workers1))")
            println("=== Workers 2     : theoretical=$(@sprintf("%5.3f", theoretical_supply2)) actual=$(@sprintf("%5.3f", workers2)) -> new=$(@sprintf("%5.3f", new_workers2))")
            println("=== Workers 12_1  : theoretical=$(@sprintf("%5.3f", theoretical_supply12_1)) actual=$(@sprintf("%5.3f", workers12_1)) -> new=$(@sprintf("%5.3f", new_workers12_1))")
            println("=== Workers 12_2  : theoretical=$(@sprintf("%5.3f", theoretical_supply12_2)) actual=$(@sprintf("%5.3f", workers12_2)) -> new=$(@sprintf("%5.3f", new_workers12_2))")
            println("=== Workers 12_12 : theoretical=$(@sprintf("%5.3f", theoretical_supply12_12)) actual=$(@sprintf("%5.3f", workers12_12)) -> new=$(@sprintf("%5.3f", new_workers12_12))")
            println("=== Demand 1     rate choice: theoretical=$(@sprintf("%6.3f", theoretical_demand1)) actual=$(@sprintf("%6.3f", demand1*p.rate_step))")
            println("=== Demand 2     rate choice: theoretical=$(@sprintf("%6.3f", theoretical_demand2)) actual=$(@sprintf("%6.3f", demand2*p.rate_step))")
            println("=== Demand 12_1  rate choice: theoretical=$(@sprintf("%6.3f", theoretical_demand12_1)) actual=$(@sprintf("%6.3f", demand12_1*p.rate_step))")
            println("=== Demand 12_2  rate choice: theoretical=$(@sprintf("%6.3f", theoretical_demand12_2)) actual=$(@sprintf("%6.3f", demand12_2*p.rate_step))")
            println("=== Demand 12_12 rate choice: theoretical=$(@sprintf("%6.3f", theoretical_demand12_12)) actual=$(@sprintf("%6.3f", demand12_12*p.rate_step))")
        end
        # print iteration results if requested
        if verbose && stage_iter % iter_print == 0
            println("================== Iteration $stage_iter ==================")
            result_metrics(p, firms, workers, demand, sim)
        end
        # check if the state has converged
        state = (workers, demand)
        if haskey(next_state, previous_state)
            @assert next_state[previous_state] == state
            verbose && println("Demand and Workers converged at iteration $stage_iter")
            break
        end
        if stage_iter >= max_iter
            error("Demand and Workers did not converge after $max_iter iterations")
        end
        next_state[previous_state] = state
        previous_state = state
        stage_iter+=1
    end
    # All states of the final loop could be the solution. Choose the one with maximal total rate of workers.
    starting_state = previous_state
    max_state = starting_state
    current_state = starting_state
    loop_count = 0
    max_worker_count = 0
    while true
        loop_count += 1
        worker_count = current_state[1].rate1 + current_state[1].rate2 + current_state[1].rate12_1 + current_state[1].rate12_2 + current_state[1].rate12_12
        if worker_count > max_worker_count
            max_worker_count = worker_count
            max_state = current_state
        end
        current_state = next_state[current_state]
        if current_state == starting_state
            break
        end
    end
    if verbose && loop_count > 10
        @warn "Workers-demand equilibrium loop is $loop_count"
    end
    workers,demand = max_state
    # Check if any worker/demand rates are 1 and set them to 0: this is because rate=1 typically only happens to encourage exploration and is not realistic
    workers = WorkerDecisions(
        workers.rate1 == 1 ? 0 : workers.rate1,
        workers.rate2 == 1 ? 0 : workers.rate2, 
        workers.rate12_1 == 1 ? 0 : workers.rate12_1,
        workers.rate12_2 == 1 ? 0 : workers.rate12_2,
        workers.rate12_12 == 1 ? 0 : workers.rate12_12
    )
    demand = DemandDecisions(
        demand.rate1 == 1 ? 0 : demand.rate1,
        demand.rate2 == 1 ? 0 : demand.rate2,
        demand.rate12_1 == 1 ? 0 : demand.rate12_1,
        demand.rate12_2 == 1 ? 0 : demand.rate12_2,
        demand.rate12_12 == 1 ? 0 : demand.rate12_12
    )
    sim = simulation(p, firms, workers, demand)
    total_iter += 1

    return workers, demand, sim, total_iter
end

function result_metrics(p::ProblemParameters, firms::FirmDecisions, worker::WorkerDecisions, demand::DemandDecisions, sim::SimulationResults)
    metrics = Dict{String, Float64}()
    metrics["Firm Profit 1"] = sim.profit1
    metrics["Firm Profit 2"] = sim.profit2
    metrics["Firm Price 1"] = firms.price1*p.price_step
    metrics["Firm Price 2"] = firms.price2*p.price_step
    metrics["Firm Pay 1"] = firms.pay1*p.price_step
    metrics["Firm Pay 2"] = firms.pay2*p.price_step
    metrics["Worker rate 1"] = worker.rate1*p.rate_step
    metrics["Worker rate 2"] = worker.rate2*p.rate_step
    metrics["Worker rate 12_1"] = worker.rate12_1*p.rate_step
    metrics["Worker rate 12_2"] = worker.rate12_2*p.rate_step
    metrics["Worker rate 12_12"] = worker.rate12_12*p.rate_step
    metrics["Workers 1"] = sim.workers1
    metrics["Workers 2"] = sim.workers2
    metrics["Workers 12"] = sim.workers12
    metrics["Worker Utilization 1"] = sim.utilization1
    metrics["Worker Utilization 2"] = sim.utilization2
    metrics["Worker Utilization 12"] = sim.utilization12
    metrics["Worker Wage 1"] = sim.wage1
    metrics["Worker Wage 2"] = sim.wage2
    metrics["Worker Wage 12"] = sim.wage12
    metrics["Demand Rate 1"] = demand.rate1*p.rate_step
    metrics["Demand Rate 2"] = demand.rate2*p.rate_step
    metrics["Demand Rate 12_1"] = demand.rate12_1*p.rate_step
    metrics["Demand Rate 12_2"] = demand.rate12_2*p.rate_step
    metrics["Demand Rate 12_12"] = demand.rate12_12*p.rate_step
    metrics["Demand Serv. Level 1"] = sim.service1
    metrics["Demand Serv. Level 2"] = sim.service2
    metrics["Demand Serv. Level 12"] = sim.service12
    metrics["Demand Utility 1"] = sim.demand1_utility
    metrics["Demand Utility 2"] = sim.demand2_utility
    metrics["Demand Utility 12"] = sim.demand12_utility
    metrics["Max Queue Pr."] = sum(sim.stationary_distribution[p.max_queue, 0:p.max_queue-1, 0:p.max_queue-1]) + 
    sum(sim.stationary_distribution[0:p.max_queue-1, p.max_queue, 0:p.max_queue-1]) + 
    sum(sim.stationary_distribution[0:p.max_queue-1, 0:p.max_queue-1, p.max_queue]) + 
    sum(sim.stationary_distribution[p.max_queue, p.max_queue, 0:p.max_queue-1]) + 
    sum(sim.stationary_distribution[p.max_queue, 0:p.max_queue-1, p.max_queue]) + 
    sum(sim.stationary_distribution[0:p.max_queue-1, p.max_queue, p.max_queue]) + 
    sum(sim.stationary_distribution[p.max_queue, p.max_queue, p.max_queue])
    return metrics

end

function print_results(p::ProblemParameters, firms::FirmDecisions, worker::WorkerDecisions, demand::DemandDecisions, sim::SimulationResults)
    metrics = result_metrics(p, firms, worker, demand, sim)
    @printf("""
Firm Profit       -- 1: %8.2f, 2: %8.2f
Firm Price        -- 1: %8.2f, 2: %8.2f
Firm Pay          -- 1: %8.2f, 2: %8.2f
Worker rate       -- 1: %8.2f, 2: %8.2f, 12_1: %8.2f, 12_2: %8.2f, 12_12: %8.2f
Workers           -- 1: %8.2f, 2: %8.2f, 12: %10.2f
Worker Utilization-- 1: %8.2f, 2: %8.2f, 12: %10.2f
Worker Wage       -- 1: %8.2f, 2: %8.2f, 12: %10.2f
Demand Rate       -- 1: %8.2f, 2: %8.2f, 12_1: %8.2f, 12_2: %8.2f, 12_12: %8.2f
Demand Serv. Level-- 1: %8.2f, 2: %8.2f, 12: %10.2f
Demand Utility    -- 1: %8.2f, 2: %8.2f, 12: %10.2f
Max Queue Pr.     -- %.3e

""",
    metrics["Firm Profit 1"], metrics["Firm Profit 2"],
    metrics["Firm Price 1"], metrics["Firm Price 2"],
    metrics["Firm Pay 1"], metrics["Firm Pay 2"],
    metrics["Worker rate 1"], metrics["Worker rate 2"], metrics["Worker rate 12_1"], metrics["Worker rate 12_2"], metrics["Worker rate 12_12"],
    metrics["Workers 1"], metrics["Workers 2"], metrics["Workers 12"],
    metrics["Worker Utilization 1"], metrics["Worker Utilization 2"], metrics["Worker Utilization 12"],
    metrics["Worker Wage 1"], metrics["Worker Wage 2"], metrics["Worker Wage 12"],
    metrics["Demand Rate 1"], metrics["Demand Rate 2"], metrics["Demand Rate 12_1"], metrics["Demand Rate 12_2"], metrics["Demand Rate 12_12"],
    metrics["Demand Serv. Level 1"], metrics["Demand Serv. Level 2"], metrics["Demand Serv. Level 12"],
    metrics["Demand Utility 1"], metrics["Demand Utility 2"], metrics["Demand Utility 12"],
    metrics["Max Queue Pr."]
    )
end

"returns initial guess of equilibrium - the goal is to create a situation where demand, supply and firms are good (positive  utility and profit)"
function initial_guess(p::ProblemParameters, seed::Int; randomness = 0.1)
    Random.seed!(seed)

    # the start is randomized and tries to have some initial demand and pay supply as much as possible given that the firms make a positive profit
    price1 = round(Int, (1 - randomness * rand()) * p.service_value / p.price_step) - 1
    price2 = round(Int, (1 - randomness * rand()) * p.service_value / p.price_step) - 1
    firms = FirmDecisions(price1=price1, price2=price2, pay1 = price1 - 1, pay2 = price2 - 1)
    # the worker supply is set to a random value close tofull possible supply
    worker = WorkerDecisions(
        rate1 = round(Int, (1-randomness*rand()) * p.firm1_supply / (p.time1 * p.rate_step)), 
        rate2 = round(Int, (1-randomness*rand()) * p.firm2_supply / (p.time2 * p.rate_step)),
        rate12_1 = 0, # the flexible workers are initially accepting both firms, with the maximum possible rate
        rate12_2 = 0,
        rate12_12 = round(Int, (1-randomness*rand()) * p.firm12_supply / (min(p.time1, p.time2) * p.rate_step))
    )
    # the demand rate is set to a random value close to the full possible demand
    demand = DemandDecisions(
        rate1 = round(Int, (1-randomness*rand()) * p.firm1_demand / p.rate_step),
        rate2 = round(Int, (1-randomness*rand()) * p.firm2_demand / p.rate_step),
        rate12_1 = 0,
        rate12_2 = 0,
        rate12_12 = round(Int, (1-randomness*rand()) * p.firm12_demand / p.rate_step)
    )
    return firms, worker, demand
end
end # module