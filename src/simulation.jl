"""
Simulation module for SAOM.
Implements the continuous-time Markov chain simulation of network and behavior dynamics.
"""

#==============================================================================#
# Parameter Map
#==============================================================================#

"""
    ParameterMap

Mapping between the full free-parameter vector θ and the included effect entries.

The canonical θ layout is: free rate parameters first (in effects-table order,
basic rates parameterized as the rate ``\\rho_m`` itself), then free objective
parameters (in effects-table order). Fixed included effects keep their
`initial_value` and are not part of θ.
"""
struct ParameterMap
    rate_entries::Vector{EffectEntry}   # included rate entries (free and fixed)
    obj_entries::Vector{EffectEntry}    # included objective entries (free and fixed)
    free::Vector{EffectEntry}           # free rate entries, then free objective entries
    index::IdDict{EffectEntry, Int}     # entry -> position in θ
end

"""
    build_param_map(effects::SienaEffects)

Build the [`ParameterMap`](@ref) for the included effects.
"""
function build_param_map(effects::SienaEffects)
    rate_entries = EffectEntry[e for e in effects.effects
                               if e.include && e.effect isa RateEffect]
    obj_entries = EffectEntry[e for e in effects.effects
                              if e.include &&
                                 (e.effect isa NetworkEffect || e.effect isa BehaviorEffect)]
    free = vcat(EffectEntry[e for e in rate_entries if !e.fix],
                EffectEntry[e for e in obj_entries if !e.fix])
    index = IdDict{EffectEntry, Int}(e => i for (i, e) in enumerate(free))
    return ParameterMap(rate_entries, obj_entries, free, index)
end

n_free_parameters(pm::ParameterMap) = length(pm.free)
n_free_rate_parameters(pm::ParameterMap) = count(e -> e.effect isa RateEffect, pm.free)

"""
    entry_value(pm::ParameterMap, θ::Vector{Float64}, entry::EffectEntry)

Current parameter value of an included entry: its θ component if free, its
`initial_value` if fixed.
"""
entry_value(pm::ParameterMap, θ::Vector{Float64}, entry::EffectEntry) =
    haskey(pm.index, entry) ? θ[pm.index[entry]] : entry.initial_value

"""
    objective_theta(pm::ParameterMap, θ::Vector{Float64})

The objective-function part of the full parameter vector.
"""
function objective_theta(pm::ParameterMap, θ::Vector{Float64})
    length(θ) == n_free_parameters(pm) ||
        throw(ArgumentError("θ has length $(length(θ)) but the model has " *
                            "$(n_free_parameters(pm)) free parameters " *
                            "($(n_free_rate_parameters(pm)) rate + " *
                            "$(n_free_parameters(pm) - n_free_rate_parameters(pm)) objective)"))
    return θ[(n_free_rate_parameters(pm) + 1):end]
end

"""
    basic_rate(pm::ParameterMap, θ::Vector{Float64}, variable::Symbol, period::Int)

Basic rate parameter ``\\rho_m`` for a dependent variable and period (1.0 if the
model has no basic rate entry for it).
"""
function basic_rate(pm::ParameterMap, θ::Vector{Float64}, variable::Symbol, period::Int)
    for entry in pm.rate_entries
        eff = entry.effect
        if eff isa BasicRateEffect && eff.variable == variable && eff.period == period
            return max(entry_value(pm, θ, entry), 1e-6)
        end
    end
    return 1.0
end

# Non-basic rate entries for a variable, with their current parameter values.
function _nonbasic_rates(pm::ParameterMap, θ::Vector{Float64}, variable::Symbol)
    entries = EffectEntry[e for e in pm.rate_entries
                          if !(e.effect isa BasicRateEffect) &&
                             target_variable(e.effect) == variable]
    values = Float64[entry_value(pm, θ, e) for e in entries]
    return entries, values
end

"""
    parameter_names(effects::SienaEffects)

Display names of the free parameters, aligned with the θ vector.
"""
function parameter_names(effects::SienaEffects)
    pm = build_param_map(effects)
    return String[e.effect isa RateEffect ? e.name : e.shortname for e in pm.free]
end

#==============================================================================#
# Objective Function
#==============================================================================#

"""
    compute_objective(effects::SienaEffects, θ::Vector{Float64},
                     state::NetworkState, data::SienaData,
                     actor::Int, alter::Int, variable::Symbol)

Objective-function change for a potential ministep, using the **objective part** of
the parameter vector (see [`objective_theta`](@ref)).

For networks the change is a toggle of the tie `actor -> alter`; since
`compute_contribution` returns the add-direction change statistic, deletions get the
sign flip ``(1 - 2 x_{ij})``. For behavior, `alter` encodes the direction (±1) and the
contributions are already directional differences. Creation effects only enter for
additions/increases, endowment effects only for deletions/decreases.
"""
function compute_objective(effects::SienaEffects, θ::Vector{Float64},
                          state::NetworkState, data::SienaData,
                          actor::Int, alter::Int, variable::Symbol)
    dep = data.dependents[variable]
    is_network = dep isa DependentNetwork
    if is_network
        adding = state.networks[variable][actor, alter] == 0
        sign = adding ? 1.0 : -1.0
    else
        adding = alter > 0
        sign = 1.0
    end

    obj = 0.0
    param_idx = 1
    for entry in get_objective_effects(effects)
        θ_k = entry.fix ? entry.initial_value : θ[param_idx]
        if target_variable(entry.effect) == variable
            etype = effect_type(entry.effect)
            use = etype == :eval ||
                  (etype == :creation && adding) ||
                  (etype == :endow && !adding)
            if use && θ_k != 0.0
                contrib = compute_contribution(entry.effect, state, data, actor, alter)
                obj += θ_k * sign * contrib
            end
        end
        entry.fix || (param_idx += 1)
    end

    return obj
end

#==============================================================================#
# Choice Probabilities
#==============================================================================#

"""
    compute_network_choice_probs(effects::SienaEffects, θ::Vector{Float64},
                                state::NetworkState, data::SienaData,
                                actor::Int, variable::Symbol)

Compute multinomial-logit probabilities for all possible tie toggles of an actor
(plus the no-change option). Returns `(probabilities, alters)` where alter 0 is the
no-change option. For two-mode networks the alters are the events (columns).
"""
function compute_network_choice_probs(effects::SienaEffects, θ::Vector{Float64},
                                     state::NetworkState, data::SienaData,
                                     actor::Int, variable::Symbol)
    net = state.networks[variable]
    dep = data.dependents[variable]::DependentNetwork
    n_alters = size(net, 2)
    onemode = dep.type == :onemode

    objectives = Float64[0.0]     # no-change option
    valid_alters = Int[0]

    for alter in 1:n_alters
        if onemode && alter == actor && !dep.allow_self_loops
            continue
        end
        obj = compute_objective(effects, θ, state, data, actor, alter, variable)
        push!(objectives, obj)
        push!(valid_alters, alter)
    end

    max_obj = maximum(objectives)
    exp_obj = exp.(objectives .- max_obj)
    probs = exp_obj ./ sum(exp_obj)

    return probs, valid_alters
end

"""
    compute_behavior_choice_probs(effects::SienaEffects, θ::Vector{Float64},
                                 state::NetworkState, data::SienaData,
                                 actor::Int, variable::Symbol)

Compute probabilities for behavior changes (-1, 0, +1) for an actor.
Returns (probabilities, directions).
"""
function compute_behavior_choice_probs(effects::SienaEffects, θ::Vector{Float64},
                                      state::NetworkState, data::SienaData,
                                      actor::Int, variable::Symbol)
    beh = state.behaviors[variable]
    dep = data.dependents[variable]::DependentBehavior
    current = beh[actor]

    objectives = Float64[]
    valid_directions = Int[]

    for dir in (-1, 0, 1)
        new_val = current + dir
        if new_val < dep.min_val || new_val > dep.max_val
            continue
        end
        # Contributions are s_ki(z_i + d) - s_ki(z_i), so the no-change option is
        # exactly 0 on the same baseline.
        obj = dir == 0 ? 0.0 :
              compute_objective(effects, θ, state, data, actor, dir, variable)
        push!(objectives, obj)
        push!(valid_directions, dir)
    end

    if isempty(objectives)
        return [1.0], [0]
    end

    max_obj = maximum(objectives)
    exp_obj = exp.(objectives .- max_obj)
    probs = exp_obj ./ sum(exp_obj)

    return probs, valid_directions
end

#==============================================================================#
# Mini-Step Execution
#==============================================================================#

"""
    execute_network_ministep!(state::NetworkState, effects::SienaEffects,
                             θ::Vector{Float64}, data::SienaData,
                             actor::Int, variable::Symbol, rng::AbstractRNG)

Execute a network mini-step for the given actor. Returns the chosen alter (0 for
no change).
"""
function execute_network_ministep!(state::NetworkState, effects::SienaEffects,
                                  θ::Vector{Float64}, data::SienaData,
                                  actor::Int, variable::Symbol, rng::AbstractRNG)
    probs, alters = compute_network_choice_probs(effects, θ, state, data, actor, variable)

    u = rand(rng)
    cumsum_p = 0.0
    chosen_alter = 0
    for i in eachindex(probs)
        cumsum_p += probs[i]
        if u <= cumsum_p
            chosen_alter = alters[i]
            break
        end
    end

    if chosen_alter > 0
        net = state.networks[variable]
        net[actor, chosen_alter] = 1 - net[actor, chosen_alter]
    end

    return chosen_alter
end

"""
    execute_behavior_ministep!(state::NetworkState, effects::SienaEffects,
                              θ::Vector{Float64}, data::SienaData,
                              actor::Int, variable::Symbol, rng::AbstractRNG)

Execute a behavior mini-step for the given actor. Returns the chosen direction.
"""
function execute_behavior_ministep!(state::NetworkState, effects::SienaEffects,
                                   θ::Vector{Float64}, data::SienaData,
                                   actor::Int, variable::Symbol, rng::AbstractRNG)
    probs, directions = compute_behavior_choice_probs(effects, θ, state, data, actor, variable)

    u = rand(rng)
    cumsum_p = 0.0
    chosen_dir = 0
    for i in eachindex(probs)
        cumsum_p += probs[i]
        if u <= cumsum_p
            chosen_dir = directions[i]
            break
        end
    end

    if chosen_dir != 0
        state.behaviors[variable][actor] += chosen_dir
    end

    return chosen_dir
end

#==============================================================================#
# Period Simulation
#==============================================================================#

"""
    SimulationResult

Result of simulating a period.

# Fields
- `final_state::NetworkState`: snapshot of the state at the end of the period
- `n_network_changes::Dict{Symbol, Int}`: number of tie changes per network
- `n_behavior_changes::Dict{Symbol, Int}`: number of behavior changes per behavior
"""
struct SimulationResult
    final_state::NetworkState
    n_network_changes::Dict{Symbol, Int}
    n_behavior_changes::Dict{Symbol, Int}
end

_n_variable_actors(state::NetworkState, data::SienaData, variable::Symbol) =
    data.dependents[variable] isa DependentNetwork ?
        size(state.networks[variable], 1) : length(state.behaviors[variable])

# Per-actor rates for one dependent variable.
function _variable_actor_rates!(rates::Vector{Float64}, state::NetworkState,
                                data::SienaData, variable::Symbol, λ_basic::Float64,
                                nb_entries::Vector{EffectEntry}, nb_θ::Vector{Float64})
    n = _n_variable_actors(state, data, variable)
    resize!(rates, n)
    if isempty(nb_entries)
        fill!(rates, λ_basic)
    else
        for i in 1:n
            rates[i] = actor_rate(λ_basic, nb_entries, nb_θ, state, data, i)
        end
    end
    return rates
end

function _sample_categorical(rng::AbstractRNG, weights::Vector{Float64}, total::Float64)
    u = rand(rng) * total
    acc = 0.0
    for i in eachindex(weights)
        acc += weights[i]
        u <= acc && return i
    end
    return length(weights)
end

"""
    simulate_period!(state::NetworkState, effects::SienaEffects,
                    θ_obj::Vector{Float64}, rate_params::Dict{Symbol, Float64},
                    data::SienaData, rng::AbstractRNG; kwargs...)

Simulate one period of network/behavior dynamics over the unit time interval.

# Arguments
- `θ_obj`: objective-function parameters (see [`objective_theta`](@ref))
- `rate_params`: basic rate ``\\rho`` per dependent variable for this period
- `nonbasic_rates`: optional `Dict(variable => (entries, values))` of non-basic rate
  effect entries and their parameters
- `conditional`/`target_changes`: if `conditional=true`, simulate until the target
  number of changes is reached instead of until time 1
- `max_steps`: safety cap on the number of ministeps
"""
function simulate_period!(state::NetworkState, effects::SienaEffects,
                         θ_obj::Vector{Float64}, rate_params::Dict{Symbol, Float64},
                         data::SienaData, rng::AbstractRNG;
                         nonbasic_rates::Dict{Symbol, Tuple{Vector{EffectEntry}, Vector{Float64}}} =
                             Dict{Symbol, Tuple{Vector{EffectEntry}, Vector{Float64}}}(),
                         conditional::Bool=false,
                         target_changes::Union{Nothing, Dict{Symbol, Int}}=nothing,
                         max_steps::Int=100_000)

    n_network_changes = Dict{Symbol, Int}()
    n_behavior_changes = Dict{Symbol, Int}()
    for (name, dep) in data.dependents
        if dep isa DependentNetwork
            n_network_changes[name] = 0
        else
            n_behavior_changes[name] = 0
        end
    end

    variables = collect(keys(data.dependents))
    empty_entries = (EffectEntry[], Float64[])
    actor_rates = Dict(v => Float64[] for v in variables)
    var_totals = zeros(length(variables))

    step = 0
    while step < max_steps
        step += 1

        # Per-actor rates for every variable
        total_rate = 0.0
        for (k, v) in enumerate(variables)
            λ = get(rate_params, v, 1.0)
            nb_entries, nb_θ = get(nonbasic_rates, v, empty_entries)
            rates = _variable_actor_rates!(actor_rates[v], state, data, v, λ,
                                           nb_entries, nb_θ)
            var_totals[k] = sum(rates)
            total_rate += var_totals[k]
        end

        total_rate <= 0 && break

        # Waiting time
        dt = sample_waiting_time(total_rate, rng)
        state.time += dt
        if !conditional && state.time >= 1.0
            break
        end

        # Which variable changes, and which actor moves
        k = _sample_categorical(rng, var_totals, total_rate)
        selected_var = variables[k]
        actor = _sample_categorical(rng, actor_rates[selected_var], var_totals[k])

        dep = data.dependents[selected_var]
        if dep isa DependentNetwork
            change = execute_network_ministep!(state, effects, θ_obj, data, actor,
                                               selected_var, rng)
            change > 0 && (n_network_changes[selected_var] += 1)
        else
            change = execute_behavior_ministep!(state, effects, θ_obj, data, actor,
                                                selected_var, rng)
            change != 0 && (n_behavior_changes[selected_var] += 1)
        end

        # Conditional termination
        if conditional && !isnothing(target_changes)
            reached = true
            for (name, target) in target_changes
                current = get(n_network_changes, name, 0) + get(n_behavior_changes, name, 0)
                if current < target
                    reached = false
                    break
                end
            end
            reached && break
        end
    end

    step >= max_steps &&
        @warn "simulate_period! hit the max_steps limit ($max_steps); period may be incomplete"

    return SimulationResult(snapshot(state), n_network_changes, n_behavior_changes)
end

#==============================================================================#
# Full Simulation
#==============================================================================#

"""
    simulate_saom(data::SienaData, effects::SienaEffects, θ::Vector{Float64};
                 seed::Union{Int, Nothing}=nothing)

Simulate the SAOM across all periods. `θ` is the **full** free-parameter vector
(rate parameters first, then objective parameters — see [`build_param_map`](@ref)).

Each period starts from the corresponding *observed* wave (as in unconditional
Method-of-Moments estimation), so per-period results are conditionally independent
given the data.

Returns `(final_state, results::Vector{SimulationResult})` with one result per period.
"""
function simulate_saom(data::SienaData, effects::SienaEffects, θ::Vector{Float64};
                      seed::Union{Int, Nothing}=nothing,
                      rng::Union{AbstractRNG, Nothing}=nothing)
    data.n_waves >= 2 ||
        throw(ArgumentError("simulate_saom requires at least 2 observation waves"))
    r = isnothing(rng) ? (isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)) : rng

    pm = build_param_map(effects)
    θ_obj = objective_theta(pm, θ)

    state = NetworkState()
    all_results = SimulationResult[]
    for period in 1:(data.n_waves - 1)
        initialize!(state, data, period)
        rate_params = Dict{Symbol, Float64}()
        nonbasic = Dict{Symbol, Tuple{Vector{EffectEntry}, Vector{Float64}}}()
        for name in keys(data.dependents)
            rate_params[name] = basic_rate(pm, θ, name, period)
            nonbasic[name] = _nonbasic_rates(pm, θ, name)
        end
        result = simulate_period!(state, effects, θ_obj, rate_params, data, r;
                                  nonbasic_rates=nonbasic)
        push!(all_results, result)
    end

    return state, all_results
end
