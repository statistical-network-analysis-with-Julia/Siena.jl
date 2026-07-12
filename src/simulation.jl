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
# Objective Effect Set (tuple-backed, statically dispatched)
#==============================================================================#

"""
    ObjectiveEffectSpec

One included objective effect with the entry metadata the simulation hot loop needs
(target variable, effect type, fix status, θ-index). The `effect` field is
concretely typed, so `compute_contribution` calls through a spec are statically
dispatched.
"""
struct ObjectiveEffectSpec{E<:AbstractEffect}
    effect::E
    variable::Symbol
    etype::Symbol           # :eval, :creation, or :endow
    fix::Bool
    initial_value::Float64
    param_idx::Int          # index into the objective part of θ (0 if fixed)
end

"""
    ObjectiveEffectSet

The included objective effects of a model, stored as a tuple of
[`ObjectiveEffectSpec`](@ref)s (mirroring `ERGM.TermSet`) so that the per-candidate
contribution loop compiles to statically dispatched calls instead of dynamic dispatch
through an abstractly-typed effects table.

Build one with [`build_objective_set`](@ref) and pass it to [`simulate_period!`](@ref),
[`compute_objective`](@ref), or the choice-probability functions in place of the
`SienaEffects` argument; results are identical, evaluation is just faster.
"""
struct ObjectiveEffectSet{T<:Tuple}
    specs::T

    function ObjectiveEffectSet(specs::T) where {T<:Tuple}
        all(s -> s isa ObjectiveEffectSpec, specs) ||
            throw(ArgumentError("all elements must be ObjectiveEffectSpecs"))
        new{T}(specs)
    end
end

Base.length(oset::ObjectiveEffectSet) = length(oset.specs)

"""
    build_objective_set(effects::SienaEffects)

Snapshot the included objective effects (network and behavior effects, in effects-table
order) into an [`ObjectiveEffectSet`](@ref). The spec's `param_idx` follows the layout
of [`objective_theta`](@ref): consecutive indices over the non-fixed objective entries.

The set is a snapshot: rebuild it after changing inclusion/fix flags or initial values.
"""
function build_objective_set(effects::SienaEffects)
    specs = ObjectiveEffectSpec[]
    param_idx = 0
    for entry in get_objective_effects(effects)
        idx = 0
        if !entry.fix
            param_idx += 1
            idx = param_idx
        end
        push!(specs, ObjectiveEffectSpec(entry.effect, target_variable(entry.effect),
                                         effect_type(entry.effect), entry.fix,
                                         entry.initial_value, idx))
    end
    return ObjectiveEffectSet(Tuple(specs))
end

# Whether the spec enters the objective for this ministep direction.
@inline function _spec_active(spec::ObjectiveEffectSpec, adding::Bool)
    etype = spec.etype
    return etype == :eval || (etype == :creation && adding) ||
           (etype == :endow && !adding)
end

# Left fold of the objective over the spec tuple: same accumulation order (and hence
# bitwise-identical result) as the original loop over the effects table, but with
# statically dispatched compute_contribution calls.
@inline _objective_fold(::Tuple{}, acc::Float64, θ, state, data, actor, alter,
                        variable, adding, sgn) = acc

@inline function _objective_fold(specs::Tuple, acc::Float64, θ, state, data,
                                 actor, alter, variable, adding, sgn)
    spec = specs[1]
    if spec.variable == variable
        θ_k = spec.fix ? spec.initial_value : θ[spec.param_idx]
        if _spec_active(spec, adding) && θ_k != 0.0
            acc += θ_k * sgn * compute_contribution(spec.effect, state, data,
                                                    actor, alter)
        end
    end
    return _objective_fold(Base.tail(specs), acc, θ, state, data, actor, alter,
                           variable, adding, sgn)
end

# As _objective_fold, but also writes each free effect's θ-gradient (sign * change
# statistic) into `col` at position n_rate + param_idx. Mirrors the score path.
@inline _objective_contrib_fold(::Tuple{}, acc::Float64, col, n_rate, θ, state, data,
                                actor, alter, variable, adding, sgn) = acc

@inline function _objective_contrib_fold(specs::Tuple, acc::Float64, col, n_rate, θ,
                                         state, data, actor, alter, variable,
                                         adding, sgn)
    spec = specs[1]
    if spec.variable == variable && _spec_active(spec, adding)
        θ_k = spec.fix ? spec.initial_value : θ[spec.param_idx]
        contrib = compute_contribution(spec.effect, state, data, actor, alter)
        acc += θ_k * sgn * contrib
        spec.fix || (col[n_rate + spec.param_idx] = sgn * contrib)
    end
    return _objective_contrib_fold(Base.tail(specs), acc, col, n_rate, θ, state, data,
                                   actor, alter, variable, adding, sgn)
end

# Toggle direction and sign for one candidate ministep.
@inline function _ministep_direction(state::NetworkState, data::SienaData,
                                     actor::Int, alter::Int, variable::Symbol)
    dep = data.dependents[variable]
    if dep isa DependentNetwork
        adding = state.networks[variable][actor, alter] == 0
        return adding, adding ? 1.0 : -1.0
    else
        return alter > 0, 1.0
    end
end

#==============================================================================#
# Objective Function
#==============================================================================#

"""
    compute_objective(effects, θ::Vector{Float64},
                     state::NetworkState, data::SienaData,
                     actor::Int, alter::Int, variable::Symbol)

Objective-function change for a potential ministep, using the **objective part** of
the parameter vector (see [`objective_theta`](@ref)). `effects` may be a
`SienaEffects` table or a prebuilt [`ObjectiveEffectSet`](@ref) (the simulation hot
path builds the set once per simulation and reuses it for every candidate alter).

For networks the change is a toggle of the tie `actor -> alter`; since
`compute_contribution` returns the add-direction change statistic, deletions get the
sign flip ``(1 - 2 x_{ij})``. For behavior, `alter` encodes the direction (±1) and the
contributions are already directional differences. Creation effects only enter for
additions/increases, endowment effects only for deletions/decreases.
"""
function compute_objective(oset::ObjectiveEffectSet, θ::Vector{Float64},
                          state::NetworkState, data::SienaData,
                          actor::Int, alter::Int, variable::Symbol)
    adding, sgn = _ministep_direction(state, data, actor, alter, variable)
    return _objective_fold(oset.specs, 0.0, θ, state, data, actor, alter, variable,
                           adding, sgn)
end

compute_objective(effects::SienaEffects, θ::Vector{Float64},
                  state::NetworkState, data::SienaData,
                  actor::Int, alter::Int, variable::Symbol) =
    compute_objective(build_objective_set(effects), θ, state, data, actor, alter,
                      variable)

#==============================================================================#
# Choice Probabilities
#==============================================================================#

"""
    compute_network_choice_probs(effects, θ::Vector{Float64},
                                state::NetworkState, data::SienaData,
                                actor::Int, variable::Symbol)

Compute multinomial-logit probabilities for all possible tie toggles of an actor
(plus the no-change option). Returns `(probabilities, alters)` where alter 0 is the
no-change option. For two-mode networks the alters are the events (columns).

Structurally determined dyads (structural zeros/ones of the period-start wave;
see [`DependentNetwork`](@ref)) are excluded from the candidate set: an actor
can never toggle them. Likewise, when the data have composition changes, dyads
involving actors absent in the current period are excluded (an inactive ego
gets only the no-change option).

`effects` may be a `SienaEffects` table or a prebuilt [`ObjectiveEffectSet`](@ref);
with a `SienaEffects` argument the set is built once (hoisted out of the alter loop).
"""
function compute_network_choice_probs(oset::ObjectiveEffectSet, θ::Vector{Float64},
                                     state::NetworkState, data::SienaData,
                                     actor::Int, variable::Symbol)
    net = state.networks[variable]
    dep = data.dependents[variable]::DependentNetwork
    n_alters = size(net, 2)
    onemode = dep.type == :onemode
    smask = _structural_mask(dep, state.period)
    act = state.active

    if act !== nothing && !act[actor]
        return [1.0], [0]     # inactive ego: only the no-change option
    end

    objectives = Float64[0.0]     # no-change option
    valid_alters = Int[0]

    for alter in 1:n_alters
        if onemode && alter == actor && !dep.allow_self_loops
            continue
        end
        if smask !== nothing && smask[actor, alter]
            continue    # structurally determined: not a candidate
        end
        if act !== nothing && onemode && !act[alter]
            continue    # alter absent this period: not a candidate
        end
        obj = compute_objective(oset, θ, state, data, actor, alter, variable)
        push!(objectives, obj)
        push!(valid_alters, alter)
    end

    max_obj = maximum(objectives)
    exp_obj = exp.(objectives .- max_obj)
    probs = exp_obj ./ sum(exp_obj)

    return probs, valid_alters
end

compute_network_choice_probs(effects::SienaEffects, θ::Vector{Float64},
                             state::NetworkState, data::SienaData,
                             actor::Int, variable::Symbol) =
    compute_network_choice_probs(build_objective_set(effects), θ, state, data,
                                 actor, variable)

"""
    compute_behavior_choice_probs(effects, θ::Vector{Float64},
                                 state::NetworkState, data::SienaData,
                                 actor::Int, variable::Symbol)

Compute probabilities for behavior changes (-1, 0, +1) for an actor.
Returns (probabilities, directions).

`effects` may be a `SienaEffects` table or a prebuilt [`ObjectiveEffectSet`](@ref).
"""
function compute_behavior_choice_probs(oset::ObjectiveEffectSet, θ::Vector{Float64},
                                      state::NetworkState, data::SienaData,
                                      actor::Int, variable::Symbol)
    beh = state.behaviors[variable]
    dep = data.dependents[variable]::DependentBehavior
    current = beh[actor]

    if state.active !== nothing && !state.active[actor]
        return [1.0], [0]     # inactive ego: only the no-change option
    end

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
              compute_objective(oset, θ, state, data, actor, dir, variable)
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

compute_behavior_choice_probs(effects::SienaEffects, θ::Vector{Float64},
                              state::NetworkState, data::SienaData,
                              actor::Int, variable::Symbol) =
    compute_behavior_choice_probs(build_objective_set(effects), θ, state, data,
                                  actor, variable)

#==============================================================================#
# Score Accumulation (likelihood-ratio derivative estimation)
#==============================================================================#

"""
    ScoreAccumulator(pm::ParameterMap)

Accumulator for the score function ``S = \\partial \\log P(\\text{path}; θ)/\\partial θ``
of a simulated trajectory, aligned with the full free-parameter vector θ.

Passed to [`simulate_saom`](@ref) via the `scores` keyword. Every ministep adds its
multinomial-choice score ``s_{\\text{chosen}} - \\sum_a P(a)\\, s_a`` over the
candidate set, and every waiting time adds the rate-parameter exposure terms of the
time-truncated CTMC likelihood. The accumulated scores drive the score-function
(Schweinberger–Snijders) derivative estimator ``D = \\mathrm{cov}(s, S)`` used for
standard errors.

Only meaningful for unconditional simulation (the likelihood terms assume the period
ends at time 1).
"""
struct ScoreAccumulator
    pm::ParameterMap
    scores::Vector{Float64}
end

ScoreAccumulator(pm::ParameterMap) = ScoreAccumulator(pm, zeros(n_free_parameters(pm)))

"""
    reset_scores!(sacc::ScoreAccumulator)

Zero the accumulated scores (for reuse across simulations).
"""
reset_scores!(sacc::ScoreAccumulator) = (fill!(sacc.scores, 0.0); sacc)

# Objective value of one candidate ministep plus, for every *free* objective effect
# targeting `variable`, its θ-gradient (sign * change statistic) written into `col`
# at the effect's position in the full θ vector. Mirrors `compute_objective`.
function _objective_and_contrib!(col::Vector{Float64}, oset::ObjectiveEffectSet,
                                 θ::Vector{Float64}, state::NetworkState,
                                 data::SienaData, actor::Int, alter::Int,
                                 variable::Symbol, n_rate::Int)
    adding, sgn = _ministep_direction(state, data, actor, alter, variable)
    return _objective_contrib_fold(oset.specs, 0.0, col, n_rate, θ, state, data,
                                   actor, alter, variable, adding, sgn)
end

# As compute_network_choice_probs, but also returns the matrix C of θ-gradients of
# the candidate objectives (n_free × n_candidates; the no-change column is zero).
function _network_choice_probs_contribs(oset::ObjectiveEffectSet, θ::Vector{Float64},
                                        pm::ParameterMap, state::NetworkState,
                                        data::SienaData, actor::Int, variable::Symbol)
    net = state.networks[variable]
    dep = data.dependents[variable]::DependentNetwork
    n_alters = size(net, 2)
    onemode = dep.type == :onemode
    smask = _structural_mask(dep, state.period)
    act = state.active
    n_free = n_free_parameters(pm)
    n_rate = n_free_rate_parameters(pm)

    if act !== nothing && !act[actor]
        return [1.0], [0], zeros(n_free, 1)   # inactive ego: no-change only
    end

    objectives = Float64[0.0]     # no-change option
    valid_alters = Int[0]
    cols = Vector{Float64}[zeros(n_free)]

    for alter in 1:n_alters
        if onemode && alter == actor && !dep.allow_self_loops
            continue
        end
        if smask !== nothing && smask[actor, alter]
            continue    # structurally determined: not a candidate
        end
        if act !== nothing && onemode && !act[alter]
            continue    # alter absent this period: not a candidate
        end
        col = zeros(n_free)
        obj = _objective_and_contrib!(col, oset, θ, state, data, actor, alter,
                                      variable, n_rate)
        push!(objectives, obj)
        push!(valid_alters, alter)
        push!(cols, col)
    end

    max_obj = maximum(objectives)
    exp_obj = exp.(objectives .- max_obj)
    probs = exp_obj ./ sum(exp_obj)

    return probs, valid_alters, reduce(hcat, cols)
end

# As compute_behavior_choice_probs, but also returns the θ-gradient matrix.
function _behavior_choice_probs_contribs(oset::ObjectiveEffectSet, θ::Vector{Float64},
                                         pm::ParameterMap, state::NetworkState,
                                         data::SienaData, actor::Int, variable::Symbol)
    beh = state.behaviors[variable]
    dep = data.dependents[variable]::DependentBehavior
    current = beh[actor]
    n_free = n_free_parameters(pm)
    n_rate = n_free_rate_parameters(pm)

    if state.active !== nothing && !state.active[actor]
        return [1.0], [0], zeros(n_free, 1)   # inactive ego: no-change only
    end

    objectives = Float64[]
    valid_directions = Int[]
    cols = Vector{Float64}[]

    for dir in (-1, 0, 1)
        new_val = current + dir
        if new_val < dep.min_val || new_val > dep.max_val
            continue
        end
        col = zeros(n_free)
        obj = dir == 0 ? 0.0 :
              _objective_and_contrib!(col, oset, θ, state, data, actor, dir,
                                      variable, n_rate)
        push!(objectives, obj)
        push!(valid_directions, dir)
        push!(cols, col)
    end

    if isempty(objectives)
        return [1.0], [0], zeros(n_free, 1)
    end

    max_obj = maximum(objectives)
    exp_obj = exp.(objectives .- max_obj)
    probs = exp_obj ./ sum(exp_obj)

    return probs, valid_directions, reduce(hcat, cols)
end

# Choice score for one executed ministep: gradient of the chosen candidate minus the
# probability-weighted gradient over the candidate set.
function _accumulate_choice_scores!(sacc::ScoreAccumulator, C::Matrix{Float64},
                                    probs::Vector{Float64}, chosen_idx::Int)
    sacc.scores .+= view(C, :, chosen_idx) .- C * probs
    return sacc
end

# Exposure (waiting-time / survival) part of the rate-parameter scores for one step:
# d/dρ_v of -λ_tot * exposure is -(Λ_v/ρ_v); for a non-basic rate parameter with
# score r_ki it is -Σ_i λ_{v,i} r_ki.
function _accumulate_rate_exposure_scores!(sacc::ScoreAccumulator, state::NetworkState,
                                           data::SienaData, variables::Vector{Symbol},
                                           actor_rates::Dict{Symbol, Vector{Float64}},
                                           var_totals::Vector{Float64},
                                           rate_params::Dict{Symbol, Float64},
                                           exposure::Float64)
    exposure <= 0.0 && return sacc
    for (idx, entry) in enumerate(sacc.pm.free)
        eff = entry.effect
        eff isa RateEffect || continue
        v = target_variable(eff)
        k = findfirst(==(v), variables)
        k === nothing && continue
        if eff isa BasicRateEffect
            eff.period == state.period || continue
            ρ = max(get(rate_params, v, 1.0), 1e-6)
            sacc.scores[idx] -= exposure * var_totals[k] / ρ
        else
            rates = actor_rates[v]
            total = 0.0
            for a in eachindex(rates)
                total += rates[a] * rate_score(eff, state, data, a)
            end
            sacc.scores[idx] -= exposure * total
        end
    end
    return sacc
end

# Selection part of the rate-parameter scores when actor `actor` moves on
# `selected_var`: d/dρ_v log λ_{v,i} = 1/ρ_v; for a non-basic parameter it is r_ki.
function _accumulate_rate_selection_scores!(sacc::ScoreAccumulator, state::NetworkState,
                                            data::SienaData, selected_var::Symbol,
                                            ρ::Float64, actor::Int)
    for (idx, entry) in enumerate(sacc.pm.free)
        eff = entry.effect
        eff isa RateEffect || continue
        target_variable(eff) == selected_var || continue
        if eff isa BasicRateEffect
            eff.period == state.period || continue
            sacc.scores[idx] += 1.0 / max(ρ, 1e-6)
        else
            sacc.scores[idx] += rate_score(eff, state, data, actor)
        end
    end
    return sacc
end

#==============================================================================#
# Mini-Step Execution
#==============================================================================#

"""
    execute_network_ministep!(state::NetworkState, effects,
                             θ::Vector{Float64}, data::SienaData,
                             actor::Int, variable::Symbol, rng::AbstractRNG;
                             scores::Union{Nothing, ScoreAccumulator}=nothing)

Execute a network mini-step for the given actor. Returns the chosen alter (0 for
no change). If `scores` is given, the choice score of the ministep is accumulated.
`effects` may be a `SienaEffects` table or a prebuilt [`ObjectiveEffectSet`](@ref).
"""
function execute_network_ministep!(state::NetworkState,
                                  effects::Union{SienaEffects, ObjectiveEffectSet},
                                  θ::Vector{Float64}, data::SienaData,
                                  actor::Int, variable::Symbol, rng::AbstractRNG;
                                  scores::Union{Nothing, ScoreAccumulator}=nothing)
    oset = effects isa ObjectiveEffectSet ? effects : build_objective_set(effects)
    if scores === nothing
        probs, alters = compute_network_choice_probs(oset, θ, state, data, actor,
                                                     variable)
        C = nothing
    else
        probs, alters, C = _network_choice_probs_contribs(oset, θ, scores.pm,
                                                          state, data, actor, variable)
    end

    u = rand(rng)
    cumsum_p = 0.0
    chosen_idx = 1                # option 1 is the no-change option
    for i in eachindex(probs)
        cumsum_p += probs[i]
        if u <= cumsum_p
            chosen_idx = i
            break
        end
    end
    chosen_alter = alters[chosen_idx]

    scores === nothing || _accumulate_choice_scores!(scores, C, probs, chosen_idx)

    if chosen_alter > 0
        net = state.networks[variable]
        net[actor, chosen_alter] = 1 - net[actor, chosen_alter]
    end

    return chosen_alter
end

"""
    execute_behavior_ministep!(state::NetworkState, effects,
                              θ::Vector{Float64}, data::SienaData,
                              actor::Int, variable::Symbol, rng::AbstractRNG;
                              scores::Union{Nothing, ScoreAccumulator}=nothing)

Execute a behavior mini-step for the given actor. Returns the chosen direction.
If `scores` is given, the choice score of the ministep is accumulated.
`effects` may be a `SienaEffects` table or a prebuilt [`ObjectiveEffectSet`](@ref).
"""
function execute_behavior_ministep!(state::NetworkState,
                                   effects::Union{SienaEffects, ObjectiveEffectSet},
                                   θ::Vector{Float64}, data::SienaData,
                                   actor::Int, variable::Symbol, rng::AbstractRNG;
                                   scores::Union{Nothing, ScoreAccumulator}=nothing)
    oset = effects isa ObjectiveEffectSet ? effects : build_objective_set(effects)
    if scores === nothing
        probs, directions = compute_behavior_choice_probs(oset, θ, state, data,
                                                          actor, variable)
        C = nothing
    else
        probs, directions, C = _behavior_choice_probs_contribs(oset, θ, scores.pm,
                                                               state, data, actor,
                                                               variable)
    end

    u = rand(rng)
    cumsum_p = 0.0
    chosen_idx = findfirst(==(0), directions)   # fall back to the no-change option
    for i in eachindex(probs)
        cumsum_p += probs[i]
        if u <= cumsum_p
            chosen_idx = i
            break
        end
    end
    chosen_dir = directions[chosen_idx]

    scores === nothing || _accumulate_choice_scores!(scores, C, probs, chosen_idx)

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

# Per-actor rates for one dependent variable. Actors absent in the current
# period (composition change) get rate 0: they take no ministeps.
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
    act = state.active
    if act !== nothing
        for i in 1:min(n, length(act))
            act[i] || (rates[i] = 0.0)
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
    simulate_period!(state::NetworkState, effects,
                    θ_obj::Vector{Float64}, rate_params::Dict{Symbol, Float64},
                    data::SienaData, rng::AbstractRNG; kwargs...)

Simulate one period of network/behavior dynamics over the unit time interval.

# Arguments
- `effects`: a `SienaEffects` table or a prebuilt [`ObjectiveEffectSet`](@ref)
  ([`simulate_saom`](@ref) builds the set once and reuses it across periods)
- `θ_obj`: objective-function parameters (see [`objective_theta`](@ref))
- `rate_params`: basic rate ``\\rho`` per dependent variable for this period
- `nonbasic_rates`: optional `Dict(variable => (entries, values))` of non-basic rate
  effect entries and their parameters
- `conditional`/`target_changes`: if `conditional=true`, simulate until each
  variable in `target_changes` has reached the target *distance* from its
  period-start state (network: Hamming distance; behavior: L1 distance), instead
  of until time 1 — RSiena's conditional simulation
- `max_steps`: safety cap on the number of ministeps
- `scores`: optional [`ScoreAccumulator`](@ref) collecting the score function of the
  simulated trajectory (unconditional simulation only)

Per-actor rates are cached between ministeps: a variable's rates are recomputed only
after a ministep actually changed the state, and only for variables whose rate
function is state-dependent (i.e. with non-basic rate effects). Variables with only
a basic rate have constant rates that are computed once per period.
"""
function simulate_period!(state::NetworkState,
                         effects::Union{SienaEffects, ObjectiveEffectSet},
                         θ_obj::Vector{Float64}, rate_params::Dict{Symbol, Float64},
                         data::SienaData, rng::AbstractRNG;
                         nonbasic_rates::Dict{Symbol, Tuple{Vector{EffectEntry}, Vector{Float64}}} =
                             Dict{Symbol, Tuple{Vector{EffectEntry}, Vector{Float64}}}(),
                         conditional::Bool=false,
                         target_changes::Union{Nothing, Dict{Symbol, Int}}=nothing,
                         max_steps::Int=100_000,
                         scores::Union{Nothing, ScoreAccumulator}=nothing)

    oset = effects isa ObjectiveEffectSet ? effects : build_objective_set(effects)

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
    # Rate caching: a variable's per-actor rates depend on the state only through
    # its non-basic rate effects. `dirty[k]` marks variables whose cached rates
    # must be recomputed; state-independent variables are computed once and stay
    # cached for the whole period.
    state_dependent = [!isempty(get(nonbasic_rates, v, empty_entries)[1])
                       for v in variables]
    dirty = trues(length(variables))

    # Conditional simulation (RSiena): track the distance of each conditioned
    # variable from its period-start state incrementally, and stop once every
    # target distance is reached (a toggle towards the start state decreases
    # the distance, so this is not a plain change count).
    cond_start_nets = Dict{Symbol, Matrix{Int8}}()
    cond_start_behs = Dict{Symbol, Vector{Int}}()
    cond_dist = Dict{Symbol, Int}()
    cond_done = false
    if conditional
        target_changes === nothing &&
            throw(ArgumentError("conditional simulation requires target_changes"))
        for name in keys(target_changes)
            if data.dependents[name] isa DependentNetwork
                cond_start_nets[name] = copy(state.networks[name].m)
            else
                cond_start_behs[name] = copy(state.behaviors[name])
            end
            cond_dist[name] = 0
        end
        cond_done = all(t <= 0 for t in values(target_changes))
    end

    step = 0
    while step < max_steps
        conditional && cond_done && break
        step += 1

        # Per-actor rates for every variable (cached; recomputed only when dirty)
        total_rate = 0.0
        for (k, v) in enumerate(variables)
            if dirty[k]
                λ = get(rate_params, v, 1.0)
                nb_entries, nb_θ = get(nonbasic_rates, v, empty_entries)
                rates = _variable_actor_rates!(actor_rates[v], state, data, v, λ,
                                               nb_entries, nb_θ)
                var_totals[k] = sum(rates)
                dirty[k] = false
            end
            total_rate += var_totals[k]
        end

        total_rate <= 0 && break

        # Waiting time
        dt = sample_waiting_time(total_rate, rng)
        t_before = state.time
        state.time += dt
        ended = !conditional && state.time >= 1.0
        if scores !== nothing
            # Exposure term of the rate scores: full waiting time for a completed
            # jump, censored at the end of the unit period otherwise.
            exposure = ended ? max(1.0 - t_before, 0.0) : dt
            _accumulate_rate_exposure_scores!(scores, state, data, variables,
                                              actor_rates, var_totals, rate_params,
                                              exposure)
        end
        ended && break

        # Which variable changes, and which actor moves
        k = _sample_categorical(rng, var_totals, total_rate)
        selected_var = variables[k]
        actor = _sample_categorical(rng, actor_rates[selected_var], var_totals[k])

        scores === nothing ||
            _accumulate_rate_selection_scores!(scores, state, data, selected_var,
                                               get(rate_params, selected_var, 1.0),
                                               actor)

        dep = data.dependents[selected_var]
        state_changed = false
        if dep isa DependentNetwork
            change = execute_network_ministep!(state, oset, θ_obj, data, actor,
                                               selected_var, rng; scores=scores)
            if change > 0
                n_network_changes[selected_var] += 1
                state_changed = true
            end
        else
            change = execute_behavior_ministep!(state, oset, θ_obj, data, actor,
                                                selected_var, rng; scores=scores)
            if change != 0
                n_behavior_changes[selected_var] += 1
                state_changed = true
            end
        end

        # Invalidate cached rates of state-dependent variables after a real change
        if state_changed
            for k2 in eachindex(dirty)
                state_dependent[k2] && (dirty[k2] = true)
            end
        end

        # Conditional termination: update the conditioned variable's distance
        # from the period-start state and stop once every target is reached
        if conditional && state_changed && haskey(cond_dist, selected_var)
            if dep isa DependentNetwork
                start_val = Int(cond_start_nets[selected_var][actor, change])
                new_val = state.networks[selected_var][actor, change]
                cond_dist[selected_var] += new_val == start_val ? -1 : 1
            else
                start_val = cond_start_behs[selected_var][actor]
                new_val = state.behaviors[selected_var][actor]
                cond_dist[selected_var] +=
                    abs(new_val - start_val) - abs(new_val - change - start_val)
            end
            cond_done = all(cond_dist[name] >= target
                            for (name, target) in target_changes)
            cond_done && break
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

If `scores` is given, the score function of the simulated trajectory is accumulated
across all periods (see [`ScoreAccumulator`](@ref)).

If `condvar` is given, every period is simulated *conditionally* on the observed
amount of change of that dependent variable (RSiena's conditional simulation): the
period runs until the variable's distance from the period-start observation reaches
`cond_targets[period]` instead of until time 1. Score accumulation is not supported
with conditional simulation.
"""
function simulate_saom(data::SienaData, effects::SienaEffects, θ::Vector{Float64};
                      seed::Union{Int, Nothing}=nothing,
                      rng::Union{AbstractRNG, Nothing}=nothing,
                      scores::Union{Nothing, ScoreAccumulator}=nothing,
                      condvar::Union{Symbol, Nothing}=nothing,
                      cond_targets::Union{Nothing, Vector{Int}}=nothing)
    data.n_waves >= 2 ||
        throw(ArgumentError("simulate_saom requires at least 2 observation waves"))
    if condvar !== nothing
        scores === nothing ||
            throw(ArgumentError("score accumulation is not supported with " *
                                "conditional simulation"))
        haskey(data.dependents, condvar) ||
            throw(ArgumentError("unknown conditioning variable :$condvar"))
        (cond_targets !== nothing && length(cond_targets) == data.n_waves - 1) ||
            throw(ArgumentError("cond_targets must give one target distance per period"))
    end
    r = isnothing(rng) ? (isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)) : rng

    pm = build_param_map(effects)
    θ_obj = objective_theta(pm, θ)
    # Build the tuple-backed objective set once; every ministep of every period
    # reuses it (statically dispatched contribution calls, no per-step filtering).
    oset = build_objective_set(effects)

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
        if condvar === nothing
            result = simulate_period!(state, oset, θ_obj, rate_params, data, r;
                                      nonbasic_rates=nonbasic, scores=scores)
        else
            result = simulate_period!(state, oset, θ_obj, rate_params, data, r;
                                      nonbasic_rates=nonbasic, conditional=true,
                                      target_changes=Dict(condvar =>
                                                          cond_targets[period]))
        end
        push!(all_results, result)
    end

    return state, all_results
end
