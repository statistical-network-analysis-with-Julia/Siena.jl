"""
Base types and utilities for SAOM effects.
"""

#==============================================================================#
# Effect Types
#==============================================================================#

"""
    AbstractEffect

Abstract base type for all SAOM effects.
"""
abstract type AbstractEffect end

"""
    NetworkEffect <: AbstractEffect

Effect on network dynamics (evaluation or endowment).
"""
abstract type NetworkEffect <: AbstractEffect end

"""
    BehaviorEffect <: AbstractEffect

Effect on behavior dynamics.
"""
abstract type BehaviorEffect <: AbstractEffect end

"""
    RateEffect <: AbstractEffect

Effect on the rate function.
"""
abstract type RateEffect <: AbstractEffect end

#==============================================================================#
# Effect Properties
#==============================================================================#

"""
    effect_name(e::AbstractEffect)

Return the canonical name of the effect.
"""
function effect_name end

"""
    effect_type(e::AbstractEffect)

Return the type of effect (:eval, :endow, :creation, :rate).
"""
function effect_type end

"""
    target_variable(e::AbstractEffect)

Return the name of the variable this effect applies to.
"""
function target_variable end

"""
    interaction_with(e::AbstractEffect)

Return the name of the covariate/variable this effect interacts with, or nothing.
"""
function interaction_with end

interaction_with(::AbstractEffect) = nothing

#==============================================================================#
# Effect Computation
#==============================================================================#

"""
    evaluate_actor(effect::AbstractEffect, state::NetworkState, data::SienaData, actor::Int)

Actor `actor`'s evaluation-function component ``s_{ki}(x)`` for this effect (RSiena's
``s_{ki}``). This is the primitive from which everything else derives:

- `compute_statistic` sums it over all actors (the Method-of-Moments target statistic);
- the generic `compute_contribution` fallback differences it across a tie toggle
  (networks) or a behavior step (behavior).

Every concrete effect must implement this method.
"""
function evaluate_actor end

"""
    compute_contribution(effect::AbstractEffect, state::NetworkState,
                        data::SienaData, actor::Int, alter::Int)

Change statistic of the effect for actor `actor`.

For network effects this is the **add-direction** change
``s_{ki}(x \\text{ with tie } i \\to j) - s_{ki}(x \\text{ without tie})``, independent
of whether the tie currently exists (the simulation applies the sign flip for
deletions). For behavior effects, `alter` encodes the direction (±1) and the value is
``s_{ki}(z_i + d) - s_{ki}(z_i)``.

A generic fallback based on [`evaluate_actor`](@ref) is provided; effects may add a
closed-form method for speed (verified against the fallback in the test suite).
"""
function compute_contribution end

"""
    compute_statistic(effect::AbstractEffect, state::NetworkState, data::SienaData)

Compute the network statistic associated with this effect:
``s_k(x) = \\sum_i s_{ki}(x)`` (see [`evaluate_actor`](@ref)).
"""
function compute_statistic(e::AbstractEffect, state::NetworkState, data::SienaData)
    s = 0.0
    for i in 1:_n_focal_actors(e, state)
        s += evaluate_actor(e, state, data, i)
    end
    return s
end

_n_focal_actors(e::AbstractEffect, state::NetworkState) =
    e isa BehaviorEffect ? length(state.behaviors[target_variable(e)]) :
                           size(state.networks[target_variable(e)], 1)

# Generic fallback: brute-force toggle. Correct by construction (add-direction,
# state-independent) but O(cost of evaluate_actor) per call; hot effects should
# provide a closed form.
function compute_contribution(e::NetworkEffect, state::NetworkState,
                              data::SienaData, actor::Int, alter::Int)
    net = state.networks[target_variable(e)]
    old = net[actor, alter]
    net[actor, alter] = 1
    with_tie = evaluate_actor(e, state, data, actor)
    net[actor, alter] = 0
    without_tie = evaluate_actor(e, state, data, actor)
    net[actor, alter] = old
    return with_tie - without_tie
end

function compute_contribution(e::BehaviorEffect, state::NetworkState,
                              data::SienaData, actor::Int, direction::Int)
    direction == 0 && return 0.0
    beh = state.behaviors[target_variable(e)]
    old = beh[actor]
    current = evaluate_actor(e, state, data, actor)
    beh[actor] = old + direction
    changed = evaluate_actor(e, state, data, actor)
    beh[actor] = old
    return changed - current
end

#==============================================================================#
# Effect Entry (for effects table)
#==============================================================================#

"""
    EffectEntry

An entry in the effects table, combining an effect with its inclusion status and parameters.

# Fields
- `effect::AbstractEffect`: The effect itself
- `name::String`: Display name
- `shortname::String`: Short name for output
- `include::Bool`: Whether effect is included in model
- `fix::Bool`: Whether parameter is fixed
- `test::Bool`: Whether to perform score test
- `initial_value::Float64`: Initial parameter value
- `parameter::Int`: Parameter group (for rate effects)
"""
mutable struct EffectEntry
    effect::AbstractEffect
    name::String
    shortname::String
    include::Bool
    fix::Bool
    test::Bool
    initial_value::Float64
    parameter::Int  # For rate effects: which period

    function EffectEntry(effect::AbstractEffect;
                         name::String=string(effect_name(effect)),
                         shortname::String=string(effect_name(effect)),
                         include::Bool=false,
                         fix::Bool=false,
                         test::Bool=false,
                         initial_value::Float64=0.0,
                         parameter::Int=0)
        new(effect, name, shortname, include, fix, test, initial_value, parameter)
    end
end

#==============================================================================#
# Effects Object (collection of effects)
#==============================================================================#

"""
    SienaEffects

Collection of effects for SAOM estimation.

# Fields
- `effects::Vector{EffectEntry}`: All effect entries
- `variable_names::Vector{Symbol}`: Names of dependent variables
"""
mutable struct SienaEffects
    effects::Vector{EffectEntry}
    variable_names::Vector{Symbol}

    function SienaEffects(variable_names::Vector{Symbol}=Symbol[])
        new(EffectEntry[], variable_names)
    end
end

Base.length(se::SienaEffects) = length(se.effects)
Base.iterate(se::SienaEffects, state=1) = state > length(se) ? nothing : (se.effects[state], state + 1)
Base.getindex(se::SienaEffects, i) = se.effects[i]

"""
    add_effect!(effects::SienaEffects, entry::EffectEntry)

Add an effect entry.
"""
function add_effect!(effects::SienaEffects, entry::EffectEntry)
    push!(effects.effects, entry)
    effects
end

"""
    get_included_effects(effects::SienaEffects)

Return vector of included effect entries.
"""
function get_included_effects(effects::SienaEffects)
    filter(e -> e.include, effects.effects)
end

"""
    get_rate_effects(effects::SienaEffects)

Return vector of rate effect entries.
"""
function get_rate_effects(effects::SienaEffects)
    filter(e -> e.effect isa RateEffect && e.include, effects.effects)
end

"""
    get_objective_effects(effects::SienaEffects)

Return vector of objective function effect entries (network and behavior effects).
"""
function get_objective_effects(effects::SienaEffects)
    filter(e -> (e.effect isa NetworkEffect || e.effect isa BehaviorEffect) && e.include,
           effects.effects)
end

"""
    n_rate_parameters(effects::SienaEffects)

Return number of rate parameters to estimate.
"""
function n_rate_parameters(effects::SienaEffects)
    sum(e -> e.include && !e.fix, get_rate_effects(effects))
end

"""
    n_objective_parameters(effects::SienaEffects)

Return number of objective function parameters to estimate.
"""
function n_objective_parameters(effects::SienaEffects)
    sum(e -> e.include && !e.fix, get_objective_effects(effects))
end

function Base.show(io::IO, effects::SienaEffects)
    included = get_included_effects(effects)
    print(io, "SienaEffects($(length(effects)) total, $(length(included)) included)")
end

"""
    effects_table(effects::SienaEffects)

Return a DataFrame representation of the effects.
"""
function effects_table(effects::SienaEffects)
    df = DataFrame(
        name = String[],
        shortname = String[],
        type = Symbol[],
        variable = Symbol[],
        include = Bool[],
        fix = Bool[],
        test = Bool[],
        initial = Float64[]
    )

    for entry in effects.effects
        push!(df, (
            name = entry.name,
            shortname = entry.shortname,
            type = effect_type(entry.effect),
            variable = target_variable(entry.effect),
            include = entry.include,
            fix = entry.fix,
            test = entry.test,
            initial = entry.initial_value
        ))
    end

    df
end
