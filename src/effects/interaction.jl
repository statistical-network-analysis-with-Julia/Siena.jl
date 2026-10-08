"""
User-defined interaction effects — RSiena's `includeInteraction()`.

RSiena builds an interaction of two or three effects of the same dependent variable
as an *elementary* effect: its change statistic is the product of the components'
change statistics, and its target statistic is built from the components' tie (or
ego) statistics. Which effects may interact follows RSiena's `interactionType`
column (`ego`, `dyadic`, or neither for network effects; `OK` or not for behaviour
effects); the rules are checked when the interaction is created and again against
the data by `validate_effects`.
"""

#==============================================================================#
# Interaction types and tie statistics of network effects
#==============================================================================#

# RSiena's `interactionType` of a network effect for a network of the given kind:
# `:ego` (the tie statistic depends on ego only), `:dyadic` (on ego and alter
# only), or `:none`.
_interaction_type(::NetworkEffect, kind::Symbol) = :none
_interaction_type(::OutdegreeEffect, kind::Symbol) = kind === :undirected ? :ego : :dyadic
_interaction_type(::ReciprocityEffect, ::Symbol) = :dyadic
_interaction_type(e::IndegreePopularityEffect, kind::Symbol) =
    (!e.sqrt && kind !== :undirected) ? :dyadic : :none
_interaction_type(::OutdegreePopularityEffect, ::Symbol) = :dyadic
_interaction_type(::IndegreeActivityEffect, ::Symbol) = :ego
_interaction_type(::Union{GWESPEffect, GWESPBackwardEffect, GWESPMixedEffect}, ::Symbol) = :dyadic
_interaction_type(::EgoEffect, ::Symbol) = :ego
_interaction_type(::EgoSqEffect, kind::Symbol) = kind === :undirected ? :dyadic : :ego
_interaction_type(::Union{AlterEffect, AlterSqEffect, SimilarityEffect, SameEffect,
                          DifferenceEffect, DifferenceSqEffect, AbsDifferenceEffect,
                          HigherEffect, EgoTimesAlterEffect, EgoPlusAlterEffect,
                          DyadCovariateEffect, SameXRecipEffect, SimXRecipEffect,
                          SimXTransTripEffect}, ::Symbol) = :dyadic

# RSiena's `tieStatistic` of an ego or dyadic effect: the contribution of the
# existing tie i -> j to the effect's statistic.
_tie_statistic(e::OutdegreeEffect, state, data, i::Int, j::Int) =
    _undirected(data, e.variable) ? 0.5 : 1.0
_tie_statistic(e::ReciprocityEffect, state, data, i::Int, j::Int) =
    Float64(state.networks[e.variable][j, i])
function _tie_statistic(e::IndegreePopularityEffect, state, data, i::Int, j::Int)
    d = Float64(_col_sum(state.networks[e.variable], j))
    return e.sqrt ? sqrt(d) : d
end
function _tie_statistic(e::OutdegreePopularityEffect, state, data, i::Int, j::Int)
    net = state.networks[e.variable]
    if e.parm > 0
        d = Float64(_row_sum(net, j))
        return e.sqrt ? sqrt(d) : d
    end
    start = _period_start_network(data, e.variable, state.period)
    d0 = Float64(sum(view(start, j, :)) - start[j, j])
    return e.parm == 0 ? d0 : Float64(_row_sum(net, j)) + d0
end
function _tie_statistic(e::IndegreeActivityEffect, state, data, i::Int, j::Int)
    net = state.networks[e.variable]
    d = Float64(_col_sum(net, i))
    if e.parm <= 0
        start = _period_start_network(data, e.variable, state.period)
        d0 = Float64(sum(view(start, :, i)) - start[i, i])
        d = e.parm == 0 ? d0 : d + d0
    end
    return e.sqrt ? sqrt(d) : d
end
_tie_statistic(e::GWESPEffect, state, data, i::Int, j::Int) =
    _gwesp_weight(e.alpha, _esp_count(state.networks[e.variable], i, j, true, false))
_tie_statistic(e::GWESPBackwardEffect, state, data, i::Int, j::Int) =
    _gwesp_weight(e.alpha, _esp_count(state.networks[e.variable], i, j, false, true))
_tie_statistic(e::GWESPMixedEffect, state, data, i::Int, j::Int) =
    _gwesp_weight(e.alpha, _esp_count(state.networks[e.variable], i, j, true, true))
function _tie_statistic(e::SimXTransTripEffect, state, data, i::Int, j::Int)
    tp = _esp_count(state.networks[e.variable], i, j, true, false)
    return _attr_similarity(state, data, e.covariate, i, j) * tp
end
# For the remaining ego and dyadic effects the change statistic depends on the
# dyad's attributes (and the reverse tie) only, so it is the tie statistic.
_tie_statistic(e::NetworkEffect, state, data, i::Int, j::Int) =
    compute_contribution(e, state, data, i, j)

# RSiena's `egoStatistic` of a component: the sum of its tie statistics over ego's
# ties when it has them, else the actor statistic on RSiena's scale (cycle3 counts
# a cycle once, ...).
_statistic_scale(::NetworkEffect, data) = 1.0
_statistic_scale(::CyclicTripletsEffect, data) = 1 / 3
_statistic_scale(::TransitiveTriadsEffect, data) = 1 / 3
_statistic_scale(::FourCyclesEffect, data) = 1 / 2
_statistic_scale(e::NbrDist2Effect, data) = _undirected(data, e.variable) ? 0.5 : 1.0

#==============================================================================#
# Network interactions
#==============================================================================#

"""
    InteractionEffect(variable, components) <: NetworkEffect

Interaction of two or three network effects of the same dependent network —
RSiena's user-defined interaction (`includeInteraction`, short name `unspInt`).
Create it with [`include_interaction!`](@ref).

As in RSiena it is an *elementary* effect: the change statistic of the tie
``i \\to j`` is the **product of the components' change statistics**. The target
statistic is RSiena's: with all components but one of type *ego*, the product of
the ego components' tie statistics and the remaining component's actor statistic;
otherwise ``\\sum_j x_{ij} \\prod_k t_k(i, j)`` with ``t_k`` the components' tie
statistics.

RSiena's rules decide which effects may interact. Each effect is of interaction
type *ego* (`egoX`, `egoSqX`, `inAct`, `inActSqrt`; `density` on an undirected
network), *dyadic* (`density`, `recip`, `inPop`, `outPop`, the GWESP effects, `altX`,
`simX`, `sameX`, `diffX`, `higher`, `egoXaltX`, `X`, …) or neither (`transTrip`,
`cycle3`, `outAct`, …). A two-way interaction needs at least one ego effect, or two
dyadic effects; a three-way interaction needs at least two ego effects, or three
ego/dyadic effects.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
effects = get_effects(data)
include_interaction!(effects, :friendship, :egosmoke1, :recip)
entry = only(e for e in effects if e.effect isa InteractionEffect)
entry.shortname                       # "egosmoke1_x_recip"
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(entry.effect, state, data)
```
"""
struct InteractionEffect{T<:Tuple} <: NetworkEffect
    variable::Symbol
    components::T

    function InteractionEffect(variable::Symbol, components::T) where {T<:Tuple}
        2 <= length(components) <= 3 || throw(ArgumentError(
            "an interaction has two or three components, got $(length(components))"))
        for c in components
            c isa NetworkEffect || throw(ArgumentError(
                "interaction components must be network effects"))
            c isa Union{InteractionEffect, EndowmentEffect, CreationEffect} &&
                throw(ArgumentError("interactions of interactions or of " *
                                    "endowment/creation effects are not supported"))
            target_variable(c) == variable || throw(ArgumentError(
                "invalid network interaction specification: must all be same network"))
        end
        new{T}(variable, components)
    end
end

effect_name(::InteractionEffect) = :unspInt
effect_type(::InteractionEffect) = :eval
target_variable(e::InteractionEffect) = e.variable

@inline _contribution_product(::Tuple{}, state, data, actor, alter) = 1.0
@inline _contribution_product(cs::Tuple, state, data, actor, alter) =
    compute_contribution(cs[1], state, data, actor, alter) *
    _contribution_product(Base.tail(cs), state, data, actor, alter)

compute_contribution(e::InteractionEffect, state::NetworkState, data::SienaData,
                     actor::Int, alter::Int) =
    _contribution_product(e.components, state, data, actor, alter)

function evaluate_actor(e::InteractionEffect, state::NetworkState, data::SienaData,
                        actor::Int)
    kind = _network_kind(data.dependents[e.variable]::DependentNetwork)
    net = state.networks[e.variable]
    types = map(c -> _interaction_type(c, kind), e.components)
    n_ego = count(==(:ego), types)
    if n_ego == length(types) - 1
        # RSiena: ego components enter by their tie statistic, the remaining
        # component by its ego (actor) statistic.
        stat = 1.0
        for (c, t) in zip(e.components, types)
            if t === :ego
                stat *= _tie_statistic(c, state, data, actor, actor)
            elseif t === :dyadic
                s = 0.0
                for j in net.outneighbors[actor]
                    s += _tie_statistic(c, state, data, actor, j)
                end
                stat *= s
            else
                stat *= evaluate_actor(c, state, data, actor) * _statistic_scale(c, data)
            end
        end
        return stat
    end
    total = 0.0
    for j in net.outneighbors[actor]
        p = 1.0
        for c in e.components
            p *= _tie_statistic(c, state, data, actor, j)
        end
        total += p
    end
    return total
end

# RSiena's rules (`sienaeffects.r`): which network effects may interact.
function _check_network_interaction(components, kind::Symbol)
    types = [_interaction_type(c, kind) for c in components]
    n_ego = count(==(:ego), types)
    n_dyadic = count(==(:dyadic), types)
    names = join((string(effect_name(c)) for c in components), " x ")
    if length(types) == 2
        (n_ego >= 1 || n_dyadic == 2) || throw(ArgumentError(
            "invalid network interaction specification ($names): must be at least " *
            "one ego or both dyadic effects (RSiena's interactionType; ego: egoX, " *
            "egoSqX, inAct, inActSqrt; dyadic: density, recip, inPop, outPop, " *
            "gwesp*, altX, simX, sameX, diffX, X, ...)"))
    else
        (n_ego >= 2 || n_ego + n_dyadic == 3) || throw(ArgumentError(
            "invalid network 3-way interaction specification ($names): must be at " *
            "least two ego effects or all ego or dyadic effects"))
    end
    return nothing
end

#==============================================================================#
# Behaviour interactions
#==============================================================================#

# RSiena's interactionType "OK" for behaviour effects.
_behavior_interaction_ok(::BehaviorEffect) = false
_behavior_interaction_ok(::Union{LinearShapeEffect, AverageAlterEffect,
    AverageInAlterEffect, AverageRecipAlterEffect, TotalAlterEffect,
    TotalInAlterEffect, AverageAlterDist2Effect, IndegreeEffect,
    BehaviorOutdegreeEffect, RecipDegreeEffect, BehaviorCovariateEffect}) = true

"""
    BehaviorProductEffect(variable, components) <: BehaviorEffect

Interaction of two or three behaviour effects of the same dependent behaviour —
RSiena's user-defined behaviour interaction (`includeInteraction`, short name
`behUnspInt`). Create it with [`include_interaction!`](@ref).

Following RSiena, with ``\\tilde z_i`` the centred behaviour: the change statistic
for a step ``d`` is ``\\prod_k \\Delta_k / d^{K-1}`` (``\\Delta_k`` the components'
change statistics) and the actor statistic is ``\\prod_k s_{ki} / \\tilde
z_i^{K-1}`` (0 where ``\\tilde z_i = 0``). At most one component may be of an
interaction type other than RSiena's `OK` (OK: `linear`, `avAlt`, `totAlt`,
`avInAlt`, `avRecAlt`, `totInAlt`, `avAltDist2`, `indeg`, `outdeg`, `recipDeg`,
`effFrom`; not OK: `quad`, `avSim`, `totSim`, `threshold`).

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
effects = get_effects(data)
include_interaction!(effects, :alcohol, :avAltfriendship, :effFromsmoke1)
entry = only(e for e in effects if e.effect isa BehaviorProductEffect)
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(entry.effect, state, data)
```
"""
struct BehaviorProductEffect{T<:Tuple} <: BehaviorEffect
    variable::Symbol
    components::T

    function BehaviorProductEffect(variable::Symbol, components::T) where {T<:Tuple}
        2 <= length(components) <= 3 || throw(ArgumentError(
            "an interaction has two or three components, got $(length(components))"))
        for c in components
            c isa BehaviorEffect && !(c isa BehaviorProductEffect) ||
                throw(ArgumentError("behaviour interaction components must be " *
                                    "(non-interaction) behaviour effects"))
            target_variable(c) == variable || throw(ArgumentError(
                "invalid behavior interaction specification: must all be same " *
                "behavior variable"))
        end
        count(c -> !_behavior_interaction_ok(c), components) <= 1 ||
            throw(ArgumentError(
                "invalid behavior interaction specification (" *
                join((string(effect_name(c)) for c in components), " x ") *
                "): at most one effect with interactionType not OK is allowed " *
                "(OK: linear, avAlt, totAlt, avInAlt, avRecAlt, totInAlt, " *
                "avAltDist2, indeg, outdeg, recipDeg, effFrom)"))
        new{T}(variable, components)
    end
end

effect_name(::BehaviorProductEffect) = :behUnspInt
effect_type(::BehaviorProductEffect) = :eval
target_variable(e::BehaviorProductEffect) = e.variable

function compute_contribution(e::BehaviorProductEffect, state::NetworkState,
                              data::SienaData, actor::Int, direction::Int)
    direction == 0 && return 0.0
    p = 1.0
    for c in e.components
        p *= compute_contribution(c, state, data, actor, direction)
    end
    return p / Float64(direction)^(length(e.components) - 1)
end

function evaluate_actor(e::BehaviorProductEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    dep = _behavior_dep(data, e.variable)
    z = _centered_beh(dep, state.behaviors[e.variable][actor])
    z == 0 && return 0.0
    p = 1.0
    for c in e.components
        p *= evaluate_actor(c, state, data, actor)
    end
    return p / z^(length(e.components) - 1)
end
