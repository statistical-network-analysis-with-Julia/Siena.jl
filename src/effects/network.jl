"""
Network effects for the SAOM evaluation function.

Each effect implements `evaluate_actor` (the actor's evaluation-function component
``s_{ki}(x)``, following the RSiena manual §12) and, where a cheap exact closed form
exists, `compute_contribution` (the add-direction change statistic
``s_{ki}(x^{+ij}) - s_{ki}(x^{-ij})``). Effects without a closed form fall back to the
generic toggle-based contribution in `effects/base.jl`; the test suite verifies every
closed form against that fallback.
"""

#==============================================================================#
# Helper Functions
#==============================================================================#

# One method with `isa` branches (not one method per covariate type): the
# covariate comes out of the abstractly typed `data.covariates` Dict, and a single
# method keeps the call statically dispatched with an inferred `Float64` result
# (no boxing on the hot path).
function _get_covariate_value(cov::AbstractCovariate, actor::Int, wave::Int)
    if cov isa ConstantCovariate
        return cov.values[actor]
    elseif cov isa VaryingCovariate
        w = clamp(wave, 1, length(cov.values))
        return cov.values[w][actor]
    end
    return _not_actor_covariate(cov)
end

@noinline _not_actor_covariate(cov::AbstractCovariate)::Float64 =
    throw(ArgumentError("covariate :$(cov.name) is not an actor covariate " *
                        "(ConstantCovariate/VaryingCovariate)"))

# The similarity denominator, cached on the covariate at construction (it used to
# be recomputed -- with a `filter` allocation -- on every candidate dyad).
function _get_covariate_range(cov::AbstractCovariate)
    cov isa ConstantCovariate && return cov.range
    cov isa VaryingCovariate && return cov.range
    return 1.0
end

function _get_covariate_sim_mean(cov::AbstractCovariate)
    (cov isa ConstantCovariate || cov isa VaryingCovariate) && return cov.sim_mean
    return 0.0
end

function _get_dyad_covariate_value(cov::AbstractCovariate, i::Int, j::Int, wave::Int)
    if cov isa ConstantDyadCovariate
        return cov.values[i, j]
    elseif cov isa VaryingDyadCovariate
        w = clamp(wave, 1, length(cov.values))
        return cov.values[w][i, j]
    end
    return _not_dyadic_covariate(cov)
end

# The covariate value as the user gave it (centring undone). Effects that compare a
# covariate with a code -- a setting number, an event's setting -- must compare raw
# values: centring shifts every code by the covariate's mean.
function _raw_covariate_value(cov::AbstractCovariate, actor::Int, wave::Int)
    v = _get_covariate_value(cov, actor, wave)
    if cov isa ConstantCovariate
        return cov.centered ? v + cov.mean : v
    elseif cov isa VaryingCovariate
        return cov.centered ? v + cov.mean : v
    end
    return v
end

function _raw_dyad_covariate_value(cov::AbstractCovariate, i::Int, j::Int, wave::Int)
    v = _get_dyad_covariate_value(cov, i, j, wave)
    if cov isa ConstantDyadCovariate
        return cov.centered ? v + cov.mean : v
    elseif cov isa VaryingDyadCovariate
        return cov.centered ? v + cov.mean : v
    end
    return v
end

@noinline _not_dyadic_covariate(cov::AbstractCovariate)::Float64 =
    throw(ArgumentError("covariate :$(cov.name) is not a dyadic covariate " *
                        "(ConstantDyadCovariate/VaryingDyadCovariate)"))

# Value of `actor` on the attribute `name` that a covariate effect conditions on:
# an actor covariate, or a CO-EVOLVING dependent behaviour. A dependent behaviour is
# centred by its overall mean, as RSiena centres it when it enters network effects
# (egoX/altX/simX/... of a behaviour: network selection on the behaviour). The
# behaviour's current value is read, so during simulation it co-evolves, and in the
# moment statistics it is the value at the start of the period (RSiena's
# convention for every variable other than the effect's own target).
@inline function _attr_value(state::NetworkState, data::SienaData, name::Symbol,
                             actor::Int)
    beh = get(state.behaviors, name, nothing)
    if beh !== nothing
        return beh[actor] - (data.dependents[name]::DependentBehavior).mean_val
    end
    return _get_covariate_value(data.covariates[name], actor, state.period)
end

# Centred similarity (RSiena's sim_ij - ^sim) on a covariate or a dependent
# behaviour (whose range and mean similarity are those of the observed behaviour).
@inline function _attr_similarity(state::NetworkState, data::SienaData, name::Symbol,
                                  i::Int, j::Int)
    beh = get(state.behaviors, name, nothing)
    if beh !== nothing
        dep = data.dependents[name]::DependentBehavior
        return _centered_beh_similarity(dep, beh[i], beh[j])
    end
    return _centered_similarity(data.covariates[name], i, j, state.period)
end

# Centered similarity between actors i and j on a covariate (RSiena's sim_ij - ^sim).
function _centered_similarity(cov::AbstractCovariate, i::Int, j::Int, wave::Int)
    v1 = _get_covariate_value(cov, i, wave)
    v2 = _get_covariate_value(cov, j, wave)
    r = _get_covariate_range(cov)
    sim = r > 0 ? 1.0 - abs(v1 - v2) / r : 1.0
    return sim - _get_covariate_sim_mean(cov)
end

# Degree lookups. Simulation states store networks as `StateNetwork`s with
# incrementally maintained degree vectors, so the hot-loop lookups are O(1);
# the generic methods keep plain-matrix callers working.
_row_sum(net::StateNetwork, i::Int) = net.outdeg[i]
_col_sum(net::StateNetwork, j::Int) = net.indeg[j]
_row_sum(net::AbstractMatrix{Int}, i::Int) = @views sum(net[i, :])
_col_sum(net::AbstractMatrix{Int}, j::Int) = @views sum(net[:, j])

# Whether a variable is an undirected (symmetric) one-mode network. Target
# statistics of a few effects follow RSiena's symmetric-data conventions.
function _undirected(data::SienaData, variable::Symbol)
    dep = get(data.dependents, variable, nothing)
    return dep isa DependentNetwork && _is_undirected(dep)
end

# Sum of the actor components (the generic statistic), for effects that override
# `compute_statistic` only in special cases.
_actor_sum(e::AbstractEffect, state::NetworkState, data::SienaData) =
    sum(i -> evaluate_actor(e, state, data, i), 1:_n_focal_actors(e, state); init=0.0)

#==============================================================================#
# Basic Structural Effects
#==============================================================================#

"""
    OutdegreeEffect <: NetworkEffect

Basic outdegree effect (density): ``s_i = x_{i+}``. RSiena: density (short name
`:outdegree` in [`get_effects`](@ref); `:density` is accepted too).

On an undirected network this is RSiena's *degree (density)* effect, and the
target statistic counts each edge once (``\\frac12 \\sum_i x_{i+}``), as RSiena's does.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(OutdegreeEffect(:friendship), state, data)
```
"""
struct OutdegreeEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::OutdegreeEffect) = :density
effect_type(::OutdegreeEffect) = :eval
target_variable(e::OutdegreeEffect) = e.variable

function evaluate_actor(e::OutdegreeEffect, state::NetworkState, data::SienaData, actor::Int)
    return Float64(_row_sum(state.networks[e.variable], actor))
end

function compute_contribution(e::OutdegreeEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return 1.0
end

function compute_statistic(e::OutdegreeEffect, state::NetworkState, data::SienaData)
    s = _actor_sum(e, state, data)
    return _undirected(data, e.variable) ? s / 2 : s
end

"""
    ReciprocityEffect <: NetworkEffect

Reciprocity effect: ``s_i = \\sum_j x_{ij} x_{ji}``. RSiena: recip

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(ReciprocityEffect(:friendship), state, data)
```
"""
struct ReciprocityEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::ReciprocityEffect) = :recip
effect_type(::ReciprocityEffect) = :eval
target_variable(e::ReciprocityEffect) = e.variable

function evaluate_actor(e::ReciprocityEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        j == actor && continue
        count += net[actor, j] * net[j, actor]
    end
    return Float64(count)
end

function compute_contribution(e::ReciprocityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    return Float64(net[alter, actor])
end

#==============================================================================#
# Triadic Effects
#==============================================================================#

"""
    TransitiveTripletsEffect <: NetworkEffect

Transitive triplets: ``s_i = \\sum_{j \\ne h} x_{ij} x_{ih} x_{jh}``. RSiena: transTrip

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(TransitiveTripletsEffect(:friendship), state, data)
```
"""
struct TransitiveTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveTripletsEffect) = :transTrip
effect_type(::TransitiveTripletsEffect) = :eval
target_variable(e::TransitiveTripletsEffect) = e.variable

function evaluate_actor(e::TransitiveTripletsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    count = 0
    for j in net.outneighbors[actor]
        j == actor && continue
        for h in net.outneighbors[actor]
            (h == actor || h == j) && continue
            count += net[j, h]
        end
    end
    return Float64(count)
end

function compute_contribution(e::TransitiveTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    count = 0
    for h in net.outneighbors[actor]
        (h == actor || h == alter) && continue
        count += net[alter, h] + net[h, alter]
    end
    return Float64(count)
end

"""
    TransitiveTiesEffect <: NetworkEffect

Transitive ties: ``s_i = \\sum_j x_{ij} \\, I(\\exists h: x_{ih} x_{hj} = 1)``.
RSiena: transTies

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(TransitiveTiesEffect(:friendship), state, data)
```
"""
struct TransitiveTiesEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveTiesEffect) = :transTies
effect_type(::TransitiveTiesEffect) = :eval
target_variable(e::TransitiveTiesEffect) = e.variable

function evaluate_actor(e::TransitiveTiesEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[h, j] == 1
                count += 1
                break
            end
        end
    end
    return Float64(count)
end

# Number of two-paths i -> k -> h with k != skip (add-direction helpers need the
# count that does not go through the toggled alter).
@inline function _twopaths_excluding(net::StateNetwork, i::Int, h::Int, skip::Int)
    c = 0
    for k in net.outneighbors[i]
        (k == i || k == h || k == skip) && continue
        c += net[k, h]
    end
    return c
end

# Closed form (add direction): the tie i -> j is itself transitive when a two-path
# i -> k -> j exists; and it makes each existing tie i -> h transitive when j -> h
# and no other two-path i -> k -> h exists.
function compute_contribution(e::TransitiveTiesEffect, state::NetworkState,
                              data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    change = _twopaths_excluding(net, actor, alter, alter) > 0 ? 1.0 : 0.0
    for h in net.outneighbors[actor]
        (h == actor || h == alter) && continue
        if net[alter, h] == 1 && _twopaths_excluding(net, actor, h, alter) == 0
            change += 1.0
        end
    end
    return change
end

"""
    TransitiveTriadsEffect <: NetworkEffect

Transitive triads of an **undirected** network (RSiena: `transTriads`, defined for
symmetric networks only): ``s_i = \\tfrac12 \\sum_{j \\ne h} x_{ij} x_{ih} x_{jh}``, the
number of triangles containing ``i``. The change statistic of the tie ``i–j`` is the
number of their common neighbours (RSiena's two-path count), and the target
statistic is the number of triangles, ``\\frac16 \\sum_{i,j} x_{ij}\\,\\#\\{h: x_{ih}
x_{hj}=1\\}``, as in RSiena.

Before 0.2 this name was an alias of [`TransitiveTiesEffect`](@ref) (`transTies`), a
different statistic; it is now RSiena's `transTriads` and is refused on directed
networks (use `transTrip`/`transTies` there).

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
sym(x) = max.(x, x')
waves = [sym(NetworkCore.as_matrix(w)) for w in s50.friendship]  # 0/1 matrices
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, waves; directed=false))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(TransitiveTriadsEffect(:friendship), state, data)
```
"""
struct TransitiveTriadsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveTriadsEffect) = :transTriads
effect_type(::TransitiveTriadsEffect) = :eval
target_variable(e::TransitiveTriadsEffect) = e.variable

function evaluate_actor(e::TransitiveTriadsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    count = 0
    for j in net.outneighbors[actor]
        j == actor && continue
        for h in net.outneighbors[actor]
            (h == actor || h == j) && continue
            count += net[j, h]
        end
    end
    return count / 2
end

# RSiena's contribution: the number of two-paths i -> h -> j.
function compute_contribution(e::TransitiveTriadsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    count = 0
    for h in net.outneighbors[actor]
        (h == actor || h == alter) && continue
        count += net[h, alter]
    end
    return Float64(count)
end

function compute_statistic(e::TransitiveTriadsEffect, state::NetworkState,
                           data::SienaData)
    # Each triangle is counted once per member by the actor sum.
    return _actor_sum(e, state, data) / 3
end

"""
    TransitiveMediatedTripletsEffect <: NetworkEffect

Transitive mediated triplets (ego in the mediating position):
``s_i = \\sum_{j \\ne h} x_{ji} x_{ih} x_{jh}``. RSiena: transMedTrip

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(TransitiveMediatedTripletsEffect(:friendship), state, data)
```
"""
struct TransitiveMediatedTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveMediatedTripletsEffect) = :transMedTrip
effect_type(::TransitiveMediatedTripletsEffect) = :eval
target_variable(e::TransitiveMediatedTripletsEffect) = e.variable

function evaluate_actor(e::TransitiveMediatedTripletsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[j, actor] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[j, h] == 1
                count += 1
            end
        end
    end
    return Float64(count)
end

function compute_contribution(e::TransitiveMediatedTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for h in 1:n
        (h == actor || h == alter) && continue
        if net[h, actor] == 1 && net[h, alter] == 1
            count += 1
        end
    end
    return Float64(count)
end

"""
    TransitiveRecipTripletsEffect <: NetworkEffect

Transitive reciprocated triplets:
``s_i = \\sum_{j \\ne h} x_{ij} x_{ji} x_{ih} x_{hj}``. RSiena: transRecTrip

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(TransitiveRecipTripletsEffect(:friendship), state, data)
```
"""
struct TransitiveRecipTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveRecipTripletsEffect) = :transRecTrip
effect_type(::TransitiveRecipTripletsEffect) = :eval
target_variable(e::TransitiveRecipTripletsEffect) = e.variable

function evaluate_actor(e::TransitiveRecipTripletsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0 || net[j, actor] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[h, j] == 1
                count += 1
            end
        end
    end
    return Float64(count)
end

function compute_contribution(e::TransitiveRecipTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    # Toggled tie as the i -> j side of the mutual dyad
    if net[alter, actor] == 1
        for h in 1:n
            (h == actor || h == alter) && continue
            if net[actor, h] == 1 && net[h, alter] == 1
                count += 1.0
            end
        end
    end
    # Toggled tie as the two-path leg i -> h with h = alter
    for h in 1:n
        (h == actor || h == alter) && continue
        if net[actor, h] == 1 && net[h, actor] == 1 && net[alter, h] == 1
            count += 1.0
        end
    end
    return count
end

"""
    CyclicTripletsEffect <: NetworkEffect

Three-cycles: ``s_i = \\sum_{j \\ne h} x_{ij} x_{jh} x_{hi}``. RSiena: cycle3

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(CyclicTripletsEffect(:friendship), state, data)
```
"""
struct CyclicTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::CyclicTripletsEffect) = :cycle3
effect_type(::CyclicTripletsEffect) = :eval
target_variable(e::CyclicTripletsEffect) = e.variable

function evaluate_actor(e::CyclicTripletsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[j, h] == 1 && net[h, actor] == 1
                count += 1
            end
        end
    end
    return Float64(count)
end

function compute_contribution(e::CyclicTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for h in 1:n
        (h == actor || h == alter) && continue
        if net[alter, h] == 1 && net[h, actor] == 1
            count += 1
        end
    end
    return Float64(count)
end

# RSiena's cycle3 target statistic counts each 3-cycle once (sum_i s_i counts it
# three times, once per member).
function compute_statistic(e::CyclicTripletsEffect, state::NetworkState, data::SienaData)
    n = size(state.networks[e.variable], 1)
    return sum(evaluate_actor(e, state, data, i) for i in 1:n) / 3.0
end

"""
    BalanceSimpleEffect <: NetworkEffect

Structural balance (similarity of outgoing tie patterns with alters), with the
similarity sum normalized by ``n-2``:
``s_i = \\frac{1}{n-2} \\sum_j x_{ij} \\sum_{h \\ne i,j} (1 - |x_{ih} - x_{jh}|)``.

!!! warning "Not RSiena's balance"
    RSiena's `balance` subtracts a data-dependent centering constant ``b_0``
    (the mean tie-value dissimilarity over the observations) from each term
    instead of dividing by ``n-2``, so its statistic — and hence its parameter —
    is on a different scale. The short name here is therefore `:balanceSimple`:
    `:balance` is reserved for a numerically equivalent implementation and is
    deliberately not defined. Estimates from this effect are not comparable with
    RSiena's `balance` estimates.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(BalanceSimpleEffect(:friendship), state, data)
```
"""
struct BalanceSimpleEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::BalanceSimpleEffect) = :balanceSimple
effect_type(::BalanceSimpleEffect) = :eval
target_variable(e::BalanceSimpleEffect) = e.variable

function evaluate_actor(e::BalanceSimpleEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    n <= 2 && return 0.0
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            total += 1.0 - abs(net[actor, h] - net[j, h])
        end
    end
    return total / (n - 2)
end

"""
    BetweennessEffect <: NetworkEffect

Betweenness (brokerage): ``s_i = \\sum_{j \\ne h} x_{hi} x_{ij} (1 - x_{hj})``.
RSiena: between

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(BetweennessEffect(:friendship), state, data)
```
"""
struct BetweennessEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::BetweennessEffect) = :between
effect_type(::BetweennessEffect) = :eval
target_variable(e::BetweennessEffect) = e.variable

function evaluate_actor(e::BetweennessEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for h in 1:n
        (h == actor || net[h, actor] == 0) && continue
        for j in 1:n
            (j == actor || j == h) && continue
            if net[actor, j] == 1 && net[h, j] == 0
                count += 1
            end
        end
    end
    return Float64(count)
end

function compute_contribution(e::BetweennessEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for h in 1:n
        (h == actor || h == alter) && continue
        if net[h, actor] == 1 && net[h, alter] == 0
            count += 1
        end
    end
    return Float64(count)
end

"""
    NbrDist2Effect <: NetworkEffect

Number of actors at distance exactly 2:
``s_i = \\#\\{j \\ne i : x_{ij} = 0, \\exists h: x_{ih} x_{hj} = 1\\}``. RSiena: nbrDist2

On an undirected network the target statistic counts each pair once (half the
actor sum), as RSiena does.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(NbrDist2Effect(:friendship), state, data)
```
"""
struct NbrDist2Effect <: NetworkEffect
    variable::Symbol
end

effect_name(::NbrDist2Effect) = :nbrDist2
effect_type(::NbrDist2Effect) = :eval
target_variable(e::NbrDist2Effect) = e.variable

function evaluate_actor(e::NbrDist2Effect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 1) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[h, j] == 1
                count += 1
                break
            end
        end
    end
    return Float64(count)
end

# Closed form (add direction): j itself stops being at distance two if it was;
# and each out-neighbour h of j that ego is not tied to, and cannot reach through
# another intermediary, becomes an actor at distance two.
function compute_contribution(e::NbrDist2Effect, state::NetworkState,
                              data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    change = _twopaths_excluding(net, actor, alter, alter) > 0 ? -1.0 : 0.0
    for h in net.outneighbors[alter]
        (h == actor || h == alter) && continue
        if net[actor, h] == 0 && _twopaths_excluding(net, actor, h, alter) == 0
            change += 1.0
        end
    end
    return change
end

function compute_statistic(e::NbrDist2Effect, state::NetworkState, data::SienaData)
    s = _actor_sum(e, state, data)
    return _undirected(data, e.variable) ? s / 2 : s
end

"""
    DenseTriadsEffect <: NetworkEffect

Dense triads (tie-weighted, following RSiena):
``s_i = \\sum_j x_{ij} \\#\\{h \\ne i,j : \\text{triad } (i,j,h) \\text{ has} \\ge c
\\text{ arcs}\\}`` with ``c = 5``. RSiena: denseTriads

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(DenseTriadsEffect(:friendship), state, data)
```
"""
struct DenseTriadsEffect <: NetworkEffect
    variable::Symbol
    c::Int
    DenseTriadsEffect(variable::Symbol; c::Int=5) = new(variable, c)
end

effect_name(::DenseTriadsEffect) = :denseTriads
effect_type(::DenseTriadsEffect) = :eval
target_variable(e::DenseTriadsEffect) = e.variable

function evaluate_actor(e::DenseTriadsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            ties = net[actor, j] + net[j, actor] + net[actor, h] + net[h, actor] +
                   net[j, h] + net[h, j]
            ties >= e.c && (count += 1)
        end
    end
    return Float64(count)
end

"""
    SharedInEffect <: NetworkEffect

Ties to alters with shared in-neighbors:
``s_i = \\sum_j x_{ij} \\#\\{h \\ne i,j : x_{hi} x_{hj} = 1\\}``.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(SharedInEffect(:friendship), state, data)
```
"""
struct SharedInEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::SharedInEffect) = :sharedInNbrs
effect_type(::SharedInEffect) = :eval
target_variable(e::SharedInEffect) = e.variable

function evaluate_actor(e::SharedInEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[h, actor] == 1 && net[h, j] == 1
                count += 1
            end
        end
    end
    return Float64(count)
end

function compute_contribution(e::SharedInEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for h in 1:n
        (h == actor || h == alter) && continue
        if net[h, actor] == 1 && net[h, alter] == 1
            count += 1
        end
    end
    return Float64(count)
end

"""
    SharedOutEffect <: NetworkEffect

Ties to alters with shared out-neighbors:
``s_i = \\sum_j x_{ij} \\#\\{h \\ne i,j : x_{ih} x_{jh} = 1\\}``.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(SharedOutEffect(:friendship), state, data)
```
"""
struct SharedOutEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::SharedOutEffect) = :sharedOutNbrs
effect_type(::SharedOutEffect) = :eval
target_variable(e::SharedOutEffect) = e.variable

function evaluate_actor(e::SharedOutEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[j, h] == 1
                count += 1
            end
        end
    end
    return Float64(count)
end

#==============================================================================#
# Degree-Based Effects
#==============================================================================#

"""
    IndegreePopularityEffect <: NetworkEffect

Indegree popularity: ``s_i = \\sum_j x_{ij} f(x_{+j})`` with ``f`` the identity or
square root. RSiena: inPop, inPopSqrt

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(IndegreePopularityEffect(:friendship), state, data)
```
"""
struct IndegreePopularityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    IndegreePopularityEffect(variable::Symbol; sqrt::Bool=false) = new(variable, sqrt)
end

effect_name(e::IndegreePopularityEffect) = e.sqrt ? :inPopSqrt : :inPop
effect_type(::IndegreePopularityEffect) = :eval
target_variable(e::IndegreePopularityEffect) = e.variable

function evaluate_actor(e::IndegreePopularityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    total = 0.0
    # Over ego's out-ties (also correct for a two-mode network, whose alters are
    # the events/columns).
    for j in net.outneighbors[actor]
        indeg = _col_sum(net, j)
        total += e.sqrt ? sqrt(Float64(indeg)) : Float64(indeg)
    end
    return total
end

function compute_contribution(e::IndegreePopularityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    indeg = _col_sum(net, alter) - net[actor, alter]  # alter's indegree without the tie
    d = Float64(indeg + 1)                            # ... and with it
    return e.sqrt ? sqrt(d) : d
end

# RSiena's internal effect parameter for outPop/outPopSqrt/inAct/inActSqrt
# (`OutdegreePopularityEffect.cpp`, `IndegreeActivityEffect.cpp`): with `parm > 0`
# the moment statistic is the evaluation statistic at the end of the period; with
# `parm == 0` the degree in it is the one at the START of the period (the observed
# period-start wave); with `parm < 0` it is the sum of the two.
"""
    OutdegreePopularityEffect(variable; sqrt=false, parm=sqrt ? 0 : 1) <: NetworkEffect

Outdegree popularity: ``s_i = \\sum_j x_{ij} f(x_{j+})`` with ``f`` the identity
(RSiena: `outPop`) or the square root (RSiena: `outPopSqrt`). The change statistic
of the tie ``i \\to j`` is ``f(x_{j+})``.

`parm` is RSiena's internal effect parameter, with RSiena's defaults (1 for `outPop`,
0 for `outPopSqrt`). It changes only the **moment (target) statistic**, exactly as in
RSiena 1.6.6:

- `parm ≥ 1`: ``\\sum_{ij} x_{ij}(t_1)\\, f(x_{j+}(t_1))``, the manual's definition at the
  end of the period;
- `parm = 0`: ``\\sum_{ij} x_{ij}(t_1)\\, x_{j+}(t_0)`` — the alter's out-degree at the
  START of the period, and *not* square-rooted even for `outPopSqrt` (this is what
  RSiena's code computes for its default `outPopSqrt`);
- `parm < 0`: ``\\sum_{ij} x_{ij}(t_1)\\, (x_{j+}(t_1) + x_{j+}(t_0))``.

Before 0.2, `sqrt=true` always used the `parm ≥ 1` statistic, so a model copied from
RSiena (default parm 0) solved different moment equations.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(OutdegreePopularityEffect(:friendship), state, data)
```
"""
struct OutdegreePopularityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    parm::Int
    OutdegreePopularityEffect(variable::Symbol; sqrt::Bool=false,
                              parm::Int=sqrt ? 0 : 1) = new(variable, sqrt, parm)
end

effect_name(e::OutdegreePopularityEffect) = e.sqrt ? :outPopSqrt : :outPop
effect_type(::OutdegreePopularityEffect) = :eval
target_variable(e::OutdegreePopularityEffect) = e.variable

function evaluate_actor(e::OutdegreePopularityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        outdeg = _row_sum(net, j)
        total += e.sqrt ? sqrt(Float64(outdeg)) : Float64(outdeg)
    end
    return total
end

function compute_contribution(e::OutdegreePopularityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = Float64(_row_sum(net, alter))
    return e.sqrt ? sqrt(outdeg) : outdeg
end

# The observed network at the start of the state's period (RSiena's `pStart`).
_period_start_network(data::SienaData, variable::Symbol, period::Int) =
    (data.dependents[variable]::DependentNetwork).networks[period]

function compute_statistic(e::OutdegreePopularityEffect, state::NetworkState,
                           data::SienaData)
    e.parm > 0 && return _actor_sum(e, state, data)
    net = state.networks[e.variable]
    start = _period_start_network(data, e.variable, state.period)
    n = size(net, 1)
    total = 0.0
    for i in 1:n, j in 1:n
        (j == i || net[i, j] == 0) && continue
        d0 = Float64(sum(view(start, j, :)) - start[j, j])
        total += e.parm == 0 ? d0 : Float64(_row_sum(net, j)) + d0
    end
    return total
end

"""
    IndegreeActivityEffect(variable; sqrt=false, parm=sqrt ? 0 : 1) <: NetworkEffect

Indegree activity: ``s_i = x_{i+} f(x_{+i})`` with ``f`` the identity (RSiena:
`inAct`) or the square root (RSiena: `inActSqrt`). The change statistic of a tie
``i \\to j`` is ``f(x_{+i})``.

`parm` is RSiena's internal effect parameter, with RSiena's defaults (1 for `inAct`,
0 for `inActSqrt`); it changes only the **moment (target) statistic**, exactly as in
RSiena 1.6.6:

- `parm ≥ 1`: ``\\sum_i x_{i+}(t_1)\\, f(x_{+i}(t_1))``, the manual's definition;
- `parm = 0`: ``\\sum_i x_{i+}(t_1)\\, f(x_{+i}(t_0))`` — ego's in-degree at the START
  of the period (RSiena's default `inActSqrt`);
- `parm < 0`: ``\\sum_i x_{i+}(t_1)\\, f(x_{+i}(t_1) + x_{+i}(t_0))``.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(IndegreeActivityEffect(:friendship), state, data)
```
"""
struct IndegreeActivityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    parm::Int
    IndegreeActivityEffect(variable::Symbol; sqrt::Bool=false,
                           parm::Int=sqrt ? 0 : 1) = new(variable, sqrt, parm)
end

effect_name(e::IndegreeActivityEffect) = e.sqrt ? :inActSqrt : :inAct
effect_type(::IndegreeActivityEffect) = :eval
target_variable(e::IndegreeActivityEffect) = e.variable

function evaluate_actor(e::IndegreeActivityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    indeg = Float64(_col_sum(net, actor))
    outdeg = Float64(_row_sum(net, actor))
    return outdeg * (e.sqrt ? sqrt(indeg) : indeg)
end

function compute_contribution(e::IndegreeActivityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    indeg = Float64(_col_sum(net, actor))
    return e.sqrt ? sqrt(indeg) : indeg
end

function compute_statistic(e::IndegreeActivityEffect, state::NetworkState,
                           data::SienaData)
    e.parm > 0 && return _actor_sum(e, state, data)
    net = state.networks[e.variable]
    start = _period_start_network(data, e.variable, state.period)
    n = size(net, 1)
    total = 0.0
    for i in 1:n
        d0 = Float64(sum(view(start, :, i)) - start[i, i])
        d = e.parm == 0 ? d0 : Float64(_col_sum(net, i)) + d0
        total += _row_sum(net, i) * (e.sqrt ? sqrt(d) : d)
    end
    return total
end

"""
    OutdegreeActivityEffect <: NetworkEffect

Outdegree activity: ``s_i = x_{i+} f(x_{i+})`` (i.e. ``x_{i+}^2`` or
``x_{i+}^{3/2}``). RSiena: outAct, outActSqrt

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(OutdegreeActivityEffect(:friendship), state, data)
```
"""
struct OutdegreeActivityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    OutdegreeActivityEffect(variable::Symbol; sqrt::Bool=false) = new(variable, sqrt)
end

effect_name(e::OutdegreeActivityEffect) = e.sqrt ? :outActSqrt : :outAct
effect_type(::OutdegreeActivityEffect) = :eval
target_variable(e::OutdegreeActivityEffect) = e.variable

_outact_value(e::OutdegreeActivityEffect, d::Float64) = e.sqrt ? d^1.5 : d^2

function evaluate_actor(e::OutdegreeActivityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    return _outact_value(e, Float64(_row_sum(net, actor)))
end

function compute_contribution(e::OutdegreeActivityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    d = Float64(_row_sum(net, actor) - net[actor, alter])  # outdegree without the tie
    return _outact_value(e, d + 1) - _outact_value(e, d)
end

"""
    OutdegreeTruncEffect <: NetworkEffect

Truncated outdegree: ``s_i = \\min(x_{i+}, c)``. RSiena: outTrunc

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(OutdegreeTruncEffect(:friendship), state, data)
```
"""
struct OutdegreeTruncEffect <: NetworkEffect
    variable::Symbol
    c::Int
    OutdegreeTruncEffect(variable::Symbol; c::Int=1) = new(variable, c)
end

effect_name(::OutdegreeTruncEffect) = :outTrunc
effect_type(::OutdegreeTruncEffect) = :eval
target_variable(e::OutdegreeTruncEffect) = e.variable

function evaluate_actor(e::OutdegreeTruncEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    return Float64(min(_row_sum(net, actor), e.c))
end

function compute_contribution(e::OutdegreeTruncEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    d = _row_sum(net, actor) - net[actor, alter]
    return d < e.c ? 1.0 : 0.0
end

"""
    IndegreeTruncEffect <: NetworkEffect

Truncated indegree popularity: ``s_i = \\sum_j x_{ij} \\min(x_{+j}, c)``.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(IndegreeTruncEffect(:friendship), state, data)
```
"""
struct IndegreeTruncEffect <: NetworkEffect
    variable::Symbol
    c::Int
    IndegreeTruncEffect(variable::Symbol; c::Int=1) = new(variable, c)
end

effect_name(::IndegreeTruncEffect) = :inTrunc
effect_type(::IndegreeTruncEffect) = :eval
target_variable(e::IndegreeTruncEffect) = e.variable

function evaluate_actor(e::IndegreeTruncEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += min(_col_sum(net, j), e.c)
    end
    return Float64(total)
end

function compute_contribution(e::IndegreeTruncEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    d = _col_sum(net, alter) - net[actor, alter]
    return Float64(min(d + 1, e.c))
end

"""
    DegreeAssortativityEffect <: NetworkEffect

Degree activity plus popularity of an **undirected** network:
``s_i = \\sum_j x_{ij} (d_i + d_j)`` with ``d`` the degree, i.e. RSiena's
`inPop` plus `outAct`, ``\\sum_j x_{ij} x_{+j} + x_{i+}^2``, which is how its change
statistic is computed. RSiena: `degPlus` (defined
for symmetric networks only, and refused on directed ones). Before 0.2 the effect
was offered on directed networks with ``d`` the in- plus out-degree, which is not an
RSiena effect.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
sym(x) = max.(x, x')
waves = [sym(NetworkCore.as_matrix(w)) for w in s50.friendship]  # 0/1 matrices
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, waves; directed=false))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(DegreeAssortativityEffect(:friendship), state, data)
```
"""
struct DegreeAssortativityEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::DegreeAssortativityEffect) = :degPlus
effect_type(::DegreeAssortativityEffect) = :eval
target_variable(e::DegreeAssortativityEffect) = e.variable

function evaluate_actor(e::DegreeAssortativityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = Float64(_row_sum(net, actor))^2
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _col_sum(net, j)
    end
    return total
end

function compute_contribution(e::DegreeAssortativityEffect, state::NetworkState,
                              data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    x = net[actor, alter]
    return Float64(_col_sum(net, alter) - x + 1 + 2 * (_row_sum(net, actor) - x) + 1)
end

#==============================================================================#
# Isolate Effects
#==============================================================================#

"""
    IsolateNetEffect <: NetworkEffect

Network isolate (total isolate, as in RSiena):
``s_i = I(x_{i+} = 0 \\text{ and } x_{+i} = 0)``. RSiena: isolateNet

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(IsolateNetEffect(:friendship), state, data)
```
"""
struct IsolateNetEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::IsolateNetEffect) = :isolateNet
effect_type(::IsolateNetEffect) = :eval
target_variable(e::IsolateNetEffect) = e.variable

function evaluate_actor(e::IsolateNetEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    return (_row_sum(net, actor) == 0 && _col_sum(net, actor) == 0) ? 1.0 : 0.0
end

function compute_contribution(e::IsolateNetEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = _row_sum(net, actor) - net[actor, alter]
    indeg = _col_sum(net, actor)
    return (outdeg == 0 && indeg == 0) ? -1.0 : 0.0
end

"""
    OutIsolateEffect <: NetworkEffect

Out-isolate: ``s_i = I(x_{i+} = 0)``. RSiena: `outIso` (internal parameter 1). Before
0.2 its short name was `outIsolate`, which in RSiena is a *behaviour* effect.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(OutIsolateEffect(:friendship), state, data)
```
"""
struct OutIsolateEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::OutIsolateEffect) = :outIso
effect_type(::OutIsolateEffect) = :eval
target_variable(e::OutIsolateEffect) = e.variable

function evaluate_actor(e::OutIsolateEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    return _row_sum(net, actor) == 0 ? 1.0 : 0.0
end

function compute_contribution(e::OutIsolateEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = _row_sum(net, actor) - net[actor, alter]
    return outdeg == 0 ? -1.0 : 0.0
end

"""
    InIsolateEffect <: NetworkEffect

Ties to in-isolates: ``s_i = \\sum_j x_{ij} I(x_{+j} - x_{ij} = 0)`` — ties to alters
whose only incoming tie is from ego.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(InIsolateEffect(:friendship), state, data)
```
"""
struct InIsolateEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::InIsolateEffect) = :inIsolate
effect_type(::InIsolateEffect) = :eval
target_variable(e::InIsolateEffect) = e.variable

function evaluate_actor(e::InIsolateEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        if _col_sum(net, j) - net[actor, j] == 0
            count += 1
        end
    end
    return Float64(count)
end

function compute_contribution(e::InIsolateEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    indeg = _col_sum(net, alter) - net[actor, alter]
    return indeg == 0 ? 1.0 : 0.0
end

#==============================================================================#
# GWESP / Shared Partner Effects
#==============================================================================#

# GWESP weight: exp(alpha) * (1 - (1 - exp(-alpha))^sp). The default alpha is
# RSiena's: internal parameter 69, alpha = parm / 100 = 0.69 (not log 2 = 0.6931).
_gwesp_weight(alpha::Float64, sp::Int) =
    sp == 0 ? 0.0 : exp(alpha) * (1.0 - (1.0 - exp(-alpha))^sp)

"""
    GWESPEffect <: NetworkEffect

Geometrically weighted edgewise shared partners, forward-forward (two-path closure,
the GWESP analogue of transTrip): shared partners of the tie ``i \\to j`` are actors
``h`` with ``x_{ih} = x_{hj} = 1``. RSiena: `gwespFF` on directed networks and
`gwesp` on undirected ones. `alpha` defaults to RSiena's 0.69 (internal parameter
69); before 0.2 it was ``\\log 2``.

As in RSiena, the GWESP effects (`gwespFF`, `gwespBB`, `gwespFB`) are *elementary*
effects: the change statistic of the tie ``i \\to j`` is the weight
``e^{\\alpha}(1 - (1 - e^{-\\alpha})^{sp_{ij}})`` of that tie alone, not the full
change of the actor statistic (which would also count the shared partners the new tie
adds to ego's other ties). Before 0.2 the full difference was used, which is a
different model with the same target statistic.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(GWESPEffect(:friendship), state, data)
```
"""
struct GWESPEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWESPEffect(variable::Symbol; alpha::Float64=0.69) = new(variable, alpha)
end

effect_name(::GWESPEffect) = :gwespFF
effect_type(::GWESPEffect) = :eval
target_variable(e::GWESPEffect) = e.variable

function _esp_count(net::AbstractMatrix{Int}, i::Int, j::Int, ego_out::Bool, alter_out::Bool)
    n = size(net, 1)
    sp = 0
    for h in 1:n
        (h == i || h == j) && continue
        ego_tie = ego_out ? net[i, h] : net[h, i]
        alter_tie = alter_out ? net[j, h] : net[h, j]
        if ego_tie == 1 && alter_tie == 1
            sp += 1
        end
    end
    return sp
end

# RSiena implements the GWESP effects as ELEMENTARY effects: the change statistic
# of the tie i -> j is the weight of that tie's own shared partners only; the change
# in the shared-partner counts of ego's other ties is deliberately not included
# (RSiena manual, GWESP: "not implemented as an evaluation effect, but as an
# elementary effect").
function compute_contribution(e::GWESPEffect, state::NetworkState,
                              data::SienaData, actor::Int, alter::Int)
    return _gwesp_weight(e.alpha, _esp_count(state.networks[e.variable], actor, alter,
                                             true, false))
end

# Sparse version for simulation states: only ego's neighbours can be shared partners.
function _esp_count(net::StateNetwork, i::Int, j::Int, ego_out::Bool, alter_out::Bool)
    sp = 0
    for h in (ego_out ? net.outneighbors[i] : net.inneighbors[i])
        (h == i || h == j) && continue
        sp += alter_out ? net[j, h] : net[h, j]
    end
    return sp
end

function evaluate_actor(e::GWESPEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        # FF: i -> h and h -> j
        total += _gwesp_weight(e.alpha, _esp_count(net, actor, j, true, false))
    end
    return total
end

"""
    GWESPBackwardEffect <: NetworkEffect

GWESP backward-backward: shared partners of the tie ``i \\to j`` are actors ``h``
with ``x_{hi} = x_{jh} = 1``. RSiena: gwespBB

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(GWESPBackwardEffect(:friendship), state, data)
```
"""
struct GWESPBackwardEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWESPBackwardEffect(variable::Symbol; alpha::Float64=0.69) = new(variable, alpha)
end

effect_name(::GWESPBackwardEffect) = :gwespBB
effect_type(::GWESPBackwardEffect) = :eval
target_variable(e::GWESPBackwardEffect) = e.variable

# RSiena implements the GWESP effects as ELEMENTARY effects: the change statistic
# of the tie i -> j is the weight of that tie's own shared partners only; the change
# in the shared-partner counts of ego's other ties is deliberately not included
# (RSiena manual, GWESP: "not implemented as an evaluation effect, but as an
# elementary effect").
function compute_contribution(e::GWESPBackwardEffect, state::NetworkState,
                              data::SienaData, actor::Int, alter::Int)
    return _gwesp_weight(e.alpha, _esp_count(state.networks[e.variable], actor, alter,
                                             false, true))
end

function evaluate_actor(e::GWESPBackwardEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        # BB: h -> i and j -> h
        total += _gwesp_weight(e.alpha, _esp_count(net, actor, j, false, true))
    end
    return total
end

"""
    GWESPMixedEffect <: NetworkEffect

GWESP forward-backward: shared partners of the tie ``i \\to j`` are actors ``h``
with ``x_{ih} = x_{jh} = 1`` (shared out-alters). RSiena: gwespFB

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(GWESPMixedEffect(:friendship), state, data)
```
"""
struct GWESPMixedEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWESPMixedEffect(variable::Symbol; alpha::Float64=0.69) = new(variable, alpha)
end

effect_name(::GWESPMixedEffect) = :gwespFB
effect_type(::GWESPMixedEffect) = :eval
target_variable(e::GWESPMixedEffect) = e.variable

# RSiena implements the GWESP effects as ELEMENTARY effects: the change statistic
# of the tie i -> j is the weight of that tie's own shared partners only; the change
# in the shared-partner counts of ego's other ties is deliberately not included
# (RSiena manual, GWESP: "not implemented as an evaluation effect, but as an
# elementary effect").
function compute_contribution(e::GWESPMixedEffect, state::NetworkState,
                              data::SienaData, actor::Int, alter::Int)
    return _gwesp_weight(e.alpha, _esp_count(state.networks[e.variable], actor, alter,
                                             true, true))
end

function evaluate_actor(e::GWESPMixedEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        # FB: i -> h and j -> h
        total += _gwesp_weight(e.alpha, _esp_count(net, actor, j, true, true))
    end
    return total
end

"""
    GWDSPEffect <: NetworkEffect

Geometrically weighted dyadwise shared partners (two-paths, regardless of the direct
tie): ``s_i = \\sum_{j \\ne i} w(\\#\\{h : x_{ih} x_{hj} = 1\\})``. RSiena: gwdspFF

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(GWDSPEffect(:friendship), state, data)
```
"""
struct GWDSPEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWDSPEffect(variable::Symbol; alpha::Float64=0.69) = new(variable, alpha)
end

effect_name(::GWDSPEffect) = :gwdspFF
effect_type(::GWDSPEffect) = :eval
target_variable(e::GWDSPEffect) = e.variable

# Closed form (add direction): the tie i -> j adds one two-path i -> j -> h for
# every out-neighbour h of j.
function compute_contribution(e::GWDSPEffect, state::NetworkState,
                              data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    change = 0.0
    for h in net.outneighbors[alter]
        (h == actor || h == alter) && continue
        t = _twopaths_excluding(net, actor, h, alter)
        change += _gwesp_weight(e.alpha, t + 1) - _gwesp_weight(e.alpha, t)
    end
    return change
end

function evaluate_actor(e::GWDSPEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        j == actor && continue
        total += _gwesp_weight(e.alpha, _esp_count(net, actor, j, true, false))
    end
    return total
end

#==============================================================================#
# Covariate Effects
#==============================================================================#

"""
    EgoEffect <: NetworkEffect

Ego covariate effect: ``s_i = v_i x_{i+}``. RSiena: egoX

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(EgoEffect(:friendship, :smoke1), state, data)
compute_statistic(EgoEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct EgoEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoEffect) = :egoX
effect_type(::EgoEffect) = :eval
target_variable(e::EgoEffect) = e.variable
interaction_with(e::EgoEffect) = e.covariate

function evaluate_actor(e::EgoEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    v = _attr_value(state, data, e.covariate, actor)
    return v * _row_sum(net, actor)
end

function compute_contribution(e::EgoEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _attr_value(state, data, e.covariate, actor)
end

"""
    EgoSqEffect <: NetworkEffect

Squared ego covariate effect: ``s_i = v_i^2 x_{i+}``. RSiena: egoSqX

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(EgoSqEffect(:friendship, :smoke1), state, data)
compute_statistic(EgoSqEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct EgoSqEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoSqEffect) = :egoSqX
effect_type(::EgoSqEffect) = :eval
target_variable(e::EgoSqEffect) = e.variable
interaction_with(e::EgoSqEffect) = e.covariate

function evaluate_actor(e::EgoSqEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    v = _attr_value(state, data, e.covariate, actor)
    return v^2 * _row_sum(net, actor)
end

function compute_contribution(e::EgoSqEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    v = _attr_value(state, data, e.covariate, actor)
    return v^2
end

"""
    AlterEffect <: NetworkEffect

Alter covariate effect: ``s_i = \\sum_j x_{ij} v_j``. RSiena: altX

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(AlterEffect(:friendship, :smoke1), state, data)
compute_statistic(AlterEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct AlterEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::AlterEffect) = :altX
effect_type(::AlterEffect) = :eval
target_variable(e::AlterEffect) = e.variable
interaction_with(e::AlterEffect) = e.covariate

function evaluate_actor(e::AlterEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _attr_value(state, data, e.covariate, j)
    end
    return total
end

function compute_contribution(e::AlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _attr_value(state, data, e.covariate, alter)
end

"""
    AlterSqEffect <: NetworkEffect

Squared alter covariate effect: ``s_i = \\sum_j x_{ij} v_j^2``. RSiena: altSqX

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(AlterSqEffect(:friendship, :smoke1), state, data)
compute_statistic(AlterSqEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct AlterSqEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::AlterSqEffect) = :altSqX
effect_type(::AlterSqEffect) = :eval
target_variable(e::AlterSqEffect) = e.variable
interaction_with(e::AlterSqEffect) = e.covariate

function evaluate_actor(e::AlterSqEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _attr_value(state, data, e.covariate, j)^2
    end
    return total
end

function compute_contribution(e::AlterSqEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    v = _attr_value(state, data, e.covariate, alter)
    return v^2
end

"""
    SimilarityEffect <: NetworkEffect

Covariate similarity: ``s_i = \\sum_j x_{ij} (\\text{sim}_{ij} - \\widehat{sim})``
with ``\\text{sim}_{ij} = 1 - |v_i - v_j|/r_V`` and ``\\widehat{sim}`` the observed
mean similarity (as in RSiena). RSiena: simX

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(SimilarityEffect(:friendship, :smoke1), state, data)
compute_statistic(SimilarityEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct SimilarityEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SimilarityEffect) = :simX
effect_type(::SimilarityEffect) = :eval
target_variable(e::SimilarityEffect) = e.variable
interaction_with(e::SimilarityEffect) = e.covariate

function evaluate_actor(e::SimilarityEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _attr_similarity(state, data, e.covariate, actor, j)
    end
    return total
end

function compute_contribution(e::SimilarityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _attr_similarity(state, data, e.covariate, actor, alter)
end

"""
    SameEffect <: NetworkEffect

Same covariate value: ``s_i = \\sum_j x_{ij} I(v_i = v_j)``. RSiena: sameX

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(SameEffect(:friendship, :smoke1), state, data)
compute_statistic(SameEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct SameEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SameEffect) = :sameX
effect_type(::SameEffect) = :eval
target_variable(e::SameEffect) = e.variable
interaction_with(e::SameEffect) = e.covariate

function evaluate_actor(e::SameEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    v1 = _attr_value(state, data, e.covariate, actor)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        v1 == _attr_value(state, data, e.covariate, j) && (count += 1)
    end
    return Float64(count)
end

function compute_contribution(e::SameEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    v1 = _attr_value(state, data, e.covariate, actor)
    v2 = _attr_value(state, data, e.covariate, alter)
    return v1 == v2 ? 1.0 : 0.0
end

"""
    DifferenceEffect <: NetworkEffect

Difference effect (alter minus ego): ``s_i = \\sum_j x_{ij} (v_j - v_i)``. RSiena:
diffX. Before 0.2 the sign was reversed (ego minus alter), so coefficients had the
opposite sign to RSiena's under the same name.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(DifferenceEffect(:friendship, :smoke1), state, data)
compute_statistic(DifferenceEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct DifferenceEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::DifferenceEffect) = :diffX
effect_type(::DifferenceEffect) = :eval
target_variable(e::DifferenceEffect) = e.variable
interaction_with(e::DifferenceEffect) = e.covariate

function evaluate_actor(e::DifferenceEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    v1 = _attr_value(state, data, e.covariate, actor)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _attr_value(state, data, e.covariate, j) - v1
    end
    return total
end

function compute_contribution(e::DifferenceEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _attr_value(state, data, e.covariate, alter) -
           _attr_value(state, data, e.covariate, actor)
end

"""
    DifferenceSqEffect <: NetworkEffect

Squared difference effect: ``s_i = \\sum_j x_{ij} (v_i - v_j)^2``. RSiena: diffSqX

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(DifferenceSqEffect(:friendship, :smoke1), state, data)
compute_statistic(DifferenceSqEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct DifferenceSqEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::DifferenceSqEffect) = :diffSqX
effect_type(::DifferenceSqEffect) = :eval
target_variable(e::DifferenceSqEffect) = e.variable
interaction_with(e::DifferenceSqEffect) = e.covariate

function evaluate_actor(e::DifferenceSqEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    v1 = _attr_value(state, data, e.covariate, actor)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += (v1 - _attr_value(state, data, e.covariate, j))^2
    end
    return total
end

function compute_contribution(e::DifferenceSqEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    d = _attr_value(state, data, e.covariate, actor) -
        _attr_value(state, data, e.covariate, alter)
    return d^2
end

"""
    AbsDifferenceEffect <: NetworkEffect

Absolute difference effect: ``s_i = \\sum_j x_{ij} |v_i - v_j|``. RSiena: absDiffX

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(AbsDifferenceEffect(:friendship, :smoke1), state, data)
compute_statistic(AbsDifferenceEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct AbsDifferenceEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::AbsDifferenceEffect) = :absDiffX
effect_type(::AbsDifferenceEffect) = :eval
target_variable(e::AbsDifferenceEffect) = e.variable
interaction_with(e::AbsDifferenceEffect) = e.covariate

function evaluate_actor(e::AbsDifferenceEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    v1 = _attr_value(state, data, e.covariate, actor)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += abs(v1 - _attr_value(state, data, e.covariate, j))
    end
    return total
end

function compute_contribution(e::AbsDifferenceEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return abs(_attr_value(state, data, e.covariate, actor) -
               _attr_value(state, data, e.covariate, alter))
end

"""
    HigherEffect <: NetworkEffect

Ego higher than alter: ``s_i = \\sum_j x_{ij} \\big(I(v_i > v_j) + \\tfrac12 I(v_i =
v_j)\\big)``. RSiena: higher (ties count one half). Before 0.2 ties counted 0.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(HigherEffect(:friendship, :smoke1), state, data)
compute_statistic(HigherEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct HigherEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::HigherEffect) = :higher
effect_type(::HigherEffect) = :eval
target_variable(e::HigherEffect) = e.variable
interaction_with(e::HigherEffect) = e.covariate

function evaluate_actor(e::HigherEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    v1 = _attr_value(state, data, e.covariate, actor)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _higher_score(v1, _attr_value(state, data, e.covariate, j))
    end
    return total
end

_higher_score(vi::Float64, vj::Float64) = vi > vj ? 1.0 : (vi == vj ? 0.5 : 0.0)

function compute_contribution(e::HigherEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _higher_score(_attr_value(state, data, e.covariate, actor),
                         _attr_value(state, data, e.covariate, alter))
end

"""
    EgoTimesAlterEffect <: NetworkEffect

Ego × alter interaction: ``s_i = \\sum_j x_{ij} v_i v_j``. RSiena: egoXaltX

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(EgoTimesAlterEffect(:friendship, :smoke1), state, data)
compute_statistic(EgoTimesAlterEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct EgoTimesAlterEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoTimesAlterEffect) = :egoXaltX
effect_type(::EgoTimesAlterEffect) = :eval
target_variable(e::EgoTimesAlterEffect) = e.variable
interaction_with(e::EgoTimesAlterEffect) = e.covariate

function evaluate_actor(e::EgoTimesAlterEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    v1 = _attr_value(state, data, e.covariate, actor)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += v1 * _attr_value(state, data, e.covariate, j)
    end
    return total
end

function compute_contribution(e::EgoTimesAlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _attr_value(state, data, e.covariate, actor) *
           _attr_value(state, data, e.covariate, alter)
end

"""
    EgoPlusAlterEffect <: NetworkEffect

Ego + alter sum effect: ``s_i = \\sum_j x_{ij} (v_i + v_j)``. RSiena: egoPlusAltX

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(EgoPlusAlterEffect(:friendship, :smoke1), state, data)
compute_statistic(EgoPlusAlterEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct EgoPlusAlterEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoPlusAlterEffect) = :egoPlusAltX
effect_type(::EgoPlusAlterEffect) = :eval
target_variable(e::EgoPlusAlterEffect) = e.variable
interaction_with(e::EgoPlusAlterEffect) = e.covariate

function evaluate_actor(e::EgoPlusAlterEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    v1 = _attr_value(state, data, e.covariate, actor)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += v1 + _attr_value(state, data, e.covariate, j)
    end
    return total
end

function compute_contribution(e::EgoPlusAlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _attr_value(state, data, e.covariate, actor) +
           _attr_value(state, data, e.covariate, alter)
end

"""
    DyadCovariateEffect <: NetworkEffect

Dyadic covariate effect: ``s_i = \\sum_j x_{ij} w_{ij}``. RSiena: X

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_covariate!(data, ConstantDyadCovariate(:distance, [abs(i - j) / 50 for i in 1:50, j in 1:50]))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(DyadCovariateEffect(:friendship, :distance), state, data)
```
"""
struct DyadCovariateEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::DyadCovariateEffect) = :X
effect_type(::DyadCovariateEffect) = :eval
target_variable(e::DyadCovariateEffect) = e.variable
interaction_with(e::DyadCovariateEffect) = e.covariate

function evaluate_actor(e::DyadCovariateEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    total = 0.0
    for j in net.outneighbors[actor]     # one-mode alters or two-mode events
        total += _get_dyad_covariate_value(cov, actor, j, state.period)
    end
    return total
end

function compute_contribution(e::DyadCovariateEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _get_dyad_covariate_value(data.covariates[e.covariate], actor, alter, state.period)
end

"""
    SameXRecipEffect <: NetworkEffect

Same covariate × reciprocity: ``s_i = \\sum_j x_{ij} x_{ji} I(v_i = v_j)``.
RSiena: sameXRecip

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(SameXRecipEffect(:friendship, :smoke1), state, data)
compute_statistic(SameXRecipEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct SameXRecipEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SameXRecipEffect) = :sameXRecip
effect_type(::SameXRecipEffect) = :eval
target_variable(e::SameXRecipEffect) = e.variable
interaction_with(e::SameXRecipEffect) = e.covariate

function evaluate_actor(e::SameXRecipEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    v1 = _attr_value(state, data, e.covariate, actor)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0 || net[j, actor] == 0) && continue
        v1 == _attr_value(state, data, e.covariate, j) && (count += 1)
    end
    return Float64(count)
end

function compute_contribution(e::SameXRecipEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    net[alter, actor] == 0 && return 0.0
    v1 = _attr_value(state, data, e.covariate, actor)
    v2 = _attr_value(state, data, e.covariate, alter)
    return v1 == v2 ? 1.0 : 0.0
end

"""
    SimXRecipEffect <: NetworkEffect

Centered similarity × reciprocity:
``s_i = \\sum_j x_{ij} x_{ji} (\\text{sim}_{ij} - \\widehat{sim})``. RSiena: `simRecipX`
(the short name was `simXRecip` before 0.2, which is not an RSiena name).

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(SimXRecipEffect(:friendship, :smoke1), state, data)
compute_statistic(SimXRecipEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct SimXRecipEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SimXRecipEffect) = :simRecipX
effect_type(::SimXRecipEffect) = :eval
target_variable(e::SimXRecipEffect) = e.variable
interaction_with(e::SimXRecipEffect) = e.covariate

function evaluate_actor(e::SimXRecipEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0 || net[j, actor] == 0) && continue
        total += _attr_similarity(state, data, e.covariate, actor, j)
    end
    return total
end

function compute_contribution(e::SimXRecipEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    net[alter, actor] == 0 && return 0.0
    return _attr_similarity(state, data, e.covariate, actor, alter)
end

"""
    SimXTransTripEffect <: NetworkEffect

Centered similarity × transitive triplets:
``s_i = \\sum_j x_{ij} (\\text{sim}_{ij} - \\widehat{sim}) \\, \\#\\{h : x_{ih} x_{hj} = 1\\}``.
RSiena: simXTransTrip

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(SimXTransTripEffect(:friendship, :smoke1), state, data)
compute_statistic(SimXTransTripEffect(:friendship, :alcohol), state, data)  # a co-evolving behaviour
```
"""
struct SimXTransTripEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SimXTransTripEffect) = :simXTransTrip
effect_type(::SimXTransTripEffect) = :eval
target_variable(e::SimXTransTripEffect) = e.variable
interaction_with(e::SimXTransTripEffect) = e.covariate

# RSiena's change statistic (`SimilarityTransitiveTripletsEffect.cpp`): the tie's
# own term sim(i, j) * #{two-paths i -> h -> j}, plus sim(i, h) for every h with
# i -> h and h -> j. (This is RSiena's definition; it is not the plain toggle
# difference of the statistic, which would use j -> h in the second term.)
function compute_contribution(e::SimXTransTripEffect, state::NetworkState,
                              data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    tp = 0
    extra = 0.0
    for h in net.outneighbors[actor]
        (h == actor || h == alter) && continue
        if net[h, alter] == 1
            tp += 1
            extra += _attr_similarity(state, data, e.covariate, actor, h)
        end
    end
    return _attr_similarity(state, data, e.covariate, actor, alter) * tp + extra
end

function evaluate_actor(e::SimXTransTripEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        tp = 0
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[h, j] == 1
                tp += 1
            end
        end
        tp > 0 && (total += _attr_similarity(state, data, e.covariate, actor, j) * tp)
    end
    return total
end

#==============================================================================#
# Endowment/Creation Effects
#==============================================================================#

"""
    EndowmentEffect <: NetworkEffect

Wrapper for an endowment effect: the wrapped effect's change statistic enters the
objective function only for tie dissolution (handled in `compute_objective`).
Not yet supported in Method-of-Moments estimation.

# Example
```julia
using Siena
e = EndowmentEffect(ReciprocityEffect(:friendship))
effect_type(e), effect_name(e)
```
"""
struct EndowmentEffect{E<:NetworkEffect} <: NetworkEffect
    base_effect::E
end

effect_name(e::EndowmentEffect) = Symbol(string(effect_name(e.base_effect)), "Endow")
effect_type(::EndowmentEffect) = :endow
target_variable(e::EndowmentEffect) = target_variable(e.base_effect)
interaction_with(e::EndowmentEffect) = interaction_with(e.base_effect)

function compute_contribution(e::EndowmentEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return compute_contribution(e.base_effect, state, data, actor, alter)
end

function compute_statistic(e::EndowmentEffect, state::NetworkState, data::SienaData)
    throw(ArgumentError("endowment effects are not yet supported in Method-of-Moments " *
                        "estimation (effect $(effect_name(e)))"))
end

"""
    CreationEffect <: NetworkEffect

Wrapper for a creation effect: the wrapped effect's change statistic enters the
objective function only for tie creation (handled in `compute_objective`).
Not yet supported in Method-of-Moments estimation.

# Example
```julia
using Siena
e = CreationEffect(ReciprocityEffect(:friendship))
effect_type(e), effect_name(e)
```
"""
struct CreationEffect{E<:NetworkEffect} <: NetworkEffect
    base_effect::E
end

effect_name(e::CreationEffect) = Symbol(string(effect_name(e.base_effect)), "Create")
effect_type(::CreationEffect) = :creation
target_variable(e::CreationEffect) = target_variable(e.base_effect)
interaction_with(e::CreationEffect) = interaction_with(e.base_effect)

function compute_contribution(e::CreationEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return compute_contribution(e.base_effect, state, data, actor, alter)
end

function compute_statistic(e::CreationEffect, state::NetworkState, data::SienaData)
    throw(ArgumentError("creation effects are not yet supported in Method-of-Moments " *
                        "estimation (effect $(effect_name(e)))"))
end

#==============================================================================#
# Multiplex Effects
#==============================================================================#

"""
    CrossNetworkReciprocityEffect <: NetworkEffect

Reciprocity from another network: ``s_i = \\sum_j x_{ij} z_{ji}``. RSiena: crprodRecip

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
fr = [NetworkCore.as_matrix(w) for w in s50.friendship]
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, fr))
add_dependent!(data, DependentNetwork(:advice, [fr[2], fr[3], fr[1]]))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(CrossNetworkReciprocityEffect(:friendship, :advice), state, data)
```
"""
struct CrossNetworkReciprocityEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkReciprocityEffect) = :crprodRecip
effect_type(::CrossNetworkReciprocityEffect) = :eval
target_variable(e::CrossNetworkReciprocityEffect) = e.variable
interaction_with(e::CrossNetworkReciprocityEffect) = e.other_network

function evaluate_actor(e::CrossNetworkReciprocityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    other = state.networks[e.other_network]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        count += other[j, actor]
    end
    return Float64(count)
end

function compute_contribution(e::CrossNetworkReciprocityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(state.networks[e.other_network][alter, actor])
end

"""
    CrossNetworkActivityEffect <: NetworkEffect

Ego's outdegree in another network: ``s_i = \\sum_j x_{ij} z_{i+}``. RSiena: crprodAct

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
fr = [NetworkCore.as_matrix(w) for w in s50.friendship]
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, fr))
add_dependent!(data, DependentNetwork(:advice, [fr[2], fr[3], fr[1]]))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(CrossNetworkActivityEffect(:friendship, :advice), state, data)
```
"""
struct CrossNetworkActivityEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkActivityEffect) = :crprodAct
effect_type(::CrossNetworkActivityEffect) = :eval
target_variable(e::CrossNetworkActivityEffect) = e.variable
interaction_with(e::CrossNetworkActivityEffect) = e.other_network

function evaluate_actor(e::CrossNetworkActivityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    other = state.networks[e.other_network]
    return Float64(_row_sum(net, actor) * _row_sum(other, actor))
end

function compute_contribution(e::CrossNetworkActivityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(_row_sum(state.networks[e.other_network], actor))
end

"""
    CrossNetworkPopularityEffect <: NetworkEffect

Alter's indegree in another network: ``s_i = \\sum_j x_{ij} z_{+j}``. RSiena: crprodPop

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
fr = [NetworkCore.as_matrix(w) for w in s50.friendship]
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, fr))
add_dependent!(data, DependentNetwork(:advice, [fr[2], fr[3], fr[1]]))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(CrossNetworkPopularityEffect(:friendship, :advice), state, data)
```
"""
struct CrossNetworkPopularityEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkPopularityEffect) = :crprodPop
effect_type(::CrossNetworkPopularityEffect) = :eval
target_variable(e::CrossNetworkPopularityEffect) = e.variable
interaction_with(e::CrossNetworkPopularityEffect) = e.other_network

function evaluate_actor(e::CrossNetworkPopularityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    other = state.networks[e.other_network]
    n = size(net, 1)
    total = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _col_sum(other, j)
    end
    return Float64(total)
end

function compute_contribution(e::CrossNetworkPopularityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(_col_sum(state.networks[e.other_network], alter))
end

"""
    CrossNetworkTiesEffect <: NetworkEffect

Tie in another network: ``s_i = \\sum_j x_{ij} z_{ij}``. RSiena: crprod

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
fr = [NetworkCore.as_matrix(w) for w in s50.friendship]
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, fr))
add_dependent!(data, DependentNetwork(:advice, [fr[2], fr[3], fr[1]]))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(CrossNetworkTiesEffect(:friendship, :advice), state, data)
```
"""
struct CrossNetworkTiesEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkTiesEffect) = :crprod
effect_type(::CrossNetworkTiesEffect) = :eval
target_variable(e::CrossNetworkTiesEffect) = e.variable
interaction_with(e::CrossNetworkTiesEffect) = e.other_network

function evaluate_actor(e::CrossNetworkTiesEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    other = state.networks[e.other_network]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        count += other[actor, j]
    end
    return Float64(count)
end

function compute_contribution(e::CrossNetworkTiesEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(state.networks[e.other_network][actor, alter])
end
