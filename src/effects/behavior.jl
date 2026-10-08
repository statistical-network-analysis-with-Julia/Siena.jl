"""
Behavior effects for the SAOM evaluation function.

Each effect implements `evaluate_actor`, the actor's evaluation-function component
``s_{ki}(x, z)``. Behavior values are centered on the overall observed mean
(`dep.mean_val`) and similarity scores are centered on the observed mean similarity
(`dep.sim_mean`), following RSiena. Change statistics are obtained through the generic
difference fallback in `effects/base.jl` (``s_{ki}(z_i + d) - s_{ki}(z_i)``), which
makes the no-change option's change statistic exactly 0.
"""

#==============================================================================#
# Helpers
#==============================================================================#

function _get_beh_covariate_value(cov::AbstractCovariate, actor::Int, wave::Int)
    return _get_covariate_value(cov, actor, wave)
end

_behavior_dep(data::SienaData, name::Symbol) = data.dependents[name]::DependentBehavior

# Centered behavior value
_centered_beh(dep::DependentBehavior, z::Int) = z - dep.mean_val

_behavior_range(dep::DependentBehavior) = Float64(dep.max_val - dep.min_val)

# Centered similarity between two behavior values (RSiena's sim_ij - ^sim)
function _centered_beh_similarity(dep::DependentBehavior, zi::Int, zj::Int)
    r = _behavior_range(dep)
    sim = r > 0 ? 1.0 - abs(zi - zj) / r : 1.0
    return sim - dep.sim_mean
end

#==============================================================================#
# Basic Shape Effects
#==============================================================================#

"""
    LinearShapeEffect <: BehaviorEffect

Linear shape: ``s_i = \\tilde z_i``. RSiena: linear

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(LinearShapeEffect(:alcohol), state, data)
```
"""
struct LinearShapeEffect <: BehaviorEffect
    variable::Symbol
end

effect_name(::LinearShapeEffect) = :linear
effect_type(::LinearShapeEffect) = :eval
target_variable(e::LinearShapeEffect) = e.variable

function evaluate_actor(e::LinearShapeEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    dep = _behavior_dep(data, e.variable)
    return _centered_beh(dep, state.behaviors[e.variable][actor])
end

"""
    QuadraticShapeEffect <: BehaviorEffect

Quadratic shape: ``s_i = \\tilde z_i^2``. RSiena: quad

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(QuadraticShapeEffect(:alcohol), state, data)
```
"""
struct QuadraticShapeEffect <: BehaviorEffect
    variable::Symbol
end

effect_name(::QuadraticShapeEffect) = :quad
effect_type(::QuadraticShapeEffect) = :eval
target_variable(e::QuadraticShapeEffect) = e.variable

function evaluate_actor(e::QuadraticShapeEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    dep = _behavior_dep(data, e.variable)
    return _centered_beh(dep, state.behaviors[e.variable][actor])^2
end

"""
    CubicShapeEffect <: BehaviorEffect

Cubic shape: ``s_i = \\tilde z_i^3``.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(CubicShapeEffect(:alcohol), state, data)
```
"""
struct CubicShapeEffect <: BehaviorEffect
    variable::Symbol
end

effect_name(::CubicShapeEffect) = :cubic
effect_type(::CubicShapeEffect) = :eval
target_variable(e::CubicShapeEffect) = e.variable

function evaluate_actor(e::CubicShapeEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    dep = _behavior_dep(data, e.variable)
    return _centered_beh(dep, state.behaviors[e.variable][actor])^3
end

#==============================================================================#
# Network Influence Effects - Average-based
#==============================================================================#

"""
    AverageAlterEffect <: BehaviorEffect

Average alter: ``s_i = \\tilde z_i \\cdot (\\sum_j x_{ij} \\tilde z_j) / x_{i+}``
(0 for actors without outgoing ties). RSiena: avAlt

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(AverageAlterEffect(:alcohol, :friendship), state, data)
```
"""
struct AverageAlterEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageAlterEffect) = :avAlt
effect_type(::AverageAlterEffect) = :eval
target_variable(e::AverageAlterEffect) = e.variable
interaction_with(e::AverageAlterEffect) = e.network

function evaluate_actor(e::AverageAlterEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    n = size(net, 1)
    total = 0.0
    outdeg = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _centered_beh(dep, beh[j])
        outdeg += 1
    end
    outdeg == 0 && return 0.0
    return _centered_beh(dep, beh[actor]) * total / outdeg
end

"""
    AverageSimilarityEffect <: BehaviorEffect

Average similarity:
``s_i = x_{i+}^{-1} \\sum_j x_{ij} (\\text{sim}_{ij} - \\widehat{sim})``. RSiena: avSim

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(AverageSimilarityEffect(:alcohol, :friendship), state, data)
```
"""
struct AverageSimilarityEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageSimilarityEffect) = :avSim
effect_type(::AverageSimilarityEffect) = :eval
target_variable(e::AverageSimilarityEffect) = e.variable
interaction_with(e::AverageSimilarityEffect) = e.network

function evaluate_actor(e::AverageSimilarityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    n = size(net, 1)
    total = 0.0
    outdeg = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _centered_beh_similarity(dep, beh[actor], beh[j])
        outdeg += 1
    end
    outdeg == 0 && return 0.0
    return total / outdeg
end

"""
    AverageInAlterEffect <: BehaviorEffect

Average in-alter: like avAlt over incoming ties. RSiena: avInAlt

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(AverageInAlterEffect(:alcohol, :friendship), state, data)
```
"""
struct AverageInAlterEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageInAlterEffect) = :avInAlt
effect_type(::AverageInAlterEffect) = :eval
target_variable(e::AverageInAlterEffect) = e.variable
interaction_with(e::AverageInAlterEffect) = e.network

function evaluate_actor(e::AverageInAlterEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    n = size(net, 1)
    total = 0.0
    indeg = 0
    for j in 1:n
        (j == actor || net[j, actor] == 0) && continue
        total += _centered_beh(dep, beh[j])
        indeg += 1
    end
    indeg == 0 && return 0.0
    return _centered_beh(dep, beh[actor]) * total / indeg
end

"""
    AverageRecipAlterEffect <: BehaviorEffect

Average reciprocal alter: like avAlt over mutual ties. RSiena: avRecAlt

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(AverageRecipAlterEffect(:alcohol, :friendship), state, data)
```
"""
struct AverageRecipAlterEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageRecipAlterEffect) = :avRecAlt
effect_type(::AverageRecipAlterEffect) = :eval
target_variable(e::AverageRecipAlterEffect) = e.variable
interaction_with(e::AverageRecipAlterEffect) = e.network

function evaluate_actor(e::AverageRecipAlterEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    n = size(net, 1)
    total = 0.0
    deg = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0 || net[j, actor] == 0) && continue
        total += _centered_beh(dep, beh[j])
        deg += 1
    end
    deg == 0 && return 0.0
    return _centered_beh(dep, beh[actor]) * total / deg
end

"""
    AverageAttHigherSimpleEffect <: BehaviorEffect

Proportion of alters with strictly higher behavior:
``s_i = x_{i+}^{-1} \\#\\{j : x_{ij} = 1, z_j > z_i\\}``.

!!! warning "Not RSiena's avAttHigher"
    This is a *simplified* attraction-to-higher-alters effect, not RSiena's
    `avAttHigher`, whose statistic is a different function of the behavior
    differences. Its short name is therefore `:avAttHigherSimple`: the RSiena short
    name `:avAttHigher` is reserved for a numerically equivalent implementation and
    is deliberately not defined. Estimates from this effect are not comparable with
    RSiena's `avAttHigher` estimates.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(AverageAttHigherSimpleEffect(:alcohol, :friendship), state, data)
```
"""
struct AverageAttHigherSimpleEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageAttHigherSimpleEffect) = :avAttHigherSimple
effect_type(::AverageAttHigherSimpleEffect) = :eval
target_variable(e::AverageAttHigherSimpleEffect) = e.variable
interaction_with(e::AverageAttHigherSimpleEffect) = e.network

function evaluate_actor(e::AverageAttHigherSimpleEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    n = size(net, 1)
    count = 0
    outdeg = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        beh[j] > beh[actor] && (count += 1)
        outdeg += 1
    end
    outdeg == 0 && return 0.0
    return count / outdeg
end

"""
    AverageAttLowerSimpleEffect <: BehaviorEffect

Proportion of alters with strictly lower behavior:
``s_i = x_{i+}^{-1} \\#\\{j : x_{ij} = 1, z_j < z_i\\}``.

!!! warning "Not RSiena's avAttLower"
    This is a *simplified* attraction-to-lower-alters effect, not RSiena's
    `avAttLower`, whose statistic is a different function of the behavior
    differences. Its short name is therefore `:avAttLowerSimple`: the RSiena short
    name `:avAttLower` is reserved for a numerically equivalent implementation and is
    deliberately not defined. Estimates from this effect are not comparable with
    RSiena's `avAttLower` estimates.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(AverageAttLowerSimpleEffect(:alcohol, :friendship), state, data)
```
"""
struct AverageAttLowerSimpleEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageAttLowerSimpleEffect) = :avAttLowerSimple
effect_type(::AverageAttLowerSimpleEffect) = :eval
target_variable(e::AverageAttLowerSimpleEffect) = e.variable
interaction_with(e::AverageAttLowerSimpleEffect) = e.network

function evaluate_actor(e::AverageAttLowerSimpleEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    n = size(net, 1)
    count = 0
    outdeg = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        beh[j] < beh[actor] && (count += 1)
        outdeg += 1
    end
    outdeg == 0 && return 0.0
    return count / outdeg
end

#==============================================================================#
# Network Influence Effects - Total-based
#==============================================================================#

"""
    TotalAlterEffect <: BehaviorEffect

Total alter: ``s_i = \\tilde z_i \\sum_j x_{ij} \\tilde z_j``. RSiena: totAlt

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(TotalAlterEffect(:alcohol, :friendship), state, data)
```
"""
struct TotalAlterEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::TotalAlterEffect) = :totAlt
effect_type(::TotalAlterEffect) = :eval
target_variable(e::TotalAlterEffect) = e.variable
interaction_with(e::TotalAlterEffect) = e.network

function evaluate_actor(e::TotalAlterEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _centered_beh(dep, beh[j])
    end
    return _centered_beh(dep, beh[actor]) * total
end

"""
    TotalSimilarityEffect <: BehaviorEffect

Total similarity: ``s_i = \\sum_j x_{ij} (\\text{sim}_{ij} - \\widehat{sim})``.
RSiena: totSim

!!! note "RSiena's change statistic"
    As in RSiena 1.6.6, the change statistic of a behaviour step is the change of
    the summed similarities **minus** ``x_{i+}\\,\\widehat{sim}``, in either
    direction — not the plain difference of the statistic above. A positive `totSim`
    parameter therefore also makes any behaviour change less likely for actors with
    out-ties. Before 0.2 Siena.jl used the plain difference, which gives different
    estimates from RSiena under the same name; the RSiena manual's advice to include
    `outdeg` alongside `totSim` applies.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(TotalSimilarityEffect(:alcohol, :friendship), state, data)
```
"""
struct TotalSimilarityEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::TotalSimilarityEffect) = :totSim
effect_type(::TotalSimilarityEffect) = :eval
target_variable(e::TotalSimilarityEffect) = e.variable
interaction_with(e::TotalSimilarityEffect) = e.network

# RSiena's change statistic (`SimilarityEffect::calculateChangeContribution` with
# `average = false`): the change of the summed similarities MINUS
# ``x_{i+}\,\widehat{sim}`` -- for a step in either direction. This is not the
# difference of the (centred) statistic, in which the centring constant cancels; it
# is what RSiena 1.6.6 simulates under the name `totSim`, and it makes every
# behaviour step of an actor with out-ties less attractive.
function compute_contribution(e::TotalSimilarityEffect, state::NetworkState,
                              data::SienaData, actor::Int, direction::Int)
    direction == 0 && return 0.0
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    r = _behavior_range(dep)
    outdeg = _row_sum(net, actor)
    outdeg == 0 && return 0.0
    zi = beh[actor]
    change = 0.0
    if r > 0
        for j in net.outneighbors[actor]
            j == actor && continue
            change += (abs(zi - beh[j]) - abs(zi + direction - beh[j])) / r
        end
    end
    return change - outdeg * dep.sim_mean
end

function evaluate_actor(e::TotalSimilarityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _centered_beh_similarity(dep, beh[actor], beh[j])
    end
    return total
end

"""
    TotalInAlterEffect <: BehaviorEffect

Total in-alter: ``s_i = \\tilde z_i \\sum_j x_{ji} \\tilde z_j``. RSiena: totInAlt

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(TotalInAlterEffect(:alcohol, :friendship), state, data)
```
"""
struct TotalInAlterEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::TotalInAlterEffect) = :totInAlt
effect_type(::TotalInAlterEffect) = :eval
target_variable(e::TotalInAlterEffect) = e.variable
interaction_with(e::TotalInAlterEffect) = e.network

function evaluate_actor(e::TotalInAlterEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[j, actor] == 0) && continue
        total += _centered_beh(dep, beh[j])
    end
    return _centered_beh(dep, beh[actor]) * total
end

#==============================================================================#
# Distance-2 Influence Effects
#==============================================================================#

"""
    AverageAlterDist2Effect <: BehaviorEffect

Average alter at distance 2, RSiena's `avAltDist2`: the average over ego's alters of
each alter's average alter, excluding ego,

``s_i = \\tilde z_i \\, \\frac{1}{x_{i+}} \\sum_j x_{ij}
\\frac{\\sum_{h \\ne i} x_{jh} \\tilde z_h}{x_{j+} - x_{ji}}``

(an alter whose only out-tie is to ego contributes 0; ``s_i = 0`` for an actor
without out-ties), with ``\\tilde z`` the centred behaviour. Before 0.2 the effect
averaged over the actors at geodesic distance exactly 2, a different statistic.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(AverageAlterDist2Effect(:alcohol, :friendship), state, data)
```
"""
struct AverageAlterDist2Effect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageAlterDist2Effect) = :avAltDist2
effect_type(::AverageAlterDist2Effect) = :eval
target_variable(e::AverageAlterDist2Effect) = e.variable
interaction_with(e::AverageAlterDist2Effect) = e.network

function evaluate_actor(e::AverageAlterDist2Effect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    n = size(net, 1)

    total = 0.0
    n_alters = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        n_alters += 1
        alter_sum = 0.0
        for h in 1:n
            (h == j || h == actor || net[j, h] == 0) && continue
            alter_sum += _centered_beh(dep, beh[h])
        end
        denom = _row_sum(net, j) - net[j, actor]
        total += denom > 0 ? alter_sum / denom : alter_sum
    end
    n_alters == 0 && return 0.0
    return _centered_beh(dep, beh[actor]) * total / n_alters
end

#==============================================================================#
# Degree Effects on Behavior
#==============================================================================#

"""
    IndegreeEffect <: BehaviorEffect

Indegree effect on behavior: ``s_i = \\tilde z_i x_{+i}``. RSiena: indeg

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(IndegreeEffect(:alcohol, :friendship), state, data)
```
"""
struct IndegreeEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::IndegreeEffect) = :indeg
effect_type(::IndegreeEffect) = :eval
target_variable(e::IndegreeEffect) = e.variable
interaction_with(e::IndegreeEffect) = e.network

function evaluate_actor(e::IndegreeEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    z = _centered_beh(dep, state.behaviors[e.variable][actor])
    return z * _col_sum(net, actor)
end

"""
    BehaviorOutdegreeEffect <: BehaviorEffect

Outdegree effect on behavior: ``s_i = \\tilde z_i x_{i+}``. RSiena: outdeg

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(BehaviorOutdegreeEffect(:alcohol, :friendship), state, data)
```
"""
struct BehaviorOutdegreeEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::BehaviorOutdegreeEffect) = :outdeg
effect_type(::BehaviorOutdegreeEffect) = :eval
target_variable(e::BehaviorOutdegreeEffect) = e.variable
interaction_with(e::BehaviorOutdegreeEffect) = e.network

function evaluate_actor(e::BehaviorOutdegreeEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    z = _centered_beh(dep, state.behaviors[e.variable][actor])
    return z * _row_sum(net, actor)
end

"""
    RecipDegreeEffect <: BehaviorEffect

Reciprocal degree effect: ``s_i = \\tilde z_i \\#\\{j : x_{ij} x_{ji} = 1\\}``.
RSiena: recipDeg

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(RecipDegreeEffect(:alcohol, :friendship), state, data)
```
"""
struct RecipDegreeEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::RecipDegreeEffect) = :recipDeg
effect_type(::RecipDegreeEffect) = :eval
target_variable(e::RecipDegreeEffect) = e.variable
interaction_with(e::RecipDegreeEffect) = e.network

function evaluate_actor(e::RecipDegreeEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    n = size(net, 1)
    recip = 0
    for j in 1:n
        j == actor && continue
        recip += net[actor, j] * net[j, actor]
    end
    z = _centered_beh(dep, state.behaviors[e.variable][actor])
    return z * recip
end

#==============================================================================#
# Covariate Effects on Behavior
#==============================================================================#

"""
    BehaviorCovariateEffect <: BehaviorEffect

Effect from covariate: ``s_i = \\tilde z_i v_i``. RSiena: effFrom

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(BehaviorCovariateEffect(:alcohol, :smoke1), state, data)
```
"""
struct BehaviorCovariateEffect <: BehaviorEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::BehaviorCovariateEffect) = :effFrom
effect_type(::BehaviorCovariateEffect) = :eval
target_variable(e::BehaviorCovariateEffect) = e.variable
interaction_with(e::BehaviorCovariateEffect) = e.covariate

function evaluate_actor(e::BehaviorCovariateEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    dep = _behavior_dep(data, e.variable)
    z = _centered_beh(dep, state.behaviors[e.variable][actor])
    return z * _get_beh_covariate_value(data.covariates[e.covariate], actor, state.period)
end

"""
    CovariateInteractionEffect <: BehaviorEffect

Covariate × quadratic behavior interaction: ``s_i = \\tilde z_i^2 v_i``.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(CovariateInteractionEffect(:alcohol, :smoke1), state, data)
```
"""
struct CovariateInteractionEffect <: BehaviorEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::CovariateInteractionEffect) = :covInt
effect_type(::CovariateInteractionEffect) = :eval
target_variable(e::CovariateInteractionEffect) = e.variable
interaction_with(e::CovariateInteractionEffect) = e.covariate

function evaluate_actor(e::CovariateInteractionEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    dep = _behavior_dep(data, e.variable)
    z = _centered_beh(dep, state.behaviors[e.variable][actor])
    return z^2 * _get_beh_covariate_value(data.covariates[e.covariate], actor, state.period)
end

#==============================================================================#
# Behavior-Behavior Effects
#==============================================================================#

"""
    BehaviorInteractionEffect <: BehaviorEffect

Effect of one behavior on another: ``s_i = \\tilde z_i \\tilde w_i``. RSiena: behBeh

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_dependent!(data, DependentBehavior(:smoke, [s50.smoke[:, w] for w in 1:3]))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(BehaviorInteractionEffect(:alcohol, :smoke), state, data)
```
"""
struct BehaviorInteractionEffect <: BehaviorEffect
    variable::Symbol
    other_behavior::Symbol
end

effect_name(::BehaviorInteractionEffect) = :behBeh
effect_type(::BehaviorInteractionEffect) = :eval
target_variable(e::BehaviorInteractionEffect) = e.variable
interaction_with(e::BehaviorInteractionEffect) = e.other_behavior

function evaluate_actor(e::BehaviorInteractionEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    dep = _behavior_dep(data, e.variable)
    other_dep = _behavior_dep(data, e.other_behavior)
    z = _centered_beh(dep, state.behaviors[e.variable][actor])
    w = _centered_beh(other_dep, state.behaviors[e.other_behavior][actor])
    return z * w
end

"""
    BehaviorSimilarityEffect <: BehaviorEffect

Similarity in another behavior:
``s_i = \\tilde z_i \\cdot x_{i+}^{-1} \\sum_j x_{ij} \\text{sim}^w_{ij}`` where
``\\text{sim}^w`` is similarity on the other behavior. RSiena: simBeh

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_dependent!(data, DependentBehavior(:smoke, [s50.smoke[:, w] for w in 1:3]))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(BehaviorSimilarityEffect(:alcohol, :smoke, :friendship), state, data)
```
"""
struct BehaviorSimilarityEffect <: BehaviorEffect
    variable::Symbol
    other_behavior::Symbol
    network::Symbol
end

effect_name(::BehaviorSimilarityEffect) = :simBeh
effect_type(::BehaviorSimilarityEffect) = :eval
target_variable(e::BehaviorSimilarityEffect) = e.variable
interaction_with(e::BehaviorSimilarityEffect) = e.other_behavior

function evaluate_actor(e::BehaviorSimilarityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    other = state.behaviors[e.other_behavior]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    other_dep = _behavior_dep(data, e.other_behavior)
    n = size(net, 1)

    total = 0.0
    outdeg = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _centered_beh_similarity(other_dep, other[actor], other[j])
        outdeg += 1
    end
    outdeg == 0 && return 0.0
    return _centered_beh(dep, beh[actor]) * total / outdeg
end

#==============================================================================#
# Threshold Effects
#==============================================================================#

"""
    ThresholdEffect <: BehaviorEffect

Threshold effect: ``s_i = I(z_i \\ge c)``.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(ThresholdEffect(:alcohol, 3), state, data)
```
"""
struct ThresholdEffect <: BehaviorEffect
    variable::Symbol
    threshold::Int
end

effect_name(::ThresholdEffect) = :threshold
effect_type(::ThresholdEffect) = :eval
target_variable(e::ThresholdEffect) = e.variable

function evaluate_actor(e::ThresholdEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    return state.behaviors[e.variable][actor] >= e.threshold ? 1.0 : 0.0
end

"""
    PropThresholdEffect <: BehaviorEffect

Proportional threshold: ``s_i = \\tilde z_i`` if the proportion of alters at the
behavior maximum is at least the threshold, else 0.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(PropThresholdEffect(:alcohol, :friendship, 0.5), state, data)
```
"""
struct PropThresholdEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
    threshold::Float64
end

effect_name(::PropThresholdEffect) = :propThreshold
effect_type(::PropThresholdEffect) = :eval
target_variable(e::PropThresholdEffect) = e.variable
interaction_with(e::PropThresholdEffect) = e.network

function evaluate_actor(e::PropThresholdEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    n = size(net, 1)

    n_above = 0
    outdeg = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        beh[j] >= dep.max_val && (n_above += 1)
        outdeg += 1
    end
    outdeg == 0 && return 0.0
    prop = n_above / outdeg
    return prop >= e.threshold ? _centered_beh(dep, beh[actor]) : 0.0
end

#==============================================================================#
# Isolate Effects
#==============================================================================#

"""
    BehaviorIsolateEffect <: BehaviorEffect

Isolate effect on behavior: ``s_i = \\tilde z_i I(x_{i+} = x_{+i} = 0)``.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(BehaviorIsolateEffect(:alcohol, :friendship), state, data)
```
"""
struct BehaviorIsolateEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::BehaviorIsolateEffect) = :behIsolate
effect_type(::BehaviorIsolateEffect) = :eval
target_variable(e::BehaviorIsolateEffect) = e.variable
interaction_with(e::BehaviorIsolateEffect) = e.network

function evaluate_actor(e::BehaviorIsolateEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    isolate = _row_sum(net, actor) == 0 && _col_sum(net, actor) == 0
    return isolate ? _centered_beh(dep, state.behaviors[e.variable][actor]) : 0.0
end

#==============================================================================#
# Feedback Effects
#==============================================================================#

"""
    FeedbackEffect <: BehaviorEffect

Product of (uncentered) similarities with all alters:
``s_i = \\prod_j (1 - |z_i - z_j|/r_Z)^{x_{ij}}`` (0 for isolates). A Siena.jl
effect with no RSiena counterpart; its short name is `:simProd` (it was `:feedback`
before 0.2, which in RSiena names an unrelated effect for continuous behaviour).

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(FeedbackEffect(:alcohol, :friendship), state, data)
```
"""
struct FeedbackEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::FeedbackEffect) = :simProd
effect_type(::FeedbackEffect) = :eval
target_variable(e::FeedbackEffect) = e.variable
interaction_with(e::FeedbackEffect) = e.network

function evaluate_actor(e::FeedbackEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = _behavior_dep(data, e.variable)
    n = size(net, 1)
    r = _behavior_range(dep)
    r == 0 && return 0.0

    prod_sim = 1.0
    outdeg = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        prod_sim *= 1.0 - abs(beh[actor] - beh[j]) / r
        outdeg += 1
    end
    return outdeg == 0 ? 0.0 : prod_sim
end

#==============================================================================#
# Main Effect (for compatibility)
#==============================================================================#

"""
    MainBehaviorEffect <: BehaviorEffect

Main effect (constant tendency): ``s_i = \\tilde z_i`` (identical to
[`LinearShapeEffect`](@ref); kept for compatibility).

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
state = initialize!(NetworkState(), data, 2; period=1)   # observed wave 2
compute_statistic(MainBehaviorEffect(:alcohol), state, data)
```
"""
struct MainBehaviorEffect <: BehaviorEffect
    variable::Symbol
end

effect_name(::MainBehaviorEffect) = :main
effect_type(::MainBehaviorEffect) = :eval
target_variable(e::MainBehaviorEffect) = e.variable

function evaluate_actor(e::MainBehaviorEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    dep = _behavior_dep(data, e.variable)
    return _centered_beh(dep, state.behaviors[e.variable][actor])
end
