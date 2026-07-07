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
    AverageAttHigherEffect <: BehaviorEffect

Proportion of alters with strictly higher behavior:
``s_i = x_{i+}^{-1} \\#\\{j : x_{ij} = 1, z_j > z_i\\}``.
(Simplified relative to RSiena's avAttHigher.)
"""
struct AverageAttHigherEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageAttHigherEffect) = :avAttHigher
effect_type(::AverageAttHigherEffect) = :eval
target_variable(e::AverageAttHigherEffect) = e.variable
interaction_with(e::AverageAttHigherEffect) = e.network

function evaluate_actor(e::AverageAttHigherEffect, state::NetworkState,
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
    AverageAttLowerEffect <: BehaviorEffect

Proportion of alters with strictly lower behavior:
``s_i = x_{i+}^{-1} \\#\\{j : x_{ij} = 1, z_j < z_i\\}``.
(Simplified relative to RSiena's avAttLower.)
"""
struct AverageAttLowerEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageAttLowerEffect) = :avAttLower
effect_type(::AverageAttLowerEffect) = :eval
target_variable(e::AverageAttLowerEffect) = e.variable
interaction_with(e::AverageAttLowerEffect) = e.network

function evaluate_actor(e::AverageAttLowerEffect, state::NetworkState,
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
"""
struct TotalSimilarityEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::TotalSimilarityEffect) = :totSim
effect_type(::TotalSimilarityEffect) = :eval
target_variable(e::TotalSimilarityEffect) = e.variable
interaction_with(e::TotalSimilarityEffect) = e.network

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

Average alter at distance 2:
``s_i = \\tilde z_i \\cdot \\text{mean}\\{\\tilde z_k : k \\text{ at distance 2}\\}``.
RSiena: avAltDist2
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
    count = 0
    for k in 1:n
        (k == actor || net[actor, k] == 1) && continue
        at_dist2 = false
        for j in 1:n
            (j == actor || j == k) && continue
            if net[actor, j] == 1 && net[j, k] == 1
                at_dist2 = true
                break
            end
        end
        if at_dist2
            total += _centered_beh(dep, beh[k])
            count += 1
        end
    end
    count == 0 && return 0.0
    return _centered_beh(dep, beh[actor]) * total / count
end

#==============================================================================#
# Degree Effects on Behavior
#==============================================================================#

"""
    IndegreeEffect <: BehaviorEffect

Indegree effect on behavior: ``s_i = \\tilde z_i x_{+i}``. RSiena: indeg
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
``s_i = \\prod_j (1 - |z_i - z_j|/r_Z)^{x_{ij}}`` (0 for isolates).
"""
struct FeedbackEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::FeedbackEffect) = :feedback
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
