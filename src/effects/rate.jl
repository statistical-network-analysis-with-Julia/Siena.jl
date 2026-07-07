"""
Rate effects for SAOM.

The SAOM rate function is multiplicative:
``\\lambda_i = \\rho_m \\prod_k \\exp(\\alpha_k r_{ki}(x))``
where ``\\rho_m`` is the basic rate parameter for the period and each non-basic rate
effect contributes a factor ``\\exp(\\alpha_k r_{ki})``. Every rate effect implements
`rate_score` returning the score ``r_{ki}``; the basic rate effect has score 1 and its
parameter is the rate ``\\rho_m`` itself (not its logarithm).
"""

#==============================================================================#
# Basic Rate Effects
#==============================================================================#

"""
    BasicRateEffect <: RateEffect

Basic rate parameter for a period. The associated parameter is the rate ``\\rho_m``
itself (a positive number, as in RSiena).
"""
struct BasicRateEffect <: RateEffect
    variable::Symbol
    period::Int
end

effect_name(e::BasicRateEffect) = Symbol("rate$(e.variable)$(e.period)")
effect_type(::BasicRateEffect) = :rate
target_variable(e::BasicRateEffect) = e.variable

"""
    rate_score(e::RateEffect, state::NetworkState, data::SienaData, actor::Int)

Score ``r_{ki}(x)`` of the rate effect for a given actor. The effect's multiplicative
contribution to the actor's rate is ``\\exp(\\alpha_k r_{ki})``; the same score defines
the effect's Method-of-Moments statistic ``\\sum_i r_{ki} d_i`` (with ``d_i`` the
amount of change by actor ``i``).
"""
function rate_score end

rate_score(::BasicRateEffect, state::NetworkState, data::SienaData, actor::Int) = 1.0

"""
    compute_rate(e::RateEffect, θ::Float64, state::NetworkState,
                data::SienaData, actor::Int)

Multiplicative rate factor ``\\exp(\\theta \\cdot r_{ki})`` of a (non-basic) rate
effect for a given actor.
"""
function compute_rate(e::RateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    return exp(θ * rate_score(e, state, data, actor))
end

#==============================================================================#
# Covariate Rate Effects
#==============================================================================#

"""
    CovariateRateEffect <: RateEffect

Rate depends on a covariate value: score ``v_i``.
"""
struct CovariateRateEffect <: RateEffect
    variable::Symbol
    covariate::Symbol
    period::Int
end

effect_name(e::CovariateRateEffect) = Symbol("rate$(e.covariate)")
effect_type(::CovariateRateEffect) = :rate
target_variable(e::CovariateRateEffect) = e.variable
interaction_with(e::CovariateRateEffect) = e.covariate

function rate_score(e::CovariateRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    return _get_covariate_value(data.covariates[e.covariate], actor, state.period)
end

"""
    CovariateSqRateEffect <: RateEffect

Rate depends on the squared covariate value: score ``v_i^2``.
"""
struct CovariateSqRateEffect <: RateEffect
    variable::Symbol
    covariate::Symbol
    period::Int
end

effect_name(e::CovariateSqRateEffect) = Symbol("rateSq$(e.covariate)")
effect_type(::CovariateSqRateEffect) = :rate
target_variable(e::CovariateSqRateEffect) = e.variable
interaction_with(e::CovariateSqRateEffect) = e.covariate

function rate_score(e::CovariateSqRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    return _get_covariate_value(data.covariates[e.covariate], actor, state.period)^2
end

#==============================================================================#
# Degree-Based Rate Effects
#==============================================================================#

"""
    OutdegreeRateEffect <: RateEffect

Rate depends on outdegree: score ``x_{i+}``.
"""
struct OutdegreeRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::OutdegreeRateEffect) = :outRateX
effect_type(::OutdegreeRateEffect) = :rate
target_variable(e::OutdegreeRateEffect) = e.variable
interaction_with(e::OutdegreeRateEffect) = e.network

function rate_score(e::OutdegreeRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    return Float64(_row_sum(state.networks[e.network], actor))
end

"""
    IndegreeRateEffect <: RateEffect

Rate depends on indegree: score ``x_{+i}``.
"""
struct IndegreeRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::IndegreeRateEffect) = :inRateX
effect_type(::IndegreeRateEffect) = :rate
target_variable(e::IndegreeRateEffect) = e.variable
interaction_with(e::IndegreeRateEffect) = e.network

function rate_score(e::IndegreeRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    return Float64(_col_sum(state.networks[e.network], actor))
end

"""
    OutdegreeLogRateEffect <: RateEffect

Rate depends on log(outdegree + 1).
"""
struct OutdegreeLogRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::OutdegreeLogRateEffect) = :outRateLog
effect_type(::OutdegreeLogRateEffect) = :rate
target_variable(e::OutdegreeLogRateEffect) = e.variable
interaction_with(e::OutdegreeLogRateEffect) = e.network

function rate_score(e::OutdegreeLogRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    return log(_row_sum(state.networks[e.network], actor) + 1)
end

"""
    IndegreeLogRateEffect <: RateEffect

Rate depends on log(indegree + 1).
"""
struct IndegreeLogRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::IndegreeLogRateEffect) = :inRateLog
effect_type(::IndegreeLogRateEffect) = :rate
target_variable(e::IndegreeLogRateEffect) = e.variable
interaction_with(e::IndegreeLogRateEffect) = e.network

function rate_score(e::IndegreeLogRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    return log(_col_sum(state.networks[e.network], actor) + 1)
end

"""
    OutdegreeInvRateEffect <: RateEffect

Rate depends on 1/(outdegree + 1).
"""
struct OutdegreeInvRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::OutdegreeInvRateEffect) = :outRateInv
effect_type(::OutdegreeInvRateEffect) = :rate
target_variable(e::OutdegreeInvRateEffect) = e.variable
interaction_with(e::OutdegreeInvRateEffect) = e.network

function rate_score(e::OutdegreeInvRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    return 1.0 / (_row_sum(state.networks[e.network], actor) + 1)
end

"""
    IndegreeInvRateEffect <: RateEffect

Rate depends on 1/(indegree + 1).
"""
struct IndegreeInvRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::IndegreeInvRateEffect) = :inRateInv
effect_type(::IndegreeInvRateEffect) = :rate
target_variable(e::IndegreeInvRateEffect) = e.variable
interaction_with(e::IndegreeInvRateEffect) = e.network

function rate_score(e::IndegreeInvRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    return 1.0 / (_col_sum(state.networks[e.network], actor) + 1)
end

"""
    OutdegreeSqRateEffect <: RateEffect

Rate depends on squared outdegree: score ``x_{i+}^2``.
"""
struct OutdegreeSqRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::OutdegreeSqRateEffect) = :outRateSq
effect_type(::OutdegreeSqRateEffect) = :rate
target_variable(e::OutdegreeSqRateEffect) = e.variable
interaction_with(e::OutdegreeSqRateEffect) = e.network

function rate_score(e::OutdegreeSqRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    return Float64(_row_sum(state.networks[e.network], actor))^2
end

#==============================================================================#
# Reciprocity-Based Rate Effects
#==============================================================================#

"""
    RecipDegreeRateEffect <: RateEffect

Rate depends on the number of reciprocated ties.
"""
struct RecipDegreeRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::RecipDegreeRateEffect) = :recipRateX
effect_type(::RecipDegreeRateEffect) = :rate
target_variable(e::RecipDegreeRateEffect) = e.variable
interaction_with(e::RecipDegreeRateEffect) = e.network

function rate_score(e::RecipDegreeRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    net = state.networks[e.network]
    n = size(net, 1)
    recip = 0
    for j in 1:n
        if j != actor && net[actor, j] == 1 && net[j, actor] == 1
            recip += 1
        end
    end
    return Float64(recip)
end

"""
    OutRecipRateEffect <: RateEffect

Rate depends on outgoing ties that are reciprocated.
Same as RecipDegreeRateEffect but named differently in RSiena.
"""
const OutRecipRateEffect = RecipDegreeRateEffect

#==============================================================================#
# Behavior-Based Rate Effects
#==============================================================================#

"""
    BehaviorRateEffect <: RateEffect

Rate depends on the actor's own (mean-centered) behavior value.
"""
struct BehaviorRateEffect <: RateEffect
    variable::Symbol
    behavior::Symbol
    period::Int
end

effect_name(e::BehaviorRateEffect) = Symbol("behRate$(e.behavior)")
effect_type(::BehaviorRateEffect) = :rate
target_variable(e::BehaviorRateEffect) = e.variable
interaction_with(e::BehaviorRateEffect) = e.behavior

function rate_score(e::BehaviorRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    haskey(state.behaviors, e.behavior) || return 0.0
    dep = data.dependents[e.behavior]
    z = state.behaviors[e.behavior][actor]
    return dep isa DependentBehavior ? z - dep.mean_val : Float64(z)
end

"""
    AverageAlterRateEffect <: RateEffect

Rate depends on the average (mean-centered) behavior of out-alters.
"""
struct AverageAlterRateEffect <: RateEffect
    variable::Symbol
    behavior::Symbol
    network::Symbol
    period::Int
end

effect_name(e::AverageAlterRateEffect) = Symbol("avAltRate$(e.behavior)")
effect_type(::AverageAlterRateEffect) = :rate
target_variable(e::AverageAlterRateEffect) = e.variable
interaction_with(e::AverageAlterRateEffect) = e.behavior

function rate_score(e::AverageAlterRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    net = state.networks[e.network]
    haskey(state.behaviors, e.behavior) || return 0.0
    beh = state.behaviors[e.behavior]
    dep = data.dependents[e.behavior]
    center = dep isa DependentBehavior ? dep.mean_val : 0.0

    n = size(net, 1)
    total = 0.0
    outdeg = 0
    for j in 1:n
        if j != actor && net[actor, j] == 1
            total += beh[j] - center
            outdeg += 1
        end
    end
    return outdeg == 0 ? 0.0 : total / outdeg
end

"""
    TotalAlterRateEffect <: RateEffect

Rate depends on the total (mean-centered) behavior of out-alters.
"""
struct TotalAlterRateEffect <: RateEffect
    variable::Symbol
    behavior::Symbol
    network::Symbol
    period::Int
end

effect_name(e::TotalAlterRateEffect) = Symbol("totAltRate$(e.behavior)")
effect_type(::TotalAlterRateEffect) = :rate
target_variable(e::TotalAlterRateEffect) = e.variable
interaction_with(e::TotalAlterRateEffect) = e.behavior

function rate_score(e::TotalAlterRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    net = state.networks[e.network]
    haskey(state.behaviors, e.behavior) || return 0.0
    beh = state.behaviors[e.behavior]
    dep = data.dependents[e.behavior]
    center = dep isa DependentBehavior ? dep.mean_val : 0.0

    n = size(net, 1)
    total = 0.0
    for j in 1:n
        if j != actor && net[actor, j] == 1
            total += beh[j] - center
        end
    end
    return total
end

#==============================================================================#
# Similarity-Based Rate Effects
#==============================================================================#

"""
    SimilarityRateEffect <: RateEffect

Rate depends on average behavior similarity with out-alters.
"""
struct SimilarityRateEffect <: RateEffect
    variable::Symbol
    behavior::Symbol
    network::Symbol
    period::Int
end

effect_name(e::SimilarityRateEffect) = Symbol("simRate$(e.behavior)")
effect_type(::SimilarityRateEffect) = :rate
target_variable(e::SimilarityRateEffect) = e.variable
interaction_with(e::SimilarityRateEffect) = e.behavior

function rate_score(e::SimilarityRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    net = state.networks[e.network]
    haskey(state.behaviors, e.behavior) || return 0.0
    beh = state.behaviors[e.behavior]
    dep = data.dependents[e.behavior]
    beh_range = dep isa DependentBehavior ? (dep.max_val - dep.min_val) : 1
    beh_range == 0 && (beh_range = 1)

    n = size(net, 1)
    total_sim = 0.0
    outdeg = 0
    ego_val = beh[actor]
    for j in 1:n
        if j != actor && net[actor, j] == 1
            total_sim += 1.0 - abs(ego_val - beh[j]) / beh_range
            outdeg += 1
        end
    end
    return outdeg == 0 ? 0.0 : total_sim / outdeg
end

#==============================================================================#
# Setting/Group Rate Effects
#==============================================================================#

"""
    SettingRateEffect <: RateEffect

Rate multiplier for actors in a particular setting: score
``I(v_i = \\text{setting value})``.
"""
struct SettingRateEffect <: RateEffect
    variable::Symbol
    setting::Symbol  # Covariate indicating setting membership
    setting_value::Int  # The setting value to match
    period::Int
end

effect_name(e::SettingRateEffect) = Symbol("settingRate$(e.setting)")
effect_type(::SettingRateEffect) = :rate
target_variable(e::SettingRateEffect) = e.variable
interaction_with(e::SettingRateEffect) = e.setting

function rate_score(e::SettingRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    cov_val = _get_covariate_value(data.covariates[e.setting], actor, state.period)
    return round(Int, cov_val) == e.setting_value ? 1.0 : 0.0
end

#==============================================================================#
# Ego x Alter Interaction Rate Effects
#==============================================================================#

"""
    EgoAlterRateEffect <: RateEffect

Rate depends on the sum of ego × alter covariate products over outgoing ties.
"""
struct EgoAlterRateEffect <: RateEffect
    variable::Symbol
    covariate::Symbol
    network::Symbol
    period::Int
end

effect_name(e::EgoAlterRateEffect) = Symbol("egoAltRate$(e.covariate)")
effect_type(::EgoAlterRateEffect) = :rate
target_variable(e::EgoAlterRateEffect) = e.variable
interaction_with(e::EgoAlterRateEffect) = e.covariate

function rate_score(e::EgoAlterRateEffect, state::NetworkState,
                    data::SienaData, actor::Int)
    net = state.networks[e.network]
    cov = data.covariates[e.covariate]
    ego_val = _get_covariate_value(cov, actor, state.period)

    n = size(net, 1)
    total = 0.0
    for j in 1:n
        if j != actor && net[actor, j] == 1
            total += ego_val * _get_covariate_value(cov, j, state.period)
        end
    end
    return total
end

#==============================================================================#
# Rate Function Computation
#==============================================================================#

"""
    actor_rate(λ_basic::Float64, effects::Vector{EffectEntry}, θ::Vector{Float64},
              state::NetworkState, data::SienaData, actor::Int)

Compute the rate ``\\lambda_i = \\lambda \\prod_k \\exp(\\theta_k r_{ki})`` for an
actor, given the basic rate and the non-basic rate effect entries with their
parameters.
"""
function actor_rate(λ_basic::Float64, effects::Vector{EffectEntry}, θ::Vector{Float64},
                    state::NetworkState, data::SienaData, actor::Int)
    λ = λ_basic
    for (k, entry) in enumerate(effects)
        λ *= exp(θ[k] * rate_score(entry.effect::RateEffect, state, data, actor))
    end
    return λ
end

"""
    sample_waiting_time(rate::Float64, rng::AbstractRNG)

Sample waiting time from exponential distribution with given rate.
"""
function sample_waiting_time(rate::Float64, rng::AbstractRNG)
    return -log(rand(rng)) / rate
end
