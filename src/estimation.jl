"""
Estimation module for SAOM.

Implements unconditional Method-of-Moments estimation via Robbins-Monro stochastic
approximation. The full parameter vector θ contains the free rate parameters first
(basic rates as the rate itself) and then the free objective parameters (see
`build_param_map`).

Moment statistics, aligned with θ:
- basic rate for period m: the distance between the state at the end of period m and
  the observation at the start of the period, ``\\sum_i d_i`` (network: Hamming
  distance; behavior: L1 distance);
- non-basic rate effect: ``\\sum_m \\sum_i r_{ki}(x(t_m)) d_i``;
- objective effect: ``\\sum_m s_k`` evaluated at the end of each period.

Targets replace the simulated end state with the observed next wave; each simulated
period starts from the observed wave (unconditional MoM, Snijders 2001).
"""

#==============================================================================#
# Result Type
#==============================================================================#

"""
    SienaResult

Result of SAOM estimation.

# Fields
- `effects::SienaEffects`: The effects object
- `parameter_names::Vector{String}`: Names of the free parameters (θ order)
- `estimates::Vector{Float64}`: Estimates of the full free-parameter vector
- `standard_errors::Vector{Float64}`: Standard errors
- `t_ratios::Vector{Float64}`: Convergence t-ratios (deviation / sd of simulated statistic)
- `covariance::Matrix{Float64}`: Covariance matrix of the estimates
- `converged::Bool`: Whether all per-parameter |t-ratios| are below the threshold
  (default 0.1) *and* the overall maximum convergence ratio is below its threshold
  (default 0.25), the RSiena publication standard
- `tconv_max::Float64`: Overall maximum convergence ratio (RSiena's `tconv.max`)
- `diverged::Bool`: Whether a parameter hit the divergence clamp during estimation
  (estimates are then unreliable)
- `n_iterations::Int`: Number of Robbins-Monro iterations used
- `rate_estimates::Dict{Symbol, Vector{Float64}}`: Basic rate estimates per variable and period
- `targets::Vector{Float64}`: Observed target statistics
- `simulated_means::Vector{Float64}`: Mean simulated statistics at the estimates (phase 3)
"""
struct SienaResult
    effects::SienaEffects
    parameter_names::Vector{String}
    estimates::Vector{Float64}
    standard_errors::Vector{Float64}
    t_ratios::Vector{Float64}
    covariance::Matrix{Float64}
    converged::Bool
    tconv_max::Float64
    diverged::Bool
    n_iterations::Int
    rate_estimates::Dict{Symbol, Vector{Float64}}
    targets::Vector{Float64}
    simulated_means::Vector{Float64}
end

function Base.show(io::IO, result::SienaResult)
    println(io, "SAOM Estimation Results")
    println(io, "=======================")
    println(io, "Converged: $(result.converged) " *
                "(max |t-ratio| = $(round(maximum(abs.(result.t_ratios)), digits=3)), " *
                "overall max convergence ratio = $(round(result.tconv_max, digits=3)))")
    result.diverged &&
        println(io, "WARNING: divergence detected (estimates hit the parameter clamp)")
    println(io, "Iterations: $(result.n_iterations)")
    pm = build_param_map(result.effects)
    n_rate = n_free_rate_parameters(pm)

    if n_rate > 0
        println(io)
        println(io, "Rate Parameters:")
        println(io, "----------------")
        for i in 1:n_rate
            @printf(io, "%-28s %8.4f (%6.4f)\n", result.parameter_names[i],
                    result.estimates[i], result.standard_errors[i])
        end
    end

    println(io)
    println(io, "Objective Function Parameters:")
    println(io, "------------------------------")
    idx = (n_rate + 1):length(result.estimates)
    est = result.estimates[idx]
    se = result.standard_errors[idx]
    z = [se[k] > 0 ? est[k] / se[k] : NaN for k in eachindex(est)]
    p = [isnan(zk) ? NaN : 2 * ccdf(Normal(), abs(zk)) for zk in z]
    print_coeftable(io, result.parameter_names[idx], est, se, p; z_values=z)
end

#==============================================================================#
# Moment Statistics
#==============================================================================#

# Observed states at the start of each period (used for rate scores and distances).
function _observed_start_states(data::SienaData)
    return [initialize!(NetworkState(), data, p) for p in 1:(data.n_waves - 1)]
end

# Observed states at the end of each period (period set to the starting wave so that
# varying covariates use the values of the period).
function _observed_end_states(data::SienaData)
    return [initialize!(NetworkState(), data, p + 1; period=p)
            for p in 1:(data.n_waves - 1)]
end

# Rate moment for one period: sum_i r_ki(x(t_m)) * d_i(end vs start).
# Structurally determined dyads (period-start mask) cannot change under the
# model, so they are excluded from the observed distances as well; likewise
# dyads/actors inactive in the period (composition change).
function _rate_distance_statistic(eff::RateEffect, start_state::NetworkState,
                                  end_state::NetworkState, data::SienaData)
    v = target_variable(eff)
    dep = data.dependents[v]
    act = start_state.active
    total = 0.0
    if dep isa DependentNetwork
        x0 = start_state.networks[v]
        x1 = end_state.networks[v]
        smask = _structural_mask(dep, start_state.period)
        onemode = dep.type == :onemode
        n, m = size(x0)
        for i in 1:n
            act !== nothing && !act[i] && continue
            d = 0
            for j in 1:m
                smask !== nothing && smask[i, j] && continue
                act !== nothing && onemode && !act[j] && continue
                d += abs(x1[i, j] - x0[i, j])
            end
            d == 0 && continue
            total += rate_score(eff, start_state, data, i) * d
        end
    else
        z0 = start_state.behaviors[v]
        z1 = end_state.behaviors[v]
        for i in eachindex(z0)
            act !== nothing && !act[i] && continue
            d = abs(z1[i] - z0[i])
            d == 0 && continue
            total += rate_score(eff, start_state, data, i) * d
        end
    end
    return total
end

# Zero out structurally determined entries (period-start mask) of every network
# variable of `state`, plus the rows/columns of actors inactive in the period
# (composition change), so that effect statistics computed on it exclude
# structurally fixed dyads and absent actors. Applied identically to observed
# (target) and simulated states, keeping the moment equation consistent.
function _zero_structural!(state::NetworkState, data::SienaData)
    act = state.active
    for (name, dep) in data.dependents
        dep isa DependentNetwork || continue
        x = state.networks[name]
        smask = _structural_mask(dep, state.period)
        if smask !== nothing
            n1, n2 = size(smask)
            @inbounds for j in 1:n2, i in 1:n1
                smask[i, j] && (x[i, j] = 0)
            end
        end
        if act !== nothing
            n1, n2 = size(x)
            for i in 1:n1
                act[i] && continue
                for j in 1:n2
                    x[i, j] = 0
                end
            end
            if dep.type == :onemode
                for j in 1:n2
                    act[j] && continue
                    for i in 1:n1
                        x[i, j] = 0
                    end
                end
            end
        end
    end
    return state
end

# Full moment vector aligned with pm.free, given the end state of every period.
#
# Objective statistics follow RSiena's convention: the effect's *target* variable is
# taken at the end of the period, while every other variable (co-evolving networks,
# behaviors) keeps its value at the start of the period.
#
# Structurally determined dyads are excluded: network entries under the
# period-start structural mask are zeroed before computing effect statistics
# (for targets and simulations alike), and rate distances skip them.
function _moment_statistics(data::SienaData, pm::ParameterMap,
                            end_states::Vector{NetworkState},
                            start_states::Vector{NetworkState})
    n_periods = data.n_waves - 1
    stats = zeros(length(pm.free))
    for p in 1:n_periods
        mixed = Dict{Symbol, NetworkState}()
        for (k, entry) in enumerate(pm.free)
            eff = entry.effect
            if eff isa RateEffect
                if !(eff isa BasicRateEffect) || eff.period == p
                    stats[k] += _rate_distance_statistic(eff, start_states[p],
                                                         end_states[p], data)
                end
            else
                v = target_variable(eff)
                st = get!(mixed, v) do
                    s = snapshot(start_states[p])
                    if haskey(end_states[p].networks, v)
                        s.networks[v] = copy(end_states[p].networks[v])
                    else
                        s.behaviors[v] = copy(end_states[p].behaviors[v])
                    end
                    _zero_structural!(s, data)
                end
                stats[k] += compute_statistic(eff, st, data)
            end
        end
    end
    return stats
end

"""
    compute_target_statistics(data::SienaData, effects::SienaEffects)

Observed target statistics for all free parameters (θ order): objective statistics
are evaluated at the end wave of each period and summed over periods; rate statistics
are the observed amounts of change.
"""
function compute_target_statistics(data::SienaData, effects::SienaEffects)
    pm = build_param_map(effects)
    return _moment_statistics(data, pm, _observed_end_states(data),
                              _observed_start_states(data))
end

"""
    compute_simulated_statistics(data::SienaData, effects::SienaEffects,
                                results::Vector{SimulationResult})

Simulated moment statistics for all free parameters, from the per-period results of
[`simulate_saom`](@ref).
"""
function compute_simulated_statistics(data::SienaData, effects::SienaEffects,
                                     results::Vector{SimulationResult})
    pm = build_param_map(effects)
    end_states = [r.final_state for r in results]
    return _moment_statistics(data, pm, end_states, _observed_start_states(data))
end

# One simulation -> moment vector (with cached pm/start states). If `scores` is
# given, the score function of the trajectory is accumulated into it. `condvar`/
# `cond_targets` switch to conditional simulation (see `simulate_saom`); if
# `times` is given, it receives the per-period elapsed simulation times (the
# stopping times, used for the conditional rate estimates).
function _simulate_moments(data::SienaData, effects::SienaEffects, pm::ParameterMap,
                           start_states::Vector{NetworkState}, θ::Vector{Float64},
                           seed::Int;
                           scores::Union{Nothing, ScoreAccumulator}=nothing,
                           condvar::Union{Symbol, Nothing}=nothing,
                           cond_targets::Union{Nothing, Vector{Int}}=nothing,
                           times::Union{Nothing, AbstractVector{Float64}}=nothing)
    _, results = simulate_saom(data, effects, θ; seed=seed, scores=scores,
                               condvar=condvar, cond_targets=cond_targets)
    if times !== nothing
        for (p, r) in enumerate(results)
            times[p] = r.final_state.time
        end
    end
    end_states = [r.final_state for r in results]
    return _moment_statistics(data, pm, end_states, start_states)
end

#==============================================================================#
# Robbins-Monro Update
#==============================================================================#

"""
    update_parameters!(θ::Vector{Float64}, score::Vector{Float64},
                      D::Matrix{Float64}, gain::Float64; max_step::Float64=2.0)

Robbins-Monro update ``θ \\leftarrow θ - \\text{gain} \\cdot D^{-1} (\\bar s - s_{obs})``,
with the step capped at `max_step` in Euclidean norm. Falls back to the pseudoinverse
if `D` is singular.
"""
function update_parameters!(θ::Vector{Float64}, score::Vector{Float64},
                           D::Matrix{Float64}, gain::Float64; max_step::Float64=2.0)
    update = try
        D \ score
    catch err
        err isa Union{SingularException, LinearAlgebra.LAPACKException} || rethrow()
        pinv(D) * score
    end
    s = norm(update)
    if s > max_step
        update .*= max_step / s
    end
    θ .-= gain .* update
    return θ
end

# Keep rate parameters positive and objective parameters in a sane range. Returns
# `true` if a divergence clamp activated (objective parameter at ±10 or basic rate
# at the upper bound); hitting the small positive rate floor is not divergence.
function _clamp_parameters!(θ::Vector{Float64}, pm::ParameterMap)
    diverged = false
    for (i, entry) in enumerate(pm.free)
        if entry.effect isa BasicRateEffect
            θ[i] > 1e3 && (diverged = true)
            θ[i] = clamp(θ[i], 0.05, 1e3)
        else
            abs(θ[i]) > 10.0 && (diverged = true)
            θ[i] = clamp(θ[i], -10.0, 10.0)
        end
    end
    return diverged
end

#==============================================================================#
# Derivative Matrix Estimation
#==============================================================================#

"""
    estimate_derivative_matrix(data::SienaData, effects::SienaEffects,
                              θ::Vector{Float64}, n_sims::Int, rng::AbstractRNG)

Estimate ``D = \\partial E[s]/\\partial θ`` by forward finite differences with
**common random numbers**: base and perturbed runs use the same simulation seeds, so
the Monte-Carlo noise largely cancels in the difference.

The simulations are independent (each is driven by its own seeded RNG) and run
multi-threaded; every simulation writes to its own slot and the results are reduced
in a fixed order, so the estimate is identical to a single-threaded run regardless
of the number of threads.
"""
function estimate_derivative_matrix(data::SienaData, effects::SienaEffects,
                                   θ::Vector{Float64}, n_sims::Int, rng::AbstractRNG;
                                   condvar::Union{Symbol, Nothing}=nothing,
                                   cond_targets::Union{Nothing, Vector{Int}}=nothing)
    pm = build_param_map(effects)
    start_states = _observed_start_states(data)
    n_params = length(θ)
    D = zeros(n_params, n_params)

    seeds = rand(rng, 1:10^8, n_sims)

    # Per-simulation results land in their own row; the mean is accumulated
    # afterwards in simulation order, so it is thread-count independent.
    sim_stats = zeros(n_sims, n_params)
    function mean_stats!(dest::Vector{Float64}, θv::Vector{Float64})
        Threads.@threads for s in 1:n_sims
            sim_stats[s, :] = _simulate_moments(data, effects, pm, start_states, θv,
                                                seeds[s]; condvar=condvar,
                                                cond_targets=cond_targets)
        end
        fill!(dest, 0.0)
        for s in 1:n_sims
            dest .+= view(sim_stats, s, :)
        end
        dest ./= n_sims
        return dest
    end

    base_stats = mean_stats!(zeros(n_params), θ)

    plus_stats = zeros(n_params)
    for j in 1:n_params
        ε = 0.1 * max(1.0, abs(θ[j]))
        θ_plus = copy(θ)
        θ_plus[j] += ε

        mean_stats!(plus_stats, θ_plus)

        D[:, j] = (plus_stats .- base_stats) ./ ε
    end

    return D
end

"""
    estimate_derivative_matrix_score(data::SienaData, effects::SienaEffects,
                                    θ::Vector{Float64}, n_sims::Int, rng::AbstractRNG)

Score-function (likelihood-ratio) estimator of ``D = \\partial E[s]/\\partial θ``
(Schweinberger & Snijders 2007), as used by RSiena: simulate `n_sims` trajectories
at θ, accumulate the score of each trajectory (see [`ScoreAccumulator`](@ref)), and
estimate ``D = \\widehat{\\mathrm{cov}}(s, S)`` across the simulations.

Unlike [`estimate_derivative_matrix`](@ref) it needs no parameter perturbations, so
the cost is `n_sims` simulations regardless of the number of parameters.

The simulations are independent (one seeded RNG per simulation) and run
multi-threaded; results are identical to a single-threaded run regardless of the
number of threads.
"""
function estimate_derivative_matrix_score(data::SienaData, effects::SienaEffects,
                                         θ::Vector{Float64}, n_sims::Int,
                                         rng::AbstractRNG)
    pm = build_param_map(effects)
    start_states = _observed_start_states(data)
    n_params = length(θ)
    stats = zeros(n_sims, n_params)
    scores = zeros(n_sims, n_params)
    # Seeds are drawn sequentially up front (same RNG stream as a serial loop);
    # each simulation then runs on its own seeded RNG and writes only its own rows.
    seeds = [rand(rng, 1:10^8) for _ in 1:n_sims]
    Threads.@threads for s in 1:n_sims
        sacc = ScoreAccumulator(pm)
        stats[s, :] = _simulate_moments(data, effects, pm, start_states, θ,
                                        seeds[s]; scores=sacc)
        scores[s, :] = sacc.scores
    end
    return cov(stats, scores)
end

#==============================================================================#
# Initial Values
#==============================================================================#

# Observed amount of change of a dependent variable over one period: Hamming
# distance (networks) or L1 distance (behavior) between the period's endpoint
# waves, excluding structurally determined dyads and actors inactive in the
# period (composition change). This is the basic-rate moment target and the
# stopping distance of conditional estimation.
function _observed_distance(data::SienaData, variable::Symbol, period::Int)
    dep = data.dependents[variable]
    act = _activity_mask(data, period)
    dist = 0
    if dep isa DependentNetwork
        x0 = dep.networks[period]
        x1 = dep.networks[period + 1]
        smask = _structural_mask(dep, period)
        onemode = dep.type == :onemode
        n, m = size(x0)
        for i in 1:n
            act !== nothing && !act[i] && continue
            for j in 1:m
                smask !== nothing && smask[i, j] && continue
                act !== nothing && onemode && !act[j] && continue
                dist += abs(x1[i, j] - x0[i, j])
            end
        end
    else
        z0 = dep.values[period]
        z1 = dep.values[period + 1]
        for i in eachindex(z0)
            act !== nothing && !act[i] && continue
            dist += abs(z1[i] - z0[i])
        end
    end
    return dist
end

# Simple data-based initial value for a basic rate parameter: observed amount of
# change per actor, inflated to account for cancelling ministeps.
function default_basic_rate(data::SienaData, variable::Symbol, period::Int)
    dep = data.dependents[variable]
    period >= data.n_waves && return 1.0
    dist = _observed_distance(data, variable, period)
    n = n_actors(dep)
    return max(0.5, 1.5 * dist / n)
end

function _initial_parameters(data::SienaData, pm::ParameterMap)
    θ = zeros(length(pm.free))
    for (i, entry) in enumerate(pm.free)
        eff = entry.effect
        if eff isa BasicRateEffect
            θ[i] = entry.initial_value > 0 ? entry.initial_value :
                   default_basic_rate(data, eff.variable, eff.period)
        else
            θ[i] = entry.initial_value
        end
    end
    return θ
end

#==============================================================================#
# Main Estimation Function
#==============================================================================#

"""
    fit_siena(data::SienaData, effects::SienaEffects;
              algorithm::SienaAlgorithm=SienaAlgorithm())

Estimate SAOM parameters using unconditional Method of Moments with Robbins-Monro
stochastic approximation. Equivalent to `siena07()` in RSiena ([`siena07`](@ref)
is kept as an alias).

Phases:
1. Derivative matrix estimation at the initial values, plus initial rough updates.
2. `n_subphases` subphases of Robbins-Monro updates with halving gain; the derivative
   matrix is re-estimated at the first and last subphase. Each subphase returns the
   Polyak-Ruppert average of its θ iterates (after a short warm start), as in RSiena.
3. Simulations at the final estimates for convergence t-ratios
   (deviation / sd of the simulated statistic), the overall maximum convergence
   ratio `tconv.max`, and standard errors via ``D^{-1} \\Sigma D^{-T}`` where `D` is
   by default the score-function (likelihood-ratio) estimate over all phase-3
   simulations (`algorithm.derivative_method = :score`; see
   [`estimate_derivative_matrix_score`](@ref)).

Convergence follows the RSiena publication standard: all per-parameter |t-ratios|
below `algorithm.convergence_threshold` (0.1) *and* `tconv.max` below
`algorithm.overall_convergence_threshold` (0.25).

# Conditional estimation
With `algorithm.conditional = true` (RSiena's `cond=TRUE`), estimation conditions
on the observed amount of change of one dependent variable (`algorithm.condvar`;
defaults to the only dependent variable, RSiena's `condvarno=1`): every simulated
period runs until that variable's distance from the period-start observation
reaches the observed distance, instead of until time 1 (Snijders 2001). The
conditioned variable's basic rate parameters leave the parameter vector — the
entries are marked fixed on the effects object — and are estimated afterwards
from the phase-3 stopping times (simulation rate x mean stopping time); they are
reported in `rate_estimates`. Conditional estimation always uses the
finite-difference derivative estimator (the trajectory score function assumes
time-1 termination).

# Returns
- `SienaResult`: Estimation results (rate parameters first, then objective parameters)
"""
function fit_siena(data::SienaData, effects::SienaEffects;
                   algorithm::SienaAlgorithm=SienaAlgorithm())
    data.n_waves >= 2 ||
        throw(ArgumentError("estimation requires at least 2 observation waves"))

    # Conditional estimation setup: fix the conditioned variable's basic rate
    # entries (they are determined by the conditioning, not by the moment
    # equations) and compute the per-period target distances.
    condvar = nothing
    cond_targets = nothing
    cond_entries = EffectEntry[]
    if algorithm.conditional
        condvar = algorithm.condvar
        if condvar === nothing
            length(data.dependents) == 1 ||
                throw(ArgumentError("conditional estimation with several dependent " *
                                    "variables requires algorithm.condvar"))
            condvar = first(keys(data.dependents))
        end
        haskey(data.dependents, condvar) ||
            throw(ArgumentError("unknown conditioning variable :$condvar"))
        cond_targets = [_observed_distance(data, condvar, p)
                        for p in 1:(data.n_waves - 1)]
        all(>(0), cond_targets) ||
            throw(ArgumentError("conditional estimation requires observed change " *
                                "in :$condvar in every period"))
        for entry in effects.effects
            eff = entry.effect
            if entry.include && eff isa BasicRateEffect && eff.variable == condvar
                entry.fix = true
                entry.initial_value > 0 ||
                    (entry.initial_value = default_basic_rate(data, condvar, eff.period))
                push!(cond_entries, entry)
            end
        end
    end

    pm = build_param_map(effects)
    isempty(pm.free) && throw(ArgumentError("the model has no free parameters"))
    for entry in pm.free
        if effect_type(entry.effect) in (:endow, :creation)
            throw(ArgumentError("endowment/creation effects are not yet supported in " *
                                "estimation: $(entry.shortname) (fix them or exclude them)"))
        end
    end

    rng = isnothing(algorithm.seed) ? Random.default_rng() : MersenneTwister(algorithm.seed)
    start_states = _observed_start_states(data)

    θ = _initial_parameters(data, pm)
    targets = _moment_statistics(data, pm, _observed_end_states(data), start_states)
    n_params = length(θ)

    if algorithm.verbose
        println("Starting SAOM estimation")
        println("Free parameters: $n_params " *
                "($(n_free_rate_parameters(pm)) rate, " *
                "$(n_params - n_free_rate_parameters(pm)) objective)")
    end

    total_iterations = 0
    diverged = false
    sim(θv) = _simulate_moments(data, effects, pm, start_states, θv,
                                rand(rng, 1:10^8);
                                condvar=condvar, cond_targets=cond_targets)

    function clamp_tracked!()
        if _clamp_parameters!(θ, pm) && !diverged
            diverged = true
            @warn "parameter estimate hit the divergence clamp " *
                  "(|θ| = 10 for objective parameters); the model is likely " *
                  "diverging and the estimates are unreliable"
        end
    end

    #==========================================================================
    # Phase 1: derivative matrix at initial values + rough updates
    ==========================================================================#
    algorithm.verbose && println("\n--- Phase 1 ---")
    D = estimate_derivative_matrix(data, effects, θ, algorithm.derivative_sims, rng;
                                   condvar=condvar, cond_targets=cond_targets)
    D += 0.01 * I

    gain = algorithm.initial_gain
    for _ in 1:algorithm.phase1_iterations
        total_iterations += 1
        score = sim(θ) .- targets
        update_parameters!(θ, score, D, gain)
        clamp_tracked!()
    end

    #==========================================================================
    # Phase 2: subphases with halving gain and Polyak-Ruppert averaging
    ==========================================================================#
    algorithm.verbose && println("\n--- Phase 2 ---")
    for subphase in 1:algorithm.n_subphases
        algorithm.verbose && println("  Subphase $subphase")
        if subphase == 1 || subphase == algorithm.n_subphases
            D = estimate_derivative_matrix(data, effects, θ, algorithm.derivative_sims,
                                           rng; condvar=condvar,
                                           cond_targets=cond_targets)
            D += 0.01 * I
        end
        gain = max(algorithm.initial_gain * 0.5^subphase, algorithm.min_gain)
        # Polyak-Ruppert averaging: the subphase result is the average of the θ
        # iterates over the subphase (excluding a short warm start), not the last
        # iterate — this suppresses most of the Robbins-Monro Monte-Carlo noise.
        n_iter = algorithm.phase1_iterations
        n_warm = n_iter ÷ 4
        θ_sum = zeros(n_params)
        n_avg = 0
        for it in 1:n_iter
            total_iterations += 1
            score = sim(θ) .- targets
            update_parameters!(θ, score, D, gain)
            clamp_tracked!()
            if it > n_warm
                θ_sum .+= θ
                n_avg += 1
            end
        end
        n_avg > 0 && (θ .= θ_sum ./ n_avg)
    end

    #==========================================================================
    # Phase 3: convergence check and standard errors at fixed θ
    ==========================================================================#
    algorithm.verbose && println("\n--- Phase 3 ---")
    n3 = algorithm.phase3_iterations
    # The trajectory score function assumes unconditional (time-1) termination,
    # so conditional estimation falls back to the finite-difference estimator.
    use_score = algorithm.derivative_method == :score && condvar === nothing
    phase3_stats = zeros(n3, n_params)
    phase3_scores = zeros(n3, n_params)
    # Conditional estimation: collect the per-period stopping times for the
    # post-hoc rate estimates of the conditioned variable.
    phase3_times = condvar === nothing ? nothing : zeros(n3, data.n_waves - 1)
    # Phase-3 simulations are independent at fixed θ: pre-draw one seed per
    # simulation (same RNG stream as a serial loop), then run them multi-threaded.
    # Each iteration writes only its own rows, so the results are identical to a
    # single-threaded run regardless of the thread count.
    phase3_seeds = [rand(rng, 1:10^8) for _ in 1:n3]
    algorithm.verbose &&
        println("  Running $n3 simulations on $(Threads.nthreads()) thread(s)")
    Threads.@threads for iter in 1:n3
        if use_score
            sacc = ScoreAccumulator(pm)
            phase3_stats[iter, :] = _simulate_moments(data, effects, pm, start_states,
                                                      θ, phase3_seeds[iter];
                                                      scores=sacc)
            phase3_scores[iter, :] = sacc.scores
        else
            times = phase3_times === nothing ? nothing :
                    view(phase3_times, iter, :)
            phase3_stats[iter, :] = _simulate_moments(data, effects, pm, start_states,
                                                      θ, phase3_seeds[iter];
                                                      condvar=condvar,
                                                      cond_targets=cond_targets,
                                                      times=times)
        end
    end
    total_iterations += n3

    mean_sim_stats = vec(mean(phase3_stats, dims=1))
    deviations = mean_sim_stats .- targets
    sd_stats = vec(std(phase3_stats, dims=1))
    Sigma = cov(phase3_stats)

    # Convergence: per-parameter t-ratios (deviation / sd of the simulated statistic)
    # and the overall maximum convergence ratio tconv.max = sqrt(ē' Σ⁻¹ ē), the
    # maximum t-ratio over all linear combinations of the statistics (RSiena).
    conv_stats = ConvergenceStats(n_params)
    update_convergence!(conv_stats, deviations, sd_stats)
    conv_stats.tconv_max = sqrt(max(_quad_form_inv(deviations, Sigma), 0.0))
    converged = is_converged(conv_stats, algorithm.convergence_threshold,
                             algorithm.overall_convergence_threshold)

    # Derivative matrix for the standard errors: score-function (likelihood-ratio)
    # estimator over all phase-3 simulations (RSiena) or finite differences,
    # regularized like the phase-1/2 derivative estimates.
    D_final = use_score ? cov(phase3_stats, phase3_scores) :
              estimate_derivative_matrix(data, effects, θ, algorithm.derivative_sims,
                                         rng; condvar=condvar,
                                         cond_targets=cond_targets)
    D_final += 0.01 * I

    # Covariance of the estimates: D^{-1} Σ D^{-T}
    param_cov = try
        D_inv = inv(D_final)
        D_inv * Sigma * D_inv'
    catch err
        err isa Union{SingularException, LinearAlgebra.LAPACKException} || rethrow()
        @warn "derivative matrix is singular; using pseudoinverse for standard errors"
        D_pinv = pinv(D_final)
        D_pinv * Sigma * D_pinv'
    end
    se = sqrt.(max.(diag(param_cov), 0.0))

    if algorithm.verbose
        println("\n--- Results ---")
        println("Converged: $converged")
        println("Max |t-ratio|: $(round(conv_stats.max_t_ratio, digits=3))")
        println("Overall max convergence ratio: $(round(conv_stats.tconv_max, digits=3))")
        diverged && println("WARNING: divergence detected during estimation")
    end

    # Conditional rate estimates: with simulation rate ρ the stopping time of a
    # trajectory scales as 1/ρ, so the rate that makes the expected stopping time
    # equal to the unit period length is ρ_sim * E[T] (Snijders 2001). Store the
    # estimate on the fixed entries so that `basic_rate`/`rate_estimates` and any
    # subsequent simulation at the result pick it up.
    if condvar !== nothing
        mean_times = vec(mean(phase3_times, dims=1))
        for entry in cond_entries
            entry.initial_value *= mean_times[entry.effect.period]
        end
    end

    # Basic rate estimates per variable and period
    rate_estimates = Dict{Symbol, Vector{Float64}}()
    for (name, dep) in data.dependents
        if any(e -> e.effect isa BasicRateEffect && target_variable(e.effect) == name,
               pm.rate_entries)
            rate_estimates[name] = [basic_rate(pm, θ, name, p)
                                    for p in 1:(data.n_waves - 1)]
        end
    end

    return SienaResult(
        effects,
        parameter_names(effects),
        θ,
        se,
        conv_stats.t_ratios,
        param_cov,
        converged,
        conv_stats.tconv_max,
        diverged,
        total_iterations,
        rate_estimates,
        targets,
        mean_sim_stats
    )
end

"""
    siena07(data::SienaData, effects::SienaEffects;
            algorithm::SienaAlgorithm=SienaAlgorithm())

Alias for [`fit_siena`](@ref), keeping the RSiena function name.
"""
siena07(data::SienaData, effects::SienaEffects; kwargs...) =
    fit_siena(data, effects; kwargs...)

# x' Σ⁻¹ x with a pseudoinverse fallback for singular Σ.
function _quad_form_inv(x::Vector{Float64}, Σ::Matrix{Float64})
    return try
        dot(x, Σ \ x)
    catch err
        err isa Union{SingularException, LinearAlgebra.LAPACKException} || rethrow()
        dot(x, pinv(Σ) * x)
    end
end

#==============================================================================#
# Coefficient Access Functions (StatsAPI method extensions)
#==============================================================================#

"""
    coef(result::SienaResult)

Return the parameter estimates (rate parameters first, then objective parameters).
Extends `StatsAPI.coef`.
"""
StatsAPI.coef(result::SienaResult) = result.estimates

"""
    stderror(result::SienaResult)

Return standard errors. Extends `StatsAPI.stderror`.
"""
StatsAPI.stderror(result::SienaResult) = result.standard_errors

"""
    vcov(result::SienaResult)

Return covariance matrix. Extends `StatsAPI.vcov`.
"""
StatsAPI.vcov(result::SienaResult) = result.covariance

"""
    confint(result::SienaResult; level::Float64=0.95)

Compute confidence intervals. Extends `StatsAPI.confint`.
"""
function StatsAPI.confint(result::SienaResult; level::Float64=0.95)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    lower = result.estimates .- z .* result.standard_errors
    upper = result.estimates .+ z .* result.standard_errors
    return hcat(lower, upper)
end
