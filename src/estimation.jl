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
- `converged::Bool`: Whether all convergence t-ratios are below the threshold
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
    n_iterations::Int
    rate_estimates::Dict{Symbol, Vector{Float64}}
    targets::Vector{Float64}
    simulated_means::Vector{Float64}
end

function Base.show(io::IO, result::SienaResult)
    println(io, "SAOM Estimation Results")
    println(io, "=======================")
    println(io, "Converged: $(result.converged) " *
                "(max |t-ratio| = $(round(maximum(abs.(result.t_ratios)), digits=3)))")
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
    for i in (n_rate + 1):length(result.estimates)
        est = result.estimates[i]
        se = result.standard_errors[i]
        sig = se > 0 && abs(est / se) > 1.96 ? "*" : ""
        @printf(io, "%-28s %8.4f (%6.4f) %s\n", result.parameter_names[i], est, se, sig)
    end
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
function _rate_distance_statistic(eff::RateEffect, start_state::NetworkState,
                                  end_state::NetworkState, data::SienaData)
    v = target_variable(eff)
    dep = data.dependents[v]
    total = 0.0
    if dep isa DependentNetwork
        x0 = start_state.networks[v]
        x1 = end_state.networks[v]
        n, m = size(x0)
        for i in 1:n
            d = 0
            for j in 1:m
                d += abs(x1[i, j] - x0[i, j])
            end
            d == 0 && continue
            total += rate_score(eff, start_state, data, i) * d
        end
    else
        z0 = start_state.behaviors[v]
        z1 = end_state.behaviors[v]
        for i in eachindex(z0)
            d = abs(z1[i] - z0[i])
            d == 0 && continue
            total += rate_score(eff, start_state, data, i) * d
        end
    end
    return total
end

# Full moment vector aligned with pm.free, given the end state of every period.
#
# Objective statistics follow RSiena's convention: the effect's *target* variable is
# taken at the end of the period, while every other variable (co-evolving networks,
# behaviors) keeps its value at the start of the period.
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
                    s
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

# One simulation -> moment vector (with cached pm/start states).
function _simulate_moments(data::SienaData, effects::SienaEffects, pm::ParameterMap,
                           start_states::Vector{NetworkState}, θ::Vector{Float64},
                           seed::Int)
    _, results = simulate_saom(data, effects, θ; seed=seed)
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

# Keep rate parameters positive and objective parameters in a sane range.
function _clamp_parameters!(θ::Vector{Float64}, pm::ParameterMap)
    for (i, entry) in enumerate(pm.free)
        if entry.effect isa BasicRateEffect
            θ[i] = clamp(θ[i], 0.05, 1e3)
        else
            θ[i] = clamp(θ[i], -10.0, 10.0)
        end
    end
    return θ
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
"""
function estimate_derivative_matrix(data::SienaData, effects::SienaEffects,
                                   θ::Vector{Float64}, n_sims::Int, rng::AbstractRNG)
    pm = build_param_map(effects)
    start_states = _observed_start_states(data)
    n_params = length(θ)
    D = zeros(n_params, n_params)

    seeds = rand(rng, 1:10^8, n_sims)

    base_stats = zeros(n_params)
    for s in 1:n_sims
        base_stats .+= _simulate_moments(data, effects, pm, start_states, θ, seeds[s])
    end
    base_stats ./= n_sims

    for j in 1:n_params
        ε = 0.1 * max(1.0, abs(θ[j]))
        θ_plus = copy(θ)
        θ_plus[j] += ε

        plus_stats = zeros(n_params)
        for s in 1:n_sims
            plus_stats .+= _simulate_moments(data, effects, pm, start_states, θ_plus,
                                             seeds[s])
        end
        plus_stats ./= n_sims

        D[:, j] = (plus_stats .- base_stats) ./ ε
    end

    return D
end

#==============================================================================#
# Initial Values
#==============================================================================#

# Simple data-based initial value for a basic rate parameter: observed amount of
# change per actor, inflated to account for cancelling ministeps.
function default_basic_rate(data::SienaData, variable::Symbol, period::Int)
    dep = data.dependents[variable]
    period >= data.n_waves && return 1.0
    if dep isa DependentNetwork
        x0 = dep.networks[period]
        x1 = dep.networks[period + 1]
        dist = sum(abs.(x1 .- x0))
        n = size(x0, 1)
    else
        z0 = dep.values[period]
        z1 = dep.values[period + 1]
        dist = sum(abs.(z1 .- z0))
        n = length(z0)
    end
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
    siena07(data::SienaData, effects::SienaEffects;
           algorithm::SienaAlgorithm=SienaAlgorithm())

Estimate SAOM parameters using unconditional Method of Moments with Robbins-Monro
stochastic approximation. Equivalent to `siena07()` in RSiena.

Phases:
1. Derivative matrix estimation at the initial values, plus initial rough updates.
2. `n_subphases` subphases of Robbins-Monro updates with halving gain; the derivative
   matrix is re-estimated at the first and last subphase.
3. Simulations at the final estimates for convergence t-ratios
   (deviation / sd of the simulated statistic) and standard errors via
   ``D^{-1} \\Sigma D^{-T}``.

# Returns
- `SienaResult`: Estimation results (rate parameters first, then objective parameters)
"""
function siena07(data::SienaData, effects::SienaEffects;
                algorithm::SienaAlgorithm=SienaAlgorithm())
    data.n_waves >= 2 ||
        throw(ArgumentError("estimation requires at least 2 observation waves"))

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
    sim(θv) = _simulate_moments(data, effects, pm, start_states, θv, rand(rng, 1:10^8))

    #==========================================================================
    # Phase 1: derivative matrix at initial values + rough updates
    ==========================================================================#
    algorithm.verbose && println("\n--- Phase 1 ---")
    D = estimate_derivative_matrix(data, effects, θ, algorithm.derivative_sims, rng)
    D += 0.01 * I

    gain = algorithm.initial_gain
    for _ in 1:algorithm.phase1_iterations
        total_iterations += 1
        score = sim(θ) .- targets
        update_parameters!(θ, score, D, gain)
        _clamp_parameters!(θ, pm)
    end

    #==========================================================================
    # Phase 2: subphases with halving gain
    ==========================================================================#
    algorithm.verbose && println("\n--- Phase 2 ---")
    for subphase in 1:algorithm.n_subphases
        algorithm.verbose && println("  Subphase $subphase")
        if subphase == 1 || subphase == algorithm.n_subphases
            D = estimate_derivative_matrix(data, effects, θ, algorithm.derivative_sims, rng)
            D += 0.01 * I
        end
        gain = max(algorithm.initial_gain * 0.5^subphase, algorithm.min_gain)
        for _ in 1:algorithm.phase1_iterations
            total_iterations += 1
            score = sim(θ) .- targets
            update_parameters!(θ, score, D, gain)
            _clamp_parameters!(θ, pm)
        end
    end

    #==========================================================================
    # Phase 3: convergence check and standard errors at fixed θ
    ==========================================================================#
    algorithm.verbose && println("\n--- Phase 3 ---")
    n3 = algorithm.phase3_iterations
    phase3_stats = zeros(n3, n_params)
    for iter in 1:n3
        total_iterations += 1
        phase3_stats[iter, :] = sim(θ)
        if algorithm.verbose && iter % 200 == 0
            println("  Iteration $iter / $n3")
        end
    end

    mean_sim_stats = vec(mean(phase3_stats, dims=1))
    deviations = mean_sim_stats .- targets
    sd_stats = vec(std(phase3_stats, dims=1))

    # Convergence t-ratios: deviation / sd of the simulated statistic
    conv_stats = ConvergenceStats(n_params)
    update_convergence!(conv_stats, deviations, sd_stats)
    converged = is_converged(conv_stats, algorithm.convergence_threshold)

    # Covariance of the estimates: D^{-1} Σ D^{-T}
    Sigma = cov(phase3_stats)
    D_final = estimate_derivative_matrix(data, effects, θ, algorithm.derivative_sims, rng)
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
        total_iterations,
        rate_estimates,
        targets,
        mean_sim_stats
    )
end

#==============================================================================#
# Coefficient Access Functions
#==============================================================================#

"""
    coef(result::SienaResult)

Return the parameter estimates (rate parameters first, then objective parameters).
"""
coef(result::SienaResult) = result.estimates

"""
    stderror(result::SienaResult)

Return standard errors.
"""
stderror(result::SienaResult) = result.standard_errors

"""
    vcov(result::SienaResult)

Return covariance matrix.
"""
vcov(result::SienaResult) = result.covariance

"""
    confint(result::SienaResult; level::Float64=0.95)

Compute confidence intervals.
"""
function confint(result::SienaResult; level::Float64=0.95)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    lower = result.estimates .- z .* result.standard_errors
    upper = result.estimates .+ z .* result.standard_errors
    return hcat(lower, upper)
end
