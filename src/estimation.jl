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
    SienaScoreTest

Score-type tests (Schweinberger 2012) of effects held fixed at their initial value
with `include_effects!(...; fix=true, test=true)` — RSiena's `test=TRUE`. Returned
in `result.score_test` by [`fit_siena`](@ref).

The tested statistics are simulated in phase 3 together with the estimated ones.
With ``e`` the deviations (simulated minus observed), ``D`` the score-function
derivative matrix and ``Σ`` the covariance of the statistics, partitioned into the
estimated (1) and tested (2) parts, the test uses the deviation of the tested
statistics orthogonalised against the estimated ones,
``o = e_2 - D_{21} D_{11}^{-1} e_1``, whose variance is
``V = Σ_{22} - D_{21}D_{11}^{-1}Σ_{12} - Σ_{21}D_{11}^{-T}D_{21}' +
D_{21}D_{11}^{-1}Σ_{11}D_{11}^{-T}D_{21}'``, and ``o' V^{-1} o`` is referred to a
chi-square distribution — RSiena's `EvaluateTestStatistic`.

# Fields
- `names::Vector{String}`: Short names of the tested effects
- `fixed_values::Vector{Float64}`: The values they were held at
- `chisq::Float64`, `df::Int`, `p_value::Float64`: The joint test
- `effect_chisq::Vector{Float64}`, `effect_p_values::Vector{Float64}`: One-df tests
  of each effect alone (the other tested effects left out, as RSiena does)
- `one_sided::Vector{Float64}`: Signed one-df statistics (``\\approx N(0,1)``;
  positive when the parameter would be estimated above its fixed value)
- `one_step::Vector{Float64}`: One-step estimates of the tested parameters

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
include_effects!(effects, :friendship, [:transTrip]; fix=true, test=true)
fit = fit_siena(data, effects; rng=MersenneTwister(3),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
fit.score_test.p_value < 0.05       # transitivity is strongly needed
```
"""
struct SienaScoreTest
    names::Vector{String}
    fixed_values::Vector{Float64}
    chisq::Float64
    df::Int
    p_value::Float64
    effect_chisq::Vector{Float64}
    effect_p_values::Vector{Float64}
    one_sided::Vector{Float64}
    one_step::Vector{Float64}
end

"""
    SienaResult

Result of SAOM estimation.

# Fields
- `data::SienaData`: Independent snapshot of the fitted data for `gof(result)`.
- `derivative_matrix::Matrix{Float64}`: Raw final moment derivative, in parameter order.
- `phase3_cov::Matrix{Float64}`: Final simulated statistic covariance, in parameter order.
- `condition_number::Float64`: Two-norm condition number of the raw derivative matrix.
- `n_refinements::Int`: Newton updates performed after Robbins-Monro.
- `convergence_threshold::Float64`: Required per-parameter convergence threshold.
- `overall_convergence_threshold::Float64`: Required overall convergence threshold.
- `effects::SienaEffects`: The effects object that was estimated — with
  `algorithm.model_type != :standard` this holds only the effects of the *simulated*
  dependent variables (see [`restrict_effects`](@ref)), which is what θ is aligned with
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
- `n_iterations::Int`: Number of Robbins-Monro iterations used (phases 1 and 2;
  capped by `algorithm.max_iterations` when that budget is set)
- `rate_estimates::Dict{Symbol, Vector{Float64}}`: Basic rate estimates per variable and period
- `targets::Vector{Float64}`: Observed target statistics
- `simulated_means::Vector{Float64}`: Mean simulated statistics at the estimates (phase 3)
- `n_simulations_run::Int`: Total number of period simulations actually performed
  over all phases (Robbins-Monro iterations, derivative estimation and phase 3)
- `n_threads_used::Int`: Number of threads the independent simulations actually ran
  on — `Threads.nthreads()` with `algorithm.threaded = true`, and 1 with
  `algorithm.threaded = false`
- `model_type::Symbol`: The `algorithm.model_type` the fit ran under (`:standard`,
  `:networkonly` or `:behavioronly`); simulations from the result (e.g.
  [`siena_gof`](@ref)) freeze the same dependent variables
- `conditional::Bool`, `condvar`: Whether the fit was conditional, and on which
  variable
- `rate_standard_errors::Dict{Symbol, Vector{Float64}}`: Standard errors of the basic
  rates per variable and period (for the conditioning variable, RSiena's simulation
  rate × sd of the stopping time)
- `n_validations::Int`: Independent phase-3 batches run (1 + re-entries)
- `derivative_method::Symbol`: `:score` or `:finite_difference` — how the
  derivative matrix behind the standard errors was estimated
- `score_test::Union{Nothing, SienaScoreTest}`: Score-type tests of the effects
  included with `fix=true, test=true` (see [`SienaScoreTest`](@ref))
- `phase3_period_stats`, `phase3_period_scores` (simulations × parameters × periods)
  and `period_targets` (parameters × periods): the per-period phase-3 statistics,
  trajectory scores and targets, kept for [`siena_time_test`](@ref)

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = fit_siena(data, effects; rng=MersenneTwister(2),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
fit.rate_estimates[:friendship]        # conditional rate estimates, one per period
```
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
    n_simulations_run::Int
    n_threads_used::Int
    model_type::Symbol
    data::SienaData
    derivative_matrix::Matrix{Float64}
    phase3_cov::Matrix{Float64}
    condition_number::Float64
    n_refinements::Int
    convergence_threshold::Float64
    overall_convergence_threshold::Float64
    conditional::Bool
    condvar::Union{Nothing, Symbol}
    rate_standard_errors::Dict{Symbol, Vector{Float64}}
    n_validations::Int
    derivative_method::Symbol
    score_test::Union{Nothing, SienaScoreTest}
    phase3_period_stats::Array{Float64, 3}
    phase3_period_scores::Array{Float64, 3}
    period_targets::Matrix{Float64}
end

"""
    SienaConvergenceError <: Exception

The final independent simulation batch did not validate a Siena fit. It is raised
only when the fit was asked to fail on non-convergence with
`SienaAlgorithm(allow_unconverged=false)`; by default (`allow_unconverged=true`, as in
RSiena) an unconverged fit is returned with a warning and `converged = false`. The
diagnostic [`SienaResult`](@ref) is available as `error.result`; it is not a
converged estimate.
# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
# An impossible convergence threshold forces the error.
alg = SienaAlgorithm(verbose=false, phase3_iterations=100, allow_unconverged=false,
                     convergence_threshold=1e-6, revalidate_max=0, refine_max=0)
err = try
    fit_siena(data, effects; algorithm=alg, rng=MersenneTwister(1)); nothing
catch e
    e
end
err isa SienaConvergenceError && !err.result.converged      # true
```
"""
struct SienaConvergenceError <: Exception
    result::SienaResult
end

_convergence_summary(r::SienaResult) = string(
    "Siena fit did not converge: max |t-ratio| = ", maximum(abs, r.t_ratios),
    " (required < ", r.convergence_threshold, "), tconv_max = ", r.tconv_max,
    " (required < ", r.overall_convergence_threshold, "); derivative condition = ",
    r.condition_number, ".")

function Base.showerror(io::IO, err::SienaConvergenceError)
    print(io, _convergence_summary(err.result),
          " This error is raised because the algorithm sets allow_unconverged=false; ",
          "inspect error.result, or leave allow_unconverged at its default (true) to ",
          "get the unconverged fit back with a warning.")
end

function Base.show(io::IO, result::SienaResult)
    println(io, "SAOM Estimation Results")
    println(io, "=======================")
    println(io, "Converged: $(result.converged) " *
                "(max |t-ratio| = $(round(maximum(abs.(result.t_ratios)), digits=3)), " *
                "overall max convergence ratio = $(round(result.tconv_max, digits=3)))")
    result.diverged &&
        println(io, "WARNING: divergence detected (estimates hit the parameter clamp)")
    result.converged ||
        println(io, "WARNING: the fit did NOT converge; do not report these estimates " *
                    "(re-run with the result's estimates as initial values, or more " *
                    "iterations)")
    println(io, "Estimation: ", result.conditional ?
            "conditional on the observed change of :$(result.condvar)" : "unconditional")
    println(io, "Iterations: $(result.n_iterations)")
    println(io, "Simulations: $(result.n_simulations_run) " *
                "on $(result.n_threads_used) thread(s)")
    pm = build_param_map(result.effects)
    n_rate = n_free_rate_parameters(pm)

    if n_rate > 0 || result.conditional
        println(io)
        println(io, "Rate Parameters:")
        println(io, "----------------")
        for i in 1:n_rate
            @printf(io, "%-28s %8.4f (%6.4f)\n", result.parameter_names[i],
                    result.estimates[i], result.standard_errors[i])
        end
        if result.conditional
            v = result.condvar
            for (p, ρ) in enumerate(result.rate_estimates[v])
                @printf(io, "%-28s %8.4f (%6.4f)\n",
                        "Rate $v (period $p, cond.)", ρ,
                        result.rate_standard_errors[v][p])
            end
        end
    end

    println(io)
    println(io, "Objective Function Parameters:")
    println(io, "------------------------------")
    idx = (n_rate + 1):length(result.estimates)
    est = result.estimates[idx]
    se = result.standard_errors[idx]
    z, p = z_pvalues(est, se)
    print_coeftable(io, result.parameter_names[idx], est, se, p; z_values=z)
    if result.score_test !== nothing
        println(io)
        show(io, result.score_test)
    end
end

#==============================================================================#
# The shared result-metadata protocol (NetworkCore.jl `src/results.jl`)
#==============================================================================#
#
# `fit_metadata(fit)` collects these accessors, so what the SAOM fit actually
# did is programmatically inspectable rather than buried in `siena07`'s
# docstring.

estimand(::SienaResult) = :saom

"""
    objective(::SienaResult) -> Symbol

`:moment` — Method of Moments by Robbins-Monro stochastic approximation. This is
the ONLY estimator implemented (RSiena's Maximum Likelihood and Bayesian methods
are not), so no likelihood is ever evaluated.
"""
objective(::SienaResult) = :moment

"""
    is_exact(::SienaResult) -> Bool

Always `false`. Method of Moments solves simulated moment equations, not a
likelihood; the moments themselves are Monte-Carlo estimates from finitely many
simulated trajectories. There is no formula for which this collapses to exact ML.
"""
is_exact(::SienaResult) = false

"""
    se_method(::SienaResult) -> Symbol

`:sandwich` — the method-of-moments covariance `D⁻¹ Σ D⁻ᵀ`, where `Σ` is the
Monte-Carlo covariance of the phase-3 simulated statistics and `D` the estimated
derivative matrix (by default the score-function/likelihood-ratio estimator over
all phase-3 simulations; `algorithm.derivative_method = :finite_difference`
switches to finite differences, and conditional estimation always falls back to
them). Both factors are simulation estimates. No ridge is added. An unidentified
derivative matrix gives undefined standard errors — see `approximations(fit)`.
"""
se_method(::SienaResult) = :sandwich

"""
    missing_method(::SienaResult) -> Symbol

`:rejected`. Missing (`NA`) tie values are not handled: a `SienaData` carries no
unobserved-tie mask, so no dyad can be flagged as missing and none is imputed,
dropped or conditioned on. Structurally determined values (10/11) are a different
thing and *are* honoured.
"""
missing_method(::SienaResult) = :rejected

function approximations(result::SienaResult)
    out = [
        "Method of Moments by stochastic approximation: the moments are " *
        "Monte-Carlo estimates from simulated trajectories, so the estimates " *
        "carry Monte-Carlo error",
        "standard errors are D⁻¹ Σ D⁻ᵀ with BOTH factors estimated from the " *
        "phase-3 simulations; unidentified derivative matrices produce undefined " *
        "standard errors instead of an implicit ridge",
    ]
    result.model_type === :standard ||
        push!(out, "model_type = :$(result.model_type): the other dependent " *
                   "variables were FROZEN at their period-start values, and " *
                   "their effects are not in the estimated parameter vector")
    result.derivative_method === :finite_difference &&
        push!(out, "standard errors use a central finite-difference derivative " *
                   "with common random numbers (derivative_method = " *
                   ":finite_difference), not RSiena's score-function estimator")
    result.n_validations > 1 &&
        push!(out, "the first phase-3 validation failed and refinement was " *
                   "re-entered $(result.n_validations - 1) time(s); the reported " *
                   "diagnostics come from a fresh, independent batch")
    for (name, cov) in sort!(collect(result.data.covariates); by=first)
        cov.n_imputed > 0 &&
            push!(out, "covariate :$name had $(cov.n_imputed) missing value(s) " *
                       "imputed by the mean (missing=:mean); unlike RSiena, those " *
                       "actors still enter the target statistics")
    end
    cc = result.data.composition_change
    (cc !== nothing && !isempty(cc.changes)) &&
        push!(out, "composition change is handled per whole period: an actor " *
                   "contributes to a period only when present at both of its " *
                   "observation waves (RSiena's within-period joining/leaving " *
                   "times are not supported)")
    result.diverged &&
        pushfirst!(out, "divergence detected: a parameter hit the clamp during " *
                        "estimation and the estimates are unreliable")
    result.converged ||
        pushfirst!(out, "the fit did NOT meet the RSiena convergence standard " *
                        "(max |t-ratio| < $(result.convergence_threshold), " *
                        "tconv.max < $(result.overall_convergence_threshold)); " *
                        "achieved tconv.max = $(round(result.tconv_max, digits=3))")
    return out
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
    return vec(sum(_moment_statistics_by_period(data, pm, end_states, start_states);
                   dims=2))
end

# The moment vector of every period separately (n_free × n_periods); the moment
# statistics are their row sums.
function _moment_statistics_by_period(data::SienaData, pm::ParameterMap,
                                      end_states::Vector{NetworkState},
                                      start_states::Vector{NetworkState})
    n_periods = data.n_waves - 1
    stats = zeros(length(pm.free), n_periods)
    for p in 1:n_periods
        mixed = Dict{Symbol, NetworkState}()
        for (k, entry) in enumerate(pm.free)
            eff = entry.effect
            if eff isa RateEffect
                if !(eff isa BasicRateEffect) || eff.period == p
                    stats[k, p] += _rate_distance_statistic(eff, start_states[p],
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
                stats[k, p] += compute_statistic(eff, st, data)
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
# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]
w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
effects = get_effects(data)
include_effects!(effects, :advice, [:outdegree, :recip])
compute_target_statistics(data, effects)      # [rate distance, outdegree, recip]
```
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
# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]
w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
effects = get_effects(data)
include_effects!(effects, :advice, [:outdegree, :recip])
_, periods = simulate_saom(data, effects, [2.0, -1.0, 0.5]; rng=Siena.Random.Xoshiro(1))
compute_simulated_statistics(data, effects, periods)    # distance, outdegree, recip
```
"""
function compute_simulated_statistics(data::SienaData, effects::SienaEffects,
                                     results::Vector{SimulationResult})
    pm = build_param_map(effects)
    end_states = [r.final_state for r in results]
    return _moment_statistics(data, pm, end_states, _observed_start_states(data))
end

# Every simulation the estimator runs goes through here, so an optional atomic
# `counter` gives the exact number of simulations a fit performed (reported as
# `SienaResult.n_simulations_run`).
const SimCounter = Threads.Atomic{Int}

# Run `body(i)` for `i in 1:n`, multi-threaded when `threaded` and strictly serially
# on the calling thread otherwise. The loop bodies are independent (each writes only
# its own slot and draws from its own pre-seeded RNG), so the results are identical
# either way — `threaded` only controls *where* the work runs. The threaded branch
# goes through NetworkCore's `spawn_all`, so a simulation that throws reaches the
# caller as its own exception (an `ArgumentError`, say), exactly as in the serial
# branch, and not wrapped in a `CompositeException`/`TaskFailedException`. The
# work is split into a few strided chunks per thread to balance simulations of
# unequal length.
function _run_simulations!(body::F, n::Int, threaded::Bool) where {F}
    if threaded && n > 1
        n_chunks = min(n, 4 * Threads.nthreads())
        spawn_all(n_chunks) do c
            for i in c:n_chunks:n
                body(i)
            end
        end
    else
        for i in 1:n
            body(i)
        end
    end
    return nothing
end

# Number of threads the independent simulations of a fit actually run on.
_threads_used(alg::SienaAlgorithm) = alg.threaded ? Threads.nthreads() : 1

# One simulation -> moment vector (with cached pm/start states). If `scores` is
# given, the score function of the trajectory is accumulated into it. `condvar`/
# `cond_targets` switch to conditional simulation (see `simulate_saom`); if
# `times` is given, it receives the per-period elapsed simulation times (the
# stopping times, used for the conditional rate estimates); `variables` restricts the
# ministeps to the simulated variables (`model_type`). `counter`, if given, is
# incremented by one.
function _simulate_moments(data::SienaData, effects::SienaEffects, pm::ParameterMap,
                           start_states::Vector{NetworkState}, θ::Vector{Float64},
                           seed::Int;
                           scores::Union{Nothing, ScoreAccumulator}=nothing,
                           condvar::Union{Symbol, Nothing}=nothing,
                           cond_targets::Union{Nothing, Vector{Int}}=nothing,
                           variables::Union{Nothing, Vector{Symbol}}=nothing,
                           times::Union{Nothing, AbstractVector{Float64}}=nothing,
                           period_stats::Union{Nothing, AbstractMatrix{Float64}}=nothing,
                           counter::Union{Nothing, SimCounter}=nothing)
    counter === nothing || Threads.atomic_add!(counter, 1)
    _, results = simulate_saom(data, effects, θ; rng=MersenneTwister(seed), scores=scores,
                               condvar=condvar, cond_targets=cond_targets,
                               variables=variables, validate=false)
    if times !== nothing
        for (p, r) in enumerate(results)
            times[p] = r.final_state.time
        end
    end
    end_states = [r.final_state for r in results]
    by_period = _moment_statistics_by_period(data, pm, end_states, start_states)
    period_stats === nothing || (period_stats .= by_period)
    return vec(sum(by_period; dims=2))
end

# RSiena's score-function derivative estimator from per-period statistics and
# scores (n_sims × n_params × n_periods arrays): D = Σ_p cov(s_p, S_p). Statistics
# of one period depend only on that period's trajectory, so the cross-period
# covariances are zero in expectation; leaving them out removes their noise.
function _score_derivative(pstats::Array{Float64, 3}, pscores::Array{Float64, 3})
    n_params = size(pstats, 2)
    D = zeros(n_params, n_params)
    for p in axes(pstats, 3)
        D .+= cov(view(pstats, :, :, p), view(pscores, :, :, p))
    end
    return D
end

#==============================================================================#
# Robbins-Monro Update
#==============================================================================#

"""
    update_parameters!(θ::Vector{Float64}, score::AbstractVector{<:Real},
                      D::AbstractMatrix{<:Real}, gain::Real; max_step::Real=2.0)

Robbins-Monro update ``θ \\leftarrow θ - \\text{gain} \\cdot D^{-1} (\\bar s - s_{obs})``,
with the step capped at `max_step` in Euclidean norm. Falls back to the pseudoinverse
if `D` is singular. `D` may be any matrix type, dense or structured (a `Diagonal`,
for example): which container LinearAlgebra returns for a mixed broadcast is an
implementation detail that changes between Julia versions.
"""
function update_parameters!(θ::Vector{Float64}, score::AbstractVector{<:Real},
                           D::AbstractMatrix{<:Real}, gain::Real; max_step::Real=2.0)
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
                              θ::Vector{Float64}, n_sims::Int, rng::AbstractRNG;
                              central=true, step=0.1, ...)

Estimate ``D = \\partial E[s]/\\partial θ`` (``D_{kj} = \\partial E[s_k]/\\partial θ_j``)
by finite differences with **common random numbers**: every perturbed run reuses the
base run's simulation seeds, so the Monte-Carlo noise largely cancels in the
difference.

`central=true` (the default) takes central differences with step
``ε_j = `` `step` ``\\cdot \\max(1, |θ_j|)``; the first-order curvature bias cancels.
Before 0.2 this was a forward difference, whose bias inflated ``D`` by up to a third
and made standard errors 15–25 % too small. A much smaller step does not help: with
common random numbers the difference of two discrete trajectories has a variance of
order ``1/ε``, so the default `step = 0.1` balances bias against noise.
`central=false` keeps the cheaper forward difference, which the estimator uses only
as a phase-1/phase-2 preconditioner under `derivative_method = :finite_difference`,
where its bias changes step sizes but not the solution of the moment equations.

The simulations are independent (each is driven by its own seeded RNG) and run
multi-threaded unless `threaded=false`; every simulation writes to its own slot and
the results are reduced in a fixed order, so the estimate is identical to a
single-threaded run regardless of the number of threads.

# Example
```julia
using Siena, Random
w1 = [0 1 0 0; 0 0 1 0; 1 0 0 1; 0 1 0 0]; w2 = [0 1 1 0; 0 0 1 0; 0 0 0 1; 1 1 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
effects = get_effects(data)
include_effects!(effects, :advice, [:outdegree, :recip])
D = estimate_derivative_matrix(data, effects, [2.0, -1.0, 0.5], 20, MersenneTwister(1))
size(D)            # (3, 3)
```
"""
function estimate_derivative_matrix(data::SienaData, effects::SienaEffects,
                                   θ::Vector{Float64}, n_sims::Int, rng::AbstractRNG;
                                   condvar::Union{Symbol, Nothing}=nothing,
                                   cond_targets::Union{Nothing, Vector{Int}}=nothing,
                                   variables::Union{Nothing, Vector{Symbol}}=nothing,
                                   threaded::Bool=true,
                                   central::Bool=true,
                                   step::Float64=0.1,
                                   counter::Union{Nothing, SimCounter}=nothing)
    pm = build_param_map(effects)
    start_states = _observed_start_states(data)
    n_params = length(θ)
    D = zeros(n_params, n_params)

    seeds = rand(rng, 1:10^8, n_sims)

    # Per-simulation results land in their own row; the mean is accumulated
    # afterwards in simulation order, so it is thread-count independent.
    sim_stats = zeros(n_sims, n_params)
    function mean_stats!(dest::Vector{Float64}, θv::Vector{Float64})
        _run_simulations!(n_sims, threaded) do s
            sim_stats[s, :] = _simulate_moments(data, effects, pm, start_states, θv,
                                                seeds[s]; condvar=condvar,
                                                cond_targets=cond_targets,
                                                variables=variables,
                                                counter=counter)
        end
        fill!(dest, 0.0)
        for s in 1:n_sims
            dest .+= view(sim_stats, s, :)
        end
        dest ./= n_sims
        return dest
    end

    plus_stats = zeros(n_params)
    minus_stats = zeros(n_params)
    if central
        for j in 1:n_params
            ε = step * max(1.0, abs(θ[j]))
            θ_plus = copy(θ)
            θ_plus[j] += ε
            θ_minus = copy(θ)
            θ_minus[j] -= ε
            if pm.free[j].effect isa BasicRateEffect
                θ_minus[j] = max(θ_minus[j], 1e-3)
            end
            mean_stats!(plus_stats, θ_plus)
            mean_stats!(minus_stats, θ_minus)
            D[:, j] = (plus_stats .- minus_stats) ./ (θ_plus[j] - θ_minus[j])
        end
    else
        base_stats = mean_stats!(zeros(n_params), θ)
        for j in 1:n_params
            ε = step * max(1.0, abs(θ[j]))
            θ_plus = copy(θ)
            θ_plus[j] += ε
            mean_stats!(plus_stats, θ_plus)
            D[:, j] = (plus_stats .- base_stats) ./ ε
        end
    end

    return D
end

"""
    estimate_derivative_matrix_score(data::SienaData, effects::SienaEffects,
                                    θ::Vector{Float64}, n_sims::Int, rng::AbstractRNG)

Score-function (likelihood-ratio) estimator of ``D = \\partial E[s]/\\partial θ``
(Schweinberger & Snijders 2007), as used by RSiena: simulate `n_sims` trajectories
at θ, accumulate the score of each trajectory period by period (see
[`ScoreAccumulator`](@ref)), and estimate
``D = \\sum_m \\widehat{\\mathrm{cov}}(s_m, S_m)`` across the simulations, pairing
each period's statistics with that period's score only (RSiena's
`derivativeFromScoresAndDeviations`; the cross-period terms are zero in
expectation, and before 0.2 their noise was included).

Unlike [`estimate_derivative_matrix`](@ref) it needs no parameter perturbations, so
the cost is `n_sims` simulations regardless of the number of parameters. Pass
`condvar`/`cond_targets` for the conditional estimator's derivative (see
[`ScoreAccumulator`](@ref) for why the score function stays valid there).

# Example
```julia
using Siena, Random
w1 = [0 1 0 0; 0 0 1 0; 1 0 0 1; 0 1 0 0]; w2 = [0 1 1 0; 0 0 1 0; 0 0 0 1; 1 1 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
effects = get_effects(data)
include_effects!(effects, :advice, [:outdegree, :recip])
D = estimate_derivative_matrix_score(data, effects, [2.0, -1.0, 0.5], 200,
                                     MersenneTwister(1))
size(D)            # (3, 3)
```

The simulations are independent (one seeded RNG per simulation) and run
multi-threaded unless `threaded=false`; results are identical to a single-threaded
run regardless of the number of threads.
"""
function estimate_derivative_matrix_score(data::SienaData, effects::SienaEffects,
                                         θ::Vector{Float64}, n_sims::Int,
                                         rng::AbstractRNG;
                                         condvar::Union{Symbol, Nothing}=nothing,
                                         cond_targets::Union{Nothing, Vector{Int}}=nothing,
                                         variables::Union{Nothing, Vector{Symbol}}=nothing,
                                         threaded::Bool=true,
                                         counter::Union{Nothing, SimCounter}=nothing)
    pm = build_param_map(effects)
    start_states = _observed_start_states(data)
    batch = _phase3_batch(data, effects, pm, start_states, θ, n_sims, rng;
                          condvar=condvar, cond_targets=cond_targets,
                          variables=variables, threaded=threaded, counter=counter)
    return _score_derivative(batch.pstats, batch.pscores)
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

"""
    default_basic_rate(data::SienaData, variable::Symbol, period::Int) -> Float64

Data-based starting value of a basic rate parameter: the observed amount of change
of `variable` in `period` per actor, inflated by 1.5 for cancelling ministeps (at
least 0.5). [`get_effects`](@ref) uses it as the initial value of the basic rates,
and conditional estimation as the simulation rate of the conditioning variable.

# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]; w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
default_basic_rate(data, :advice, 1)        # 1.5 * 2 changes / 3 actors = 1.0
```
"""
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
              algorithm::SienaAlgorithm=SienaAlgorithm(), rng=algorithm.rng)

Estimate SAOM parameters by the Method of Moments with Robbins-Monro stochastic
approximation (Snijders 2001) — the estimator RSiena's `siena07()` uses by default.

# Relation to RSiena's `siena07()`
[`siena07`](@ref) is an alias of this function. The estimand is RSiena's: the same
moment equations (targets pinned against `RSiena:::getTargets` for every effect
offered under an RSiena name), conditional estimation by default when one dependent
variable is simulated (RSiena's `cond = NA`), the score-function derivative for the
standard errors, and the publication convergence standard. It is **not** a drop-in
numerical copy of `siena07()`:

- only the Method of Moments is implemented (no Maximum Likelihood, Bayesian or
  GMoM estimation, no multi-group data, no `siena08` meta-analysis);
- the Robbins-Monro schedule is simpler (fixed subphase length, no diagonalisation
  or Dolby option, finite-difference preconditioner in phases 1-2) and is followed
  by capped Newton refinement, so Monte-Carlo paths differ;
- the effects whose formula only approximates RSiena's carry a `Simple` suffix
  (e.g. [`BalanceSimpleEffect`](@ref)); interactions are not implemented;
- dyads whose structural status changes between waves do not get RSiena's
  correction (see [`DependentNetwork`](@ref)), missing (`NA`) tie values are
  refused, and composition change is wave-integer only (see
  [`add_composition_change!`](@ref)).

The checked-in golden fixtures compare the estimates and standard errors with
RSiena 1.6.6 for an unconditional and a conditional s50 network model, a
co-evolution model and an undirected model, within multiples of RSiena's own
seed-to-seed spread.

# Phases
1. A finite-difference derivative matrix at the initial values (the preconditioner),
   plus `phase1_iterations` Robbins-Monro updates.
2. `n_subphases` subphases of Robbins-Monro updates with halving gain; each returns
   the Polyak-Ruppert average of its iterates.
3. Up to `refine_max` capped Newton updates, each from `2 * phase3_iterations`
   simulations with the score-function derivative, until the moment deviations are
   below half the convergence thresholds.
4. Phase 3: `phase3_iterations` fresh simulations at the final estimate give the
   convergence t-ratios, `tconv.max`, and the standard errors ``D^{-1} Σ D^{-T}``
   (``D`` by the score function, `algorithm.derivative_method = :score`, or by
   central finite differences). If this independent batch fails the convergence
   standard, it drives one more Newton step and a fresh batch validates again, up to
   `revalidate_max` times; the reported diagnostics always come from a batch that was
   not used to choose the estimate.

Convergence follows the RSiena publication standard: all per-parameter |t-ratios|
below `algorithm.convergence_threshold` (0.1) *and* `tconv.max` below
`algorithm.overall_convergence_threshold` (0.25). An unconverged fit is returned with
a warning (`allow_unconverged=true`, the default, as RSiena does) — `show` and
`approximations(fit)` flag it and its estimates should not be reported — or
throws [`SienaConvergenceError`](@ref) with `allow_unconverged=false`.

# Restricted models (`algorithm.model_type`)
With `model_type = :networkonly` (or `:behavioronly`) only the network (behavior)
dependent variables take ministeps; the others are *frozen* at their period-start
values for the whole simulated period. A frozen variable stays in the simulation
state and is still read by the effects of the simulated variables, but it never
changes — so its own rate and objective effects have constant moments, are not
identified, and are dropped from the estimated parameter vector (see
[`simulated_variables`](@ref) and [`restrict_effects`](@ref)). `algorithm.condvar`
must be one of the simulated variables. This is **not** RSiena's `modelType` (see the
warning in [`SienaAlgorithm`](@ref)).

# Conditional estimation
Conditional estimation (RSiena's `cond=TRUE`, and the default when one dependent
variable is simulated and the data have no composition change) conditions on the observed amount of change of one dependent
variable (`algorithm.condvar`, defaulting to the only one, RSiena's `condvarno=1`):
every simulated period runs until that variable's distance from the period-start
observation reaches the observed distance, instead of until time 1. The conditioned
variable's basic rates leave the parameter vector and are estimated afterwards from
the phase-3 stopping times (simulation rate × mean stopping time), with RSiena's
standard error (simulation rate × standard deviation of the stopping time); both are
reported in `rate_estimates` and `rate_standard_errors`. The basic rates of the
other dependent variables are rescaled to the same time unit, as RSiena does. The
derivative matrix is the score-function estimator in this case too (the conditional
stopping rule is a stopping time of the simulated chain).

# Score-type tests
Effects included with `fix=true, test=true` are held at their initial value and
tested with the generalised score-type test of Schweinberger (2012), RSiena's
`test=TRUE`: their statistics are simulated in phase 3 alongside the estimated ones,
and `result.score_test` (a [`SienaScoreTest`](@ref)) holds the joint and per-effect
chi-square tests, the signed one-df statistics and one-step estimates.

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_nodeset!(data, NodeSet(50))
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
result = fit_siena(data, effects; rng=MersenneTwister(1),
                   algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
result.conditional        # true: one dependent network, RSiena's default
```

# Returns
- [`SienaResult`](@ref): estimates (rate parameters first, then objective parameters)
"""
function fit_siena(data::SienaData, effects::SienaEffects;
                   algorithm::SienaAlgorithm=SienaAlgorithm(),
                   rng::AbstractRNG=algorithm.rng)
    data.n_waves >= 2 ||
        throw(ArgumentError("estimation requires at least 2 observation waves"))

    # `model_type`: the dependent variables that co-evolve. The others are frozen
    # (no ministeps, values held at the period start) but stay in the state and
    # readable by the effects of the simulated variables. A frozen variable's own
    # rate and objective effects are unidentified -- its moments are constant -- so
    # they leave the model: the restricted effects object is what the estimator
    # (parameter map, targets, simulations, result) works with from here on.
    sim_vars = simulated_variables(data, algorithm.model_type)
    restricted = length(sim_vars) < length(data.dependents)
    effects = deepcopy(restrict_effects(effects, sim_vars))
    validate_effects(data, effects)
    sim_variables = restricted ? sim_vars : nothing

    # RSiena's cond = NA: conditional exactly when one dependent variable is
    # simulated.
    # (RSiena never conditions when the data have composition change.)
    has_cc = data.composition_change !== nothing &&
             !isempty(data.composition_change.changes)
    conditional = algorithm.conditional === nothing ?
                  (length(sim_vars) == 1 && !has_cc) : algorithm.conditional

    # Conditional estimation setup: fix the conditioned variable's basic rate
    # entries (they are determined by the conditioning, not by the moment
    # equations) and compute the per-period target distances.
    condvar = nothing
    cond_targets = nothing
    cond_entries = EffectEntry[]
    if conditional
        condvar = algorithm.condvar
        if condvar === nothing
            length(sim_vars) == 1 ||
                throw(ArgumentError("conditional estimation with several dependent " *
                                    "variables requires algorithm.condvar"))
            condvar = first(sim_vars)
        end
        haskey(data.dependents, condvar) ||
            throw(ArgumentError("unknown conditioning variable :$condvar"))
        condvar in sim_vars ||
            throw(ArgumentError("the conditioning variable :$condvar is not simulated " *
                                "under model_type = :$(algorithm.model_type) (it is " *
                                "frozen, so it cannot condition the simulation); " *
                                "condition on one of " *
                                join((":$v" for v in sim_vars), ", ")))
        cond_targets = [_observed_distance(data, condvar, p)
                        for p in 1:(data.n_waves - 1)]
        all(>(0), cond_targets) ||
            throw(ArgumentError("conditional estimation requires observed change " *
                                "in :$condvar in every period" *
                                (algorithm.conditional === nothing ?
                                 " (it is the default for one dependent variable; " *
                                 "pass conditional=false for unconditional estimation)" :
                                 "")))
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
    if isempty(pm.free)
        throw(ArgumentError(restricted ?
            "the model has no free parameters left after the model_type = " *
            ":$(algorithm.model_type) restriction: only the effects of " *
            join((":$v" for v in sim_vars), ", ") * " are estimated (the effects of " *
            "the frozen variables are dropped, they are not identified)" :
            "the model has no free parameters"))
    end
    for entry in effects.effects
        entry.include || continue
        if !entry.fix && effect_type(entry.effect) in (:endow, :creation)
            throw(ArgumentError("endowment/creation effects are not yet supported in " *
                                "estimation: $(entry.shortname) (fix them or exclude them)"))
        end
        entry.test && !entry.fix && throw(ArgumentError(
            "effect $(entry.shortname) has test=true but is not fixed; a score-type " *
            "test needs fix=true (see include_effects!)"))
    end

    start_states = _observed_start_states(data)

    θ = _initial_parameters(data, pm)
    targets = _moment_statistics(data, pm, _observed_end_states(data), start_states)
    n_params = length(θ)

    n_threads = _threads_used(algorithm)
    sim_counter = SimCounter(0)

    if algorithm.verbose
        println("Starting SAOM estimation")
        println("Free parameters: $n_params " *
                "($(n_free_rate_parameters(pm)) rate, " *
                "$(n_params - n_free_rate_parameters(pm)) objective)")
        println("Estimation: ", condvar === nothing ? "unconditional" :
                "conditional on the observed change of :$condvar")
        println("Simulations per Robbins-Monro iteration: $(algorithm.n_simulations)")
        restricted &&
            println("Model type: :$(algorithm.model_type) — simulating " *
                    join((":$v" for v in sim_vars), ", ") *
                    " (the other dependent variables stay fixed)")
        println("Execution: $(algorithm.threaded ? "threaded" : "serial") " *
                "on $n_threads thread(s)")
        algorithm.max_iterations === nothing ||
            println("Robbins-Monro iteration budget: $(algorithm.max_iterations)")
    end

    total_iterations = 0
    diverged = false

    # One Robbins-Monro step's simulated moment vector: the average of
    # `algorithm.n_simulations` independent simulations at θ (RSiena's default is a
    # single simulation per iteration; averaging more reduces the update noise).
    function sim(θv)
        seeds = rand(rng, 1:10^8, algorithm.n_simulations)
        draws = zeros(algorithm.n_simulations, n_params)
        _run_simulations!(algorithm.n_simulations, algorithm.threaded) do i
            draws[i, :] = _simulate_moments(data, effects, pm, start_states, θv, seeds[i];
                                            condvar=condvar, cond_targets=cond_targets,
                                            variables=sim_variables, counter=sim_counter)
        end
        return vec(mean(draws, dims=1))
    end

    # Robbins-Monro iteration budget (`algorithm.max_iterations`): the number of
    # phase-1/phase-2 iterations still allowed. Phase-3 simulations are not
    # iterations and do not draw on it.
    budget_left() = algorithm.max_iterations === nothing ? typemax(Int) :
                    algorithm.max_iterations - total_iterations
    budget_hit = false
    function note_budget!()
        budget_hit && return
        budget_hit = true
        @warn "reached the Robbins-Monro iteration budget " *
              "(max_iterations = $(algorithm.max_iterations)) before finishing " *
              "phase 2; the remaining phase-1/phase-2 iterations are skipped and " *
              "the estimates are likely unconverged"
    end

    function clamp_tracked!()
        if _clamp_parameters!(θ, pm) && !diverged
            diverged = true
            @warn "parameter estimate hit the divergence clamp " *
                  "(|θ| = 10 for objective parameters); the model is likely " *
                  "diverging and the estimates are unreliable"
        end
    end

    # Phase-1/phase-2 preconditioner, as in RSiena: the score-function derivative
    # from `derivative_sims` simulations (a cheap forward finite difference with
    # common random numbers under derivative_method = :finite_difference), mixed
    # with its diagonal by `algorithm.diagonalize` (RSiena's `diagonalize`, 0.2 for
    # MoM) so that a noisy off-diagonal does not throw the iterates around. It
    # changes the Robbins-Monro step sizes, not the solution; inference never uses it.
    function preconditioner()
        Dp = if algorithm.derivative_method == :score
            estimate_derivative_matrix_score(data, effects, θ, algorithm.derivative_sims,
                rng; condvar=condvar, cond_targets=cond_targets,
                variables=sim_variables, threaded=algorithm.threaded,
                counter=sim_counter)
        else
            estimate_derivative_matrix(data, effects, θ, algorithm.derivative_sims, rng;
                condvar=condvar, cond_targets=cond_targets, variables=sim_variables,
                threaded=algorithm.threaded, central=false, counter=sim_counter)
        end
        # Built element-wise into a dense matrix: the container type of a broadcast
        # mixing a Matrix with a Diagonal is not stable across Julia versions.
        w = algorithm.diagonalize
        P = Matrix{Float64}(undef, size(Dp))
        for j in axes(Dp, 2), i in axes(Dp, 1)
            P[i, j] = i == j ? Dp[i, j] : (1 - w) * Dp[i, j]
        end
        return P
    end

    #==========================================================================
    # Phase 1: derivative matrix at initial values + rough updates
    ==========================================================================#
    algorithm.verbose && println("\n--- Phase 1 ---")
    D = preconditioner()

    gain = algorithm.initial_gain
    n_phase1 = min(algorithm.phase1_iterations, budget_left())
    n_phase1 < algorithm.phase1_iterations && note_budget!()
    for _ in 1:n_phase1
        total_iterations += 1
        score = sim(θ) .- targets
        update_parameters!(θ, score, D, gain)
        clamp_tracked!()
    end
    algorithm.verbose && println("  θ = ", round.(θ; digits=4))

    #==========================================================================
    # Phase 2: subphases with halving gain and Polyak-Ruppert averaging
    ==========================================================================#
    algorithm.verbose && println("\n--- Phase 2 ---")
    for subphase in 1:algorithm.n_subphases
        # The iteration budget is exhausted: skip the rest of phase 2 and go to
        # phase 3 with the parameters reached so far.
        if budget_left() <= 0
            note_budget!()
            break
        end
        algorithm.verbose && println("  Subphase $subphase")
        if subphase == 1 || subphase == algorithm.n_subphases
            D = preconditioner()
        end
        gain = max(algorithm.initial_gain * 0.5^subphase, algorithm.min_gain)
        # Polyak-Ruppert averaging: the subphase result is the average of the θ
        # iterates over the subphase (excluding a short warm start), not the last
        # iterate -- this suppresses most of the Robbins-Monro Monte-Carlo noise.
        n_iter = min(algorithm.phase1_iterations, budget_left())
        n_iter < algorithm.phase1_iterations && note_budget!()
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
        algorithm.verbose && println("    θ = ", round.(θ; digits=4))
    end

    # Newton refinement solves simulated moment equations; it is not a likelihood
    # maximization, so NetworkCore.newton_fit's objective line search does not apply.
    # The derivative is the score-function estimator from the same batch (valid for
    # conditional simulation too). Stop using the refinement batch only: final
    # validation draws below are never used to select the estimate.
    n_refinements = 0
    for iteration in 1:algorithm.refine_max
        stats, D_ref = _refinement_batch(data, effects, pm, start_states, θ,
            2 * algorithm.phase3_iterations, rng; condvar=condvar,
            cond_targets=cond_targets, variables=sim_variables,
            threaded=algorithm.threaded, counter=sim_counter)
        dev = vec(mean(stats, dims=1)) .- targets
        sigma = cov(stats)
        cs = ConvergenceStats(n_params)
        update_convergence!(cs, dev, sqrt.(max.(diag(sigma), 0.0)))
        cs.tconv_max = sqrt(max(_quad_form_inv(dev, sigma), 0.0))
        algorithm.verbose &&
            println("  Refinement batch $iteration: max |t| = " *
                    "$(round(cs.max_t_ratio, digits=3)), tconv.max = " *
                    "$(round(cs.tconv_max, digits=3))")
        # A stricter training threshold leaves room for independent validation noise.
        if is_converged(cs, algorithm.convergence_threshold / 2,
                        algorithm.overall_convergence_threshold / 2)
            break
        end
        update_parameters!(θ, dev, D_ref, 1.0)
        clamp_tracked!()
        n_refinements += 1
    end

    #==========================================================================
    # Phase 3: convergence check and standard errors at fixed θ, re-entering
    # refinement when the independent validation batch fails
    ==========================================================================#
    algorithm.verbose && println("\n--- Phase 3 ---")
    n3 = algorithm.phase3_iterations
    use_score = algorithm.derivative_method == :score

    # Score-type tests: the tested (fixed) effects are simulated as free parameters
    # held at their fixed values, so that their statistics and scores are collected
    # with those of the estimated parameters. The simulation itself is unchanged.
    tested = [k for (k, e) in enumerate(effects.effects) if e.include && e.fix && e.test]
    ext_effects, pm_ext, main_idx, test_idx = if isempty(tested)
        effects, pm, collect(1:n_params), Int[]
    else
        _extended_model(effects, pm, tested)
    end
    targets_ext = isempty(tested) ? targets :
        _moment_statistics(data, pm_ext, _observed_end_states(data), start_states)
    ψ0 = [pm_ext.free[i].initial_value for i in test_idx]
    θ_ext(θv) = (t = zeros(length(pm_ext.free)); t[main_idx] = θv; t[test_idx] = ψ0; t)

    phase3_stats = zeros(0, 0)
    phase3_pstats = phase3_pscores = zeros(0, 0, 0)
    D_score = D_ext = Sigma_ext = zeros(0, 0)
    dev_ext = Float64[]
    phase3_times = nothing
    deviations = mean_sim_stats = sd_stats = Float64[]
    Sigma = zeros(0, 0)
    conv_stats = ConvergenceStats(n_params)
    converged = false
    n_validations = 0
    while true
        n_validations += 1
        algorithm.verbose &&
            println("  Validation batch $n_validations: $n3 simulations on " *
                    "$n_threads thread(s)")
        batch = _phase3_batch(data, ext_effects, pm_ext, start_states, θ_ext(θ), n3,
            rng; condvar=condvar, cond_targets=cond_targets, variables=sim_variables,
            threaded=algorithm.threaded, counter=sim_counter)
        phase3_times = batch.times
        phase3_pstats = batch.pstats[:, main_idx, :]
        phase3_pscores = batch.pscores[:, main_idx, :]
        phase3_stats = batch.stats[:, main_idx]
        D_ext = _score_derivative(batch.pstats, batch.pscores)
        D_score = D_ext[main_idx, main_idx]
        Sigma_ext = cov(batch.stats)
        dev_ext = vec(mean(batch.stats, dims=1)) .- targets_ext

        mean_sim_stats = vec(mean(phase3_stats, dims=1))
        deviations = mean_sim_stats .- targets
        sd_stats = vec(std(phase3_stats, dims=1))
        Sigma = cov(phase3_stats)

        # Convergence: per-parameter t-ratios (deviation / sd of the simulated
        # statistic) and the overall maximum convergence ratio
        # tconv.max = sqrt(ē' Σ⁻¹ ē), the maximum t-ratio over all linear
        # combinations of the statistics (RSiena).
        conv_stats = ConvergenceStats(n_params)
        update_convergence!(conv_stats, deviations, sd_stats)
        conv_stats.tconv_max = sqrt(max(_quad_form_inv(deviations, Sigma), 0.0))
        converged = is_converged(conv_stats, algorithm.convergence_threshold,
                                 algorithm.overall_convergence_threshold)
        (converged || diverged || n_validations > algorithm.revalidate_max) && break
        # Re-enter refinement: the failed batch drives one Newton step, and the next
        # iteration validates the new estimate on fresh simulations.
        algorithm.verbose &&
            println("  validation failed (max |t| = " *
                    "$(round(conv_stats.max_t_ratio, digits=3)), tconv.max = " *
                    "$(round(conv_stats.tconv_max, digits=3))); refining again")
        update_parameters!(θ, deviations, D_score, 1.0)
        clamp_tracked!()
        n_refinements += 1
    end

    # Derivative matrix for the standard errors: score-function (likelihood-ratio)
    # estimator over all phase-3 simulations (RSiena) or central finite
    # differences. Preserve the raw estimate; no ridge is added.
    D_final = use_score ? D_score :
              estimate_derivative_matrix(data, effects, θ, algorithm.derivative_sims,
                                         rng; condvar=condvar,
                                         cond_targets=cond_targets,
                                         variables=sim_variables,
                                         threaded=algorithm.threaded,
                                         counter=sim_counter)

    # Preserve the raw derivative and report conditioning. A pseudoinverse can
    # stabilize parameter updates, but cannot identify an unidentified parameter:
    # reporting finite/zero SEs from it would be scientifically misleading.
    condition_number = cond(D_final)
    identified = isfinite(condition_number) && condition_number < 1e10
    param_cov = if identified
        left = D_final \ Sigma
        Matrix(Symmetric((D_final \ left')'))
    else
        @warn "derivative matrix is singular or ill-conditioned; standard errors are undefined" condition_number
        fill(NaN, n_params, n_params)
    end
    converged &= identified && !diverged

    # Score-type tests of the fixed effects (RSiena's test = TRUE).
    score_test = isempty(tested) ? nothing :
        _score_test(D_ext, Sigma_ext, dev_ext, main_idx, test_idx, ψ0,
                    String[pm_ext.free[i].shortname for i in test_idx])

    if algorithm.verbose
        println("\n--- Results ---")
        println("Converged: $converged")
        println("Max |t-ratio|: $(round(conv_stats.max_t_ratio, digits=3))")
        println("Overall max convergence ratio: $(round(conv_stats.tconv_max, digits=3))")
        diverged && println("WARNING: divergence detected during estimation")
    end

    θ_est = copy(θ)
    rate_se = Dict{Symbol, Vector{Float64}}()
    # Conditional rate estimates: with simulation rate ρ the stopping time of a
    # trajectory scales as 1/ρ, so the rate that makes the expected stopping time
    # equal to the unit period length is ρ_sim * E[T] (Snijders 2001), with RSiena's
    # standard error ρ_sim * sd(T). The basic rates of the other dependent variables
    # were estimated in the same simulated time unit and are rescaled by E[T] too
    # (RSiena's `theta[posj] * rate`). The estimate is stored on the fixed entries
    # so that `basic_rate`/`rate_estimates` and any later simulation pick it up.
    if condvar !== nothing
        mean_times = vec(mean(phase3_times, dims=1))
        sd_times = vec(std(phase3_times, dims=1))
        cond_se = zeros(data.n_waves - 1)
        for entry in cond_entries
            p = entry.effect.period
            cond_se[p] = entry.initial_value * sd_times[p]
            entry.initial_value *= mean_times[p]
        end
        rate_se[condvar] = cond_se
        for (i, entry) in enumerate(pm.free)
            eff = entry.effect
            eff isa BasicRateEffect || continue
            f = mean_times[eff.period]
            θ_est[i] *= f
            param_cov[i, :] .*= f
            param_cov[:, i] .*= f
        end
    end
    se = sqrt.(max.(diag(param_cov), 0.0))

    # Basic rate estimates (and standard errors) per variable and period
    rate_estimates = Dict{Symbol, Vector{Float64}}()
    for (name, dep) in data.dependents
        if any(e -> e.effect isa BasicRateEffect && target_variable(e.effect) == name,
               pm.rate_entries)
            rate_estimates[name] = [basic_rate(pm, θ_est, name, p)
                                    for p in 1:(data.n_waves - 1)]
            if !haskey(rate_se, name)
                rate_se[name] = [let i = findfirst(e -> e.effect isa BasicRateEffect &&
                                                   e.effect.variable == name &&
                                                   e.effect.period == p, pm.free)
                                     i === nothing ? NaN : se[i]
                                 end for p in 1:(data.n_waves - 1)]
            end
        end
    end

    result = SienaResult(
        effects,
        parameter_names(effects),
        θ_est,
        se,
        conv_stats.t_ratios,
        param_cov,
        converged,
        conv_stats.tconv_max,
        diverged,
        total_iterations,
        rate_estimates,
        targets,
        mean_sim_stats,
        sim_counter[],
        n_threads,
        algorithm.model_type,
        deepcopy(data),
        D_final,
        Sigma,
        condition_number,
        n_refinements,
        algorithm.convergence_threshold,
        algorithm.overall_convergence_threshold,
        condvar !== nothing,
        condvar,
        rate_se,
        n_validations,
        use_score ? :score : :finite_difference,
        score_test,
        phase3_pstats,
        phase3_pscores,
        _moment_statistics_by_period(data, pm, _observed_end_states(data), start_states)
    )
    if !converged
        algorithm.allow_unconverged || throw(SienaConvergenceError(result))
        @warn _convergence_summary(result) * " The unconverged fit is returned " *
              "(converged = false); its estimates should not be reported. " *
              "SienaAlgorithm(allow_unconverged=false) raises SienaConvergenceError instead."
    end
    return result
end

"""
    siena07(data::SienaData, effects::SienaEffects;
            algorithm::SienaAlgorithm=SienaAlgorithm())

Alias for [`fit_siena`](@ref), keeping the RSiena function name for the same step of
the workflow. See [`fit_siena`](@ref) for how far the correspondence with RSiena's
`siena07()` goes — the estimation method is the same, but the two are not
numerically interchangeable in general.
# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = siena07(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
fit.converged
```
"""
const siena07 = fit_siena

"""
    fit_siena(algorithm::SienaAlgorithm, data::SienaData, effects::SienaEffects; rng)
    fit_siena(algorithm::SienaAlgorithm; data::SienaData, effects::SienaEffects, rng)

RSiena's argument order: `siena07(myalgorithm, data = mydata, effects = myeff)` puts
the algorithm first, and both spellings of it are accepted (with `siena07` too). The
fit is exactly `fit_siena(data, effects; algorithm=algorithm, rng=rng)`. Passing the
algorithm both positionally and as `algorithm=` is an `ArgumentError`, as is any
other argument order (the message names the accepted ones).

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
alg = SienaAlgorithm(verbose=false, phase3_iterations=200)
fit = siena07(alg, data, effects; rng=MersenneTwister(1))           # RSiena's order
fit2 = siena07(alg; data=data, effects=effects, rng=MersenneTwister(1))
fit.estimates == fit2.estimates                                      # true
```
"""
function fit_siena(algorithm::SienaAlgorithm, data::SienaData, effects::SienaEffects;
                   kwargs...)
    haskey(kwargs, :algorithm) && throw(ArgumentError(
        "the algorithm was given twice, positionally (RSiena's order) and as " *
        "`algorithm=`; pass it once"))
    return fit_siena(data, effects; algorithm=algorithm, kwargs...)
end

function fit_siena(algorithm::SienaAlgorithm; data=nothing, effects=nothing, kwargs...)
    data isa SienaData && effects isa SienaEffects || throw(ArgumentError(
        "fit_siena/siena07 with the algorithm first (RSiena's order) needs " *
        "`data=` (a SienaData) and `effects=` (a SienaEffects), as in R's " *
        "siena07(alg, data = mydata, effects = myeff); got data::" *
        "$(typeof(data)) and effects::$(typeof(effects))"))
    return fit_siena(algorithm, data, effects; kwargs...)
end

# Any other argument list: say which orders exist instead of a bare MethodError.
function fit_siena(args...; kwargs...)
    throw(ArgumentError(
        "fit_siena/siena07 takes (data::SienaData, effects::SienaEffects; " *
        "algorithm=SienaAlgorithm(), rng=...) or, in RSiena's order, " *
        "(algorithm::SienaAlgorithm, data::SienaData, effects::SienaEffects; rng=...); " *
        "got positional arguments of type (" * join(string.(typeof.(args)), ", ") * ")"))
end

# A singular covariance must not hide a discrepancy in its null space.
function _quad_form_inv(x::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real})
    n = length(x)
    size(Σ) == (n, n) || throw(DimensionMismatch("covariance must match the deviation vector"))
    all(isfinite, Σ) && all(isfinite, x) || return Inf
    # Work in per-moment standard deviation units. Rank/null-space tolerances
    # on the raw covariance change their meaning when a covariate's units change.
    active = Int[]
    for i in 1:n
        variance = Σ[i, i]
        variance < 0 && return Inf
        if variance == 0
            # A truly deterministic statistic permits exactly zero discrepancy.
            x[i] == 0 || return Inf
            all(iszero, view(Σ, i, :)) && all(iszero, view(Σ, :, i)) || return Inf
        else
            push!(active, i)
        end
    end
    isempty(active) && return 0.0
    scales = sqrt.(diag(Σ)[active])
    y = x[active] ./ scales
    all(isfinite, y) || return Inf
    k = length(active)
    correlation = Matrix{Float64}(undef, k, k)
    for j in 1:k, i in 1:k
        # Two divisions avoid overflowing or underflowing scales[i]*scales[j].
        correlation[i, j] = (Σ[active[i], active[j]] / scales[i]) / scales[j]
    end
    all(isfinite, correlation) || return Inf
    f = eigen(Symmetric(correlation))
    cutoff = maximum(abs, f.values) * k * eps(Float64)
    minimum(f.values) < -cutoff && return Inf
    z = f.vectors' * y
    null_tolerance = sqrt(eps(Float64)) * norm(y)
    total = 0.0
    for i in eachindex(z)
        if f.values[i] > cutoff
            total += abs2(z[i]) / f.values[i]
        elseif abs(z[i]) > null_tolerance
            return Inf
        end
    end
    return total
end

# Score-function refinement batch (unconditional or conditional simulation). These
# observations never become phase 3; keeping fresh final draws protects the
# convergence/SE diagnostics from selection.
function _refinement_batch(data, effects, pm, start_states, θ, n, rng;
                           condvar=nothing, cond_targets=nothing, variables=nothing,
                           threaded=true, counter=nothing)
    batch = _phase3_batch(data, effects, pm, start_states, θ, n, rng;
                          condvar=condvar, cond_targets=cond_targets,
                          variables=variables, threaded=threaded, counter=counter)
    return batch.stats, _score_derivative(batch.pstats, batch.pscores)
end

# `n` independent simulations at θ: moment statistics (total and per period),
# trajectory scores (total and per period), and -- for conditional simulation --
# the per-period stopping times. One seed per simulation is drawn up front from
# `rng`, so results do not depend on the thread count.
function _phase3_batch(data, effects, pm, start_states, θ, n, rng;
                       condvar=nothing, cond_targets=nothing, variables=nothing,
                       threaded=true, counter=nothing)
    p = length(θ)
    n_periods = data.n_waves - 1
    stats = zeros(n, p)
    scores = zeros(n, p)
    pstats = zeros(n, p, n_periods)
    pscores = zeros(n, p, n_periods)
    times = condvar === nothing ? nothing : zeros(n, n_periods)
    seeds = [rand(rng, 1:10^8) for _ in 1:n]
    _run_simulations!(n, threaded) do i
        acc = ScoreAccumulator(pm)
        t = times === nothing ? nothing : view(times, i, :)
        stats[i, :] = _simulate_moments(data, effects, pm, start_states, θ, seeds[i];
            scores=acc, condvar=condvar, cond_targets=cond_targets,
            variables=variables, times=t, period_stats=view(pstats, i, :, :),
            counter=counter)
        scores[i, :] = acc.scores
        for q in 1:n_periods
            pscores[i, :, q] = acc.period_scores[q]
        end
    end
    return (; stats, scores, pstats, pscores, times)
end

# The model with the tested (fixed) effects freed, for phase 3 of a fit with
# score-type tests: returns the extended effects, its parameter map, the positions
# of the original free parameters in it, and the positions of the tested effects.
function _extended_model(effects::SienaEffects, pm::ParameterMap, tested::Vector{Int})
    ext = deepcopy(effects)
    for k in tested
        ext.effects[k].fix = false
    end
    pm_ext = build_param_map(ext)
    pos(entries, e) = findfirst(x -> x === e, entries)
    k_of_ext = [pos(ext.effects, e) for e in pm_ext.free]
    k_of_main = [pos(effects.effects, e) for e in pm.free]
    main_idx = [findfirst(==(k), k_of_ext) for k in k_of_main]
    test_idx = [findfirst(==(k), k_of_ext) for k in tested]
    return ext, pm_ext, main_idx, test_idx
end


# RSiena's EvaluateTestStatistic (method of moments) for the parameters in `est`
# (estimated) and `tst` (tested): returns (chisq, one_sided).
function _score_statistic(D, Σ, dev, est::Vector{Int}, tst::Vector{Int})
    d11 = D[est, est]; d21 = D[tst, est]
    s11 = Σ[est, est]; s12 = Σ[est, tst]; s21 = Σ[tst, est]; s22 = Σ[tst, tst]
    id11 = try
        inv(d11)
    catch err
        err isa Union{SingularException, LinearAlgebra.LAPACKException} || rethrow()
        return NaN, NaN
    end
    rg = d21 * id11
    ov = dev[tst] .- rg * dev[est]
    v2 = s21 .- rg * s11
    v9 = s22 .- rg * s12 .- v2 * id11' * d21'
    v9 = (v9 + v9') / 2
    chisq = try
        max(dot(ov, v9 \ ov), 0.0)
    catch err
        err isa Union{SingularException, LinearAlgebra.LAPACKException} || rethrow()
        NaN
    end
    one_sided = length(tst) == 1 && v9[1, 1] > 0 ? -ov[1] / sqrt(v9[1, 1]) : NaN
    return chisq, one_sided
end

function _score_test(D, Σ, dev, main_idx, test_idx, ψ0, names)
    q = length(test_idx)
    chisq, os = _score_statistic(D, Σ, dev, main_idx, test_idx)
    p = isfinite(chisq) ? ccdf(Chisq(q), chisq) : NaN
    eff_chisq = zeros(q); eff_p = zeros(q); one_sided = zeros(q)
    for (k, t) in enumerate(test_idx)
        c, o = _score_statistic(D, Σ, dev, main_idx, [t])
        eff_chisq[k] = c
        eff_p[k] = isfinite(c) ? ccdf(Chisq(1), c) : NaN
        one_sided[k] = o
    end
    # One-step estimator (RSiena): theta - D^{-1} e over all parameters.
    one_step = try
        ψ0 .- (D \ dev)[test_idx]
    catch err
        err isa Union{SingularException, LinearAlgebra.LAPACKException} || rethrow()
        fill(NaN, q)
    end
    return SienaScoreTest(names, ψ0, chisq, q, p, eff_chisq, eff_p, one_sided, one_step)
end

function Base.show(io::IO, t::SienaScoreTest)
    println(io, "Score-type test (Schweinberger 2012) of $(t.df) fixed effect(s):")
    @printf(io, "  joint: chi-squared = %.3f, df = %d, p = %s\n", t.chisq, t.df,
            format_pvalue(t.p_value))
    for k in eachindex(t.names)
        @printf(io, "  %-24s fixed at %6.3f: chi-squared = %7.3f, p = %s, one-sided z = %6.3f, one-step estimate = %7.3f\n",
                t.names[k], t.fixed_values[k], t.effect_chisq[k],
                format_pvalue(t.effect_p_values[k]), t.one_sided[k], t.one_step[k])
    end
end

#==============================================================================#
# Coefficient Access Functions (StatsAPI method extensions)
#==============================================================================#

"""
    coef(result::SienaResult)

Return the parameter estimates (rate parameters first, then objective parameters).
Extends `StatsAPI.coef`.
# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
coef(fit)          # objective parameters (a conditional fit has no rate in θ)
```
"""
StatsAPI.coef(result::SienaResult) = result.estimates

"""
    coefnames(result::SienaResult) -> Vector{String}

The parameter labels, in `coef(result)` order (R's `names(coef(fit))`): the same
labels as `coeftable(result).names` and `result.parameter_names` (a copy, so changing
it leaves the fit alone). Extends `StatsAPI.coefnames`.
# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
coefnames(fit)                          # ["outdegree", "recip"]
coefnames(fit) == coeftable(fit).names  # true
```
"""
StatsAPI.coefnames(result::SienaResult) = copy(result.parameter_names)

"""
    stderror(result::SienaResult)

Return standard errors. Extends `StatsAPI.stderror`.
# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
stderror(fit)
```
"""
StatsAPI.stderror(result::SienaResult) = result.standard_errors

"""
    vcov(result::SienaResult)

Return covariance matrix. Extends `StatsAPI.vcov`.
# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
vcov(fit)
```
"""
StatsAPI.vcov(result::SienaResult) = result.covariance

"""
    confint(result::SienaResult; level::Float64=0.95)

Compute confidence intervals. Extends `StatsAPI.confint`.
# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
confint(fit; level=0.9)
```
"""
function StatsAPI.confint(result::SienaResult; level::Float64=0.95)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    lower = result.estimates .- z .* result.standard_errors
    upper = result.estimates .+ z .* result.standard_errors
    return hcat(lower, upper)
end

"""
    coeftable(result::SienaResult)

Return the shared coefficient table in fitted parameter order. Rate parameters are
positive waiting-rate parameters; their normal Wald tests against zero are generally
not appropriate, so their z and p entries are undefined. Convergence t-ratios remain
available separately as `result.t_ratios`.
# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
coeftable(fit)
```
"""
function StatsAPI.coeftable(result::SienaResult)
    z, p = z_pvalues(result.estimates, result.standard_errors)
    for (i, entry) in enumerate(build_param_map(result.effects).free)
        if entry.effect isa BasicRateEffect
            z[i] = NaN
            p[i] = NaN
        end
    end
    return CoefficientTable(result.parameter_names, result.estimates,
                            result.standard_errors; z_values=z, p_values=p)
end
