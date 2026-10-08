"""
Algorithm configuration for SAOM estimation.
"""

#==============================================================================#
# Algorithm Configuration
#==============================================================================#

"""
    SienaAlgorithm

Configuration for the SAOM estimation algorithm.

Every field below is read by [`fit_siena`](@ref) and changes what the estimator
actually does; the settings that were in effect for a fit are reported back on the
[`SienaResult`](@ref) (`n_simulations_run`, `n_threads_used`, `n_iterations`).

# Fields
- `n_subphases::Int`: Number of subphases in phase 2
- `phase1_iterations::Int`: Robbins-Monro iterations in phase 1 (also the number of
  iterations per phase-2 subphase)
- `phase3_iterations::Int`: Simulations in phase 3
- `initial_gain::Float64`: Initial gain parameter (a)
- `min_gain::Float64`: Minimum gain parameter
- `max_iterations::Union{Int, Nothing}`: Budget for the total number of
  Robbins-Monro iterations in phases 1 and 2. `nothing` (the default) means no
  budget, i.e. all `phase1_iterations * (1 + n_subphases)` iterations are run. When
  the budget is exhausted the remaining phase-1/phase-2 iterations are skipped, a
  warning is emitted, and the estimation proceeds to phase 3 (the estimates are then
  typically unconverged). Phase-3 simulations are not iterations and do not count
  against the budget.
- `convergence_threshold::Float64`: Per-parameter convergence criterion — every
  convergence t-ratio must satisfy |t| < threshold (RSiena publication standard: 0.1)
- `rng::AbstractRNG`: Random number generator; defaults to `Random.default_rng()`.
  A fit consumes this generator. Pass a fresh seeded generator to repeat a fit.
- `allow_unconverged::Bool`: What happens when the final, independent phase-3
  batch does not meet the convergence standard. `true` (the default, RSiena's
  behaviour) returns the fit with a warning; `converged = false` is then shown by
  `show`, listed by `approximations`, and the estimates should not be reported.
  `false` throws [`SienaConvergenceError`](@ref), carrying the result.
- `refine_max::Int`: Maximum Newton refinement updates after phase 2 (default 5;
  zero disables refinement). Each uses `2 * phase3_iterations` new simulations.
  Phase 3 always uses a separate, independent batch for final diagnostics.
- `revalidate_max::Int`: How often a failed phase-3 validation re-enters
  refinement (default 2): the failed batch drives one more Newton step and a
  fresh, independent phase-3 batch validates the new estimate. The reported
  diagnostics always come from a batch that was not used to choose the estimate.
- `model_type::Symbol`: Which dependent variables co-evolve — `:standard` (the
  default: every dependent variable takes ministeps), `:networkonly` (only
  [`DependentNetwork`](@ref) variables take ministeps; behavior variables stay
  fixed at their period-start values, still readable by network effects) or
  `:behavioronly` (the mirror image: only [`DependentBehavior`](@ref) variables
  take ministeps; networks are frozen and exogenous). The effects of a frozen
  variable are dropped from the estimated parameter vector — a variable that never
  changes contributes a constant moment, so its parameters are unidentified (see
  [`simulated_variables`](@ref) and [`fit_siena`](@ref))
- `conditional::Union{Bool, Nothing}`: Conditional estimation (RSiena's `cond`):
  condition every simulated period on the observed amount of change of one
  dependent variable instead of on the unit time length (see [`fit_siena`](@ref)).
  `nothing` (the default) is RSiena's `cond = NA`: conditional when exactly one
  dependent variable is simulated (and the data have no composition change),
  unconditional otherwise
- `condvar::Union{Symbol, Nothing}`: Name of the conditioning variable for
  conditional estimation (RSiena's `condvarno`); `nothing` (the default) uses
  the only dependent variable and errors if there are several
- `n_simulations::Int`: Number of simulations averaged into the simulated moment
  vector of each Robbins-Monro iteration (phases 1 and 2). The default 1 is RSiena's
  behaviour; larger values reduce the Monte-Carlo noise of every update at a
  proportional cost.
- `threaded::Bool`: Run the independent simulations (phase 3 and the derivative
  estimators) multi-threaded. `true` (the default) uses `Threads.nthreads()` threads;
  `false` runs them strictly serially on the calling thread. Results are identical
  either way — each simulation is driven by its own pre-drawn seed.
- `verbose::Bool`: Print progress
- `derivative_sims::Int`: Simulations behind the phase-1/phase-2 derivative matrix
  (the Robbins-Monro preconditioner; default 100), and per perturbation of the
  finite-difference derivative under `derivative_method = :finite_difference`
- `diagonalize::Float64`: RSiena's `diagonalize`: the Robbins-Monro updates use
  ``(1 - w) D + w\\,\\mathrm{diag}(D)`` (default 0.2, RSiena's value for MoM)
- `overall_convergence_threshold::Float64`: Threshold for the overall maximum
  convergence ratio `tconv.max` (RSiena standard: 0.25)
- `derivative_method::Symbol`: Estimator for the phase-3 derivative matrix used for
  standard errors — `:score` (score-function / likelihood-ratio estimator over all
  phase-3 simulations, as in RSiena; also for conditional estimation) or
  `:finite_difference` (central differences with common random numbers, for
  cross-checking)

# Example
```julia
using Siena, Random
alg = SienaAlgorithm(rng=MersenneTwister(1), verbose=false, n_subphases=3)
alg.conditional === nothing        # RSiena's cond = NA
```

!!! warning "Not RSiena's modelType"
    `model_type` here selects the **subset of dependent variables that co-evolve**
    (`:standard` / `:networkonly` / `:behavioronly`), which is *not* what RSiena's
    `modelType` means. RSiena's `modelType` selects the *kind of network model*
    driving a ministep — 1 standard actor-oriented, 2 forcing, 3 initiative,
    4 pairwise-forcing, ... . Siena.jl applies RSiena's *default* for each kind of
    network and nothing else: the standard actor-oriented ministep (type 1) for
    directed networks and the forcing model (type 2: unilateral initiative,
    reciprocal confirmation) for undirected ones (`DependentNetwork(...;
    directed=false)`). The other model types (initiative, pairwise, double-step,
    ...) are **not implemented**. The name is kept because it reads naturally in
    Julia, but a Siena.jl `model_type` value never corresponds to an RSiena
    `modelType` value.
"""
mutable struct SienaAlgorithm
    n_subphases::Int
    phase1_iterations::Int
    phase3_iterations::Int
    initial_gain::Float64
    min_gain::Float64
    max_iterations::Union{Int, Nothing}
    convergence_threshold::Float64
    rng::AbstractRNG
    allow_unconverged::Bool
    refine_max::Int
    revalidate_max::Int
    model_type::Symbol
    conditional::Union{Bool, Nothing}
    condvar::Union{Symbol, Nothing}
    n_simulations::Int
    threaded::Bool
    verbose::Bool
    derivative_sims::Int
    diagonalize::Float64
    overall_convergence_threshold::Float64
    derivative_method::Symbol

    function SienaAlgorithm(;
        n_subphases::Int=4,
        phase1_iterations::Int=50,
        phase3_iterations::Int=1000,
        initial_gain::Float64=0.2,
        min_gain::Float64=0.0005,
        max_iterations::Union{Int, Nothing}=nothing,
        convergence_threshold::Float64=0.1,
        rng::AbstractRNG=Random.default_rng(),
        allow_unconverged::Bool=true,
        refine_max::Int=5,
        revalidate_max::Int=2,
        model_type::Symbol=:standard,
        conditional::Union{Bool, Nothing}=nothing,
        condvar::Union{Symbol, Nothing}=nothing,
        n_simulations::Int=1,
        threaded::Bool=true,
        verbose::Bool=true,
        derivative_sims::Int=100,
        diagonalize::Float64=0.2,
        overall_convergence_threshold::Float64=0.25,
        derivative_method::Symbol=:score
    )
        if model_type ∉ (:standard, :networkonly, :behavioronly)
            throw(ArgumentError("model_type must be :standard, :networkonly or " *
                                ":behavioronly (which dependent variables co-evolve), " *
                                "got :$model_type. Note this is not RSiena's " *
                                "`modelType` (Siena.jl uses RSiena's default network " *
                                "model type for each kind of network; the others " *
                                "are not implemented)."))
        end
        if derivative_method ∉ (:score, :finite_difference)
            throw(ArgumentError("derivative_method must be :score or :finite_difference"))
        end
        if n_simulations < 1
            throw(ArgumentError("n_simulations must be >= 1, got $n_simulations"))
        end
        if max_iterations !== nothing && max_iterations < 1
            throw(ArgumentError("max_iterations must be >= 1 or `nothing` (no " *
                                "budget), got $max_iterations"))
        end
        for (fname, fval) in ((:n_subphases, n_subphases),
                              (:phase1_iterations, phase1_iterations),
                              (:phase3_iterations, phase3_iterations),
                              (:derivative_sims, derivative_sims))
            fval >= 1 || throw(ArgumentError("$fname must be >= 1, got $fval"))
        end
        phase3_iterations >= 2 || throw(ArgumentError("phase3_iterations must be >= 2"))
        refine_max >= 0 || throw(ArgumentError("refine_max must be >= 0"))
        0 <= diagonalize <= 1 || throw(ArgumentError("diagonalize must be in [0, 1]"))
        revalidate_max >= 0 || throw(ArgumentError("revalidate_max must be >= 0"))
        for (label, value) in ((:initial_gain, initial_gain), (:min_gain, min_gain),
                               (:convergence_threshold, convergence_threshold),
                               (:overall_convergence_threshold, overall_convergence_threshold))
            isfinite(value) && value > 0 ||
                throw(ArgumentError("$label must be finite and positive"))
        end
        new(n_subphases, phase1_iterations, phase3_iterations,
            initial_gain, min_gain, max_iterations, convergence_threshold,
            rng, allow_unconverged, refine_max, revalidate_max, model_type,
            conditional, condvar, n_simulations, threaded, verbose,
            derivative_sims, diagonalize, overall_convergence_threshold, derivative_method)
    end
end

function Base.show(io::IO, alg::SienaAlgorithm)
    print(io, "SienaAlgorithm(")
    print(io, "n_subphases=$(alg.n_subphases), ")
    print(io, "phase1=$(alg.phase1_iterations), ")
    print(io, "phase3=$(alg.phase3_iterations), ")
    print(io, "n_simulations=$(alg.n_simulations), ")
    print(io, "threaded=$(alg.threaded), ")
    print(io, "max_iterations=$(alg.max_iterations), ")
    print(io, "conditional=$(alg.conditional === nothing ? "auto" : alg.conditional), ")
    print(io, "model_type=:$(alg.model_type))")
end

#==============================================================================#
# Gain Sequence
#==============================================================================#

"""
    GainSequence

Manages the gain parameter sequence for Robbins-Monro algorithm.
# Example
```julia
using Siena
gs = GainSequence(0.2, 0.001)
next_gain!(gs), next_gain!(gs)      # (0.2, 0.1)
```
"""
mutable struct GainSequence
    initial::Float64
    minimum::Float64
    current::Float64
    iteration::Int

    function GainSequence(initial::Float64, minimum::Float64)
        new(initial, minimum, initial, 0)
    end
end

"""
    next_gain!(gs::GainSequence)

Get the next gain value and update the iteration counter.
# Example
```julia
using Siena
gs = GainSequence(0.2, 0.05)
[next_gain!(gs) for _ in 1:5]        # 0.2, 0.1, 0.0667, 0.05, 0.05
```
"""
function next_gain!(gs::GainSequence)
    gs.iteration += 1
    gs.current = max(gs.minimum, gs.initial / gs.iteration)
    return gs.current
end

"""
    reset_gain!(gs::GainSequence)

Reset the gain sequence.
# Example
```julia
using Siena
gs = GainSequence(0.2, 0.05)
next_gain!(gs); next_gain!(gs)
reset_gain!(gs).current        # 0.2
```
"""
function reset_gain!(gs::GainSequence)
    gs.iteration = 0
    gs.current = gs.initial
    gs
end

#==============================================================================#
# Phase Management
#==============================================================================#

"""
    EstimationPhase

Current phase of the estimation algorithm.
# Example
```julia
using Siena
PhaseState(SienaAlgorithm(verbose=false)).phase == PHASE_1      # true
```
"""
@enum EstimationPhase begin
    PHASE_1  # Initial rough estimation
    PHASE_2  # Refinement with subphases
    PHASE_3  # Final estimation and standard errors
end

"""
    PHASE_1

The first [`EstimationPhase`](@ref): the derivative matrix is estimated at the
initial values and a few rough Robbins-Monro updates follow.

# Example
```julia
using Siena
PHASE_1 < PHASE_2 < PHASE_3          # true
```
"""
PHASE_1

"""
    PHASE_2

The second [`EstimationPhase`](@ref): Robbins-Monro subphases with decreasing gain
and Polyak-Ruppert averaging.

# Example
```julia
using Siena
Int(PHASE_2)          # 1
```
"""
PHASE_2

"""
    PHASE_3

The third [`EstimationPhase`](@ref): simulations at the final estimate for the
convergence check and the standard errors.

# Example
```julia
using Siena
string(PHASE_3)       # "PHASE_3"
```
"""
PHASE_3

"""
    PhaseState

State of the current estimation phase.
# Example
```julia
using Siena
ps = PhaseState(SienaAlgorithm(verbose=false))
ps.phase, ps.subphase         # (PHASE_1, 1)
```
"""
mutable struct PhaseState
    phase::EstimationPhase
    subphase::Int
    iteration::Int
    converged::Bool
    gain_seq::GainSequence

    function PhaseState(alg::SienaAlgorithm)
        new(PHASE_1, 1, 0, false,
            GainSequence(alg.initial_gain, alg.min_gain))
    end
end

"""
    advance_phase!(ps::PhaseState, alg::SienaAlgorithm)

Advance to the next phase or subphase.
"""
function advance_phase!(ps::PhaseState, alg::SienaAlgorithm)
    if ps.phase == PHASE_1
        ps.phase = PHASE_2
        ps.subphase = 1
        ps.iteration = 0
        reset_gain!(ps.gain_seq)
    elseif ps.phase == PHASE_2
        if ps.subphase < alg.n_subphases
            ps.subphase += 1
            ps.iteration = 0
            # Reduce gain between subphases
            ps.gain_seq.initial *= 0.5
            reset_gain!(ps.gain_seq)
        else
            ps.phase = PHASE_3
            ps.iteration = 0
        end
    end
    ps
end

#==============================================================================#
# Convergence Checking
#==============================================================================#

"""
    ConvergenceStats

Statistics for convergence checking.

# Fields
- `t_ratios::Vector{Float64}`: per-parameter convergence t-ratios
- `max_t_ratio::Float64`: maximum absolute t-ratio
- `overall_convergence::Float64`: root-mean-square of the t-ratios
- `tconv_max::Float64`: overall maximum convergence ratio (RSiena's `tconv.max`) —
  the maximum convergence t-ratio over all linear combinations of the statistics,
  ``\\sqrt{\\bar e' \\Sigma^{-1} \\bar e}``
# Example
```julia
using Siena
cs = ConvergenceStats(2)
Siena.update_convergence!(cs, [0.05, -0.02], [1.0, 1.0])
cs.max_t_ratio        # 0.05
```
"""
mutable struct ConvergenceStats
    t_ratios::Vector{Float64}
    max_t_ratio::Float64
    overall_convergence::Float64
    tconv_max::Float64

    function ConvergenceStats(n_params::Int)
        new(zeros(n_params), Inf, Inf, Inf)
    end
end

"""
    update_convergence!(cs::ConvergenceStats, deviations::Vector{Float64},
                       scale::Vector{Float64})

Update convergence statistics. `deviations` are the differences between mean
simulated and target statistics; `scale` is the standard deviation of the simulated
statistics (RSiena's convergence t-ratio denominator).
"""
function update_convergence!(cs::ConvergenceStats, deviations::Vector{Float64},
                            scale::Vector{Float64})
    for i in eachindex(cs.t_ratios)
        if scale[i] > 0
            cs.t_ratios[i] = deviations[i] / scale[i]
        else
            cs.t_ratios[i] = abs(deviations[i]) < 1e-8 ? 0.0 : Inf
        end
    end
    cs.max_t_ratio = maximum(abs.(cs.t_ratios))
    cs.overall_convergence = sqrt(mean(cs.t_ratios.^2))
    cs
end

"""
    is_converged(cs::ConvergenceStats, threshold::Float64)
    is_converged(cs::ConvergenceStats, threshold::Float64, overall_threshold::Float64)

Check if the algorithm has converged: every per-parameter |t-ratio| must be below
`threshold`, and — in the three-argument form (the RSiena publication standard:
0.1 and 0.25) — the overall maximum convergence ratio `tconv.max` must be below
`overall_threshold`.
"""
function is_converged(cs::ConvergenceStats, threshold::Float64)
    return cs.max_t_ratio < threshold
end

function is_converged(cs::ConvergenceStats, threshold::Float64,
                      overall_threshold::Float64)
    return cs.max_t_ratio < threshold && cs.tconv_max < overall_threshold
end

#==============================================================================#
# Default Algorithm Creation
#==============================================================================#

"""
    siena_algorithm(; kwargs...)

Create a [`SienaAlgorithm`](@ref) with default or specified parameters.

Counterpart of RSiena's `sienaAlgorithmCreate()`. The defaults follow RSiena where
it matters for inference — conditional estimation for a single dependent variable
(`cond = NA`), 4 subphases with the gain halved per subphase, the publication
convergence standard (all |t| < 0.1, `tconv.max` < 0.25), and a returned (not
thrown) unconverged fit. The schedule itself is simpler than RSiena's: a fixed
`phase1_iterations` per subphase instead of RSiena's adaptive `n2min`/`n2max`, a
first-subphase gain of `initial_gain / 2`, no diagonalisation or Dolby option, a
finite-difference derivative as the phase-1/2 preconditioner, and capped Newton
refinement before phase 3. Settings without an implementation here (`maxlike`, ...)
are absent rather than silently ignored. Siena.jl's `model_type` is *not* RSiena's
`modelType` (see the warning in [`SienaAlgorithm`](@ref)).

# Example
```julia
using Siena, Random
alg = siena_algorithm(rng=MersenneTwister(42), conditional=false, verbose=false)
alg.conditional        # false: unconditional MoM, as RSiena's cond = FALSE
```
"""
function siena_algorithm(; kwargs...)
    SienaAlgorithm(; kwargs...)
end
