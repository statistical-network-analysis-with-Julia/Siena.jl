"""
Algorithm configuration for SAOM estimation.
"""

#==============================================================================#
# Algorithm Configuration
#==============================================================================#

"""
    SienaAlgorithm

Configuration for the SAOM estimation algorithm.

# Fields
- `n_subphases::Int`: Number of subphases in phase 2
- `phase1_iterations::Int`: Iterations in phase 1
- `phase3_iterations::Int`: Iterations in phase 3
- `initial_gain::Float64`: Initial gain parameter (a)
- `min_gain::Float64`: Minimum gain parameter
- `max_iterations::Int`: Maximum total iterations
- `convergence_threshold::Float64`: Per-parameter convergence criterion — every
  convergence t-ratio must satisfy |t| < threshold (RSiena publication standard: 0.1)
- `seed::Union{Int, Nothing}`: Random seed
- `model_type::Symbol`: Model type (:standard, :behavioronly, :networkonly)
- `conditional::Bool`: Conditional estimation (RSiena's `cond`): condition every
  simulated period on the observed amount of change of one dependent variable
  instead of on the unit time length (see [`siena07`](@ref))
- `condvar::Union{Symbol, Nothing}`: Name of the conditioning variable for
  conditional estimation (RSiena's `condvarno`); `nothing` (the default) uses
  the only dependent variable and errors if there are several
- `n_simulations::Int`: Number of simulations per iteration
- `parallel::Bool`: Use parallel processing
- `verbose::Bool`: Print progress
- `derivative_sims::Int`: Simulations per parameter for finite-difference derivative
  estimation (with common random numbers)
- `overall_convergence_threshold::Float64`: Threshold for the overall maximum
  convergence ratio `tconv.max` (RSiena standard: 0.25)
- `derivative_method::Symbol`: Estimator for the phase-3 derivative matrix used for
  standard errors — `:score` (score-function / likelihood-ratio estimator over all
  phase-3 simulations, as in RSiena) or `:finite_difference` (forward differences
  with common random numbers, for cross-checking)
"""
mutable struct SienaAlgorithm
    n_subphases::Int
    phase1_iterations::Int
    phase3_iterations::Int
    initial_gain::Float64
    min_gain::Float64
    max_iterations::Int
    convergence_threshold::Float64
    seed::Union{Int, Nothing}
    model_type::Symbol
    conditional::Bool
    condvar::Union{Symbol, Nothing}
    n_simulations::Int
    parallel::Bool
    verbose::Bool
    derivative_sims::Int
    overall_convergence_threshold::Float64
    derivative_method::Symbol

    function SienaAlgorithm(;
        n_subphases::Int=4,
        phase1_iterations::Int=50,
        phase3_iterations::Int=1000,
        initial_gain::Float64=0.2,
        min_gain::Float64=0.0005,
        max_iterations::Int=50,
        convergence_threshold::Float64=0.1,
        seed::Union{Int, Nothing}=nothing,
        model_type::Symbol=:standard,
        conditional::Bool=false,
        condvar::Union{Symbol, Nothing}=nothing,
        n_simulations::Int=1,
        parallel::Bool=false,
        verbose::Bool=true,
        derivative_sims::Int=30,
        overall_convergence_threshold::Float64=0.25,
        derivative_method::Symbol=:score
    )
        if model_type ∉ (:standard, :behavioronly, :networkonly)
            throw(ArgumentError("model_type must be :standard, :behavioronly, or :networkonly"))
        end
        if derivative_method ∉ (:score, :finite_difference)
            throw(ArgumentError("derivative_method must be :score or :finite_difference"))
        end
        new(n_subphases, phase1_iterations, phase3_iterations,
            initial_gain, min_gain, max_iterations, convergence_threshold,
            seed, model_type, conditional, condvar, n_simulations, parallel, verbose,
            derivative_sims, overall_convergence_threshold, derivative_method)
    end
end

function Base.show(io::IO, alg::SienaAlgorithm)
    print(io, "SienaAlgorithm(")
    print(io, "n_subphases=$(alg.n_subphases), ")
    print(io, "phase1=$(alg.phase1_iterations), ")
    print(io, "phase3=$(alg.phase3_iterations), ")
    print(io, "model=$(alg.model_type))")
end

#==============================================================================#
# Gain Sequence
#==============================================================================#

"""
    GainSequence

Manages the gain parameter sequence for Robbins-Monro algorithm.
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
"""
function next_gain!(gs::GainSequence)
    gs.iteration += 1
    gs.current = max(gs.minimum, gs.initial / gs.iteration)
    return gs.current
end

"""
    reset_gain!(gs::GainSequence)

Reset the gain sequence.
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
"""
@enum EstimationPhase begin
    PHASE_1  # Initial rough estimation
    PHASE_2  # Refinement with subphases
    PHASE_3  # Final estimation and standard errors
end

"""
    PhaseState

State of the current estimation phase.
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

Create a SienaAlgorithm with default or specified parameters.
Equivalent to R's sienaAlgorithmCreate().
"""
function siena_algorithm(; kwargs...)
    SienaAlgorithm(; kwargs...)
end
