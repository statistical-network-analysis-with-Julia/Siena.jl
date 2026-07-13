"""
    Siena.jl - Stochastic Actor-Oriented Models for Julia

A Julia implementation of SIENA (Simulation Investigation for Empirical Network Analysis)
for analyzing longitudinal network data. Port of RSiena (https://github.com/stocnet/rsiena).

Stochastic Actor-Oriented Models (SAOM) are statistical models for analyzing:
- Longitudinal network data (panel data with repeated network observations)
- Co-evolution of networks and behavior
- Multivariate and two-mode networks

The models use simulation-based estimation (Method of Moments) with a continuous-time
Markov chain model of network evolution.

# Key Functions

## Data Preparation
- `siena_data()`: Create a Siena data object
- `siena_nodeset()`: Define a node set
- `siena_dependent()`: Define dependent network or behavior variable
- `constant_covariate()`, `varying_covariate()`: Create actor covariates
- `constant_dyad_covariate()`, `varying_dyad_covariate()`: Create dyadic covariates
- `add_composition_change!()`: Attach actors joining/leaving (`CompositionChange`)

## Model Specification
- `get_effects()`: Create effects object from data
- `include_effects!()`: Add effects to the model (a requested effect that does not
  exist is an error, not a warning)
- `include_interaction!()`: Add interaction effects — **not implemented, throws**

## Estimation
- `fit_siena()`: Estimate model parameters (main estimation function;
  `siena07()` is the RSiena-faithful alias)
- `siena_algorithm()`: Configure estimation algorithm

## Model Assessment
- `gof()`: Goodness of fit testing (method of the shared `Networks.gof` generic)
- `siena_gof()`: Goodness of fit testing (RSiena-style, detailed result)
- `siena_time_test()`: Test for time heterogeneity

# Example

```julia
using Siena

# Create data
data = siena_data()
add_nodeset!(data, NodeSet(50))

# Add dependent network (3 waves)
networks = [rand(0:1, 50, 50) for _ in 1:3]
add_dependent!(data, DependentNetwork(:friendship, networks))

# Get effects and include structural effects
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])

# Estimate
result = fit_siena(data, effects)   # siena07(data, effects) is an alias
```

See the RSiena manual for theoretical background on Stochastic Actor-Oriented Models.
"""
module Siena

using DataFrames
using Distributions
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics
using StatsAPI
using StatsAPI: coef, stderror, vcov, confint
using StatsBase

# Shared result-presentation infrastructure (coefficient tables, GOF containers)
using Networks: print_coeftable, format_pvalue, signif_code
# The ONE ecosystem-wide `gof` generic and the shared GOF result types: Siena
# adds methods/conversions and re-exports the names, so `using Siena, Network`
# never produces colliding exports.
import Networks: gof, GOFResult, GOFStatistic

# The shared result-metadata protocol (Networks.jl `src/results.jl`): the
# generic accessors that say what a fit actually did. Imported by name because
# Siena adds methods for `SienaResult`; `fit_metadata(fit)` collects them.
import Networks: estimand, objective, is_exact, se_method, missing_method,
                 approximations

# Core types
export NodeSet, SienaData
export AbstractDependent, DependentNetwork, DependentBehavior
export AbstractCovariate, ConstantCovariate, VaryingCovariate
export ConstantDyadCovariate, VaryingDyadCovariate
export CompositionChange, NetworkState, StateNetwork
export initialize!
export add_change!, add_composition_change!, is_present

# Data creation functions
export siena_data, siena_nodeset, siena_dependent
export constant_covariate, varying_covariate
export constant_dyad_covariate, varying_dyad_covariate
export add_nodeset!, add_dependent!, add_covariate!
export n_waves, n_actors
export has_structural, is_structural_dyad, n_structural_dyads

# Effects types
export AbstractEffect, NetworkEffect, BehaviorEffect, RateEffect
export EffectEntry, SienaEffects

# Structural network effects - Basic
export OutdegreeEffect, ReciprocityEffect

# Structural network effects - Triadic
export TransitiveTripletsEffect, TransitiveTriadsEffect, TransitiveTiesEffect
export TransitiveMediatedTripletsEffect, TransitiveRecipTripletsEffect
export CyclicTripletsEffect, BalanceSimpleEffect, BetweennessEffect
export NbrDist2Effect, DenseTriadsEffect, SharedInEffect, SharedOutEffect

# Structural network effects - Degree-based
export IndegreePopularityEffect, OutdegreePopularityEffect
export IndegreeActivityEffect, OutdegreeActivityEffect
export OutdegreeTruncEffect, IndegreeTruncEffect
export DegreeAssortativityEffect

# Structural network effects - Isolate
export IsolateEffect, IsolateNetEffect, OutIsolateEffect, InIsolateEffect

# Structural network effects - GWESP family
export GWESPEffect, GWESPBackwardEffect, GWESPMixedEffect, GWDSPEffect

# Covariate network effects
export EgoEffect, EgoSqEffect, AlterEffect, AlterSqEffect
export SimilarityEffect, SameEffect, DifferenceEffect, DifferenceSqEffect
export AbsDifferenceEffect, HigherEffect
export EgoTimesAlterEffect, EgoPlusAlterEffect
export DyadCovariateEffect
export SameXRecipEffect, SimXRecipEffect, SimXTransTripEffect
export EndowmentEffect, CreationEffect

# Multiplex network effects
export CrossNetworkReciprocityEffect, CrossNetworkActivityEffect
export CrossNetworkPopularityEffect, CrossNetworkTiesEffect

# Behavior effects - Shape
export LinearShapeEffect, QuadraticShapeEffect, CubicShapeEffect

# Behavior effects - Influence
export AverageAlterEffect, TotalAlterEffect, AverageSimilarityEffect, TotalSimilarityEffect
export AverageInAlterEffect, AverageRecipAlterEffect
export AverageAttHigherSimpleEffect, AverageAttLowerSimpleEffect
export AverageAlterDist2Effect, TotalInAlterEffect

# Behavior effects - Degree-based
export IndegreeEffect, BehaviorOutdegreeEffect, RecipDegreeEffect

# Behavior effects - Covariate
export BehaviorCovariateEffect, CovariateInteractionEffect

# Behavior effects - Behavior interaction
export BehaviorInteractionEffect, BehaviorSimilarityEffect

# Behavior effects - Threshold and other
export ThresholdEffect, PropThresholdEffect
export BehaviorIsolateEffect, FeedbackEffect, MainBehaviorEffect

# Rate effects - Basic
export BasicRateEffect, CovariateRateEffect

# Rate effects - Degree-based
export OutdegreeRateEffect, IndegreeRateEffect
export OutdegreeLogRateEffect, IndegreeLogRateEffect
export OutdegreeInvRateEffect, IndegreeInvRateEffect
export OutdegreeSqRateEffect, RecipDegreeRateEffect

# Rate effects - Behavior-based
export BehaviorRateEffect, AverageAlterRateEffect, TotalAlterRateEffect
export SimilarityRateEffect

# Rate effects - Other
export SettingRateEffect, EgoAlterRateEffect, CovariateSqRateEffect

# Two-mode network effects
export TwoModeEffect
export TwoModeOutdegreeEffect, TwoModeIndegreeEffect
export FourCyclesEffect, SharedEventsEffect, GWESPTwoModeEffect
export TwoModeEgoEffect, TwoModeEventEffect
export TwoModeSameEffect, TwoModeSimilarityEffect
export TwoModeActivityEffect, TwoModePopularityAltEffect
export TwoModeTransitiveClosureEffect, TwoModeActorAssortativityEffect
export TwoModeWithinEffect

# Effects functions
export effect_name, effect_type, target_variable, interaction_with
export compute_contribution, compute_statistic, evaluate_actor, rate_score
export get_effects, include_effects!, include_interaction!
export get_included_effects, get_rate_effects, get_objective_effects
export effects_table

# Algorithm configuration
export SienaAlgorithm, siena_algorithm
export GainSequence, PhaseState, ConvergenceStats
export next_gain!, reset_gain!
export EstimationPhase, PHASE_1, PHASE_2, PHASE_3

# Simulation
export simulate_saom, simulate_period!, snapshot
export compute_objective, compute_network_choice_probs, compute_behavior_choice_probs
export SimulationResult
export ParameterMap, build_param_map, parameter_names, objective_theta, basic_rate
export simulated_variables, restrict_effects
export ObjectiveEffectSet, build_objective_set
export ScoreAccumulator, reset_scores!

# Estimation
export fit_siena, siena07, SienaResult
export coef, stderror, vcov, confint
export compute_target_statistics, compute_simulated_statistics, default_basic_rate
export estimate_derivative_matrix, estimate_derivative_matrix_score

# Goodness of fit
export AbstractGOFStatistic
export IndegreeDistribution, OutdegreeDistribution
export TriadCensus, GeodesicDistribution, BehaviorDistribution
export siena_gof, SienaGOFResult, compute_gof_statistic
export siena_gof_indegree, siena_gof_outdegree, siena_gof_triad, siena_gof_behavior
# Shared GOF interface (re-exports of the Networks.jl names: `gof(fit, ...)`
# returns the ecosystem-wide `GOFResult`)
export gof, GOFResult, GOFStatistic

# Include source files
include("types.jl")
include("effects/base.jl")
include("effects/network.jl")
include("effects/behavior.jl")
include("effects/rate.jl")
include("effects/twomode.jl")
include("algorithm.jl")
include("simulation.jl")
include("estimation.jl")
include("gof.jl")

#==============================================================================#
# Convenience Constructors (RSiena-like API)
#==============================================================================#

"""
    siena_data()

Create an empty SienaData object.
Counterpart of R's sienaDataCreate() without arguments.
"""
siena_data() = SienaData()

"""
    siena_nodeset(n::Int; names::Vector{String}=String[], id::Symbol=:actors)

Create a node set.
Counterpart of R's sienaNodeSet().
"""
siena_nodeset(n::Int; names::Vector{String}=String[], id::Symbol=:actors) =
    NodeSet(n; names=names, id=id)

"""
    siena_dependent(name::Symbol, networks::Vector{<:AbstractMatrix}; kwargs...)

Create a dependent network variable.
Counterpart of R's sienaDependent() for networks.

Matrices may contain RSiena-style structural codes (default: `10` =
structural zero, `11` = structural one, configurable via the
`structural_zero`/`structural_one` keywords); see
[`DependentNetwork`](@ref) for the semantics.
"""
siena_dependent(name::Symbol, networks::Vector{<:AbstractMatrix}; kwargs...) =
    DependentNetwork(name, networks; kwargs...)

"""
    siena_dependent(name::Symbol, values::Vector{<:AbstractVector}; kwargs...)

Create a dependent behavior variable.
Counterpart of R's sienaDependent() for behavior.
"""
siena_dependent(name::Symbol, values::Vector{<:AbstractVector{<:Integer}}; kwargs...) =
    DependentBehavior(name, values; kwargs...)

"""
    constant_covariate(name::Symbol, values::AbstractVector; kwargs...)

Create a constant covariate.
Counterpart of R's coCovar().
"""
constant_covariate(name::Symbol, values::AbstractVector; kwargs...) =
    ConstantCovariate(name, values; kwargs...)

"""
    varying_covariate(name::Symbol, values::Vector{<:AbstractVector}; kwargs...)

Create a varying covariate.
Counterpart of R's varCovar().
"""
varying_covariate(name::Symbol, values::Vector{<:AbstractVector}; kwargs...) =
    VaryingCovariate(name, values; kwargs...)

"""
    constant_dyad_covariate(name::Symbol, values::AbstractMatrix; kwargs...)

Create a constant dyadic covariate.
Counterpart of R's coDyadCovar().
"""
constant_dyad_covariate(name::Symbol, values::AbstractMatrix; kwargs...) =
    ConstantDyadCovariate(name, values; kwargs...)

"""
    varying_dyad_covariate(name::Symbol, values::Vector{<:AbstractMatrix}; kwargs...)

Create a varying dyadic covariate.
Counterpart of R's varDyadCovar().
"""
varying_dyad_covariate(name::Symbol, values::Vector{<:AbstractMatrix}; kwargs...) =
    VaryingDyadCovariate(name, values; kwargs...)

#==============================================================================#
# Effects API
#==============================================================================#

"""
    get_effects(data::SienaData)

Create a SienaEffects object with default effects for the data.
Counterpart of R's getEffects().

This creates rate effects for each period and basic structural effects
(but doesn't include them by default).
"""
function get_effects(data::SienaData)
    effects = SienaEffects(collect(keys(data.dependents)))

    # Add rate effects for each dependent variable and period. The initial value is
    # the basic rate itself (not its log), derived from the observed amount of change.
    for (name, dep) in data.dependents
        for p in 1:(data.n_waves - 1)
            rate_eff = BasicRateEffect(name, p)
            entry = EffectEntry(rate_eff;
                               name="Rate $name (period $p)",
                               shortname="rate$p",
                               include=true,
                               initial_value=default_basic_rate(data, name, p))
            add_effect!(effects, entry)
        end

        if dep isa DependentNetwork
            # Add basic structural effects (not included by default)
            add_effect!(effects, EffectEntry(OutdegreeEffect(name);
                name="Outdegree (density)", shortname="outdegree"))
            add_effect!(effects, EffectEntry(ReciprocityEffect(name);
                name="Reciprocity", shortname="recip"))
            add_effect!(effects, EffectEntry(TransitiveTripletsEffect(name);
                name="Transitive triplets", shortname="transTrip"))
            add_effect!(effects, EffectEntry(TransitiveTriadsEffect(name);
                name="Transitive ties", shortname="transTies"))
            add_effect!(effects, EffectEntry(CyclicTripletsEffect(name);
                name="3-cycles", shortname="cycle3"))
            add_effect!(effects, EffectEntry(IndegreePopularityEffect(name);
                name="Indegree popularity", shortname="inPop"))
            add_effect!(effects, EffectEntry(IndegreePopularityEffect(name; sqrt=true);
                name="Indegree popularity (sqrt)", shortname="inPopSqrt"))
            add_effect!(effects, EffectEntry(OutdegreeActivityEffect(name);
                name="Outdegree activity", shortname="outAct"))
            add_effect!(effects, EffectEntry(OutdegreeActivityEffect(name; sqrt=true);
                name="Outdegree activity (sqrt)", shortname="outActSqrt"))
            add_effect!(effects, EffectEntry(GWESPEffect(name);
                name="GWESP", shortname="gwesp"))

            # Add covariate effects for each covariate
            for (cov_name, cov) in data.covariates
                if cov isa ConstantCovariate || cov isa VaryingCovariate
                    add_effect!(effects, EffectEntry(EgoEffect(name, cov_name);
                        name="Ego $cov_name", shortname="ego$(cov_name)"))
                    add_effect!(effects, EffectEntry(AlterEffect(name, cov_name);
                        name="Alter $cov_name", shortname="alt$(cov_name)"))
                    add_effect!(effects, EffectEntry(SimilarityEffect(name, cov_name);
                        name="Similarity $cov_name", shortname="sim$(cov_name)"))
                    add_effect!(effects, EffectEntry(SameEffect(name, cov_name);
                        name="Same $cov_name", shortname="same$(cov_name)"))
                elseif cov isa ConstantDyadCovariate || cov isa VaryingDyadCovariate
                    add_effect!(effects, EffectEntry(DyadCovariateEffect(name, cov_name);
                        name="Dyadic $cov_name", shortname="dyad$(cov_name)"))
                end
            end

        elseif dep isa DependentBehavior
            # Add behavior effects
            add_effect!(effects, EffectEntry(LinearShapeEffect(name);
                name="Linear shape", shortname="linear"))
            add_effect!(effects, EffectEntry(QuadraticShapeEffect(name);
                name="Quadratic shape", shortname="quad"))

            # Add network influence effects for each network
            for (net_name, net_dep) in data.dependents
                if net_dep isa DependentNetwork
                    add_effect!(effects, EffectEntry(AverageAlterEffect(name, net_name);
                        name="Average alter ($net_name)", shortname="avAlt$(net_name)"))
                    add_effect!(effects, EffectEntry(AverageSimilarityEffect(name, net_name);
                        name="Average similarity ($net_name)", shortname="avSim$(net_name)"))
                    add_effect!(effects, EffectEntry(TotalAlterEffect(name, net_name);
                        name="Total alter ($net_name)", shortname="totAlt$(net_name)"))
                end
            end
        end
    end

    return effects
end

"""
    include_effects!(effects::SienaEffects, variable::Symbol, effect_names::Vector{Symbol};
                    initial_value=nothing, fix::Bool=false, test::Bool=false,
                    strict::Bool=true)

Include effects in the model by short name.
Counterpart of R's `includeEffects()`.

An effect name that matches no entry for `variable` is an error: a model that
silently drops an effect the analyst asked for is not reproducible. The error lists
the names that are available for the variable, which also catches typos and effects
that `get_effects` does not create for the given data.

Pass `strict=false` to opt out and get the old warn-and-skip behaviour — the
unmatched names are then reported with a warning and simply not included.

# Arguments
- `effects`: The effects object
- `variable`: Name of the dependent variable
- `effect_names`: Vector of effect short names to include
- `initial_value`: Initial parameter value (`nothing` keeps the entry's current value)
- `fix`: Whether to fix the parameter
- `test`: Whether to perform score test
- `strict`: Throw (default) rather than warn when a requested effect does not exist

# Examples
```julia
include_effects!(effects, :friendship, [:outdegree, :recip])
include_effects!(effects, :friendship, [:transTrpt])   # throws: typo, did you mean transTrip?
include_effects!(effects, :friendship, [:transTrpt]; strict=false)   # warns, includes nothing
```
"""
function include_effects!(effects::SienaEffects, variable::Symbol, effect_names::Vector{Symbol};
                         initial_value::Union{Nothing, Float64}=nothing,
                         fix::Bool=false, test::Bool=false, strict::Bool=true)
    matched = Set{Symbol}()
    for entry in effects.effects
        # Use shortname (user-facing name) for matching
        shortname_sym = Symbol(entry.shortname)
        if target_variable(entry.effect) == variable && shortname_sym in effect_names
            entry.include = true
            entry.fix = fix
            entry.test = test
            isnothing(initial_value) || (entry.initial_value = initial_value)
            push!(matched, shortname_sym)
        end
    end
    missed = setdiff(Set(effect_names), matched)
    if !isempty(missed)
        # Sorted for a deterministic message.
        missing_names = sort!(collect(missed); by=String)
        if strict
            available = sort!([entry.shortname for entry in effects.effects
                               if target_variable(entry.effect) == variable])
            throw(ArgumentError(
                "no effect(s) $(missing_names) for variable :$variable. " *
                "Available effects for :$variable: $(available). " *
                "(Pass `strict=false` to skip unavailable effects with a warning " *
                "instead of this error.)"))
        end
        @warn "effects not found for variable :$variable and NOT included: " *
              "$(missing_names)"
    end
    effects
end

"""
    include_interaction!(effects::SienaEffects, variable::Symbol,
                        effect1::Symbol, effect2::Symbol; kwargs...)

Include an interaction effect. **Not implemented — always throws.**

RSiena's `includeInteraction()` has no counterpart here yet: there is no interaction
effect type, so no interaction can be added to the model. Earlier versions warned and
returned the effects object unchanged, which let a model be estimated and published
without the interaction the analyst had asked for; it now throws instead.

Workaround: define the product effect you need as a concrete `NetworkEffect` /
`BehaviorEffect` subtype implementing `evaluate_actor` and add it with
`Siena.add_effect!(effects, EffectEntry(MyEffect(...); include=true))`.
"""
function include_interaction!(effects::SienaEffects, variable::Symbol,
                             effect1::Symbol, effect2::Symbol; kwargs...)
    throw(ArgumentError(
        "include_interaction! is not implemented: the interaction of :$effect1 and " *
        ":$effect2 for variable :$variable cannot be added to the model. Define the " *
        "interaction as a concrete effect type implementing `evaluate_actor` and add " *
        "it with `Siena.add_effect!`."))
end

#==============================================================================#
# Deprecations
#==============================================================================#

# Effects whose formula is *not* the RSiena formula of the same name have been
# renamed with a `Simple` suffix (type name and short name alike), so that an RSiena
# short name always denotes a numerically equivalent implementation. The old type
# names keep working for one release cycle; the old *short names* (`:balance`,
# `:avAttHigher`, `:avAttLower`) are gone and are reserved for the real formulas.
Base.@deprecate_binding BalanceEffect BalanceSimpleEffect
Base.@deprecate_binding AverageAttHigherEffect AverageAttHigherSimpleEffect
Base.@deprecate_binding AverageAttLowerEffect AverageAttLowerSimpleEffect

end # module
