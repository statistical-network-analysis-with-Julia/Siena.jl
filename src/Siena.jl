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
- `include_interaction!()`: Add a two- or three-way interaction of effects
  (RSiena's `includeInteraction`)

## Estimation
- `fit_siena()`: Estimate model parameters (main estimation function;
  `siena07()` is the RSiena-faithful alias)
- `siena_algorithm()`: Configure estimation algorithm

## Model Assessment
- `gof()`: Goodness of fit testing (method of the shared `NetworkCore.gof` generic;
  the preferred entry point)
- `siena_gof()`: the same test with the RSiena-style detailed result
- score-type tests of fixed effects: `include_effects!(...; fix=true, test=true)`
  (RSiena's `test=TRUE`)
- `siena_time_test()`: Score-type tests of time heterogeneity (RSiena's
  `sienaTimeTest`)

# Example

The s50 friendship panel bundled with NetworkCore (50 pupils, three waves), with
the model of RSiena's introductory script: RSiena's default effects (rates,
outdegree, reciprocity), transitive triplets, and smoking-based selection.

```julia
using Siena, NetworkCore, Random

s50 = load_dataset(:s50)
data = siena_data()
add_nodeset!(data, NodeSet(50))
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))

effects = get_effects(data)       # RSiena's defaults: rates, outdegree, recip
include_effects!(effects, :friendship, [:transTrip, :altsmoke1, :egosmoke1, :simsmoke1])

# Seeded, so the fit is reproducible; siena07(data, effects; ...) is an alias.
result = fit_siena(data, effects; rng=MersenneTwister(1),
                   algorithm=SienaAlgorithm(verbose=false))
result.converged                  # true
coeftable(result)                 # estimates, standard errors, z and p
```

See the RSiena manual for theoretical background on Stochastic Actor-Oriented Models.
"""
module Siena

using DataFrames
using Distributions
using LinearAlgebra
using PrecompileTools: @setup_workload, @compile_workload
using Printf
using Random
using SparseArrays
using Statistics
using StatsAPI
using StatsAPI: coef, coefnames, stderror, vcov, confint
using StatsBase

# Shared result-presentation infrastructure (coefficient tables, GOF containers)
using NetworkCore: print_coeftable, format_pvalue, signif_code, z_pvalues, mc_pvalue, CoefficientTable
using NetworkCore: spawn_all
# The ONE ecosystem-wide `gof` generic and the shared GOF result types: Siena
# adds methods/conversions and re-exports the names, so `using Siena, Network`
# never produces colliding exports.
import NetworkCore: gof, GOFResult, GOFStatistic

# The shared result-metadata protocol (NetworkCore.jl `src/results.jl`): the
# generic accessors that say what a fit actually did. Imported by name because
# Siena adds methods for `SienaResult`; `fit_metadata(fit)` collects them.
import NetworkCore: estimand, objective, is_exact, se_method, missing_method,
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
export IsolateNetEffect, OutIsolateEffect, InIsolateEffect

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
export InteractionEffect, BehaviorProductEffect

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
export get_effects, include_effects!, include_interaction!, add_effect!, validate_effects
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
export ScoreAccumulator, reset_scores!, MinistepWorkspace, compute_network_choice_probs!

# Estimation
export fit_siena, siena07, SienaResult, SienaConvergenceError, SienaScoreTest
export siena_time_test, SienaTimeTest
export coef, coefnames, stderror, vcov, confint, coeftable
export compute_target_statistics, compute_simulated_statistics, default_basic_rate
export estimate_derivative_matrix, estimate_derivative_matrix_score

# Goodness of fit
export AbstractGOFStatistic
export IndegreeDistribution, OutdegreeDistribution
export TriadCensus, GeodesicDistribution, BehaviorDistribution
export siena_gof, SienaGOFResult, compute_gof_statistic
export siena_gof_indegree, siena_gof_outdegree, siena_gof_triad, siena_gof_behavior
# Shared GOF interface (re-exports of the NetworkCore.jl names: `gof(fit, ...)`
# returns the ecosystem-wide `GOFResult`)
export gof, GOFResult, GOFStatistic

# Include source files
include("types.jl")
include("effects/base.jl")
include("effects/network.jl")
include("effects/behavior.jl")
include("effects/rate.jl")
include("effects/twomode.jl")
include("effects/interaction.jl")
include("effects/registry.jl")
include("algorithm.jl")
include("simulation.jl")
include("estimation.jl")
include("timetest.jl")
include("gof.jl")

#==============================================================================#
# Convenience Constructors (RSiena-like API)
#==============================================================================#

"""
    siena_data()

Create an empty SienaData object.
Counterpart of R's sienaDataCreate() without arguments.
# Example
```julia
using Siena
data = siena_data()
data.n_waves      # 0 until a dependent variable is added
```
"""
siena_data() = SienaData()

"""
    siena_nodeset(n::Int; names::Vector{String}=String[], id::Symbol=:actors)

Create a node set.
Counterpart of R's sienaNodeSet().
# Example
```julia
using Siena
siena_nodeset(4; id=:clubs)        # NodeSet(:clubs, n=4)
```
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
# Example
```julia
using Siena
siena_dependent(:advice, [[0 1; 0 0], [0 1; 1 0]]) isa DependentNetwork      # true
siena_dependent(:mood, [[1, 2], [2, 2]]) isa DependentBehavior                 # true
```
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
# Example
```julia
using Siena
constant_covariate(:age, [20, 30, 40]).values      # [-10.0, 0.0, 10.0]
```
"""
constant_covariate(name::Symbol, values::AbstractVector; kwargs...) =
    ConstantCovariate(name, values; kwargs...)

"""
    varying_covariate(name::Symbol, values::Vector{<:AbstractVector}; kwargs...)

Create a varying covariate.
Counterpart of R's varCovar().
# Example
```julia
using Siena
varying_covariate(:mood, [[1, 2, 3], [2, 3, 4]]).values[2]       # centred wave-2 values
```
"""
varying_covariate(name::Symbol, values::Vector{<:AbstractVector}; kwargs...) =
    VaryingCovariate(name, values; kwargs...)

"""
    constant_dyad_covariate(name::Symbol, values::AbstractMatrix; kwargs...)

Create a constant dyadic covariate.
Counterpart of R's coDyadCovar().
# Example
```julia
using Siena
constant_dyad_covariate(:dist, [0 1 2; 1 0 1; 2 1 0]).mean      # 4/3
```
"""
constant_dyad_covariate(name::Symbol, values::AbstractMatrix; kwargs...) =
    ConstantDyadCovariate(name, values; kwargs...)

"""
    varying_dyad_covariate(name::Symbol, values::Vector{<:AbstractMatrix}; kwargs...)

Create a varying dyadic covariate.
Counterpart of R's varDyadCovar().
# Example
```julia
using Siena
varying_dyad_covariate(:contact, [[0 1; 1 0], [0 2; 2 0]]).mean     # 1.5
```
"""
varying_dyad_covariate(name::Symbol, values::Vector{<:AbstractMatrix}; kwargs...) =
    VaryingDyadCovariate(name, values; kwargs...)

#==============================================================================#
# Effects API
#==============================================================================#

"""
    get_effects(data::SienaData)

Create the effects table for the data — the counterpart of RSiena's `getEffects()`.

Every effect Siena.jl implements that is defined for a dependent variable is
registered for it, under its RSiena short name where it has one. Which effects are
**included** follows RSiena's `getEffects()`, so the default model is RSiena's:

| Dependent variable | Included by default |
|:-------------------|:--------------------|
| directed one-mode network | basic rate of every period, `outdegree` (RSiena `density`), `recip` |
| undirected network | basic rates, `outdegree` (`density`) |
| two-mode network | basic rates, `outdegree` (`density`) |
| behaviour | basic rates, `linear`, and `quad` when the observed range (maximum minus minimum over all waves) is at least 2 |

As in RSiena, a network whose every period only adds ties (or only drops them) gets
no `outdegree` effect at all, and a behaviour whose every period only rises (or only
falls) gets no `linear` effect; see [`DependentNetwork`](@ref) and
[`DependentBehavior`](@ref) on up-only and down-only periods. Every other effect is
registered but excluded. Including a default effect again with
[`include_effects!`](@ref) changes nothing, and `include_effects!(...; include=false)`
removes one from the model (RSiena's `includeEffects(..., include = FALSE)`).

Which effects exist depends on the kind of variable, following RSiena's effect groups:

- **directed one-mode networks**: `outdegree` (RSiena `density`), `recip`,
  `transTrip`, `transMedTrip`, `transRecTrip`, `cycle3`, `transTies`, `between`,
  `nbrDist2`, `denseTriads`, `gwespFF`/`gwespBB`/`gwespFB`/`gwdspFF`, `inPop`,
  `inPopSqrt`, `outPop`, `outPopSqrt`, `inAct`, `inActSqrt`, `outAct`, `outActSqrt`,
  `outTrunc`, `isolateNet`, `outIso`, …;
- **undirected (symmetric) networks** (`DependentNetwork(...; directed=false)`):
  `outdegree` (degree), `transTriads`, `transTies`, `between`, `nbrDist2`, `gwesp`,
  `inPop`, `inPopSqrt`, `outAct`, `outActSqrt`, `degPlus`, `outTrunc`, `isolateNet`,
  `outIso`;
- **two-mode networks**: the [`TwoModeEffect`](@ref)s (`outdegree2`, `fourCycles`,
  …), never a one-mode effect;
- **behaviour**: `linear`, `quad`, the influence effects (`avAlt`, `avSim`,
  `totAlt`, …) for each network, `effFrom` for each actor covariate;
- **rate**: one basic rate per period, plus `outRate`, `inRate`, `RateX`, ….

Effects that condition on an attribute come once per actor covariate **and once
per co-evolving dependent behaviour** (network selection on the behaviour), with
the attribute's name in the short name: `egosmoke1`, `altalcohol`, `simalcohol`, …
(RSiena's `egoX`/`altX`/`simX` with `interaction1`; [`include_effects!`](@ref) also
accepts the RSiena spelling with `interaction1=`). Effects without an RSiena
counterpart are marked "(Siena.jl)" in their display name; the effects guide has the
full concordance table.

# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]; w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
add_covariate!(data, ConstantCovariate(:age, [30, 40, 50]))
effects = get_effects(data)
"transTrip" in [e.shortname for e in effects]      # true
"egoage" in [e.shortname for e in effects]         # true
[e.shortname for e in get_included_effects(effects)]   # ["rate1", "outdegree", "recip"]
```
"""
function get_effects(data::SienaData)
    effects = SienaEffects(sort!(collect(keys(data.dependents)); by=String))
    for (name, dep) in data.dependents
        effects.kinds[name] = dep isa DependentNetwork ? _network_kind(dep) : :behavior
    end
    for name in effects.variable_names
        dep = data.dependents[name]
        entries = EffectEntry[]
        _rate_entries!(entries, data, name, dep)
        if dep isa DependentNetwork
            _network_entries!(entries, data, name, dep)
        elseif dep isa DependentBehavior
            _behavior_entries!(entries, data, name, dep)
        end
        _rsiena_defaults!(entries, dep)
        foreach(e -> add_effect!(effects, e), entries)
    end
    return effects
end

# RSiena's getEffects() default model: the basic rates (already included by
# `_rate_entries!`), `density` for every network, `recip` for a directed one-mode
# network, `linear` for a behaviour and `quad` when its observed range is at least 2.
# When every period of the variable only goes up (or only goes down), RSiena does not
# offer `density`/`linear` at all.
function _rsiena_defaults!(entries::Vector{EffectEntry}, dep::AbstractDependent)
    _all_one_direction(dep) &&
        filter!(e -> !(e.effect isa Union{OutdegreeEffect, LinearShapeEffect}), entries)
    for e in entries
        e.include |= _rsiena_default(e.effect, dep)
    end
    return entries
end

_rsiena_default(::AbstractEffect, ::AbstractDependent) = false
_rsiena_default(::OutdegreeEffect, ::DependentNetwork) = true
_rsiena_default(::ReciprocityEffect, dep::DependentNetwork) =
    _network_kind(dep) === :directed
_rsiena_default(::LinearShapeEffect, ::DependentBehavior) = true
_rsiena_default(::QuadraticShapeEffect, dep::DependentBehavior) =
    _observed_range(dep) >= 2

# The observed range of a behaviour (RSiena's `range` attribute), which can be
# narrower than the declared `min_val:max_val`.
_observed_range(dep::DependentBehavior) =
    maximum(maximum, dep.values) - minimum(minimum, dep.values)

# RSiena spellings accepted for an entry registered under another short name.
const _SHORTNAME_ALIASES = Dict(:gwesp => :gwespFF, :fourCycles => :cycle4)

# Whether the entry answers to the requested name: its Siena.jl short name, or its
# RSiena short name (`effect_name`) together with the RSiena `interaction1`. A given
# `interaction1` must match in both cases, so a short name never silently selects an
# effect of another covariate or network.
function _matches(entry::EffectEntry, name::Symbol, interaction1)
    if Symbol(entry.shortname) == name
        return interaction1 === nothing || interaction_with(entry.effect) == interaction1
    end
    effect_name(entry.effect) == name || return false
    return interaction_with(entry.effect) == interaction1
end

"""
    include_effects!(effects::SienaEffects, variable::Symbol, effect_names::Vector{Symbol};
                    interaction1=nothing, initial_value=nothing, fix::Bool=false,
                    test::Bool=false, strict::Bool=true, include::Bool=true)

Include effects in the model by short name. Counterpart of R's `includeEffects()`.

[`get_effects`](@ref) already includes RSiena's default effects (`outdegree` and
`recip` for a directed network, `linear` and `quad` for a behaviour, …); naming one
of them again changes nothing. `include=false` removes the named effects from the
model instead, as RSiena's `includeEffects(..., include = FALSE)` does.

An effect is selected by its Siena.jl short name as listed by [`get_effects`](@ref)
(`:outdegree`, `:transTrip`, `:egosmoke1`, `:avAltfriendship`, …) **or** by its RSiena
short name together with RSiena's `interaction1` (`[:egoX, :altX, :simX];
interaction1=:smoke1`, `[:avAlt]; interaction1=:friendship`, `:density` for
`:outdegree`).

An effect name that matches no entry for `variable` is an error: a model that
silently drops an effect the analyst asked for is not reproducible. The error lists
the names that are available for the variable, which also catches typos and effects
that do not exist for the kind of variable (e.g. `recip` on an undirected network).
Pass `strict=false` to warn and skip unmatched names instead.

# Arguments
- `effects`: The effects object
- `variable`: Name of the dependent variable
- `effect_names`: Vector of effect short names to include
- `interaction1`: RSiena's `interaction1` (the covariate, behaviour or network the
  effect refers to) when selecting by RSiena short name
- `initial_value`: Initial parameter value (`nothing` keeps the entry's current value)
- `fix`: Hold the parameter fixed at its initial value instead of estimating it
- `test`: Request a score-type test of the fixed value (RSiena's `test=TRUE`,
  Schweinberger 2012). Requires `fix=true`; the fit then reports the test in
  `result.score_test` (see [`SienaScoreTest`](@ref))
- `strict`: Throw (default) rather than warn when a requested effect does not exist
- `include`: `false` excludes the named effects (their `fix`/`test` flags are
  cleared) instead of including them

# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]; w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
add_covariate!(data, ConstantCovariate(:age, [30, 40, 50]))
effects = get_effects(data)
include_effects!(effects, :advice, [:egoX, :altX]; interaction1=:age)  # RSiena spelling
include_effects!(effects, :advice, [:transTrip]; fix=true, test=true)  # score test
length(get_included_effects(effects))      # 6: a rate, outdegree, recip and 3 added
include_effects!(effects, :advice, [:recip]; include=false)            # opt out
length(get_included_effects(effects))      # 5
```
"""
function include_effects!(effects::SienaEffects, variable::Symbol, effect_names::Vector{Symbol};
                         interaction1::Union{Nothing, Symbol}=nothing,
                         initial_value::Union{Nothing, Real}=nothing,
                         fix::Bool=false, test::Bool=false, strict::Bool=true,
                         include::Bool=true)
    !include && (fix || test || initial_value !== nothing) && throw(ArgumentError(
        "include=false excludes effects from the model; it cannot be combined with " *
        "fix=true, test=true or initial_value"))
    test && !fix && throw(ArgumentError(
        "test=true requires fix=true: the score-type test (RSiena's test=TRUE) " *
        "evaluates an effect held fixed at its initial value (usually 0) without " *
        "estimating it. Pass fix=true, test=true."))
    candidates = [entry for entry in effects.effects
                  if target_variable(entry.effect) == variable]
    selected = EffectEntry[]
    missed = Symbol[]
    for name in effect_names
        hits = [entry for entry in candidates if _matches(entry, name, interaction1)]
        if isempty(hits) && haskey(_SHORTNAME_ALIASES, name)
            alias = _SHORTNAME_ALIASES[name]
            hits = [entry for entry in candidates if _matches(entry, alias, interaction1)]
        end
        isempty(hits) ? push!(missed, name) : append!(selected, hits)
    end
    # Strict mode is atomic: a batch with an unknown name changes nothing.
    if isempty(missed) || !strict
        for entry in selected
            entry.include = include
            entry.fix = fix
            entry.test = test
            isnothing(initial_value) || (entry.initial_value = Float64(initial_value))
        end
    end
    if !isempty(missed)
        # Sorted for a deterministic message.
        missing_names = sort!(unique(missed); by=String)
        if strict
            available = sort!([entry.shortname for entry in candidates])
            throw(ArgumentError(
                "no effect(s) $(missing_names) for variable :$variable" *
                (interaction1 === nothing ? "" : " with interaction1=:$interaction1") *
                ". Available effects for :$variable: $(available). " *
                (any(in((:outdegree, :density, :linear)), missing_names) ?
                 "(As in RSiena, get_effects offers no outdegree/density or linear " *
                 "effect when every period of the variable only goes up, or every " *
                 "period only goes down; construct the variable with " *
                 "allow_only=false to model it without that restriction.) " : "") *
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
                         effect1::Symbol, effect2::Symbol[, effect3::Symbol];
                         interaction1=nothing, initial_value=0.0, fix=false,
                         test=false)

Add (and include) a user-defined interaction of two or three effects of `variable`
— RSiena's `includeInteraction()`. The components are named like in
[`include_effects!`](@ref): by Siena.jl short name (`:egosmoke1`, `:recip`), or by
RSiena short name with `interaction1` given as a tuple aligned with the effects
(`:egoX, :recip; interaction1=(:smoke1, nothing)`). The components themselves need
not be included in the model (as in RSiena, though including the main effects is
usually advisable).

For a network the result is an [`InteractionEffect`](@ref), for a behaviour a
[`BehaviorProductEffect`](@ref); both follow RSiena's definitions (product of the
change statistics) and RSiena's rules for which effects may interact — a
combination RSiena refuses is an `ArgumentError` here too. The new entry's short
name joins the components' with `_x_` and is returned in the effects object.

# Example
```julia
using Siena, NetworkCore
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip, :egosmoke1])
include_interaction!(effects, :friendship, :egosmoke1, :recip)
include_interaction!(effects, :friendship, :egoX, :transTrip;
                     interaction1=(:smoke1, nothing))        # RSiena spelling
[e.shortname for e in get_objective_effects(effects)]
```
"""
function include_interaction!(effects::SienaEffects, variable::Symbol,
                              names::Symbol...;
                              interaction1=nothing,
                              initial_value::Real=0.0, fix::Bool=false,
                              test::Bool=false)
    2 <= length(names) <= 3 || throw(ArgumentError(
        "include_interaction! takes two or three effect names, got $(length(names))"))
    test && !fix && throw(ArgumentError("test=true requires fix=true"))
    inter = interaction1 === nothing ? ntuple(_ -> nothing, length(names)) :
            interaction1 isa Symbol ? ntuple(_ -> interaction1, length(names)) :
            Tuple(interaction1)
    length(inter) == length(names) || throw(ArgumentError(
        "interaction1 must give one entry (a Symbol or nothing) per effect"))
    candidates = [entry for entry in effects.effects
                  if target_variable(entry.effect) == variable &&
                     !(entry.effect isa Union{InteractionEffect, BehaviorProductEffect})]
    isempty(candidates) && throw(ArgumentError(
        "no effects for variable :$variable in this effects object"))
    chosen = EffectEntry[]
    for (name, i1) in zip(names, inter)
        hits = [entry for entry in candidates if _matches(entry, name, i1)]
        if isempty(hits) && haskey(_SHORTNAME_ALIASES, name)
            hits = [entry for entry in candidates
                    if _matches(entry, _SHORTNAME_ALIASES[name], i1)]
        end
        if isempty(hits) && i1 === nothing
            # an RSiena name that needs no interaction1 beyond the entry's own
            hits = [entry for entry in candidates if effect_name(entry.effect) == name]
        end
        length(hits) == 1 || throw(ArgumentError(
            (isempty(hits) ? "no effect" : "several effects match") * " :$name" *
            (i1 === nothing ? "" : " with interaction1=:$i1") *
            " for variable :$variable. Available: " *
            "$(sort!([entry.shortname for entry in candidates]))"))
        push!(chosen, hits[1])
    end
    components = Tuple(entry.effect for entry in chosen)
    all(c -> effect_type(c) == :eval, components) || throw(ArgumentError(
        "interactions are defined for evaluation effects only"))
    eff = if components[1] isa NetworkEffect
        e = InteractionEffect(variable, components)
        kind = get(effects.kinds, variable, nothing)
        kind === nothing || _check_network_interaction(components, kind)
        e
    elseif components[1] isa BehaviorEffect
        BehaviorProductEffect(variable, components)
    else
        throw(ArgumentError("interactions of rate effects are not supported"))
    end
    shortname = join((entry.shortname for entry in chosen), "_x_")
    existing = findfirst(entry -> entry.shortname == shortname &&
                         target_variable(entry.effect) == variable, effects.effects)
    entry = EffectEntry(eff; name=join((entry.name for entry in chosen), " x "),
                        shortname=shortname, include=true, fix=fix, test=test,
                        initial_value=Float64(initial_value))
    if existing === nothing
        add_effect!(effects, entry)
    else
        effects.effects[existing] = entry
    end
    return effects
end

include("network_integration.jl")

#==============================================================================#
# Precompile workload
#==============================================================================#
# A small conditional and unconditional fit of the README's effect set, a
# simulation, GOF and `show`, so the first call in a session does not pay for
# compiling the estimator. (The objective effect set is tuple-typed, so the
# README's effect combination is the one compiled; other combinations compile their
# own contribution loop on first use.) Warnings of the tiny fits go to devnull.
@setup_workload begin
    _pc_n = 8
    _pc_w = [[Int(i != j && mod(3i + 5j + w * (i + j), 7) < 2) for i in 1:_pc_n, j in 1:_pc_n]
             for w in 1:3]
    _pc_null = Base.CoreLogging.ConsoleLogger(devnull, Base.CoreLogging.Error)
    @compile_workload begin
        Base.CoreLogging.with_logger(_pc_null) do
            _pc_data = siena_data()
            add_nodeset!(_pc_data, NodeSet(_pc_n))
            add_dependent!(_pc_data, DependentNetwork(:net, _pc_w))
            add_covariate!(_pc_data, ConstantCovariate(:x, collect(1.0:_pc_n)))
            _pc_eff = get_effects(_pc_data)
            include_effects!(_pc_eff, :net, [:outdegree, :recip, :transTrip, :altx,
                                             :egox, :simx])
            for _pc_cond in (nothing, false)
                _pc_alg = SienaAlgorithm(verbose=false, phase1_iterations=2,
                                         n_subphases=1, phase3_iterations=40,
                                         derivative_sims=4, refine_max=1,
                                         revalidate_max=1, conditional=_pc_cond)
                _pc_fit = fit_siena(_pc_data, _pc_eff; algorithm=_pc_alg,
                                    rng=Random.MersenneTwister(1))
                sprint(show, _pc_fit)
                coef(_pc_fit); stderror(_pc_fit); coeftable(_pc_fit); confint(_pc_fit)
                approximations(_pc_fit)
                try
                    sprint(show, siena_time_test(_pc_fit))
                catch
                end
                sprint(show, gof(_pc_fit; n_sim=3, rng=Random.Xoshiro(2)))
                gof(_pc_fit; n_sim=3, rng=Random.MersenneTwister(2))
            end
        end
    end
end

end # module
