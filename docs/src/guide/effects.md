# Effects

This guide covers the full library of SAOM effects available in Siena.jl. Effects are the building blocks of the model, each capturing a specific social mechanism.

## Overview

Effects are organized into four categories:

1. **Network effects** -- Structural and covariate effects on tie formation
2. **Behavior effects** -- Effects on behavior dynamics
3. **Rate effects** -- Effects on the rate of change
4. **Two-mode effects** -- Effects for bipartite networks

All effects subtype [`AbstractEffect`](@ref) and implement the core interface:

```julia
# Every effect must implement:
compute_contribution(effect, state, data, actor, alter)  # Contribution to objective function
effect_name(effect)                                       # Canonical name (Symbol)
effect_type(effect)                                       # :eval, :endow, :creation, or :rate
target_variable(effect)                                   # Which dependent variable
```

## Managing Effects

### Creating an Effects Object

Use [`get_effects`](@ref) to create a [`SienaEffects`](@ref) object populated with all available effects for your data:

```julia
effects = get_effects(data)
```

This automatically creates:
- Rate effects for each dependent variable and period (included by default)
- Structural effects for each network (available but not included)
- Covariate effects for each covariate-network combination (available but not included)
- Behavior effects for each behavior variable (available but not included)

### Including Effects

Use [`include_effects!`](@ref) to include effects in the model:

```julia
# Include by short name
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])

# With options
include_effects!(effects, :friendship, [:outdegree];
    initial_value=-1.0,   # Starting value for estimation
    fix=false,            # Whether to fix the parameter
    test=false            # Whether to perform score test
)
```

### Viewing Effects

```julia
# Summary
println(effects)  # SienaEffects(N total, M included)

# Full table as DataFrame
df = effects_table(effects)

# Get included effects only
included = get_included_effects(effects)

# Get rate effects
rates = get_rate_effects(effects)

# Get objective function effects (network + behavior)
obj = get_objective_effects(effects)
```

### Interaction Effects

```julia
include_interaction!(effects, :friendship, :recip, :transTrip)
```

## Structural Network Effects

These effects depend only on the network topology, not on covariates.

### Basic Effects

| Effect | Short Name | Description | Formula (contribution) |
|--------|-----------|-------------|----------------------|
| [`OutdegreeEffect`](@ref) | `outdegree` | Density (tendency to form ties) | 1 |
| [`ReciprocityEffect`](@ref) | `recip` | Reciprocity (mutual ties) | $x_{ji}$ |

**Outdegree** is almost always included as a baseline effect. A negative coefficient indicates a sparse network (actors prefer fewer ties).

**Reciprocity** captures the tendency for ties to be mutual. A positive coefficient means actors prefer to reciprocate ties they receive.

### Triadic Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`TransitiveTripletsEffect`](@ref) | `transTrip` | Number of transitive triplets |
| [`TransitiveTiesEffect`](@ref) | `transTies` | Whether any transitive path exists |
| [`TransitiveMediatedTripletsEffect`](@ref) | `transMedTrip` | Mediated transitive triplets |
| [`TransitiveRecipTripletsEffect`](@ref) | `transRecTrip` | Reciprocated transitive triplets |
| [`CyclicTripletsEffect`](@ref) | `cycle3` | Generalized 3-cycles |
| [`BalanceEffect`](@ref) | `balance` | Structural balance |
| [`BetweennessEffect`](@ref) | `between` | Betweenness centrality tendency |
| [`NbrDist2Effect`](@ref) | `nbrDist2` | Number of actors at distance 2 |
| [`DenseTriadsEffect`](@ref) | `denseTriads` | Triads with 5+ ties |
| [`SharedInEffect`](@ref) | `sharedIn` | Shared incoming ties |
| [`SharedOutEffect`](@ref) | `sharedOut` | Shared outgoing ties |

**TransitiveTripletsEffect** counts the number of transitive triplets involving the focal tie. It captures the "friends of friends become friends" mechanism.

**TransitiveTiesEffect** is a binary version: 1 if any transitive path exists through the focal tie, 0 otherwise. Often preferred over TransitiveTripletsEffect because it is less sensitive to high-degree nodes.

**CyclicTripletsEffect** counts generalized 3-cycles. A negative coefficient indicates hierarchical ordering in the network.

### Degree-Based Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`IndegreePopularityEffect`](@ref) | `inPop` / `inPopSqrt` | Popularity of alters (linear or sqrt) |
| [`OutdegreePopularityEffect`](@ref) | `outPop` / `outPopSqrt` | Outdegree of alters |
| [`IndegreeActivityEffect`](@ref) | `inAct` / `inActSqrt` | Activity based on indegree |
| [`OutdegreeActivityEffect`](@ref) | `outAct` / `outActSqrt` | Activity based on outdegree |
| [`OutdegreeTruncEffect`](@ref) | `outTrunc` | Truncated outdegree |
| [`IndegreeTruncEffect`](@ref) | `inTrunc` | Truncated indegree |
| [`DegreeAssortativityEffect`](@ref) | `degPlus` | Degree assortativity |

**IndegreePopularityEffect** captures preferential attachment: actors with many incoming ties attract more ties. The `sqrt` variant uses the square root, which is often more stable.

**OutdegreeActivityEffect** captures whether actors with many outgoing ties send even more. Often has a negative coefficient (diminishing returns on activity).

### Isolate Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`IsolateEffect`](@ref) | `isolate` | Complete isolate (no ties in or out) |
| [`IsolateNetEffect`](@ref) | `isolateNet` | No outgoing ties |
| [`OutIsolateEffect`](@ref) | `outIsolate` | Out-isolate indicator |
| [`InIsolateEffect`](@ref) | `inIsolate` | In-isolate indicator |

### GWESP Family

Geometrically weighted effects provide smooth alternatives to simple counting:

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`GWESPEffect`](@ref) | `gwespFF` | GWESP forward (transitive shared partners) |
| [`GWESPBackwardEffect`](@ref) | `gwespBB` | GWESP backward |
| [`GWESPMixedEffect`](@ref) | `gwespFB` | GWESP mixed (two-path) |
| [`GWDSPEffect`](@ref) | `gwdspFF` | Geometrically weighted dyadwise shared partners |

GWESP effects weight additional shared partners with decreasing marginal returns, controlled by the `alpha` parameter (default `log(2)`). They are often preferred over simple transitive triplets because they are less prone to model degeneracy.

```julia
# Custom alpha parameter
gwesp = GWESPEffect(:friendship; alpha=0.5)
```

## Covariate Network Effects

These effects depend on actor-level or dyadic covariates.

### Actor Covariate Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`EgoEffect`](@ref) | `egoX` | Ego's covariate value (sender effect) |
| [`EgoSqEffect`](@ref) | `egoSqX` | Squared ego covariate |
| [`AlterEffect`](@ref) | `altX` | Alter's covariate value (receiver effect) |
| [`AlterSqEffect`](@ref) | `altSqX` | Squared alter covariate |
| [`SimilarityEffect`](@ref) | `simX` | Similarity between ego and alter |
| [`SameEffect`](@ref) | `sameX` | Exact match on covariate value |
| [`DifferenceEffect`](@ref) | `diffX` | Ego minus alter difference |
| [`DifferenceSqEffect`](@ref) | `diffSqX` | Squared difference |
| [`AbsDifferenceEffect`](@ref) | `absDiffX` | Absolute difference |
| [`HigherEffect`](@ref) | `higher` | Ego higher than alter |
| [`EgoTimesAlterEffect`](@ref) | `egoXaltX` | Ego times alter product |
| [`EgoPlusAlterEffect`](@ref) | `egoPlusAltX` | Ego plus alter sum |

**EgoEffect** captures whether actors with higher covariate values send more ties. A positive coefficient means high-value actors are more active.

**AlterEffect** captures whether actors with higher covariate values receive more ties (popularity based on the covariate).

**SimilarityEffect** captures homophily for continuous covariates. A positive coefficient means actors prefer to tie with similar others.

**SameEffect** captures homophily for categorical covariates. A positive coefficient means actors prefer to tie with others who have the same value.

### Dyadic Covariate Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`DyadCovariateEffect`](@ref) | `X` | Direct dyadic covariate effect |

### Interaction Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`SameXRecipEffect`](@ref) | `sameXRecip` | Same covariate times reciprocity |
| [`SimXRecipEffect`](@ref) | `simXRecip` | Similarity times reciprocity |
| [`SimXTransTripEffect`](@ref) | `simXTransTrip` | Similarity times transitive triplets |

These effects test whether homophily is stronger for reciprocated ties or in transitive structures.

### Endowment and Creation Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`EndowmentEffect`](@ref) | varies | Effect active only for existing ties (dissolution) |
| [`CreationEffect`](@ref) | varies | Effect active only for new ties (creation) |

Wrap any network effect to test whether it operates differently for tie creation vs. dissolution:

```julia
# Reciprocity matters more for maintaining ties than creating them
recip_endow = EndowmentEffect(ReciprocityEffect(:friendship))
```

### Multiplex Effects

For models with multiple dependent networks:

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`CrossNetworkReciprocityEffect`](@ref) | `crprodRecip` | Reciprocity from another network |
| [`CrossNetworkActivityEffect`](@ref) | `crprodAct` | Activity in another network |
| [`CrossNetworkPopularityEffect`](@ref) | `crprodPop` | Popularity in another network |
| [`CrossNetworkTiesEffect`](@ref) | `crprod` | Tie in another network |

## Behavior Effects

Behavior effects model changes in actor-level behavior variables.

### Shape Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`LinearShapeEffect`](@ref) | `linear` | Linear tendency in behavior |
| [`QuadraticShapeEffect`](@ref) | `quad` | Quadratic shape (preference for extreme/central values) |
| [`CubicShapeEffect`](@ref) | `cubic` | Cubic shape |

**LinearShapeEffect** is the baseline behavior effect, analogous to outdegree for networks. A positive coefficient indicates a general tendency toward higher values.

**QuadraticShapeEffect** captures whether actors prefer extreme values (negative coefficient) or central values (positive coefficient) relative to the range midpoint.

### Network Influence Effects

These capture how an actor's network position influences their behavior:

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`AverageAlterEffect`](@ref) | `avAlt` | Average behavior of alters |
| [`TotalAlterEffect`](@ref) | `totAlt` | Total behavior of alters |
| [`AverageSimilarityEffect`](@ref) | `avSim` | Average similarity with alters |
| [`TotalSimilarityEffect`](@ref) | `totSim` | Total similarity with alters |
| [`AverageInAlterEffect`](@ref) | `avInAlt` | Average behavior of in-alters |
| [`AverageRecipAlterEffect`](@ref) | `avRecAlt` | Average behavior of reciprocal alters |
| [`AverageAttHigherEffect`](@ref) | `avAttHigher` | Fraction of alters with higher behavior |
| [`AverageAttLowerEffect`](@ref) | `avAttLower` | Fraction of alters with lower behavior |
| [`AverageAlterDist2Effect`](@ref) | `avAltDist2` | Average behavior at distance 2 |
| [`TotalInAlterEffect`](@ref) | `totInAlt` | Total behavior of in-alters |

**AverageAlterEffect** is the primary influence effect: actors tend to adopt the average behavior of their network neighbors. A positive coefficient indicates assimilation (convergence toward alters' behavior).

**AverageSimilarityEffect** is an alternative that models the tendency to become more similar to alters, regardless of direction.

### Degree Effects on Behavior

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`IndegreeEffect`](@ref) | `indeg` | Indegree affects behavior |
| [`BehaviorOutdegreeEffect`](@ref) | `outdeg` | Outdegree affects behavior |
| [`RecipDegreeEffect`](@ref) | `recipDeg` | Reciprocal degree affects behavior |

### Covariate Effects on Behavior

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`BehaviorCovariateEffect`](@ref) | `effFrom` | Covariate influences behavior |
| [`CovariateInteractionEffect`](@ref) | `covInt` | Covariate-behavior interaction |

### Behavior-Behavior Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`BehaviorInteractionEffect`](@ref) | `behBeh` | One behavior affects another |
| [`BehaviorSimilarityEffect`](@ref) | `simBeh` | Similarity on another behavior |

### Threshold and Other Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`ThresholdEffect`](@ref) | `threshold` | Behavior above threshold |
| [`PropThresholdEffect`](@ref) | `propThreshold` | Proportion of alters above threshold |
| [`BehaviorIsolateEffect`](@ref) | `behIsolate` | Isolate effect on behavior |
| [`FeedbackEffect`](@ref) | `feedback` | Feedback from network to behavior |
| [`MainBehaviorEffect`](@ref) | `main` | Constant tendency |

## Rate Effects

Rate effects control how often actors get opportunities to change.

### Basic Rate Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`BasicRateEffect`](@ref) | `rateXp` | Basic rate parameter for period p |
| [`CovariateRateEffect`](@ref) | `rateX` | Rate depends on covariate |

**BasicRateEffect** is always included by default. It sets the overall pace of change for each period.

### Degree-Based Rate Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`OutdegreeRateEffect`](@ref) | `outRateX` | Rate depends on outdegree |
| [`IndegreeRateEffect`](@ref) | `inRateX` | Rate depends on indegree |
| [`OutdegreeLogRateEffect`](@ref) | `outRateLog` | Rate depends on log(outdegree+1) |
| [`IndegreeLogRateEffect`](@ref) | `inRateLog` | Rate depends on log(indegree+1) |
| [`OutdegreeInvRateEffect`](@ref) | `outRateInv` | Rate depends on 1/(outdegree+1) |
| [`IndegreeInvRateEffect`](@ref) | `inRateInv` | Rate depends on 1/(indegree+1) |
| [`OutdegreeSqRateEffect`](@ref) | `outRateSq` | Rate depends on outdegree squared |
| [`RecipDegreeRateEffect`](@ref) | `recipRateX` | Rate depends on reciprocal degree |

### Behavior-Based Rate Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`BehaviorRateEffect`](@ref) | `behRateX` | Rate depends on own behavior |
| [`AverageAlterRateEffect`](@ref) | `avAltRateX` | Rate depends on average alter behavior |
| [`TotalAlterRateEffect`](@ref) | `totAltRateX` | Rate depends on total alter behavior |
| [`SimilarityRateEffect`](@ref) | `simRateX` | Rate depends on similarity with alters |

### Other Rate Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`SettingRateEffect`](@ref) | `settingRateX` | Rate for specific setting/group |
| [`EgoAlterRateEffect`](@ref) | `egoAltRateX` | Rate depends on ego-alter covariate product |
| [`CovariateSqRateEffect`](@ref) | `rateSqX` | Rate depends on squared covariate |

## Two-Mode Effects

Two-mode effects are for bipartite networks connecting actors to events/affiliations.

### Basic Two-Mode Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`TwoModeOutdegreeEffect`](@ref) | `outdegree2` | Outdegree (number of affiliations) |
| [`TwoModeIndegreeEffect`](@ref) | `indegree2` / `indegreeSqrt2` | Indegree (event popularity) |

### Structural Two-Mode Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`FourCyclesEffect`](@ref) | `fourCycles` | Four-cycles (two-mode transitivity) |
| [`SharedEventsEffect`](@ref) | `sharedEvents` / `sharedEventsSqrt` | Shared event count |
| [`GWESPTwoModeEffect`](@ref) | `gwesp2` | GWESP for two-mode networks |
| [`TwoModeTransitiveClosureEffect`](@ref) | `transClosure2` | Transitive closure |
| [`TwoModeActorAssortativityEffect`](@ref) | `actAssort2` | Actor degree assortativity |

### Covariate Two-Mode Effects

| Effect | Short Name | Description |
|--------|-----------|-------------|
| [`TwoModeEgoEffect`](@ref) | `ego2X` | Actor covariate effect |
| [`TwoModeEventEffect`](@ref) | `event2X` | Event attribute effect |
| [`TwoModeSameEffect`](@ref) | `same2X` | Same covariate at event |
| [`TwoModeSimilarityEffect`](@ref) | `sim2X` | Similarity at event |
| [`TwoModeActivityEffect`](@ref) | `activity2` / `activitySqrt2` | Alter activity |
| [`TwoModePopularityAltEffect`](@ref) | `popAlt2` | Popularity-activity interaction |
| [`TwoModeWithinEffect`](@ref) | `within2X` | Within-setting ties |

## Choosing Effects by Research Question

### Network Structure

```julia
# Basic: density + reciprocity
include_effects!(effects, :net, [:outdegree, :recip])

# Add transitivity
include_effects!(effects, :net, [:transTrip])  # or :transTies or :gwesp

# Add popularity/activity
include_effects!(effects, :net, [:inPopSqrt, :outActSqrt])
```

### Homophily

```julia
# Categorical attribute (e.g., gender)
include_effects!(effects, :net, [Symbol("samegender")])

# Continuous attribute (e.g., age)
include_effects!(effects, :net, [Symbol("simage"), Symbol("egoage"), Symbol("altage")])
```

### Peer Influence

```julia
# Average alter effect (primary influence mechanism)
include_effects!(effects, :behavior, [Symbol("avAltnet")])

# Or average similarity
include_effects!(effects, :behavior, [Symbol("avSimnet")])

# Shape effects (always include with behavior)
include_effects!(effects, :behavior, [:linear, :quad])
```

### Selection and Influence (Co-Evolution)

```julia
# Selection: network effects based on behavior similarity
include_effects!(effects, :net, [Symbol("simbehavior")])

# Influence: behavior effects based on network
include_effects!(effects, :behavior, [Symbol("avAltnet")])
```

## Effect Implementation Details

### The compute_contribution Interface

Each effect implements `compute_contribution(effect, state, data, actor, alter)`:

- For **network effects**: `actor` is the ego, `alter` is the potential tie partner
- For **behavior effects**: `actor` is the ego, `alter` encodes the direction (-1 or +1)
- Return value: `Float64` contribution to the objective function

### Creating Custom Effects

To create a new effect, define a struct and implement the required methods:

```julia
struct MyCustomEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::MyCustomEffect) = :myCustom
effect_type(::MyCustomEffect) = :eval
target_variable(e::MyCustomEffect) = e.variable

function compute_contribution(e::MyCustomEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    # Your custom logic here
    net = state.networks[e.variable]
    return Float64(...)  # Must return Float64
end
```
