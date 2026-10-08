# Effects

Effects are the statistics of the evaluation function (how actors choose) and of the
rate function (how often they get to choose). This guide lists **exactly** the
effects Siena.jl implements, for which kind of dependent variable `get_effects`
offers each of them, its RSiena counterpart, and whether its target statistic is
pinned against RSiena.

## The effects table

[`get_effects`](@ref) builds the effects table for the data — the counterpart of
RSiena's `getEffects()`. Every effect that is defined for a dependent variable is
registered for it, and the effects RSiena's `getEffects()` includes are included, so
the default model is RSiena's:

| Dependent variable | Included by default |
|:--|:--|
| directed one-mode network | the basic rate of every period, `outdegree` (RSiena's `density`), `recip` |
| undirected network | the basic rates, `outdegree` |
| two-mode network | the basic rates, `outdegree` |
| behaviour | the basic rates, `linear`; `quad` when the observed range (maximum minus minimum over all waves) is at least 2 |

Everything else is registered but excluded. As in RSiena, a network whose every
period only adds ties (or every period only drops ties) gets no `outdegree` effect,
and a behaviour whose every period only rises (or only falls) gets no `linear`
effect. In such an *up-only* (*down-only*) period the simulation lets actors only add
(drop) ties or raise (lower) their behaviour, RSiena's `allowOnly = TRUE`; construct
the variable with `allow_only=false` for RSiena's `allowOnly = FALSE`. The included
sets are pinned against `getEffects()$include` on ten data sets
(`test/fixtures/s50_defaults.toml`).

What is offered depends on the kind of variable, following RSiena's effect groups:
directed one-mode networks, undirected (symmetric) networks, two-mode networks and
behaviour. An effect included for a variable it is not defined for is refused by
[`validate_effects`](@ref) (which [`fit_siena`](@ref), [`simulate_saom`](@ref) and the
GOF functions call), never computed on the wrong index set.

```julia
using Siena, NetworkCore, Random

s50 = load_dataset(:s50)
data = siena_data()
add_nodeset!(data, NodeSet(50))
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))

effects = get_effects(data)
println(effects)                               # SienaEffects(N total, M included)
table = effects_table(effects)                 # DataFrame: name, shortname, ...
first(table, 5)
```

### Selecting effects

[`include_effects!`](@ref) takes Siena.jl short names, or RSiena's short names
together with RSiena's `interaction1`:

Effects RSiena includes by default are already in the model, so an RSiena script's
habitual `includeEffects` lines translate one to one, and naming a default effect
again (`:outdegree`, `:density`, `:recip`, `:linear`, `:quad`) changes nothing.

```julia
# Siena.jl short names: the covariate/network is part of the name
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])
include_effects!(effects, :friendship, [:egosmoke1, :altsmoke1, :simsmoke1])

# RSiena spelling: shortName + interaction1
include_effects!(effects, :friendship, [:egoX, :altX, :simX]; interaction1=:alcohol)
include_effects!(effects, :alcohol, [:linear, :quad])
include_effects!(effects, :alcohol, [:avSim]; interaction1=:friendship)

# A fixed effect with a score-type test (RSiena's fix = TRUE, test = TRUE)
include_effects!(effects, :friendship, [:cycle3]; fix=true, test=true)
```

An unknown name is an `ArgumentError` that lists the effects available for the
variable (pass `strict=false` to warn and skip instead); a strict batch with an
unknown name changes nothing. `include=false` removes effects from the model, as
RSiena's `includeEffects(..., include = FALSE)`:

```julia
include_effects!(effects, :alcohol, [:quad]; include=false)   # linear shape only
```

### Effects with a non-default parameter

The table holds each effect with RSiena's default internal parameter. Add a variant
with [`add_effect!`](@ref) — the counterpart of RSiena's `setEffect(..., parameter=)`:

```julia
add_effect!(effects, EffectEntry(OutdegreeTruncEffect(:friendship; c=3);
                                 shortname="outTrunc3", include=true))
add_effect!(effects, EffectEntry(GWESPEffect(:friendship; alpha=0.5);
                                 shortname="gwespFF50", include=true))
```

### Network selection on a co-evolving behaviour

The covariate effects (`egoX`, `altX`, `simX`, `sameX`, `diffX`, `higher`, …) accept a
**dependent behaviour** as their attribute, as in RSiena: `egoalcohol`,
`altalcohol`, `simalcohol` are registered beside `egosmoke1`, …. During simulation
they read the behaviour's current value, so selection and influence co-evolve; in
the target statistics the behaviour is taken at the start of each period, RSiena's
convention. The friendship–alcohol selection-and-influence model is pinned against
RSiena (`test/fixtures/s50_coevolution.toml`).

## Concordance with RSiena

The **Pinned** column says whether the target statistic is compared with
`RSiena:::getTargets` in `test/fixtures/s50_targets.toml` (every effect offered
under an RSiena short name is). In the Siena.jl short names, `X` stands for the
covariate's or behaviour's name and `W` for the network's.

### Directed networks

| Siena.jl short name | Type | RSiena | Pinned |
|:--|:--|:--|:--|
| `outdegree` | [`OutdegreeEffect`](@ref) | `density` | yes |
| `recip` | [`ReciprocityEffect`](@ref) | `recip` | yes |
| `transTrip` | [`TransitiveTripletsEffect`](@ref) | `transTrip` | yes |
| `transMedTrip` | [`TransitiveMediatedTripletsEffect`](@ref) | `transMedTrip` | yes |
| `transRecTrip` | [`TransitiveRecipTripletsEffect`](@ref) | `transRecTrip` | yes |
| `cycle3` | [`CyclicTripletsEffect`](@ref) | `cycle3` | yes |
| `transTies` | [`TransitiveTiesEffect`](@ref) | `transTies` | yes |
| `between` | [`BetweennessEffect`](@ref) | `between` | yes |
| `nbrDist2` | [`NbrDist2Effect`](@ref) | `nbrDist2` | yes |
| `denseTriads` | [`DenseTriadsEffect`](@ref) | `denseTriads` | yes |
| `gwespFF`, `gwespBB`, `gwespFB` | [`GWESPEffect`](@ref), [`GWESPBackwardEffect`](@ref), [`GWESPMixedEffect`](@ref) | same (α = 0.69) | yes |
| `gwdspFF` | [`GWDSPEffect`](@ref) | `gwdspFF` | yes |
| `inPop`, `inPopSqrt` | [`IndegreePopularityEffect`](@ref) | same | yes |
| `outPop`, `outPopSqrt` | [`OutdegreePopularityEffect`](@ref) | same (with `parm`) | yes |
| `inAct`, `inActSqrt` | [`IndegreeActivityEffect`](@ref) | same (with `parm`) | yes |
| `outAct`, `outActSqrt` | [`OutdegreeActivityEffect`](@ref) | same | yes |
| `outTrunc` | [`OutdegreeTruncEffect`](@ref) | `outTrunc` (c = 1; `outTrunc2` is c = 5) | yes |
| `isolateNet` | [`IsolateNetEffect`](@ref) | `isolateNet` | yes |
| `outIso` | [`OutIsolateEffect`](@ref) | `outIso` | yes |
| `egoX`, `altX`, `simX`, `sameX` | [`EgoEffect`](@ref), [`AlterEffect`](@ref), [`SimilarityEffect`](@ref), [`SameEffect`](@ref) | same | yes |
| `egoSqX`, `altSqX` | [`EgoSqEffect`](@ref), [`AlterSqEffect`](@ref) | same | yes |
| `diffX`, `diffSqX`, `absDiffX` | [`DifferenceEffect`](@ref), [`DifferenceSqEffect`](@ref), [`AbsDifferenceEffect`](@ref) | same | yes |
| `higherX` | [`HigherEffect`](@ref) | `higher` | yes |
| `egoXaltX`, `egoPlusAltX` | [`EgoTimesAlterEffect`](@ref), [`EgoPlusAlterEffect`](@ref) | same | yes |
| `sameXRecip`, `simRecipX`, `simXTransTrip` | [`SameXRecipEffect`](@ref), [`SimXRecipEffect`](@ref), [`SimXTransTripEffect`](@ref) | same | yes |
| `dyadX` | [`DyadCovariateEffect`](@ref) | `X` | yes |
| `crprodW`, `crprodRecipW` | [`CrossNetworkTiesEffect`](@ref), [`CrossNetworkReciprocityEffect`](@ref) | `crprod`, `crprodRecip` | yes |
| `crprodActW`, `crprodPopW` | [`CrossNetworkActivityEffect`](@ref), [`CrossNetworkPopularityEffect`](@ref) | — (Siena.jl) | no |
| `sharedInNbrs`, `sharedOutNbrs` | [`SharedInEffect`](@ref), [`SharedOutEffect`](@ref) | — (Siena.jl) | no |
| `inTrunc`, `inIsolate` | [`IndegreeTruncEffect`](@ref), [`InIsolateEffect`](@ref) | — (Siena.jl) | no |
| `balanceSimple` | [`BalanceSimpleEffect`](@ref) | approximates `balance` | no |

`outPopSqrt`, `inActSqrt`, `outPop` and `inAct` carry RSiena's internal parameter
`parm` (defaults 0, 0, 1, 1). It changes only the *moment* statistic, exactly as in
RSiena 1.6.6: at `parm = 0` the degree in it is the one at the start of the period
(and, for `outPopSqrt`, not square-rooted), at `parm ≥ 1` the end-of-period
statistic of the RSiena manual.

### Undirected networks

`DependentNetwork(...; directed=false)`; RSiena's model type 2 (unilateral
initiative, reciprocal confirmation).

| Siena.jl short name | Type | RSiena | Pinned |
|:--|:--|:--|:--|
| `outdegree` | [`OutdegreeEffect`](@ref) | `density` (degree; an edge counts once) | yes |
| `transTriads` | [`TransitiveTriadsEffect`](@ref) | `transTriads` | yes |
| `transTies`, `between`, `nbrDist2` | as for directed networks | same (`nbrDist2` counts pairs once) | yes |
| `gwesp` | [`GWESPEffect`](@ref) | `gwesp` | yes |
| `inPop`, `inPopSqrt`, `outAct`, `outActSqrt` | as for directed networks | degree of alter / of ego | yes |
| `degPlus` | [`DegreeAssortativityEffect`](@ref) | `degPlus` | yes |
| `outTrunc`, `isolateNet`, `outIso` | as for directed networks | same | yes |
| covariate effects, `dyadX` | as for directed networks (no `…Recip`, no `simXTransTrip`) | same | yes |

### Behaviour

`W` is a network on the same actors.

| Siena.jl short name | Type | RSiena | Pinned |
|:--|:--|:--|:--|
| `linear`, `quad` | [`LinearShapeEffect`](@ref), [`QuadraticShapeEffect`](@ref) | same | yes |
| (via `add_effect!`) | [`ThresholdEffect`](@ref) | `threshold` | yes |
| `avAltW`, `avSimW`, `totAltW`, `totSimW` | [`AverageAlterEffect`](@ref), [`AverageSimilarityEffect`](@ref), [`TotalAlterEffect`](@ref), [`TotalSimilarityEffect`](@ref) | same | yes |
| `avInAltW`, `avRecAltW`, `totInAltW` | [`AverageInAlterEffect`](@ref), [`AverageRecipAlterEffect`](@ref), [`TotalInAlterEffect`](@ref) | same (directed only) | yes |
| `avAltDist2W` | [`AverageAlterDist2Effect`](@ref) | `avAltDist2` | yes |
| `indegW`, `outdegW`, `recipDegW` | [`IndegreeEffect`](@ref), [`BehaviorOutdegreeEffect`](@ref), [`RecipDegreeEffect`](@ref) | same | yes |
| `effFromX` | [`BehaviorCovariateEffect`](@ref) | `effFrom` | yes |
| `cubic`, `covIntX`, `behBehB`, `behIsolateW`, `simProdW` | [`CubicShapeEffect`](@ref), [`CovariateInteractionEffect`](@ref), [`BehaviorInteractionEffect`](@ref), [`BehaviorIsolateEffect`](@ref), [`FeedbackEffect`](@ref) | — (Siena.jl) | no |
| `avAttHigherSimpleW`, `avAttLowerSimpleW` | [`AverageAttHigherSimpleEffect`](@ref), [`AverageAttLowerSimpleEffect`](@ref) | approximate `avAttHigher`/`avAttLower` | no |
| (via `add_effect!`) | [`PropThresholdEffect`](@ref), [`BehaviorSimilarityEffect`](@ref), [`MainBehaviorEffect`](@ref) | — (Siena.jl) | no |

### Rate

| Siena.jl short name | Type | RSiena | Pinned |
|:--|:--|:--|:--|
| `rate1`, `rate2`, … | [`BasicRateEffect`](@ref) | `Rate` | yes |
| `outRate`, `outRateLog`, `outRateInv` | [`OutdegreeRateEffect`](@ref), [`OutdegreeLogRateEffect`](@ref), [`OutdegreeInvRateEffect`](@ref) | same | yes |
| `inRate`, `inRateLog`, `inRateInv`, `recipRate` | [`IndegreeRateEffect`](@ref), [`IndegreeLogRateEffect`](@ref), [`IndegreeInvRateEffect`](@ref), [`RecipDegreeRateEffect`](@ref) | same (directed only) | yes |
| `RateX` | [`CovariateRateEffect`](@ref) | `RateX` | yes |
| `outRateSq` and (via `add_effect!`) the behaviour, setting and covariate-square rate effects | [`OutdegreeSqRateEffect`](@ref), [`BehaviorRateEffect`](@ref), [`AverageAlterRateEffect`](@ref), [`TotalAlterRateEffect`](@ref), [`SimilarityRateEffect`](@ref), [`SettingRateEffect`](@ref), [`EgoAlterRateEffect`](@ref), [`CovariateSqRateEffect`](@ref) | — (Siena.jl) | no |

### Two-mode networks

RSiena's bipartite effects are offered under RSiena's names and pinned against
RSiena on a deterministic 50 × 12 affiliation panel; the remaining two-mode effects
are Siena.jl's own (no RSiena counterpart, not validated against RSiena).

| Siena.jl short name | Type | RSiena | Pinned |
|:--|:--|:--|:--|
| `outdegree` | [`OutdegreeEffect`](@ref) | `density` | yes |
| `cycle4` (`fourCycles` accepted) | [`FourCyclesEffect`](@ref) | `cycle4` (parameter 1) | yes |
| `inPop`, `inPopSqrt` | [`IndegreePopularityEffect`](@ref) | same (event popularity) | yes |
| `outAct`, `outActSqrt` | [`OutdegreeActivityEffect`](@ref) | same | yes |
| `outTrunc`, `outIso` | [`OutdegreeTruncEffect`](@ref), [`OutIsolateEffect`](@ref) | same | yes |
| `egoX`, `egoSqX` (covariate or behaviour) | [`EgoEffect`](@ref), [`EgoSqEffect`](@ref) | same | yes |
| `dyadX` (actor × event covariate) | [`DyadCovariateEffect`](@ref) | `X` | yes |
| rate: `outRate`, `outRateLog`, `outRateInv`, `RateX` | as for one-mode networks | same | yes |
| `sharedEvents`, `sharedEventsSqrt`, `gwesp2`, `activity2`, `activitySqrt2`, `popAlt2`, `transClosure2`, `actAssort2`, `same2X`, `sim2X` | [`SharedEventsEffect`](@ref), [`GWESPTwoModeEffect`](@ref), [`TwoModeActivityEffect`](@ref), [`TwoModePopularityAltEffect`](@ref), [`TwoModeTransitiveClosureEffect`](@ref), [`TwoModeActorAssortativityEffect`](@ref), [`TwoModeSameEffect`](@ref), [`TwoModeSimilarityEffect`](@ref) | — (Siena.jl) | no |
| (via `add_effect!`) | [`TwoModeOutdegreeEffect`](@ref), [`TwoModeIndegreeEffect`](@ref), [`TwoModeEgoEffect`](@ref), [`TwoModeEventEffect`](@ref) (duplicates of `density`, `inPop`, `egoX`, `X`), [`TwoModeWithinEffect`](@ref) | — (Siena.jl) | no |

Effects linking a two-mode network to a one-mode network (RSiena's `to`, `from`,
`sharedTo`, …) are not implemented.

### Interactions

[`include_interaction!`](@ref) adds a two- or three-way interaction of effects of
the same dependent variable — RSiena's `includeInteraction`:

```julia
include_interaction!(effects, :friendship, :egosmoke1, :recip)
include_interaction!(effects, :friendship, :egoX, :altX, :recip;
                     interaction1=(:smoke1, :smoke1, nothing))     # RSiena spelling
include_interaction!(effects, :alcohol, :avAltfriendship, :effFromsmoke1)
```

As in RSiena an interaction is an *elementary* effect: its change statistic is the
product of the components' change statistics ([`InteractionEffect`](@ref); for
behaviour [`BehaviorProductEffect`](@ref)). RSiena's rules decide what may interact.
Every network effect is of interaction type **ego** (`egoX`, `egoSqX`, `inAct`,
`inActSqrt`; `density` on an undirected network), **dyadic** (`density`, `recip`,
`inPop`, `outPop`, `outPopSqrt`, the GWESP effects, `altX`, `altSqX`, `simX`, `sameX`,
`diffX`, `diffSqX`, `absDiffX`, `higher`, `egoXaltX`, `egoPlusAltX`, `sameXRecip`,
`simRecipX`, `simXTransTrip`, `X`) or **neither** (`transTrip`, `cycle3`, `outAct`,
…). A two-way interaction needs at least one ego effect or two dyadic effects; a
three-way interaction at least two ego effects, or three ego/dyadic effects. For
behaviour, at most one component may be of a type other than RSiena's `OK` (OK:
`linear`, `avAlt`, `totAlt`, `avInAlt`, `avRecAlt`, `totInAlt`, `avAltDist2`,
`indeg`, `outdeg`, `recipDeg`, `effFrom`). A combination RSiena refuses is an
`ArgumentError`.

Twenty-six interaction targets (two- and three-way; directed, undirected, two-mode
and behaviour) are pinned against RSiena, and a fitted model with `egoX × recip` is
compared with RSiena fits.

### Elementary effects

RSiena implements the GWESP effects (`gwespFF`, `gwespBB`, `gwespFB`, `gwesp`) as
elementary effects: the change statistic of a tie is the weight of that tie's own
shared partners, not the full change of the actor statistic. Siena.jl does the same
(before 0.2 it used the full difference — the same target statistic, a different
model). `simXTransTrip` likewise uses RSiena's own change statistic, and `totSim`
RSiena's: the change of the summed similarities minus outdegree × mean similarity
for a step in either direction. The fitted `gwespFF` model is compared with RSiena
fits.

Because target statistics cannot detect such differences, every RSiena-named effect
is also pinned by simulation: RSiena simulates 38 small models at fixed parameters
(`test/fixtures/r/s50_dynamics.R`) and Siena.jl must reproduce the mean of every
simulated statistic within Monte-Carlo error.

### Endowment and creation effects

[`EndowmentEffect`](@ref) and [`CreationEffect`](@ref) wrap a network effect so that
it enters only tie dissolution or only tie creation. They work in simulation, but
estimation refuses them (their moment statistics are not implemented).

### RSiena effects that are not implemented

For a directed network with an actor covariate, RSiena 1.6.6 offers, among others,
these effects that Siena.jl does **not** implement: `antiInIso`, `antiIso`, `avDeg`,
`balance`, `cycle4`, `divIn_ego`/`divOut_ego`, `gwdspFB`, `gwespBF`, `gwespRR`,
`in2Plus`, `inAct.c`/`inPop.c`/`outAct.c`/`outPop.c`, `inStructEq`, `IndTies`,
`Jin`/`Jout`, `outInv`, `outMore`, `outThreshold`, `reciAct`, `reciPop`,
`sharedPop`, `transTrip1`/`transTrip2`, the `…Dist2` covariate effects (`altDist2`,
`totDist2`, `simDist2`, …), the threshold and degree-difference covariate effects
(`altLThresholdX`, `degAbsDiffX`, …), the covariate × structure interactions
(`sameXTransTrip`, `diffXTransTrip`, `homXOutAct`, `inPopX`, `outActX`, …), and the
primary-setting effects (`primary`, `primDegAct`, …). For behaviour: RSiena's
`isolate`/`outIsolate`, `avAttHigher`/`avAttLower`, the `max`/`min`/`var` alter
effects, the exposure and infection effects, the group effects and the
behaviour × covariate interactions (`avAltEgoX`, `avSimEgoX`, …). For two-mode
networks: `gwdspFB`, `antiInIso`, the mixed one-mode/two-mode effects (`to`, `from`,
…) and the covariate × structure effects.

## Writing a custom effect

A new effect is a concrete subtype of `NetworkEffect`, `BehaviorEffect` or
`RateEffect` implementing [`evaluate_actor`](@ref) (or [`rate_score`](@ref)), with
an optional closed-form [`compute_contribution`](@ref). For a regular effect the
change statistic is the add-direction difference of the actor's statistic; the test
suite checks every built-in closed form against a brute-force toggle (the elementary
effects above are the deliberate exceptions).

```julia
struct TwoStarsEffect <: NetworkEffect      # ego's number of out-two-stars
    variable::Symbol
end
Siena.effect_name(::TwoStarsEffect) = :outTwoStars
Siena.effect_type(::TwoStarsEffect) = :eval
Siena.target_variable(e::TwoStarsEffect) = e.variable
function Siena.evaluate_actor(e::TwoStarsEffect, state::NetworkState,
                              data::SienaData, actor::Int)
    d = sum(state.networks[e.variable][actor, :])
    return d * (d - 1) / 2
end

custom = get_effects(data)
add_effect!(custom, EffectEntry(TwoStarsEffect(:friendship); include=true))
validate_effects(data, custom)
```

A user-defined effect is checked like a built-in one, as a directed-network effect
unless it adds a method to `Siena._supported_kinds`.
