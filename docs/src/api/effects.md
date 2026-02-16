# Effects API Reference

This page documents all SAOM effects and effect management functions in Siena.jl.

## Effect Type Hierarchy

### AbstractEffect

```@docs
AbstractEffect
```

### NetworkEffect

```@docs
NetworkEffect
```

### BehaviorEffect

```@docs
BehaviorEffect
```

### RateEffect

```@docs
RateEffect
```

## Effect Interface

### effect_name

```@docs
effect_name
```

### effect_type

```@docs
effect_type
```

### target_variable

```@docs
target_variable
```

### interaction_with

```@docs
interaction_with
```

### compute_contribution

```@docs
compute_contribution
```

### compute_statistic

```@docs
compute_statistic
```

## Structural Network Effects -- Basic

### OutdegreeEffect

```@docs
OutdegreeEffect
```

### ReciprocityEffect

```@docs
ReciprocityEffect
```

## Structural Network Effects -- Triadic

### TransitiveTripletsEffect

```@docs
TransitiveTripletsEffect
```

### TransitiveTiesEffect

```@docs
TransitiveTiesEffect
```

### TransitiveMediatedTripletsEffect

```@docs
TransitiveMediatedTripletsEffect
```

### TransitiveRecipTripletsEffect

```@docs
TransitiveRecipTripletsEffect
```

### CyclicTripletsEffect

```@docs
CyclicTripletsEffect
```

### BalanceEffect

```@docs
BalanceEffect
```

### BetweennessEffect

```@docs
BetweennessEffect
```

### NbrDist2Effect

```@docs
NbrDist2Effect
```

### DenseTriadsEffect

```@docs
DenseTriadsEffect
```

### SharedInEffect

```@docs
SharedInEffect
```

### SharedOutEffect

```@docs
SharedOutEffect
```

## Structural Network Effects -- Degree-Based

### IndegreePopularityEffect

```@docs
IndegreePopularityEffect
```

### OutdegreePopularityEffect

```@docs
OutdegreePopularityEffect
```

### IndegreeActivityEffect

```@docs
IndegreeActivityEffect
```

### OutdegreeActivityEffect

```@docs
OutdegreeActivityEffect
```

### OutdegreeTruncEffect

```@docs
OutdegreeTruncEffect
```

### IndegreeTruncEffect

```@docs
IndegreeTruncEffect
```

### DegreeAssortativityEffect

```@docs
DegreeAssortativityEffect
```

## Structural Network Effects -- Isolate

### IsolateEffect

```@docs
IsolateEffect
```

### IsolateNetEffect

```@docs
IsolateNetEffect
```

### OutIsolateEffect

```@docs
OutIsolateEffect
```

### InIsolateEffect

```@docs
InIsolateEffect
```

## Structural Network Effects -- GWESP Family

### GWESPEffect

```@docs
GWESPEffect
```

### GWESPBackwardEffect

```@docs
GWESPBackwardEffect
```

### GWESPMixedEffect

```@docs
GWESPMixedEffect
```

### GWDSPEffect

```@docs
GWDSPEffect
```

## Covariate Network Effects

### EgoEffect

```@docs
EgoEffect
```

### EgoSqEffect

```@docs
EgoSqEffect
```

### AlterEffect

```@docs
AlterEffect
```

### AlterSqEffect

```@docs
AlterSqEffect
```

### SimilarityEffect

```@docs
SimilarityEffect
```

### SameEffect

```@docs
SameEffect
```

### DifferenceEffect

```@docs
DifferenceEffect
```

### DifferenceSqEffect

```@docs
DifferenceSqEffect
```

### AbsDifferenceEffect

```@docs
AbsDifferenceEffect
```

### HigherEffect

```@docs
HigherEffect
```

### EgoTimesAlterEffect

```@docs
EgoTimesAlterEffect
```

### EgoPlusAlterEffect

```@docs
EgoPlusAlterEffect
```

### DyadCovariateEffect

```@docs
DyadCovariateEffect
```

### SameXRecipEffect

```@docs
SameXRecipEffect
```

### SimXRecipEffect

```@docs
SimXRecipEffect
```

### SimXTransTripEffect

```@docs
SimXTransTripEffect
```

## Endowment and Creation Effects

### EndowmentEffect

```@docs
EndowmentEffect
```

### CreationEffect

```@docs
CreationEffect
```

## Multiplex Network Effects

### CrossNetworkReciprocityEffect

```@docs
CrossNetworkReciprocityEffect
```

### CrossNetworkActivityEffect

```@docs
CrossNetworkActivityEffect
```

### CrossNetworkPopularityEffect

```@docs
CrossNetworkPopularityEffect
```

### CrossNetworkTiesEffect

```@docs
CrossNetworkTiesEffect
```

## Behavior Effects -- Shape

### LinearShapeEffect

```@docs
LinearShapeEffect
```

### QuadraticShapeEffect

```@docs
QuadraticShapeEffect
```

### CubicShapeEffect

```@docs
CubicShapeEffect
```

## Behavior Effects -- Network Influence

### AverageAlterEffect

```@docs
AverageAlterEffect
```

### TotalAlterEffect

```@docs
TotalAlterEffect
```

### AverageSimilarityEffect

```@docs
AverageSimilarityEffect
```

### TotalSimilarityEffect

```@docs
TotalSimilarityEffect
```

### AverageInAlterEffect

```@docs
AverageInAlterEffect
```

### AverageRecipAlterEffect

```@docs
AverageRecipAlterEffect
```

### AverageAttHigherEffect

```@docs
AverageAttHigherEffect
```

### AverageAttLowerEffect

```@docs
AverageAttLowerEffect
```

### AverageAlterDist2Effect

```@docs
AverageAlterDist2Effect
```

### TotalInAlterEffect

```@docs
TotalInAlterEffect
```

## Behavior Effects -- Degree-Based

### IndegreeEffect

```@docs
IndegreeEffect
```

### BehaviorOutdegreeEffect

```@docs
BehaviorOutdegreeEffect
```

### RecipDegreeEffect

```@docs
RecipDegreeEffect
```

## Behavior Effects -- Covariate

### BehaviorCovariateEffect

```@docs
BehaviorCovariateEffect
```

### CovariateInteractionEffect

```@docs
CovariateInteractionEffect
```

## Behavior Effects -- Behavior Interaction

### BehaviorInteractionEffect

```@docs
BehaviorInteractionEffect
```

### BehaviorSimilarityEffect

```@docs
BehaviorSimilarityEffect
```

## Behavior Effects -- Threshold and Other

### ThresholdEffect

```@docs
ThresholdEffect
```

### PropThresholdEffect

```@docs
PropThresholdEffect
```

### BehaviorIsolateEffect

```@docs
BehaviorIsolateEffect
```

### FeedbackEffect

```@docs
FeedbackEffect
```

### MainBehaviorEffect

```@docs
MainBehaviorEffect
```

## Rate Effects -- Basic

### BasicRateEffect

```@docs
BasicRateEffect
```

### CovariateRateEffect

```@docs
CovariateRateEffect
```

## Rate Effects -- Degree-Based

### OutdegreeRateEffect

```@docs
OutdegreeRateEffect
```

### IndegreeRateEffect

```@docs
IndegreeRateEffect
```

### OutdegreeLogRateEffect

```@docs
OutdegreeLogRateEffect
```

### IndegreeLogRateEffect

```@docs
IndegreeLogRateEffect
```

### OutdegreeInvRateEffect

```@docs
OutdegreeInvRateEffect
```

### IndegreeInvRateEffect

```@docs
IndegreeInvRateEffect
```

### OutdegreeSqRateEffect

```@docs
OutdegreeSqRateEffect
```

### RecipDegreeRateEffect

```@docs
RecipDegreeRateEffect
```

## Rate Effects -- Behavior-Based

### BehaviorRateEffect

```@docs
BehaviorRateEffect
```

### AverageAlterRateEffect

```@docs
AverageAlterRateEffect
```

### TotalAlterRateEffect

```@docs
TotalAlterRateEffect
```

### SimilarityRateEffect

```@docs
SimilarityRateEffect
```

## Rate Effects -- Other

### SettingRateEffect

```@docs
SettingRateEffect
```

### EgoAlterRateEffect

```@docs
EgoAlterRateEffect
```

### CovariateSqRateEffect

```@docs
CovariateSqRateEffect
```

## Two-Mode Network Effects

### TwoModeEffect

```@docs
TwoModeEffect
```

### TwoModeOutdegreeEffect

```@docs
TwoModeOutdegreeEffect
```

### TwoModeIndegreeEffect

```@docs
TwoModeIndegreeEffect
```

### FourCyclesEffect

```@docs
FourCyclesEffect
```

### SharedEventsEffect

```@docs
SharedEventsEffect
```

### GWESPTwoModeEffect

```@docs
GWESPTwoModeEffect
```

### TwoModeEgoEffect

```@docs
TwoModeEgoEffect
```

### TwoModeEventEffect

```@docs
TwoModeEventEffect
```

### TwoModeSameEffect

```@docs
TwoModeSameEffect
```

### TwoModeSimilarityEffect

```@docs
TwoModeSimilarityEffect
```

### TwoModeActivityEffect

```@docs
TwoModeActivityEffect
```

### TwoModePopularityAltEffect

```@docs
TwoModePopularityAltEffect
```

### TwoModeTransitiveClosureEffect

```@docs
TwoModeTransitiveClosureEffect
```

### TwoModeActorAssortativityEffect

```@docs
TwoModeActorAssortativityEffect
```

### TwoModeWithinEffect

```@docs
TwoModeWithinEffect
```

## Effects Management Functions

### get_effects

```@docs
get_effects
```

### include_effects!

```@docs
include_effects!
```

### include_interaction!

```@docs
include_interaction!
```

### get_included_effects

```@docs
get_included_effects
```

### get_rate_effects

```@docs
get_rate_effects
```

### get_objective_effects

```@docs
get_objective_effects
```

### effects_table

```@docs
effects_table
```
