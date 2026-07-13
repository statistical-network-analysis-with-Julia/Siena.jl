# Behavior Effects

Effects on a dependent **behavior** -- the shape of the behavior's own
objective function, network influence (contagion), degree-based effects,
covariate effects, behavior interactions and thresholds.

See the [Effects API Reference](effects.md) for the type hierarchy and the
effect interface these all implement.

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

### AverageAttHigherSimpleEffect

```@docs
AverageAttHigherSimpleEffect
```

### AverageAttLowerSimpleEffect

```@docs
AverageAttLowerSimpleEffect
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

