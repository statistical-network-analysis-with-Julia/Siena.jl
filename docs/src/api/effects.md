# Effects API Reference

This page documents the SAOM effect type hierarchy, the effect interface
every effect implements, and the functions for managing an effects set.

The effects are documented by family:

- **[Network Effects](effects_network.md)** -- structural (basic, triadic,
  degree-based, isolate, GWESP), covariate, endowment/creation and multiplex
  effects on a dependent network.
- **[Behavior Effects](effects_behavior.md)** -- shape, network influence,
  degree-based, covariate, interaction and threshold effects on a dependent
  behavior.
- **[Rate Effects](effects_rate.md)** -- the effects entering the rate function.
- **[Two-Mode Effects](effects_twomode.md)** -- effects on a two-mode
  (bipartite) dependent network.

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

### evaluate_actor

```@docs
evaluate_actor
```

### compute_contribution

```@docs
compute_contribution
```

### compute_statistic

```@docs
compute_statistic
```

### rate_score

```@docs
rate_score
```

### compute_rate

```@docs
Siena.compute_rate
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

### Internal Effects Helpers

```@docs
Siena.add_effect!
Siena.n_rate_parameters
Siena.n_objective_parameters
```
