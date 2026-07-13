# Estimation API Reference

This page documents the functions for model estimation, simulation, and goodness of fit.

## Model Estimation

### fit_siena

The primary estimation entry point.

```@docs
fit_siena
```

### siena07

```@docs
siena07
```

### Internal update helpers

```@docs
Siena.update_parameters!
```

## Parameters and Targets

### build_param_map

```@docs
build_param_map
ParameterMap
```

### parameter_names

```@docs
parameter_names
```

### objective_theta

```@docs
objective_theta
```

### basic_rate

```@docs
basic_rate
```

### Restricted models (`model_type`)

`SienaAlgorithm`'s `model_type` selects which dependent variables co-evolve.
The frozen variables stay in the state (and readable by the effects of the
simulated ones), but they take no ministeps and their own effects leave the
parameter vector.

```@docs
simulated_variables
restrict_effects
```

### compute_target_statistics

```@docs
compute_target_statistics
```

### compute_simulated_statistics

```@docs
compute_simulated_statistics
```

### Derivative Matrix Estimators

```@docs
estimate_derivative_matrix
estimate_derivative_matrix_score
```

## Simulation

### simulate_saom

```@docs
simulate_saom
```

### Objective effect sets

The simulation hot path snapshots the included objective effects into a
tuple-backed [`ObjectiveEffectSet`](@ref) (built once per simulation), so the
per-candidate contribution loop is statically dispatched. All simulation
functions accept either a `SienaEffects` table or a prebuilt set; results are
identical.

```@docs
ObjectiveEffectSet
build_objective_set
Siena.ObjectiveEffectSpec
Siena.entry_value
```

### Score accumulation

```@docs
ScoreAccumulator
reset_scores!
```

### simulate_period!

```@docs
simulate_period!
```

### Internal ministep machinery

One iteration of the continuous-time Markov chain: an actor is selected via
the rate function, waits an exponential time, and executes a network or
behavior ministep.

```@docs
Siena.actor_rate
Siena.sample_waiting_time
Siena.execute_network_ministep!
Siena.execute_behavior_ministep!
```

### compute_objective

```@docs
compute_objective
```

### compute_network_choice_probs

```@docs
compute_network_choice_probs
```

### compute_behavior_choice_probs

```@docs
compute_behavior_choice_probs
```

## Result Accessors

### coef

```@docs
coef
```

### stderror

```@docs
stderror
```

### vcov

```@docs
vcov
```

### confint

```@docs
confint
```

## Goodness of Fit

### gof

Method of the shared `Networks.gof` generic; returns the ecosystem-wide
[`GOFResult`](@ref).

```@docs
gof(::SienaResult, ::SienaData, ::AbstractGOFStatistic)
```

### siena_gof

The RSiena-style entry point; returns the detailed [`SienaGOFResult`](@ref).

```@docs
siena_gof
```

### compute_gof_statistic

```@docs
compute_gof_statistic
```

### Convenience Functions

```@docs
siena_gof_indegree
```

```@docs
siena_gof_outdegree
```

```@docs
siena_gof_triad
```

```@docs
siena_gof_behavior
```
