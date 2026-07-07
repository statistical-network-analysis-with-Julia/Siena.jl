# Estimation API Reference

This page documents the functions for model estimation, simulation, and goodness of fit.

## Model Estimation

### siena07

```@docs
siena07
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

### compute_target_statistics

```@docs
compute_target_statistics
```

### compute_simulated_statistics

```@docs
compute_simulated_statistics
```

## Simulation

### simulate_saom

```@docs
simulate_saom
```

### simulate_period!

```@docs
simulate_period!
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

### siena_gof

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
