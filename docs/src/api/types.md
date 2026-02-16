# Types API Reference

This page documents the core data types in Siena.jl.

## Node Sets

### NodeSet

```@docs
NodeSet
```

## Data Container

### SienaData

```@docs
SienaData
```

### Data Operations

```@docs
add_nodeset!
add_dependent!
add_covariate!
```

## Dependent Variables

### AbstractDependent

```@docs
AbstractDependent
```

### DependentNetwork

```@docs
DependentNetwork
```

### DependentBehavior

```@docs
DependentBehavior
```

### Dependent Variable Queries

```@docs
n_waves
n_actors
```

## Covariates

### AbstractCovariate

```@docs
AbstractCovariate
```

### ConstantCovariate

```@docs
ConstantCovariate
```

### VaryingCovariate

```@docs
VaryingCovariate
```

### ConstantDyadCovariate

```@docs
ConstantDyadCovariate
```

### VaryingDyadCovariate

```@docs
VaryingDyadCovariate
```

## Composition Change

### CompositionChange

```@docs
CompositionChange
```

## Network State

The `NetworkState` type maintains the current state of networks and behaviors during simulation. It tracks adjacency matrices and behavior vectors as they evolve through mini-steps.

### NetworkState

```@docs
NetworkState
```

### State Initialization

```@docs
initialize!
```

## Convenience Constructors

These functions mirror the RSiena API for creating data objects.

### siena_data

```@docs
siena_data
```

### siena_nodeset

```@docs
siena_nodeset
```

### siena_dependent

```@docs
siena_dependent
```

### constant_covariate

```@docs
constant_covariate
```

### varying_covariate

```@docs
varying_covariate
```

### constant_dyad_covariate

```@docs
constant_dyad_covariate
```

### varying_dyad_covariate

```@docs
varying_dyad_covariate
```

## Algorithm Configuration

### SienaAlgorithm

```@docs
SienaAlgorithm
```

### siena_algorithm

```@docs
siena_algorithm
```

### GainSequence

```@docs
GainSequence
```

### Gain Operations

```@docs
next_gain!
reset_gain!
```

### EstimationPhase

```@docs
EstimationPhase
```

### PhaseState

```@docs
PhaseState
```

### ConvergenceStats

```@docs
ConvergenceStats
```

## Result Types

### SienaResult

```@docs
SienaResult
```

### SimulationResult

```@docs
SimulationResult
```

## Effects Management Types

### EffectEntry

```@docs
EffectEntry
```

### SienaEffects

```@docs
SienaEffects
```

## GOF Types

### GOFResult

```@docs
GOFResult
```

### AbstractGOFStatistic

```@docs
AbstractGOFStatistic
```

### IndegreeDistribution

```@docs
IndegreeDistribution
```

### OutdegreeDistribution

```@docs
OutdegreeDistribution
```

### TriadCensus

```@docs
TriadCensus
```

### GeodesicDistribution

```@docs
GeodesicDistribution
```

### BehaviorDistribution

```@docs
BehaviorDistribution
```
