# Types API Reference

This page documents the core data types in Siena.jl.

## The Siena Module

```@docs
Siena
```

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

### Structural Zeros and Ones

```@docs
has_structural
is_structural_dyad
n_structural_dyads
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

### Composition Change Operations

```@docs
add_change!
add_composition_change!
is_present
```

## Network State

The `NetworkState` type maintains the current state of networks and behaviors during simulation. It tracks adjacency matrices (as bit-packed [`StateNetwork`](@ref) matrices with cached degrees) and behavior vectors as they evolve through mini-steps.

### NetworkState

```@docs
NetworkState
```

### StateNetwork

```@docs
StateNetwork
```

### State Initialization

```@docs
initialize!
```

### snapshot

```@docs
snapshot
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

### Internal Phase and Convergence Helpers

```@docs
Siena.advance_phase!
Siena.update_convergence!
Siena.is_converged
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

### SienaGOFResult

The detailed, RSiena-style result returned by [`siena_gof`](@ref).

```@docs
SienaGOFResult
```

### GOFResult

`GOFResult` is the ecosystem-wide GOF container from Network.jl (re-exported
by Siena). It is returned by the shared [`gof`](@ref) generic, and any
[`SienaGOFResult`](@ref) can be converted to it.

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
