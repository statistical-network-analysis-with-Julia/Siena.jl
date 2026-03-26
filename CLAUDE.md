# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Siena.jl is a Julia port of [RSiena](https://github.com/stocnet/rsiena) for statistical analysis of longitudinal network data using Stochastic Actor-Oriented Models (SAOM). It implements Method of Moments estimation via Robbins-Monro stochastic approximation, modeling network evolution as a continuous-time Markov chain of actor-driven micro-steps.

## Development Commands

- **Run tests:** `julia --project=. -e 'using Pkg; Pkg.test()'`
- **Run specific test:** `julia --project=. -e 'using Siena; include("test/runtests.jl")'`
- **Build docs:** `julia --project=docs docs/make.jl`
- **Start REPL with project:** `julia --project=.`
- **Install dependencies:** `julia --project=. -e 'using Pkg; Pkg.instantiate()'`

## Architecture

### Core Types (`src/types.jl`)

- **`NodeSet`** -- set of actors/nodes with optional names
- **`SienaData`** -- top-level container holding nodesets, dependents, and covariates (mutable, built incrementally via `add_nodeset!`, `add_dependent!`, `add_covariate!`)
- **`AbstractDependent`** with subtypes `DependentNetwork` (adjacency matrices per wave) and `DependentBehavior` (integer vectors per wave)
- **`AbstractCovariate`** with subtypes `ConstantCovariate`, `VaryingCovariate`, `ConstantDyadCovariate`, `VaryingDyadCovariate`
- **`NetworkState`** -- mutable simulation state holding current network matrices and behavior vectors
- **`CompositionChange`** -- tracks actors joining/leaving

### Effects System (`src/effects/`)

Abstract hierarchy: `AbstractEffect` -> `NetworkEffect`, `BehaviorEffect`, `RateEffect`, `TwoModeEffect`.

- **`EffectEntry`** -- wraps an effect with metadata (name, shortname, include/fix/test flags, initial value)
- **`SienaEffects`** -- collection of `EffectEntry` objects; iterable
- Effects are identified by shortname symbols (e.g., `:outdegree`, `:recip`, `:transTrip`)
- Each effect type implements `compute_contribution(effect, state, data, actor, alter)` and `compute_statistic(effect, state, data)`
- 150+ effects across 5 files: `base.jl` (abstract types + `SienaEffects`), `network.jl`, `behavior.jl`, `rate.jl`, `twomode.jl`

### Simulation (`src/simulation.jl`)

Simulates the CTMC: `simulate_saom` -> `simulate_period!` -> mini-steps (network or behavior). Choice probabilities use multinomial logit over the objective function. Rate functions control actor selection and waiting times.

### Estimation (`src/estimation.jl`)

Three-phase Robbins-Monro algorithm in `siena07`:
1. **Phase 1** -- rough parameter updates with identity derivative matrix
2. **Phase 2** -- subphases with estimated derivative matrix and decaying gain
3. **Phase 3** -- fixed parameters, collecting simulations for SE estimation via `D^{-1} Sigma D^{-T}`

Result type `SienaResult` provides `coef`, `stderror`, `vcov`, `confint`.

### Algorithm (`src/algorithm.jl`)

`SienaAlgorithm` configures estimation. `GainSequence` manages Robbins-Monro gain decay. `PhaseState` tracks phase/subphase progression. `ConvergenceStats` checks t-ratios against threshold.

### Goodness of Fit (`src/gof.jl`)

`siena_gof` simulates from estimated model and compares statistics (indegree, outdegree, triad census, geodesic, behavior distributions) to observed data using Mahalanobis distance and chi-square p-values.

### RSiena-Compatible API (`src/Siena.jl`)

The main module file defines convenience constructors mirroring RSiena function names: `siena_data()`, `siena_dependent()`, `constant_covariate()`, `get_effects()`, `include_effects!()`, `siena07()`.

### Design Patterns

- **Builder pattern** for data: create empty `SienaData`, then add components via `add_*!` functions
- **Multiple dispatch** on effect types for `compute_contribution` and `compute_statistic`
- **Abstract type hierarchy** for extensibility (new effects subtype `NetworkEffect`/`BehaviorEffect`/`RateEffect`)
- Mutable structs for state (`NetworkState`, `SienaData`, `EffectEntry`); immutable for data inputs (`NodeSet`, covariates)

## Key Dependencies

- **DataFrames** -- effects table display
- **Distributions** -- `Normal`, `Chisq` for confidence intervals and GOF p-values
- **LinearAlgebra** -- matrix operations in estimation (inversion, identity)
- **SparseArrays** -- sparse matrix support
- **StatsBase** -- statistical utilities
- **Statistics** -- `mean`, `std`, `cov`
- Requires Julia >= 1.9

## Conventions

- Function names use snake_case; type names use PascalCase
- Mutating functions end with `!` (e.g., `include_effects!`, `initialize!`, `add_nodeset!`)
- Network data stored as `Vector{Matrix{Int}}` (one matrix per wave); behavior as `Vector{Vector{Int}}`
- Covariates are auto-centered by default on construction
- Effect shortnames (symbols like `:outdegree`, `:recip`, `:transTrip`) are the primary user-facing identifiers for including effects
- RSiena naming conventions preserved where possible (e.g., `siena07`, `sienaGOF` -> `siena_gof`)
- All exports declared explicitly in `src/Siena.jl`
- Tests use `@testset` blocks in `test/runtests.jl` covering types, effects, simulation, GOF, and integration
