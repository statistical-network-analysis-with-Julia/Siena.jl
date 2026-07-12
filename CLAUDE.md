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
- **`AbstractDependent`** with subtypes `DependentNetwork` (adjacency matrices per wave) and `DependentBehavior` (integer vectors per wave). `DependentNetwork` accepts RSiena-style structural codes in the matrices (default `10` = structural zero, `11` = structural one; configurable via `structural_zero`/`structural_one`, other values throw): coded entries are decoded to 0/1 face values and recorded in per-wave `structural::Vector{BitMatrix}` masks (`has_structural`, `is_structural_dyad`, `n_structural_dyads`). Structurally determined dyads are excluded from ministep candidate sets (period-start mask), from target/simulated moment statistics (`_zero_structural!` in estimation.jl), and from rate distances
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

Hot-path design: the included objective effects are snapshotted once per simulation into a tuple-backed `ObjectiveEffectSet` (mirroring `ERGM.TermSet`), so the per-candidate contribution loop is statically dispatched instead of filtering/dispatching through the effects table per ministep. Per-actor rates are cached across ministeps and recomputed only for variables with state-dependent (non-basic) rate effects after a real state change. `ScoreAccumulator` optionally collects the trajectory score function for the score-based derivative estimator.

### Estimation (`src/estimation.jl`)

Three-phase Robbins-Monro algorithm in `siena07`:
1. **Phase 1** -- rough parameter updates with identity derivative matrix
2. **Phase 2** -- subphases with estimated derivative matrix and decaying gain
3. **Phase 3** -- fixed parameters, collecting simulations for SE estimation via `D^{-1} Sigma D^{-T}` (derivative `D` from the score-function estimator by default, or finite differences with common random numbers)

Phase-3 simulations and derivative estimation are embarrassingly parallel and run under `Threads.@threads`: seeds are pre-drawn from the algorithm RNG in serial order and each simulation runs on its own seeded RNG writing to its own result slot, so results are bitwise identical regardless of `JULIA_NUM_THREADS`.

Result type `SienaResult` provides `coef`, `stderror`, `vcov`, `confint` (StatsAPI methods).

### Algorithm (`src/algorithm.jl`)

`SienaAlgorithm` configures estimation. `GainSequence` manages Robbins-Monro gain decay. `PhaseState` tracks phase/subphase progression. `ConvergenceStats` checks t-ratios against threshold.

### Goodness of Fit (`src/gof.jl`)

`siena_gof` simulates from estimated model and compares statistics (indegree, outdegree, triad census, geodesic, behavior distributions) to observed data using Mahalanobis distance and chi-square p-values.

### RSiena-Compatible API (`src/Siena.jl`)

The main module file defines convenience constructors mirroring RSiena function names: `siena_data()`, `siena_dependent()`, `constant_covariate()`, `get_effects()`, `include_effects!()`, `siena07()`.

### Network.jl Bridge (`ext/SienaNetworkExt.jl`)

Package extension on the weak dependency Network.jl (`[weakdeps]`/`[extensions]` in Project.toml; Siena keeps zero hard network-stack deps). When Network.jl is loaded it adds methods so `DependentNetwork`/`siena_dependent` accept a `Vector` of `Network` (or `BipartiteNetwork`) observations — converted via `Network.as_matrix`, preserving directedness, self-loop allowance, and one-/two-mode type, and validating equal node sets (vertex counts, mode sizes, `:vertex_names`) and directedness across waves — and so `ConstantDyadCovariate`/`VaryingDyadCovariate` (plus their snake_case wrappers) accept networks, optionally reading an edge attribute via `attr=`. Tests load Network (test target) and cover the bridge end-to-end, including a `siena07` fit from `Network` waves.

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
- **Network** (weak dependency) -- activates the `SienaNetworkExt` extension for building Siena data types from `Network` objects
- Requires Julia >= 1.12

## Conventions

- Function names use snake_case; type names use PascalCase
- Mutating functions end with `!` (e.g., `include_effects!`, `initialize!`, `add_nodeset!`)
- Network data stored as `Vector{Matrix{Int}}` (one matrix per wave); behavior as `Vector{Vector{Int}}`
- Covariates are auto-centered by default on construction
- Effect shortnames (symbols like `:outdegree`, `:recip`, `:transTrip`) are the primary user-facing identifiers for including effects
- RSiena naming conventions preserved where possible (e.g., `siena07`, `sienaGOF` -> `siena_gof`)
- All exports declared explicitly in `src/Siena.jl`
- Tests use `@testset` blocks in `test/runtests.jl` covering types, effects, simulation, GOF, and integration
