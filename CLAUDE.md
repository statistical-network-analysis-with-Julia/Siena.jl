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
1. **Phase 1** -- rough parameter updates with a finite-difference derivative matrix
2. **Phase 2** -- subphases with estimated derivative matrix and decaying gain
3. **Newton refinement** -- capped updates from batches of `2 * phase3_iterations` simulations; final validation is never used for selection
4. **Phase 3** -- fixed parameters, collecting simulations for SE estimation via `D^{-1} Sigma D^{-T}` (derivative `D` from the score-function estimator by default, or finite differences with common random numbers)

Conditional estimation (`SienaAlgorithm(conditional=true)`, RSiena's `cond=TRUE`) conditions every simulated period on the observed amount of change of one dependent variable (`condvar`, defaulting to the only one): periods run until the conditioning variable's distance from the period-start observation reaches the observed distance instead of until time 1. The conditioned variable's basic rate entries are fixed out of the moment equations and estimated from phase-3 stopping times (`rate_estimates`); the derivative estimator falls back to finite differences. Composition change attached via `add_composition_change!(data, cc)` uses RSiena's MoM semantics: actors contribute to a period only when present at both endpoint waves (no ministeps, dyads out of candidate sets, rows/columns out of the moment statistics and rate distances).

Phase-3 simulations and derivative estimation are embarrassingly parallel and run under `Threads.@threads` when `SienaAlgorithm(parallel=true)` (the default) and strictly serially on the calling thread when `parallel=false` (both go through the `_run_simulations!` helper): seeds are pre-drawn from the algorithm RNG in serial order and each simulation runs on its own seeded RNG writing to its own result slot, so results are bitwise identical regardless of `JULIA_NUM_THREADS` or `parallel`. Every algorithm field changes execution: `n_simulations` is the number of simulations averaged into each Robbins-Monro iteration, `max_iterations` (default `nothing`) is a budget on the phase-1/phase-2 iterations, and `SienaResult` reports back what actually ran (`n_iterations`, `n_simulations_run`, `n_threads_used`, `model_type`).

`model_type` (`:standard` / `:networkonly` / `:behavioronly`) selects which dependent variables co-evolve. The seam is a simulated-variable vector computed once per fit by `simulated_variables(data, model_type)` and threaded through `fit_siena` -> `_simulate_moments`/`estimate_derivative_matrix*` -> `simulate_saom` -> `simulate_period!(; variables=...)`, where it replaces `collect(keys(data.dependents))` as the vector the rate machinery (per-actor rates, totals, dirty flags, categorical variable draw) is indexed off. Non-simulated dependents stay in `NetworkState` at their period-start values: frozen, but still readable by the effects of the simulated variables (a network rate/objective effect may depend on a frozen behavior, and vice versa). Their own rate and objective effects are unidentified (constant moments) and leave the model via `restrict_effects(effects, sim_vars)`, which returns an effects object sharing the `EffectEntry` objects — the parameter map, targets, simulations, `rate_estimates` and `SienaResult.effects` are all built from it, so no other code needs a `model_type` branch. `algorithm.condvar` must be a simulated variable. Note: this is *not* RSiena's `modelType` (forcing/initiative network models, not implemented) — the docstring carries a warning admonition saying so.

Result type `SienaResult` provides `coef`, `stderror`, `vcov`, `confint` (StatsAPI methods).

### Algorithm (`src/algorithm.jl`)

`SienaAlgorithm` configures estimation. `GainSequence` manages Robbins-Monro gain decay. `PhaseState` tracks phase/subphase progression. `ConvergenceStats` checks t-ratios against threshold.

### Golden fixtures (RSiena)

Two testsets, and the difference between them matters:

- **"Golden target statistics vs RSiena (s50)"** — target statistics. These are a *deterministic* function of the observed waves, so they prove the effect FORMULAS and nothing about the estimator. Generated by `test/fixtures/r/s50_targets.R`, stored with provenance in `s50_targets.toml`, and compared at 1e-9 for 34 explicitly selected effects.
- **"Golden: RSiena siena07 fitted output (s50)"** — what `siena07` actually returns. `test/fixtures/s50_siena07.toml` freezes a real RSiena 1.6.6 fit (coefficients, SEs, t-ratios, `tconv.max`, derivative matrix, phase-3 covariance); `test/fixtures/r/s50_siena07.R` regenerates it. Loaded with Networks.jl's `load_golden`.

  Both sides are Monte-Carlo MoM estimators, so a single-run-vs-single-run comparison could only carry a tolerance too wide to test anything. The R script therefore **refits under five further RSiena seeds and freezes RSiena's own seed-to-seed sd** (`rsiena_seed_sd`) into the fixture — the tolerances are multiples of that measured width — and the Julia testset compares the **mean of five Siena.jl fits** (Monte-Carlo error sd/√5) rather than one. Rate and objective parameters get separate tolerances because their noise differs by an order of magnitude; lumping them would force one loose tolerance onto all eight.

  **Current evidence:** capped Newton refinement closes the former N1 failure
  for all five declared s50 seeds: every independent phase-3 max |t| < 0.1 and
  tconv_max < 0.25. The result retains raw `derivative_matrix`, `phase3_cov`, and
  `condition_number`; D/Σ diagonals are checked against the frozen R fixture.
  No fixed ridge is added to standard errors. Singular/ill-conditioned derivatives
  yield undefined SEs and fail convergence. Any unconverged result throws
  `SienaConvergenceError` by default; `allow_unconverged=true` returns diagnostics
  with a warning even when verbose=false. `error.result` preserves diagnostics.
  These checks validate one unconditional model, not every effect or conditional fit.

### Goodness of Fit (`src/gof.jl`)

`siena_gof` simulates from estimated model and compares statistics (indegree, outdegree, triad census, geodesic, behavior distributions) to observed data using Mahalanobis distance and Monte Carlo p-values.

### RSiena-Compatible API (`src/Siena.jl`)

The main module file defines convenience constructors mirroring RSiena function names: `siena_data()`, `siena_dependent()`, `constant_covariate()`, `get_effects()`, `include_effects!()`, `siena07()`.

### Networks.jl Bridge (`src/network_integration.jl`)

Networks.jl is a hard dependency. The `NetworkIntegration` source submodule
adds constructors accepting `Network`/`BipartiteNetwork` waves and dyadic covariates.
It preserves directedness, self-loop allowance, node sets, and one-/two-mode structure.
There is no `[extensions]` entry. Refresh local gitignored manifests with `Pkg.resolve()`
when migrating from the older extension layout.


**Missing dyads are rejected, not coerced.** The bridge honours the ecosystem conversion contract (Networks.jl `src/conversion.jl`; per-path table in `Networks.jl/docs/src/guide/conversion_invariants.md`): every `Network` → Siena-matrix method calls `Networks.require_observed` with the standard `missing=:error`/`:face` policy, and takes `report=true` to return `(result, ::Networks.ConversionReport)`.

The distinction matters and is easy to get wrong. Siena's own per-wave mask (`structural::Vector{BitMatrix}`, the RSiena `10`/`11` codes) records **structurally determined** ties — ties that are *fixed*, and correctly excluded from ministep candidate sets and moment statistics. Networks.jl's `missing_dyads` mask records **unobserved** ties — the analyst does not know their status. These are different claims, so there is no faithful encoding: mapping an unobserved dyad onto a structural zero would tell the estimator the tie is *known to be impossible*. The bridge used to go straight through `as_matrix`, writing the unobserved dyad's face value into the matrix as a plain observed `0`. It now raises unless the caller writes `missing=:face`. Pinned by the "Networks.jl bridge: conversion invariants" testset.

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
- **Networks** (hard dependency) -- shared contracts, normal p-values, Monte Carlo p-values, coefficient tables and panel conversion
- Requires Julia >= 1.12

## Conventions

- Function names use snake_case; type names use PascalCase
- Mutating functions end with `!` (e.g., `include_effects!`, `initialize!`, `add_nodeset!`)
- Network data stored as `Vector{Matrix{Int}}` (one matrix per wave); behavior as `Vector{Vector{Int}}`
- Covariates are auto-centered by default on construction
- Effect shortnames (symbols like `:outdegree`, `:recip`, `:transTrip`) are the primary user-facing identifiers for including effects
- An RSiena short name is reserved for a numerically equivalent implementation. Effects whose formula only approximates RSiena's carry a `Simple` suffix on both the type and the short name (`BalanceSimpleEffect`/`:balanceSimple`, `:avAttHigherSimple`, `:avAttLowerSimple`), and the RSiena name is left undefined rather than aliased
- RSiena naming conventions preserved where possible (e.g., `siena07`, `sienaGOF` -> `siena_gof`)
- All exports declared explicitly in `src/Siena.jl`
- Tests use `@testset` blocks in `test/runtests.jl` covering types, effects, simulation, GOF, and integration

## September inference and performance contracts

- `siena07 === fit_siena`; the fit accepts `rng::AbstractRNG` and snapshots both data and
  effects. Conditional fitting must not fix or rescale the caller's rate entries.
- `gof(result)` uses the stored data; `n_sim` and `rng` are canonical GOF keywords.
  Deprecated `seed`/`n_sims` shims remain for compatibility.
- `StateNetwork` maintains sorted in/out adjacency lists alongside degrees and its
  compact matrix. Every write, including temporary effect toggles, updates all caches.
- `TransitiveTripletsEffect` contribution visits the ego's neighbors (O(outdegree)).
  `MinistepWorkspace` reuses probability/alter/score arrays per simulated variable.
  `compute_network_choice_probs!` allocates zero bytes after compilation. A tie toggle
  can grow adjacency vector capacity; do not claim all state mutation is allocation-free.
- `coeftable` returns Networks' shared table. Positive basic rate null tests against
  zero are undefined. No likelihood/AIC/BIC is invented for Method of Moments.
