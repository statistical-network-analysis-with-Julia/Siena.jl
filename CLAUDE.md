# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Siena.jl is a Julia port of [RSiena](https://github.com/stocnet/rsiena) for statistical analysis of longitudinal network data using Stochastic Actor-Oriented Models (SAOM). It implements Method of Moments estimation via Robbins-Monro stochastic approximation, modeling network evolution as a continuous-time Markov chain of actor-driven micro-steps.

## Development Commands

- **Run tests:** `JULIA_NUM_THREADS=4 julia --project=. -e 'using Pkg; Pkg.test()'` (RSiena golden comparisons, GOF and time-test size simulations, every docstring example, Aqua). `SNWJ_LONG_TESTS=true` adds the long comparisons (six fits per `mc_close` fixture instead of three); CI's nightly run and every `workflow_dispatch` (manual or Downstream) set it.
- **Regenerate the RSiena fixtures** (R with RSiena 1.6.6): `Rscript test/fixtures/r/<name>.R > test/fixtures/<name>.toml` for `s50_targets`, `s50_defaults`, `s50_dynamics`, `s50_siena07`, `s50_siena07_cond`, `s50_siena07_undirected`, `s50_coevolution`, `s50_siena07_interaction`, `s50_time_test`, `s50_allowonly_cond`
- **Build docs:** `julia --project=docs docs/make.jl`
- **Start REPL with project:** `julia --project=.`
- **Install dependencies:** `julia --project=. -e 'using Pkg; Pkg.instantiate()'`

## Architecture

### Core Types (`src/types.jl`)

- **`NodeSet`** -- set of actors/nodes with optional names
- **`SienaData`** -- top-level container holding nodesets, dependents, and covariates (mutable, built incrementally via `add_nodeset!`, `add_dependent!`, `add_covariate!`)
- **`AbstractDependent`** with subtypes `DependentNetwork` (adjacency matrices per wave) and `DependentBehavior` (integer vectors per wave). `DependentNetwork(...; directed=false)` is an undirected network (waves must be symmetric; `_network_kind(dep)` ∈ `:directed/:undirected/:twomode`), simulated with RSiena's model type 2 (see Simulation). `DependentNetwork` accepts RSiena-style structural codes in the matrices (default `10` = structural zero, `11` = structural one; configurable via `structural_zero`/`structural_one`, other values throw): coded entries are decoded to 0/1 face values and recorded in per-wave `structural::Vector{BitMatrix}` masks (`has_structural`, `is_structural_dyad`, `n_structural_dyads`). Structurally determined dyads are excluded from ministep candidate sets (period-start mask), from target/simulated moment statistics (`_zero_structural!` in estimation.jl), and from rate distances. Both dependent types take `allow_only=true` (RSiena's `allowOnly`): per-period `uponly`/`downonly` flags (RSiena's classification; a period with no change is both) restrict the candidate sets of `compute_network_choice_probs(!)`/`compute_behavior_choice_probs` to additions (removals). Identical waves `[w, w]` therefore freeze the variable: generator panels in the tests pass `allow_only=false`
- **`AbstractCovariate`** with subtypes `ConstantCovariate`, `VaryingCovariate`, `ConstantDyadCovariate`, `VaryingDyadCovariate`. Missing values (`missing`/`NaN`) are refused unless `missing=:mean` (RSiena's mean imputation; `n_imputed` recorded, listed by `approximations`). The similarity range is cached in the `range` field at construction (hot path). One-mode dyadic covariates are centred over the off-diagonal entries (RSiena). Covariate and dependent-variable names must be distinct.
- **`NetworkState`** -- mutable simulation state holding current network matrices and behavior vectors
- **`CompositionChange`** -- tracks actors joining/leaving

### Effects System (`src/effects/`)

Abstract hierarchy: `AbstractEffect` -> `NetworkEffect`, `BehaviorEffect`, `RateEffect`, `TwoModeEffect`.

- **`EffectEntry`** -- wraps an effect with metadata (name, shortname, include/fix/test flags, initial value)
- **`SienaEffects`** -- collection of `EffectEntry` objects; iterable
- **`get_effects` includes RSiena's `getEffects()` defaults** (`_rsiena_defaults!` in `src/Siena.jl`): basic rates, `outdegree` (`density`) for every network, `recip` for a directed one-mode network, `linear` for a behaviour and `quad` when its observed range is ≥ 2; when every period of a variable is up-only or every period down-only, `outdegree`/`linear` are not registered at all (RSiena). Pinned against `getEffects()$include` on ten data sets (`s50_defaults.toml`). `include_effects!(...; include=false)` is RSiena's `includeEffects(..., include=FALSE)`. A test that needs a model without a default effect must opt out explicitly.
- Effects are identified by shortname symbols (e.g., `:outdegree`, `:recip`, `:transTrip`); `include_effects!` also matches RSiena's short name (`effect_name`) with `interaction1=` (`[:egoX]; interaction1=:smoke1`), and is atomic in strict mode
- Each effect type implements `compute_contribution(effect, state, data, actor, alter)` and `compute_statistic(effect, state, data)`
- Files: `base.jl` (abstract types + `SienaEffects`), `network.jl`, `behavior.jl`, `rate.jl`, `twomode.jl`, `interaction.jl`, `registry.jl`
- **`interaction.jl`** — RSiena's `includeInteraction`: `InteractionEffect` (network, `unspInt`) and `BehaviorProductEffect` (behaviour, `behUnspInt`), created by `include_interaction!`. Elementary effects: contribution = product of the components' contributions (behaviour: divided by `difference^(K-1)`); statistic from RSiena's `NetworkInteractionEffect::egoStatistic` (ego components' tie statistic × the remaining component's actor statistic, else Σ_j x_ij Π tie statistics; `_tie_statistic`, `_statistic_scale`). `_interaction_type(effect, kind)` is RSiena's `interactionType` (`:ego`/`:dyadic`/`:none`) and `_check_network_interaction` its rules; `SienaEffects.kinds` (filled by `get_effects`) lets `include_interaction!` check them early, `validate_effects` checks again against the data. 26 interaction targets are pinned (RSiena targets from `siena07(nsub=0)` runs, because `getTargets` does not set up interactions).
- **`totSim`** (`TotalSimilarityEffect`) uses RSiena's change statistic: Δ(sum of similarities) − outdegree × mean similarity in either direction (RSiena's `SimilarityEffect::calculateChangeContribution` with `average=false`), not the toggle difference.
- **Elementary effects**: the GWESP effects' `compute_contribution` is the toggled tie's own weight (RSiena: GWESP is elementary), and `SimXTransTripEffect` uses RSiena's own change statistic — neither is the toggle difference of `evaluate_actor`, so they are excluded from the brute-force testset and pinned by "Elementary effects…" and by the fitted `s50_siena07_interaction.toml`. When adding an effect under an RSiena name, read its `calculateContribution` in RSiena's C++: equal targets do not imply equal dynamics.
- **Two-mode networks**: RSiena's bipartite effects reuse the one-mode types whose implementation only touches ego's row and alters' columns (`OutdegreeEffect`, `IndegreePopularityEffect`, `OutdegreeActivityEffect`, `OutdegreeTruncEffect`, `OutIsolateEffect`, `EgoEffect`, `EgoSqEffect`, `DyadCovariateEffect`; they iterate `net.outneighbors[actor]`, never `1:n` with a diagonal skip) plus `FourCyclesEffect` = RSiena `cycle4`; pinned by the `twomode_*` fixture block. The other `TwoMode*` types are Siena.jl-only.
- **`registry.jl`** holds the per-kind effects tables `get_effects` builds (`_network_entries!`, `_behavior_entries!`, `_rate_entries!`) and the validity trait `_supported_kinds(effect)` (RSiena's effect groups: `:directed`, `:undirected`, `:twomode`; for behaviour/rate effects the kind of the network they read). `validate_effects(data, effects)` refuses an included effect on a variable of the wrong kind, a missing/mistyped covariate, etc.; `fit_siena`, `simulate_saom` (unless `validate=false`, as the estimator's inner loop passes) and the GOF engine call it.
- Network covariate effects (`egoX`, `altX`, `simX`, …) read their attribute through `_attr_value`/`_attr_similarity`, which accept a **dependent behaviour** (centred by its mean; RSiena's network selection on a co-evolving behaviour) as well as a covariate.
- **Undirected networks** follow RSiena: the change statistic is the directed `compute_contribution` on the symmetric state (RSiena's `calculateContribution`), the toggle is imposed in both directions, and `compute_statistic` uses RSiena's symmetric conventions (`OutdegreeEffect` and `NbrDist2Effect` halve, `TransitiveTriadsEffect` counts triangles).
- `OutdegreePopularityEffect`/`IndegreeActivityEffect` carry RSiena's `parm` (defaults: 1 for `outPop`/`inAct`, 0 for the sqrt variants); `parm ≤ 0` changes only the moment statistic to RSiena's lagged-degree version (start-of-period degrees from the observed wave)
- **Every effect with an RSiena short name is pinned** against `RSiena:::getTargets` in `s50_targets.toml` (directed + undirected catalogues); a testset asserts every RSiena name in the `get_effects` tables is in the fixture. Effects without an RSiena counterpart use non-RSiena names (`sharedInNbrs`, `inTrunc`, `simProd`, `cubic`, … and the `…Simple` approximations).

### Simulation (`src/simulation.jl`)

Simulates the CTMC: `simulate_saom` -> `simulate_period!` -> mini-steps (network or behavior). Choice probabilities use multinomial logit over the objective function. Rate functions control actor selection and waiting times.

Hot-path design: the included objective effects are snapshotted once per simulation into a tuple-backed `ObjectiveEffectSet` (mirroring `ERGM.TermSet`), so the per-candidate contribution loop is statically dispatched instead of filtering/dispatching through the effects table per ministep. Per-actor rates are cached across ministeps and recomputed only for variables with state-dependent (non-basic) rate effects after a real state change. `ScoreAccumulator` collects the trajectory score function (total and per period, `period_scores`) for the score-based derivative estimator; it is valid for conditional simulation too (the conditional stop is a stopping time; pinned by an E[S]=0 testset). For an undirected network `execute_network_ministep!` toggles `(i,j)` and `(j,i)` and conditional distances move in steps of 2.

### Estimation (`src/estimation.jl`)

Robbins-Monro algorithm in `fit_siena` (`siena07` is a `const` alias). RSiena's argument order, `siena07(alg, data, effects)` or `siena07(alg; data, effects)`, is accepted by two extra methods; any other argument list hits a catch-all method that throws an `ArgumentError` naming the accepted orders. The update `update_parameters!` and `_quad_form_inv` take any `AbstractMatrix`: on Julia nightly the preconditioner broadcast (Matrix with Diagonal) returned a `Diagonal`, and a `Matrix{Float64}` signature stopped the package from precompiling; the preconditioner is now filled element-wise into a dense matrix.
1. **Phase 1** -- derivative matrix from `derivative_sims` simulations (score function; forward FD under `derivative_method=:finite_difference`), mixed with its diagonal by `diagonalize` (RSiena's 0.2), then rough updates
2. **Phase 2** -- subphases with halving gain and Polyak-Ruppert averaging; the preconditioner is refreshed at the first and last subphase
3. **Newton refinement** -- capped updates from batches of `2 * phase3_iterations` simulations with the score derivative; final validation is never used for selection
4. **Phase 3** -- fixed parameters; convergence t-ratios, `tconv.max`, and SEs `D^{-1} Σ D^{-T}` with `D = Σ_m cov(s_m, S_m)` (per-period pairing of statistics and scores, RSiena's `derivativeFromScoresAndDeviations`) or central finite differences (`step=0.1`). A failed validation re-enters refinement (`revalidate_max`, default 2) and a fresh batch validates; unconverged fits are **returned with a warning** by default (`allow_unconverged=true`, RSiena's behaviour), `allow_unconverged=false` throws `SienaConvergenceError`.

Conditional estimation is the default when one dependent variable is simulated (`SienaAlgorithm(conditional=nothing)` = RSiena's `cond = NA`; `true`/`false` force it). Periods run until the conditioning variable's distance from the period-start observation reaches the observed distance. Its basic rate entries are fixed (on the fit's own effects copy) at `default_basic_rate` and estimated from phase-3 stopping times (`rate_estimates`, with `rate_standard_errors` = simulation rate × sd of the stopping time, RSiena's `vrate`); the other variables' basic rates are rescaled by the mean stopping time (RSiena's `theta[posj] * rate`). The score-function derivative is used here too. Effects with `fix && test` get a score-type test (Schweinberger 2012, RSiena's `EvaluateTestStatistic`): phase 3 runs the *extended* model (`_extended_model`: tested entries unfixed at their fixed value, so the simulation is unchanged) and `result.score_test::SienaScoreTest` holds joint/per-effect chi-squares, one-sided z and one-step estimates. Composition change attached via `add_composition_change!(data, cc)` is a whole-period listwise approximation (not RSiena's within-period times): actors contribute to a period only when present at both endpoint waves (no ministeps, dyads out of candidate sets, rows/columns out of the moment statistics and rate distances); `approximations` says so.

Phase-3 simulations and derivative estimation are embarrassingly parallel and run on tasks (NetworkCore's `spawn_all`, a few strided chunks per thread, so a failing simulation raises its own exception, not a `TaskFailedException`) when `SienaAlgorithm(threaded=true)` (the default; the ecosystem's keyword) and strictly serially on the calling thread when `threaded=false` (both go through the `_run_simulations!` helper): seeds are pre-drawn from the algorithm RNG in serial order and each simulation runs on its own seeded RNG writing to its own result slot, so results are bitwise identical regardless of `JULIA_NUM_THREADS` or `threaded`. Every algorithm field changes execution: `n_simulations` is the number of simulations averaged into each Robbins-Monro iteration, `max_iterations` (default `nothing`) is a budget on the phase-1/phase-2 iterations, and `SienaResult` reports back what actually ran (`n_iterations`, `n_simulations_run`, `n_threads_used`, `model_type`).

`model_type` (`:standard` / `:networkonly` / `:behavioronly`) selects which dependent variables co-evolve. The seam is a simulated-variable vector computed once per fit by `simulated_variables(data, model_type)` and threaded through `fit_siena` -> `_simulate_moments`/`estimate_derivative_matrix*` -> `simulate_saom` -> `simulate_period!(; variables=...)`, where it replaces `collect(keys(data.dependents))` as the vector the rate machinery (per-actor rates, totals, dirty flags, categorical variable draw) is indexed off. Non-simulated dependents stay in `NetworkState` at their period-start values: frozen, but still readable by the effects of the simulated variables (a network rate/objective effect may depend on a frozen behavior, and vice versa). Their own rate and objective effects are unidentified (constant moments) and leave the model via `restrict_effects(effects, sim_vars)`, which returns an effects object sharing the `EffectEntry` objects — the parameter map, targets, simulations, `rate_estimates` and `SienaResult.effects` are all built from it, so no other code needs a `model_type` branch. `algorithm.condvar` must be a simulated variable. Note: this is *not* RSiena's `modelType` (Siena.jl uses RSiena's default network model type per network kind — 1 for directed, 2 for undirected — and implements no other) — the docstring carries a warning admonition saying so.

Result type `SienaResult` provides `coef`, `coefnames`, `stderror`, `vcov`, `confint`, `coeftable` (StatsAPI methods; `coefnames` is a copy of `parameter_names`).

### Time heterogeneity (`src/timetest.jl`)

`siena_time_test(result)` is RSiena's `sienaTimeTest`: period dummies for every non-basic-rate free effect are score-tested from the per-period phase-3 statistics/scores kept on the result (`phase3_period_stats`, `phase3_period_scores`, `period_targets`), with `D = Σ_m cov(G_m, S_m)` over the extended moment vector and `_score_statistic`/`_transformed_score` (RSiena's `ScoreTest`, `partial.scoreTest`, `transformed.scoreTest`). No new simulations. `sienaTimeFix` (estimating the dummies) is not implemented. Pinned against RSiena and by a size/power simulation.

### Algorithm (`src/algorithm.jl`)

`SienaAlgorithm` configures estimation. `GainSequence` manages Robbins-Monro gain decay. `PhaseState` tracks phase/subphase progression. `ConvergenceStats` checks t-ratios against threshold.

### Golden fixtures (RSiena)

All generated by checked-in `test/fixtures/r/*.R` scripts (RSiena 1.6.6) with a `[provenance]` block, loaded with NetworkCore.jl's `load_golden`:

- **`s50_targets.toml`** — target statistics (`RSiena:::getTargets`): the original 34, plus `directed_*`/`undirected_*` blocks with one target per effect offered under an RSiena short name (92 directed on s50 + `fr2` + alcohol + smoke1 + dyadic `dc`; 49 undirected on the symmetrised s50; RSiena `parm` variants), compared at 1e-9 through the spec → constructor map `rsiena_effect` in the test file. Deterministic: they prove the effect FORMULAS, not the estimator. A testset asserts that every RSiena name in the `get_effects` tables is pinned.
- **`s50_dynamics.toml`** — the CHANGE-STATISTIC pin. RSiena simulates 38 small models at fixed θ (`simOnly`, unconditional, `n3 = 1000`; each model = base effects + a few effects with non-zero θ; entry format `variable|shortName|interaction1|parm|type|theta`) and the mean/sd of every simulated statistic is frozen; "Golden dynamics vs RSiena" simulates the same models (300 sims, fixed seed) and requires |z| < 4.5 on every mean, and asserts that every RSiena-named effect in the `get_effects` tables (per kind: directed/und/twomode/behavior), plus `unspInt`/`behUnspInt`, appears in some model. **A new effect under an RSiena name must be added to a model in `s50_dynamics.R`** — targets alone do not validate it (this pin found `totSim`; reading the C++ found GWESP/`simXTransTrip`).
- **`s50_siena07.toml`** — one unconditional RSiena fit plus RSiena's seed-to-seed sd; the Julia test averages five fits (tolerances re-measured 2026-10: coefficients 0.025, rates 0.15).
- **`s50_siena07_cond.toml`** (conditional + score-tested `cycle3`), **`s50_siena07_undirected.toml`** (symmetrised s50, model type 2, conditional), **`s50_coevolution.toml`** (friendship–alcohol with selection on the behaviour, unconditional; TWENTY RSiena fits, `n_fits`), **`s50_siena07_interaction.toml`** (elementary `gwespFF` + `egoX × recip`), **`s50_allowonly_cond.toml`** (conditional fits on s50 panels whose period 1 is up-only or down-only; twenty RSiena fits per panel; pins the rate SE of the restricted period, RSiena's `vrate` = sd of the stopping times, which is already an sd) and **`s50_time_test.toml`** (`sienaTimeTest` joint/effect/individual/period statistics and one-step estimates) — each the MEAN of six (or `n_fits`) RSiena fits with per-fit seed sds; the Julia tests run three fits by default and six under `SNWJ_LONG_TESTS=true` (`fixture_seeds` in the test file) and compare through `mc_close` (4 combined Monte-Carlo sds of the difference of the two means, both from RSiena's per-fit sd: `4·sd·√(1/n_r + 1/n_j)`; separately, the Julia per-fit sd must stay below √F(n_j−1, n_r−1; 0.999) times RSiena's) for coefficients, SEs, rates, rate SEs and score-test statistics. A seed sd from six fits is uncertain by about a third: the co-evolution fixture's first six fits gave 0.036 for the first friendship rate's SE where twenty give 0.081, which made Siena.jl look 11 % off; prefer twenty fits for a new fixture.
- **`s50_defaults.toml`** — `getEffects()$include` on ten data sets (directed, undirected, two-mode, multiplex, binary behaviour, up-only, down-only, mixed), RSiena simulations (`simOnly`) of one model on panels with one up-only and one down-only period (the `allowOnly` restriction), and RSiena's full list of effect short names (`allEffects$shortName`), against which the test file's `SIENA_ONLY` allow-list (the only effects exempt from the RSiena pins, each with its reason) is checked.

The fitted fixtures settled an earlier conditional-SE disagreement with RSiena: with the score derivative, conditional SEs match RSiena's `cond=TRUE` within Monte-Carlo error.

### Goodness of Fit (`src/gof.jl`)

`gof(result)` (preferred) and `siena_gof` simulate from the estimated model (with its `model_type` restriction and conditioning), evaluate the statistic at the end of every period and sum over periods (RSiena's `join=TRUE`; `period=m` for one). Degree/behaviour/geodesic statistics use fixed, cumulative levels (`0:8` for degrees, as `sienaGOF`), never levels derived from the observation. The overall test pools the observed vector with the simulations for the mean and covariance (`pinv`), so `p = (1 + #{d_sim ≥ d_obs})/(N+1)` is an exact Monte-Carlo test; a size testset (200 replications) pins it. The engine is `_gof_core(data, effects, θ, stat; ...)`.

### RSiena-Compatible API (`src/Siena.jl`)

The main module file defines convenience constructors mirroring RSiena function names: `siena_data()`, `siena_dependent()`, `constant_covariate()`, `get_effects()`, `include_effects!()`, `siena07()`.

### NetworkCore.jl Bridge (`src/network_integration.jl`)

NetworkCore.jl is a hard dependency. The `NetworkIntegration` source submodule
adds constructors accepting `Network`/`BipartiteNetwork` waves and dyadic covariates.
It preserves directedness, self-loop allowance, node sets, and one-/two-mode structure. A two-mode `Network`/`BipartiteNetwork` (directed or undirected) becomes a `:twomode` variable with `directed=true` (actor → mode-2 node), from the incidence matrix; a directed two-mode network with a mode-2 → mode-1 arc is refused, as are waves whose `loops` flag differs and a `type=` that contradicts the networks. `DependentNetwork` itself accepts only `type ∈ (:onemode, :twomode, :bipartite)`.
There is no `[extensions]` entry. Refresh local gitignored manifests with `Pkg.resolve()`
when migrating from the older extension layout.


**Missing dyads are rejected, not coerced.** The bridge honours the ecosystem conversion contract (NetworkCore.jl `src/conversion.jl`; per-path table in `NetworkCore.jl/docs/src/guide/conversion_invariants.md`): every `Network` → Siena-matrix method calls `NetworkCore.require_observed` with the standard `missing=:error`/`:face` policy, and takes `report=true` to return `(result, ::NetworkCore.ConversionReport)`.

The distinction matters and is easy to get wrong. Siena's own per-wave mask (`structural::Vector{BitMatrix}`, the RSiena `10`/`11` codes) records **structurally determined** ties — ties that are *fixed*, and correctly excluded from ministep candidate sets and moment statistics. NetworkCore.jl's `missing_dyads` mask records **unobserved** ties — the analyst does not know their status. These are different claims, so there is no faithful encoding: mapping an unobserved dyad onto a structural zero would tell the estimator the tie is *known to be impossible*. The bridge used to go straight through `as_matrix`, writing the unobserved dyad's face value into the matrix as a plain observed `0`. It now raises unless the caller writes `missing=:face`. Pinned by the "NetworkCore.jl bridge: conversion invariants" testset.

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
- **PrecompileTools** -- the workload at the end of `src/Siena.jl` (conditional and unconditional fit of the README effect set, show, GOF, time test under a devnull logger); first fit 7.8 s → 0.35 s
- **NetworkCore** (hard dependency) -- shared contracts, normal p-values, Monte Carlo p-values, coefficient tables and panel conversion
- Requires Julia >= 1.12

## Conventions

- Function names use snake_case; type names use PascalCase
- Mutating functions end with `!` (e.g., `include_effects!`, `initialize!`, `add_nodeset!`)
- Network data stored as `Vector{Matrix{Int}}` (one matrix per wave); behavior as `Vector{Vector{Int}}`
- Covariates are auto-centered by default on construction
- Effect shortnames (symbols like `:outdegree`, `:recip`, `:transTrip`) are the primary user-facing identifiers for including effects
- An RSiena short name (`effect_name`) is reserved for RSiena's statistic and pinned in `s50_targets.toml`. Effects whose formula only approximates RSiena's carry a `Simple` suffix on both the type and the short name (`BalanceSimpleEffect`/`:balanceSimple`, `:avAttHigherSimple`, `:avAttLowerSimple`), and the RSiena name is left undefined rather than aliased; Siena.jl-only effects use names RSiena does not (`sharedInNbrs`, `inTrunc`, `simProd`, `cubic`, …). A new effect with an RSiena name needs a fixture spec.
- RSiena naming conventions preserved where possible (e.g., `siena07`, `sienaGOF` -> `siena_gof`)
- All exports declared explicitly in `src/Siena.jl`
- Tests use `@testset` blocks in `test/runtests.jl` covering types, effects, simulation, GOF, and integration; randomness through `rng=MersenneTwister(k)` (there is no `seed=` keyword); every export's docstring example is executed ("Every exported docstring carries a runnable example"), and `Aqua.test_all` runs with `undocumented_names=true`

## September inference and performance contracts

- `siena07 === fit_siena`; the fit accepts `rng::AbstractRNG` and snapshots both data and
  effects. Conditional fitting must not fix or rescale the caller's rate entries.
- `gof(result)` uses the stored data; `n_sim` and `rng` are the GOF keywords. The
  development-version spellings (`seed`, `n_sims`, `parallel`, `BalanceEffect`,
  `IsolateEffect`, `AverageAtt{Higher,Lower}Effect`) were deleted without shims
  (never released); the rename table is in `docs/src/guide/estimation.md`.
- `StateNetwork` maintains sorted in/out adjacency lists alongside degrees and its
  compact matrix. Every write, including temporary effect toggles, updates all caches.
- `TransitiveTripletsEffect` contribution visits the ego's neighbors (O(outdegree)).
  `MinistepWorkspace` reuses probability/alter/score arrays per simulated variable.
  `compute_network_choice_probs!` allocates zero bytes after compilation. A tie toggle
  can grow adjacency vector capacity; do not claim all state mutation is allocation-free.
- `coeftable` returns NetworkCore's shared table. Positive basic rate null tests against
  zero are undefined. No likelihood/AIC/BIC is invented for Method of Moments.
- Choice probabilities with similarity effects (covariate or dependent behaviour)
  allocate zero bytes (pinned): covariate accessors are single methods with `isa`
  branches (statically dispatched out of the abstract `data.covariates` Dict) and the
  similarity range is cached on the covariate.
