# Changelog

All notable changes to Siena.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

Release driven by the 2026-07 expert-panel review: inference is brought to
RSiena parity (score-function derivatives, Polyak–Ruppert averaging, the 0.1
t-ratio standard), conditional estimation and composition change are wired
end-to-end, phase 3 is threaded, and a Networks.jl bridge extension removes
the "dependency island".

### Breaking

- **Convergence standard tightened to RSiena's published criterion.**
  Convergence now requires every per-parameter |t| < 0.1 *and*
  `tconv.max` < 0.25 (previously a single max |t| < 0.25 test). Models that
  0.1.0 reported as `Converged: true` may now honestly report unconverged.
  *Migration:* pass `SienaAlgorithm(convergence_threshold=0.25)` and ignore
  `tconv_max` to reproduce the old (laxer) gate.
- **`GOFResult` renamed to `SienaGOFResult`.** `siena_gof` returns
  `SienaGOFResult`; the name `GOFResult` now refers to Networks.jl's shared
  GOF type (re-exported for the generic `gof` method). *Migration:* rename
  `GOFResult` to `SienaGOFResult` where you dispatch on `siena_gof` output.
- **Divergence is surfaced instead of silent.** Objective parameters are
  clamped at ±10 (rates at [0.05, 1e3]); hitting the clamp now sets
  `result.diverged = true` and warns (previously the clamp was silent).
  *Migration:* check `result.diverged` before trusting estimates.
- **Minimum Julia raised to 1.12**; package UUID regenerated. *Migration:*
  upgrade Julia and re-resolve environments pinning the old UUID.
- **Every `SienaAlgorithm` field is now read by the estimator.** `parallel`
  defaults to `true` (phase 3 and the derivative estimators run threaded;
  results are unchanged, seeds are pre-drawn per simulation),
  `max_iterations` is now `Union{Int, Nothing}` and defaults to `nothing` (a
  *budget* on the phase-1/phase-2 iterations rather than the old, unread
  `Int=50`), `n_simulations` averages that many simulations into each
  Robbins-Monro iteration, and `model_type` really restricts the model (see
  Added). Out-of-range values (`n_simulations=0`, ...) are rejected instead
  of silently misbehaving. *Migration:* drop `max_iterations=50` (it now
  *caps* the schedule and would truncate phase 2); pass `parallel=false` for
  strictly serial execution.
- **`SienaResult` gained fields** (`n_iterations`, `n_simulations_run`,
  `n_threads_used`, `model_type`). *Migration:* construct results only via
  `fit_siena`/`siena07`, not positionally.

### Added

- **A real RSiena golden fixture for FITTED OUTPUT, not just targets**
  (issues #8 / Siena#2). `test/fixtures/s50_siena07.toml` freezes an actual
  `siena07` run (RSiena 1.6.6, R 4.6.1) on the bundled s50 data — coefficients,
  standard errors, convergence t-ratios, the derivative matrix and the phase-3
  covariance — regenerable via `test/fixtures/r/s50_siena07.R`. The existing
  golden testset checked target statistics only, which are deterministic and
  prove the effect *formulas*, not the *estimator*.

  Because both sides are Monte-Carlo MoM estimators, the script also refits the
  model under five further RSiena seeds and freezes RSiena's own seed-to-seed
  standard deviation (`rsiena_seed_sd`) into the fixture: the tolerances are
  multiples of that measured width, not numbers chosen to make the test pass,
  and the Julia testset compares the MEAN of five Siena.jl fits (whose
  Monte-Carlo error is sd/√5) rather than a single run.

  **What it found.** Siena.jl's coefficients agree with RSiena's: every one of
  the eight parameters is within 0.28 of an RSiena standard error, the largest
  objective-parameter gap being 0.032 on reciprocity (0.16 SE). But its
  Robbins-Monro procedure is **3–19× noisier** than RSiena's at the same
  nominal budget (seed-to-seed sd 0.006–0.18 vs RSiena's 0.002–0.035), it does
  **not** reach RSiena's own published convergence standard on this model
  (`tconv.max` 0.26–0.77 across five seeds, versus RSiena's 0.13 and the 0.25
  threshold Siena.jl itself enforces), and **roughly 1 seed in 10 diverges
  outright** (2 of 24 surveyed seeds ran away to `tconv.max` ≈ 50). Right
  estimand, weaker algorithm. Increasing the budget does not help: both
  `n_simulations=5` and `phase1_iterations=200` make it *worse*. The testset
  characterizes the gap rather than hiding it behind a loose tolerance.
- **Conversion invariants for the Networks.jl bridge**: `DependentNetwork`,
  `ConstantDyadCovariate` and `VaryingDyadCovariate` built from `Network`
  waves take the ecosystem `missing=:error`/`:face` policy and a `report=true`
  keyword returning `(result, ::Networks.ConversionReport)`. See the ecosystem
  table in Networks.jl `docs/src/guide/conversion_invariants.md`.

- **`model_type` is a real control** (`:standard`, `:networkonly`,
  `:behavioronly`): it selects which dependent variables take ministeps. The
  others are *frozen* — they stay in the simulation state at their
  period-start values and remain readable by the effects of the simulated
  variables (a network effect may depend on a frozen behavior, and vice
  versa), but they never change, so their own rate and objective effects are
  unidentified and leave the estimated parameter vector. New exported
  helpers `simulated_variables(data, model_type)` and
  `restrict_effects(effects, variables)`; `simulate_saom`/`simulate_period!`
  take a `variables=` keyword; `algorithm.condvar` must be a simulated
  variable. *This is **not** RSiena's `modelType`* (which selects
  forcing/initiative/pairwise network models — those are not implemented in
  Siena.jl); the docstrings say so prominently.

- Score-function derivative estimator (Schweinberger–Snijders):
  `estimate_derivative_matrix_score` accumulates trajectory scores over all
  phase-3 simulations — now the default (`derivative_method=:score`) for
  both updates and the final SE matrix, replacing the 30-simulation
  finite-difference estimate whose noise entered the covariance
  quadratically.
- Conditional estimation (RSiena `cond=TRUE`):
  `SienaAlgorithm(conditional=true, condvar=...)` — periods simulate until
  the conditioning variable reaches its observed change; conditioned rates
  are estimated from stopping times.
- Composition change wired end-to-end: `add_composition_change!`,
  `add_change!`, `is_present`; actors contribute to a period only when
  present at both endpoints.
- Structural-zero/one support (RSiena 10/11 coding): `has_structural`,
  `is_structural_dyad`, `n_structural_dyads`; structural dyads are masked
  out of candidate sets, moments, and rate distances.
- Networks.jl bridge extension (`SienaNetworkExt`, loads with
  `using Networks`): `DependentNetwork`/`siena_dependent` accept a
  `Vector{<:AbstractNetwork}` of waves, and dyadic covariates accept
  `Network`/`BipartiteNetwork` objects (optionally extracting an edge
  attribute) — no more manual matrix wrangling between the descriptive and
  longitudinal layers.
- `fit_siena` as the primary entry point (`siena07` kept as the R-faithful
  alias); `gof` method on the shared `Networks.gof` generic returning a
  `Networks.GOFResult`.
- `tconv.max` reporting (`SienaResult.tconv_max`, printed with the result).
- StatsAPI accessors: `coef`, `stderror`, `vcov`, `confint` now extend the
  StatsAPI generics instead of package-local functions (co-loading with
  other model packages is safe).
- s50 CSV data files under `test/data/` for tests and examples;
  BenchmarkTools suite (`benchmark/`) with allocation regression tests.
- New exported simulation machinery: `StateNetwork` (bit-packed Int8 state
  with cached degrees), `ScoreAccumulator`, `snapshot`, `evaluate_actor`,
  `rate_score`, `compute_target_statistics`, `compute_simulated_statistics`,
  `estimate_derivative_matrix`, `estimate_derivative_matrix_score`, and the
  `ParameterMap` helpers.

### Changed

- Phase 2 applies Polyak–Ruppert averaging over each subphase (estimates
  carry substantially less Monte-Carlo noise at the same budget).
- GOF p-values are two-sided Monte-Carlo `(1+k)/(N+1)` with a
  Mahalanobis-distance overall p-value (replacing raw extreme-fractions and
  a "simplified chi-square"), so they can no longer be exactly 0.

### Fixed

- **The Networks.jl bridge silently dropped the missing-dyad mask.**
  `DependentNetwork`, `ConstantDyadCovariate` and `VaryingDyadCovariate` went
  through `as_matrix`, so an *unobserved* dyad was written into the Siena matrix
  as a plain `0` — an observed non-tie. Siena's own per-wave mask records
  **structural** zeros/ones (ties that are *determined*), which is a different
  claim from "unobserved", so there is no faithful encoding: the conversion now
  raises on a masked network unless `missing=:face` is passed.

- Divergence and clamping are reported instead of silently proceeding (see
  Breaking).
- GOF p-value underflow to exactly `0.0` eliminated by the `(1+k)/(N+1)`
  estimator.

### Performance

- Phase-3 simulations and derivative estimation run on threads with
  pre-drawn per-simulation seeds — results are bitwise identical across
  `JULIA_NUM_THREADS`.
- Hot-loop de-abstraction: included objective effects are snapshotted once
  per simulation into a tuple-backed, concretely typed `ObjectiveEffectSet`
  (statically dispatched contribution loop hoisted out of the alter loop),
  replacing per-ministep effects-table filtering and abstract-field
  dispatch.
- Bit-packed `StateNetwork` (Int8 storage, cached in/out-degrees) with
  rate recomputation only after real state changes.

## [0.1.0] - 2026-02-09

Initial release: stochastic actor-oriented models (Snijders) with
Robbins–Monro estimation, network/behavior effects, and Siena-style GOF.
