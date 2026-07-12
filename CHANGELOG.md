# Changelog

All notable changes to Siena.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

Release driven by the 2026-07 expert-panel review: inference is brought to
RSiena parity (score-function derivatives, Polyak–Ruppert averaging, the 0.1
t-ratio standard), conditional estimation and composition change are wired
end-to-end, phase 3 is threaded, and a Network.jl bridge extension removes
the "dependency island".

### Breaking

- **Convergence standard tightened to RSiena's published criterion.**
  Convergence now requires every per-parameter |t| < 0.1 *and*
  `tconv.max` < 0.25 (previously a single max |t| < 0.25 test). Models that
  0.1.0 reported as `Converged: true` may now honestly report unconverged.
  *Migration:* pass `SienaAlgorithm(convergence_threshold=0.25)` and ignore
  `tconv_max` to reproduce the old (laxer) gate.
- **`GOFResult` renamed to `SienaGOFResult`.** `siena_gof` returns
  `SienaGOFResult`; the name `GOFResult` now refers to Network.jl's shared
  GOF type (re-exported for the generic `gof` method). *Migration:* rename
  `GOFResult` to `SienaGOFResult` where you dispatch on `siena_gof` output.
- **Divergence is surfaced instead of silent.** Objective parameters are
  clamped at ±10 (rates at [0.05, 1e3]); hitting the clamp now sets
  `result.diverged = true` and warns (previously the clamp was silent).
  *Migration:* check `result.diverged` before trusting estimates.
- **Minimum Julia raised to 1.12**; package UUID regenerated. *Migration:*
  upgrade Julia and re-resolve environments pinning the old UUID.

### Added

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
- Network.jl bridge extension (`SienaNetworkExt`, loads with
  `using Network`): `DependentNetwork`/`siena_dependent` accept a
  `Vector{<:AbstractNetwork}` of waves, and dyadic covariates accept
  `Network`/`BipartiteNetwork` objects (optionally extracting an edge
  attribute) — no more manual matrix wrangling between the descriptive and
  longitudinal layers.
- `fit_siena` as the primary entry point (`siena07` kept as the R-faithful
  alias); `gof` method on the shared `Network.gof` generic returning a
  `Network.GOFResult`.
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
