# Changelog

All notable changes to Siena.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

**Dependency renamed:** the foundation package is now `NetworkCore` (developed as `Networks`); write `using NetworkCore` where code said `using Networks`. Types and functions keep their names.

### Breaking

- **Default effects now match RSiena**: `get_effects` includes what RSiena's
  `getEffects()` includes — the basic rates, `outdegree` (RSiena's `density`) for
  every network, `recip` for a directed one-mode network, `linear` for a behaviour
  and `quad` when its observed range is at least 2. Before, only the basic rates
  were included, so an RSiena specification translated line by line fitted a model
  without density and reciprocity. Naming a default effect again changes nothing;
  `include_effects!(...; include=false)` removes one (RSiena's
  `includeEffects(..., include = FALSE)`). Pinned against `getEffects()$include`
  on ten data sets (`s50_defaults.toml`).
- **Up-only and down-only periods are restricted, as RSiena's `allowOnly = TRUE`**:
  in a period in which the observed network only gains (loses) ties, or a behaviour
  only rises (falls), the simulation allows only such changes; when every period is
  up-only or every period down-only, `get_effects` offers no `outdegree` (`linear`)
  effect. `DependentNetwork(...; allow_only=false)` and
  `DependentBehavior(...; allow_only=false)` lift the restriction. Pinned by RSiena
  simulations on panels with one up-only and one down-only period.
- **`SienaAlgorithm(parallel=...)` is `threaded=...`**, the keyword the other
  packages use (`SienaAlgorithm.threaded`; `parallel=` is an error).
- **The development-version aliases are removed**: `seed=` (in `SienaAlgorithm`,
  `simulate_saom`, `siena_gof`, `gof`; use `rng=MersenneTwister(seed)`), `n_sims=`
  (use `n_sim=`), and the effect types `BalanceEffect`, `IsolateEffect`,
  `AverageAttHigherEffect` and `AverageAttLowerEffect` (use the `…Simple` types and
  `IsolateNetEffect`). The estimation guide has the rename table.
- **Conditional estimation is the default for one dependent variable** (RSiena's
  `cond = NA`); `SienaAlgorithm(conditional=false)` restores unconditional MoM.
  `SienaAlgorithm.conditional` is now `Union{Bool, Nothing}` (default `nothing`).
- **Unconverged fits are returned with a warning** (as RSiena does) instead of
  thrown; `allow_unconverged=false` restores `SienaConvergenceError`. A failed
  phase-3 validation first re-enters refinement (`revalidate_max`, default 2).
- **Effects corrected to RSiena's definitions under RSiena's names**: `diffX`
  (alter minus ego; the sign was reversed), `higher` (ties count one half),
  `avAltDist2` (RSiena's average of the alters' average alter), `outPopSqrt`/
  `inActSqrt`/`outPop`/`inAct` (RSiena's internal parameter `parm`, default 0 for
  the sqrt variants: lagged degrees in the moment statistic), GWESP family
  (`alpha = 0.69`, was `log 2`), `degPlus` (undirected networks only, degree-based).
- **Renamed**: the one-network shared-neighbour effects are `sharedInNbrs` and
  `sharedOutNbrs` (were `sharedIn`/`sharedOut`; RSiena's `sharedIn` is a different,
  two-network effect). `TransitiveTriadsEffect` is RSiena's `transTriads` (undirected
  only; it was an alias of `transTies`); `OutIsolateEffect` is `outIso` (was
  `outIsolate`, an RSiena behaviour name); `SimXRecipEffect` is `simRecipX`;
  `FeedbackEffect` is `simProd`; the network `IsolateEffect` is `IsolateNetEffect`;
  rate effects use RSiena's names (`outRate`, `inRate`,
  `recipRate`, `RateX`, …). `get_effects` registers directed GWESP as `gwespFF`
  (`:gwesp` still selects it).
- **The GWESP effects are elementary effects**, as in RSiena: the change statistic
  of a tie is the weight of its own shared partners (the full difference of the
  actor statistic was used before — same targets, different estimates).
  `simXTransTrip` uses RSiena's change statistic. The two-mode `fourCycles` effect
  is RSiena's `cycle4` (statistic: number of four-cycles); the registry no longer
  lists `outdegree2`, `indegree2`, `ego2X`, `event2X` (use `outdegree`, `inPop`,
  `egoX`, `dyadX`).
- **`totSim` uses RSiena's change statistic**: the change of the summed similarities
  minus outdegree × mean similarity, for a step in either direction (the plain
  difference was used before — same targets, different estimates).
- **Dyadic covariates are centred over the off-diagonal entries** (RSiena), which
  changes the outdegree coefficient of models with `X`.
- **Covariates with missing values (`missing`/`NaN`) are refused**; pass
  `missing=:mean` to impute the mean (RSiena's rule). Covariates and dependent
  variables can no longer share a name.
- **GOF**: degree distributions use fixed levels `0:8`, cumulative (RSiena's
  `sienaGOF` defaults), all periods are joined by default, and the Mahalanobis
  test pools the observed vector with the simulations (an exact Monte-Carlo test).
  `SienaGOFResult` is parametric with `String` labels and a `periods` field.
- **`get_effects` builds the full table per variable kind** and refuses
  meaningless combinations: two-mode networks get only two-mode effects, undirected
  networks only RSiena's symmetric-network effects; `validate_effects` checks a
  model against its data in `fit_siena`, `simulate_saom` and the GOF functions.
- `include_effects!(...; test=true)` requires `fix=true` (it used to be ignored),
  and a strict batch with an unknown name now changes nothing.
- `estimate_derivative_matrix` uses central differences by default.
- Minimum Julia 1.12; `SienaResult` gained fields (construct it only through
  `fit_siena`).

- **The NetworkCore bridge checks more of the panel**: `DependentNetwork(name,
  waves)` refuses waves whose `loops` flag differs (wave 1's flag used to govern a
  self-tie recorded in a later wave), a `type=` that contradicts the networks, and
  a directed two-mode network with an arc from mode 2 to mode 1 (which a two-mode
  variable cannot hold).
- **`DependentNetwork(...; type=...)` accepts only `:onemode`, `:twomode` and
  `:bipartite`** (a synonym of `:twomode`). Any other value used to be treated as
  two-mode without a word.

### Added

- **`siena07` in RSiena's argument order**: `siena07(alg, data, effects)` and
  `siena07(alg; data=data, effects=effects)` (R's `siena07(alg, data = mydata,
  effects = myeff)`) fit the same model as `siena07(data, effects; algorithm=alg)`.
  Any other argument list raises an `ArgumentError` naming the accepted orders
  instead of a `MethodError`.
- `coefnames(fit)` (StatsAPI) returns the coefficient labels.

- **Undirected networks** (`DependentNetwork(...; directed=false)`) with RSiena's
  model type 2 (unilateral initiative, reciprocal confirmation): symmetric toggles,
  RSiena's symmetric-network effects and target conventions.
- **Network selection on a co-evolving behaviour**: `egoX`, `altX`, `simX`, …
  accept a dependent behaviour (`egoalcohol`, …), as in RSiena.
- **Score-type tests** (Schweinberger 2012, RSiena's `test=TRUE`):
  `include_effects!(...; fix=true, test=true)` and `result.score_test`
  (`SienaScoreTest`).
- **Interactions** (`include_interaction!`, RSiena's `includeInteraction`): two- and
  three-way interactions of network effects (`InteractionEffect`) and of behaviour
  effects (`BehaviorProductEffect`) under RSiena's rules; it used to throw.
- **`siena_time_test`** (RSiena's `sienaTimeTest`): score-type tests of time
  heterogeneity from the fit's per-period phase-3 statistics and scores.
- **RSiena's two-mode effects** under RSiena's names: `density`, `cycle4`, `inPop`,
  `inPopSqrt`, `outAct`, `outActSqrt`, `outTrunc`, `outIso`, `egoX`, `egoSqX`, `X`
  and the rate effects, all pinned against RSiena.
- A PrecompileTools workload: the first fit of a session went from 7.8 s to 0.35 s
  (first simulation 0.68 s to under 0.01 s, first GOF 2.3 s to 0.04 s).
- RSiena spelling in `include_effects!` (`[:egoX, :altX]; interaction1=:smoke1`,
  `:density`), `add_effect!` and `validate_effects` exported.
- Conditional fits report rate standard errors (`rate_standard_errors`, RSiena's
  `vrate`) and rescale the other variables' basic rates to the same time unit.
- `SienaAlgorithm(diagonalize=0.2, revalidate_max=2)`; `derivative_sims`
  defaults to 100.
- **Simulation pins for every RSiena-named effect** (`s50_dynamics.toml`): RSiena
  simulates 38 models at fixed parameters and Siena.jl must reproduce the mean of
  every simulated statistic within Monte-Carlo error, so a wrong change statistic
  (which target statistics cannot detect) fails the suite; a test asserts no
  RSiena-named effect lacks such a pin.
- Closed-form change statistics for `transTies`, `nbrDist2` and `gwdspFF`
  (simulations with `nbrDist2` are about 20 times faster).
- RSiena fixtures: targets of every RSiena-named effect (159 directed, undirected
  and two-mode targets, 26 interaction targets), fitted conditional (with score
  test), undirected, co-evolution and GWESP-with-interaction models beside the
  unconditional one, and `sienaTimeTest` statistics; GOF and time-test size tests;
  docstring examples for every export, executed by the test suite; Aqua.
- RSiena fixture `s50_allowonly_cond.toml`: conditional fits on panels with an
  up-only or a down-only period (twenty RSiena fits per panel), pinning the
  conditional rate estimate and its standard error (RSiena's `vrate`, the sd of the
  stopping times) in the restricted period. The co-evolution fixture is now the
  mean of twenty RSiena fits instead of six.
- Every push compares each fitted RSiena fixture with three Siena.jl fits (one
  before), with the bound on their seed-to-seed spread; the nightly run uses six.

### Fixed

- **Siena did not load on Julia nightly**: the Robbins-Monro update
  (`update_parameters!`) accepted only a `Matrix{Float64}`, and on nightly the
  phase-1 preconditioner (a broadcast mixing a `Matrix` with a `Diagonal`) comes
  back as a `Diagonal`, so the precompile workload failed with a `MethodError`
  (and every package testing against Siena with it). The update takes any matrix,
  and the preconditioner is built as a dense matrix element by element.
- **Two-mode `Network` panels were refused** by `DependentNetwork(name, waves)`:
  an undirected two-mode `Network` was passed on as an undirected variable, which a
  two-mode variable cannot be. A two-mode `Network` or `BipartiteNetwork`
  (directed or undirected) now becomes RSiena's two-mode dependent variable, with
  ties from the mode-1 actors to the mode-2 nodes; pinned by RSiena's two-mode
  targets computed from a panel built of `Network`s.
- **Errors in threaded simulations arrive unwrapped**: a simulation that throws
  under `threaded=true` (the default) raises its own exception, as with
  `threaded=false`, not a `TaskFailedException`/`CompositeException`
  (NetworkCore's `spawn_all`).
- The module docstring's example, the first on the API page, fitted unseeded random
  50 × 50 matrices and emitted hundreds of warnings; it now fits the bundled s50
  data with a seeded RNG, quietly (pinned).

- **Undirected networks were simulated and estimated as directed**
  (`directed=false` was ignored).
- **Conditional estimation**: standard errors came from a 30-simulation forward
  finite difference (biased and noisy) and the default fit diverged; it now uses the
  score-function derivative, valid at the conditional stopping time.
- **The score-function derivative pairs each period's statistics with that
  period's score** (RSiena), removing cross-period noise from the standard errors.
- The phase-1/phase-2 preconditioner is a score-function derivative mixed with
  its diagonal (RSiena's `diagonalize`), which stops the Robbins-Monro iterates
  from being thrown around by a noisy derivative.
- A single `NaN` in a covariate no longer turns the whole centred covariate into
  `NaN`; two-mode variables are no longer offered one-mode effects (`inPop` threw a
  `BoundsError`, `transTrip` read events as actors).
- GOF with data-derived degree levels and an in-sample covariance rejected a
  correct model 15.7 % of the time at the 5 % level; it now holds its size.
- The NetworkCore.jl bridge no longer writes unobserved dyads as observed zeros
  (`missing=:error` by default, `missing=:face` to opt in).
- `include_effects!` ignored `interaction1` when a name matched a Siena.jl short
  name, so a covariate or network named in `interaction1` could be silently
  ignored. A given `interaction1` must now match.
- `SettingRateEffect` and `TwoModeWithinEffect` compared setting codes with the
  centred covariate (every code shifted by the covariate's mean), so actors were
  matched to the wrong setting; they now compare the covariate as given.
- `SienaConvergenceError`'s message and docstring said to set
  `allow_unconverged=true`, which is the default; they now say the error is raised
  because the algorithm sets `allow_unconverged=false`.

### Performance

- The similarity effects read a covariate range cached at construction instead of
  recomputing it per candidate dyad: a simulation of the README model went from
  9.0 ms / 14.4 MiB to 2.6 ms / 0.12 MiB, and a full fit from 53 s / 126 GiB to
  15 s / 2.9 GiB (4 threads); choice probabilities with similarity effects allocate
  nothing (pinned).
- Tuple-backed `ObjectiveEffectSet`, `StateNetwork` with cached degrees and sorted
  adjacency, reusable `MinistepWorkspace`, threaded simulations with pre-drawn seeds
  (results independent of the thread count).

### Known limitations

- Only the Method of Moments: no Maximum Likelihood, Bayesian or GMoM estimation,
  no multi-group data or `siena08` meta-analysis (groups with the same waves can
  be combined into one network with structural zeros between them).
- No period-dummy effects (`sienaTimeFix`); `siena_time_test` only tests.
- Endowment and creation effects are not estimable.
- Only RSiena's default network model types (standard for directed, forcing for
  undirected networks); no initiative/pairwise/double-step models, no absorbing or
  continuous behaviour.
- `NA` tie values are refused; missing covariates are refused or mean-imputed
  (imputed actors still enter the targets).
- Composition change is whole-wave: an actor counts in a period only when present
  at both of its waves (no within-period joining or leaving times).
- Dyads whose structural (10/11) status changes between waves do not get RSiena's
  correction.
- RSiena effects without an implementation (among them `balance`,
  `avAttHigher`/`avAttLower`, behaviour `isolate`/`outIsolate`, the effects linking
  two-mode and one-mode networks) — see the effects guide's concordance table.
- The Robbins-Monro schedule has no adaptive subphase lengths, Dolby option or
  `prevAns` warm start.
- In the s50 friendship–alcohol co-evolution model the standard error of the first
  friendship rate averages about 7 % above RSiena's (fourteen Siena.jl fits 1.16,
  twenty RSiena fits 1.08, seed-to-seed sds 0.10 and 0.08: 2.4 Monte-Carlo
  standard errors); at a common parameter vector with 4,000 simulations each, the
  derivative matrices and standard errors agree (1.13 against 1.09). The 11 %
  reported earlier was measured against six RSiena fits whose spread was
  understated. In a conditional fit whose first period is up-only (s50), the
  `transTrip` coefficient is 0.009 below RSiena's (a tenth of its standard error;
  targets, rates, rate standard errors and the other estimates agree); not yet
  explained. The `siena_time_test` one-step estimates spread about twice as much
  from fit to fit as RSiena's for some effects; the means agree.

## [0.1.0] - 2026-02-09

Development version, never released: stochastic actor-oriented models (Snijders)
with Robbins–Monro estimation, network/behavior effects, and Siena-style GOF.
