# Model Estimation

Siena estimates stochastic actor-oriented models by Method of Moments. It matches
expected simulated statistics to observed target statistics; it does not maximize
a likelihood. Each simulated period starts at the corresponding observed wave. With
one dependent variable the estimation is **conditional** on its observed amount of
change by default, as RSiena's (`cond = NA`).

## Estimation phases

1. **Phase 1:** estimate the derivative matrix from `derivative_sims` simulations
   (score function, as RSiena; forward finite differences under
   `derivative_method = :finite_difference`), mix it with its diagonal
   (`diagonalize`, RSiena's 0.2), then run `phase1_iterations` rough updates with
   `initial_gain`.
2. **Phase 2:** run `n_subphases` subphases, each of `phase1_iterations` updates.
   The gain halves between subphases and the last three quarters of each subphase's
   parameter iterates are averaged (Polyak–Ruppert averaging). The derivative is
   refreshed at the first and last subphase.
3. **Newton refinement:** use up to `refine_max` capped updates based on simulated
   moment means and score-function derivatives. Each batch contains
   `2 * phase3_iterations` simulations.
4. **Phase 3:** freeze the estimate and draw `phase3_iterations` new simulations
   for convergence diagnostics and standard errors. These observations were not
   used to select the estimate or decide when to stop refinement. If they fail the
   convergence standard, they drive one more Newton step and a fresh batch validates
   again, up to `revalidate_max` times.

The score-function derivative pairs each period's statistics with that period's
score, ``D = \sum_m \mathrm{cov}(s_m, S_m)``, as RSiena does. It is valid for
conditional estimation too: the conditional period ends at a stopping time of the
simulated chain, and the score of the stopped path still has mean zero (a testset
checks this).

The schedule is simpler than RSiena's (fixed subphase length instead of the adaptive
`n2min`/`n2max`, a first-subphase gain of `initial_gain / 2`, no Dolby option) and is
followed by Newton refinement, so Monte-Carlo paths differ from RSiena's; on the
pinned models the estimates and standard errors agree within RSiena's own
seed-to-seed spread.

The update is `θ ← θ − gain * (D \ deviation)`, capped in Euclidean step norm.
Refinement checks a stricter internal convergence target (half the configured
thresholds), leaving room for independent validation noise. Refinement cannot
guarantee convergence of an unidentified or poorly specified model.

## A reproducible fit

```julia
using Siena, NetworkCore, Random

s50 = load_dataset(:s50)
data = siena_data()
add_nodeset!(data, NodeSet(50))
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_covariate!(data, ConstantCovariate(:smoke1, s50.smoke[:, 1]))
effects = get_effects(data)
include_effects!(effects, :friendship,
    [:outdegree, :recip, :transTrip, :altsmoke1, :egosmoke1, :simsmoke1])
result = fit_siena(data, effects; rng=MersenneTwister(1),
                   algorithm=SienaAlgorithm(verbose=false))

```

All randomness flows through an `rng::AbstractRNG`, including independent per-draw
streams. `threaded=false` runs serially; `threaded=true` uses available Julia threads.
Both reduce simulation results in the same fixed order. Thread counts do not change
the Monte Carlo sample or the fitted model. Pass a fresh seeded generator
(`rng=MersenneTwister(1)`) to repeat a fit.

## Algorithm controls

| Keyword | Default | Meaning |
|:--|:--|:--|
| `n_subphases` | 4 | Phase-2 subphases |
| `phase1_iterations` | 50 | Phase-1 and each phase-2 subphase's iterations |
| `phase3_iterations` | 1000 | Final diagnostic sample size (at least 2) |
| `initial_gain` | 0.2 | Initial Robbins–Monro gain |
| `min_gain` | 0.0005 | Lower bound on gain |
| `max_iterations` | `nothing` | Total phase-1/2 iteration budget |
| `n_simulations` | 1 | Simulations averaged into each phase-1/2 update |
| `derivative_sims` | 100 | Simulations behind the phase-1/2 derivative (per perturbation for finite differences) |
| `diagonalize` | 0.2 | Weight of the diagonal in the phase-1/2 derivative (RSiena's `diagonalize`) |
| `derivative_method` | `:score` | Phase-3 derivative, or `:finite_difference` (central differences) |
| `refine_max` | 5 | Maximum Newton refinements; zero disables them |
| `revalidate_max` | 2 | Re-entries into refinement after a failed phase-3 validation |
| `convergence_threshold` | 0.1 | Required maximum absolute per-effect t-ratio |
| `overall_convergence_threshold` | 0.25 | Required overall convergence ratio |
| `allow_unconverged` | `true` | Return an unconverged fit with a warning (RSiena's behaviour); `false` throws |
| `rng` | `Random.default_rng()` | Generator used when the fit supplies no override |
| `threaded` | `true` | Thread independent simulation batches |
| `verbose` | `true` | Print phase progress; warnings remain enabled when false |
| `model_type` | `:standard` | Variables that co-evolve |
| `conditional` | `nothing` | Condition on observed change; `nothing` = RSiena's `cond = NA` (conditional iff one variable is simulated) |
| `condvar` | `nothing` | Conditioning variable; required when several variables co-evolve |

A phase-1/2 budget does not limit refinement or phase 3. For a deliberately small
mechanics check, both budgets can be made explicit:

```julia
diagnostic = fit_siena(data, effects; rng=MersenneTwister(4),
    algorithm=SienaAlgorithm(phase1_iterations=2, n_subphases=1,
        phase3_iterations=30, derivative_sims=3, refine_max=0,
        allow_unconverged=true, verbose=false))
println(diagnostic.converged)
```

This example is intentionally diagnostic. Its warning explains which final criteria
were not met; `show(diagnostic)` and `approximations(diagnostic)` flag it, and its
estimates must not be reported. With `allow_unconverged=false` the fit throws
[`SienaConvergenceError`](@ref) instead; the full result remains available as
`error.result`.

## Convergence and uncertainty

The per-effect convergence t-ratio is `(mean simulated − observed) / simulated sd`.
The denominator is the simulated statistic's standard deviation, not the standard
error of its Monte Carlo mean. The overall ratio is
`sqrt(deviation' * inv(Σ) * deviation)`, the maximum over linear combinations of
statistics. A discrepancy in the null space of a singular covariance gives an
infinite ratio instead of being hidden by a pseudoinverse.

```julia
result.t_ratios
result.tconv_max
result.converged
result.condition_number
```

Standard errors use the Method-of-Moments covariance `D⁻¹ Σ D⁻ᵀ`, with raw `D` and `Σ`
retained as `derivative_matrix` and `phase3_cov`. Both are Monte Carlo estimates.
No fixed ridge is added. If the derivative condition number is nonfinite or at least
`1e10`, standard errors are undefined and the fit fails convergence. Examine redundant
effects or increase simulation precision before trusting uncertainty estimates.

The overall convergence ratio standardizes moments by their simulated standard
deviations before checking covariance rank. This keeps its conclusion invariant
to changing moment units, including very small or large covariate scales. A nonzero
discrepancy in a zero-variance direction fails convergence; a pseudoinverse must
not discard it. The separate raw-derivative condition-number safeguard above is
still sensitive to scaling.

```julia
coef(result)
stderror(result)
vcov(result)
confint(result; level=0.95)
coeftable(result)
```

`result.parameter_names` gives the order. `coeftable` uses objective-parameter Wald
z-statistics; these differ from convergence t-ratios. Basic rate tests against zero
are undefined because zero is outside their fitted positive support. No log-likelihood,
AIC or BIC is available. The clamp flag `diverged` records whether an iterate hit a
parameter bound; `converged` and the final diagnostic ratios determine validity.

## Conditional estimation

Conditional estimation is the default for one dependent variable;
`SienaAlgorithm(conditional=true, condvar=:friendship)` requests it with several.
Each period runs until that variable's distance from the period-start observation
reaches its observed amount of change, rather than until time one. There must be
positive observed change in every period. Its basic rates leave the free moment
vector and are estimated from the phase-3 stopping times (simulation rate × mean
stopping time), with RSiena's standard error (simulation rate × sd of the stopping
time), in `result.rate_estimates` and `result.rate_standard_errors`. The basic rates
of the other dependent variables are rescaled to the same time unit. The caller's
data and effects are unchanged.

```julia
result.conditional                        # true: one dependent network
result.rate_estimates[:friendship]
result.rate_standard_errors[:friendship]
unconditional = fit_siena(data, effects; rng=MersenneTwister(1),
                          algorithm=SienaAlgorithm(verbose=false, conditional=false))
```

The conditional s50 model is pinned against six RSiena `cond=TRUE` fits
(`test/fixtures/s50_siena07_cond.toml`): coefficients, standard errors, rates and
rate standard errors agree within four combined Monte-Carlo standard deviations.
Before 0.2 the conditional standard errors came from a 30-simulation forward finite
difference — biased (15–25 % too small at large budgets, measured against RSiena) and
noisy (up to 2.2 times too large at the default budget) — and the default fit
diverged.

## Score-type tests

`include_effects!(effects, var, names; fix=true, test=true)` holds the effects at
their initial value (usually 0) and tests them with the generalised score-type test
of Schweinberger (2012), RSiena's `test=TRUE`. The tested statistics are simulated
in phase 3 with the estimated ones, and their deviation is orthogonalised against
the estimated parameters' moment equations before it is referred to a chi-square
distribution.

```julia
include_effects!(effects, :friendship, [:cycle3]; fix=true, test=true)
tested = fit_siena(data, effects; rng=MersenneTwister(5),
                   algorithm=SienaAlgorithm(verbose=false))
tested.score_test            # joint and per-effect tests, one-sided z, one-step estimates
```

## Time heterogeneity

[`siena_time_test`](@ref) is RSiena's `sienaTimeTest`: for every estimated effect
and every period after the first it tests, with the same score-type test, whether a
period dummy for that effect is needed. It uses the per-period statistics and
scores of the fit's phase-3 simulations, so it runs no new simulations.

```julia
heterogeneity = siena_time_test(result)
heterogeneity.p_value              # joint test over all effects and periods
heterogeneity.effect_p_values      # per effect
siena_time_test(result; effects=["recip", "transTrip"])
```

The s50 statistics are compared with RSiena's (six fits), and a simulation testset
checks that a time-homogeneous model is rejected about 5 % of the time and a
heterogeneous one often. RSiena's remedy, period-dummy effects (`sienaTimeFix`), is
not implemented.

## Restricting co-evolution

`model_type=:networkonly` freezes behaviors; `:behavioronly` freezes networks.
The frozen variables remain readable by effects of the simulated variables, but their
own effects leave the free parameter vector. `:standard` simulates all dependents.
This setting is different from RSiena's `modelType`. Siena.jl uses RSiena's default
network model for each kind of network (the standard model for directed networks,
the forcing model, type 2, for undirected ones); initiative, pairwise and double-step
models are not implemented.

## Practical checks

Start with a small, scientifically motivated effect set. Check every convergence
criterion, derivative conditioning, and goodness of fit. Increase simulation precision
or revisit redundant effects when the independent final diagnostics fail. A warning
or exception must not be bypassed merely to obtain interpretable coefficients.

Results hold independent copies of data and effects. To seed a subsequent fit with
previous estimates, explicitly match entries by `result.parameter_names`; do not assume
that adding effects preserves positions. Missing ties, period-dummy effects
(`sienaTimeFix`) and endowment/creation parameter estimation are not supported.
Structural-status changes between waves do not receive RSiena's correction.

## Names changed before the first release

Siena.jl 0.1.0 was a development version and was never released. Some of its
spellings were renamed for 0.2.0 without compatibility aliases; the old names now
raise an error.

| Development spelling | Use instead |
|:--|:--|
| `SienaAlgorithm(parallel=...)`, `siena_algorithm(parallel=...)` | `threaded=...`, the keyword the other packages use |
| `seed=k` in `SienaAlgorithm`, `simulate_saom`, `siena_gof` and `gof` | `rng=MersenneTwister(k)` |
| `n_sims=` in `siena_gof` and `gof` | `n_sim=` |
| `BalanceEffect` | `BalanceSimpleEffect` (not RSiena's `balance`) |
| `IsolateEffect` | `IsolateNetEffect` (RSiena's `isolateNet`) |
| `AverageAttHigherEffect`, `AverageAttLowerEffect` | `AverageAttHigherSimpleEffect`, `AverageAttLowerSimpleEffect` |
| short names `:sharedIn`, `:sharedOut` | `:sharedInNbrs`, `:sharedOutNbrs` (RSiena's `sharedIn` is a two-network effect) |
