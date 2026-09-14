# Model Estimation

Siena estimates stochastic actor-oriented models by Method of Moments. It matches
expected simulated statistics to observed target statistics; it does not maximize
a likelihood. Each simulated period starts at the corresponding observed wave.

## Estimation phases

1. **Phase 1:** estimate a finite-difference derivative using common random numbers,
   then run `phase1_iterations` rough updates with `initial_gain`.
2. **Phase 2:** run `n_subphases` subphases, each of `phase1_iterations` updates.
   The gain halves between subphases and the last three quarters of each subphase's
   parameter iterates are averaged (Polyak–Ruppert averaging). The derivative is
   refreshed at the first and last subphase.
3. **Newton refinement:** use up to `refine_max` capped updates based on simulated
   moment means and score-function derivatives. Each batch contains
   `2 * phase3_iterations` simulations. Conditional models use finite differences.
4. **Phase 3:** freeze the estimate and draw `phase3_iterations` new simulations
   for convergence diagnostics and standard errors. These observations were not
   used to select the estimate or decide when to stop refinement.

The update is `θ ← θ − gain * (D \ deviation)`, capped in Euclidean step norm.
Refinement checks a stricter internal convergence target (half the configured
thresholds), leaving room for independent validation noise. Refinement cannot
guarantee convergence of an unidentified or poorly specified model.

## A reproducible fit

```julia
using Siena, Networks, Random

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
streams. `parallel=false` runs serially; `parallel=true` uses available Julia threads.
Both reduce simulation results in the same fixed order. Thread counts do not change
the Monte Carlo sample or the fitted model. The deprecated `seed` keyword preserves
legacy repeat-per-fit behavior; use fresh generators for new code.

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
| `derivative_sims` | 30 | Finite-difference samples per perturbation |
| `derivative_method` | `:score` | Phase-3 derivative, or `:finite_difference` |
| `refine_max` | 5 | Maximum Newton refinements; zero disables them |
| `convergence_threshold` | 0.1 | Required maximum absolute per-effect t-ratio |
| `overall_convergence_threshold` | 0.25 | Required overall convergence ratio |
| `allow_unconverged` | `false` | Return a warned diagnostic result instead of throwing |
| `rng` | `Random.default_rng()` | Generator used when the fit supplies no override |
| `parallel` | `true` | Thread independent simulation batches |
| `verbose` | `true` | Print phase progress; warnings remain enabled when false |
| `model_type` | `:standard` | Variables that co-evolve |
| `conditional` | `false` | Condition on observed change counts |
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
were not met. The defaults instead throw [`SienaConvergenceError`](@ref); the full
result remains available as `error.result`.

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

`SienaAlgorithm(conditional=true, condvar=:friendship)` conditions each period on
that variable's observed amount of change rather than elapsed time one. There must
be positive observed change in every period. Its basic rates leave the free moment
vector and are estimated from phase-3 stopping times. They remain in
`result.rate_estimates`; the caller's data and effects are unchanged.

Conditional estimation uses finite-difference derivatives because the implemented
trajectory-score formula assumes fixed-time termination. It is implemented and tested,
but the fitted RSiena golden fixture currently validates unconditional estimation.

## Restricting co-evolution

`model_type=:networkonly` freezes behaviors; `:behavioronly` freezes networks.
The frozen variables remain readable by effects of the simulated variables, but their
own effects leave the free parameter vector. `:standard` simulates all dependents.
This setting is different from RSiena's `modelType`: forcing, initiative and pairwise
network model types are not implemented.

## Practical checks

Start with a small, scientifically motivated effect set. Check every convergence
criterion, derivative conditioning, and goodness of fit. Increase simulation precision
or revisit redundant effects when the independent final diagnostics fail. A warning
or exception must not be bypassed merely to obtain interpretable coefficients.

Results hold independent copies of data and effects. To seed a subsequent fit with
previous estimates, explicitly match entries by `result.parameter_names`; do not assume
that adding effects preserves positions. Missing ties, interaction effects, and
endowment/creation parameter estimation are not supported. Structural-status changes
between waves do not receive RSiena's correction.
