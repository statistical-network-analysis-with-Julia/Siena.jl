# Getting Started

This tutorial fits a stochastic actor-oriented model to the bundled `s50` friendship
panel: 50 actors observed at three waves, with smoking at wave 1 as a covariate.
The same model is covered by the package's RSiena reference fit.

## Prepare the environment

```@raw html
<p>Use Julia <strong>1.12+</strong> and the <a href="/getting-started/">workspace installation guide</a> for this <strong>unreleased 0.2.0 development version</strong>. From the prepared workspace:</p>
```

```bash
julia --project=Siena.jl
```

```@raw html
<p><a href="/Networks.jl/dev/">Networks.jl</a> supplies the panel and the network-to-Siena bridge. The examples below assume the environment is already prepared.</p>
```

## Using Networks.jl objects

`Networks.load_dataset(:s50)` returns fresh friendship networks and alcohol/smoking
matrices. `DependentNetwork` accepts network objects directly; matrix waves are also
accepted. The bridge is part of Siena's ordinary source and preserves directedness,
node sets, self-loop policy and one-/two-mode structure.

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
```

Basic rates are included once per period. The six objective parameters model density,
reciprocity, transitive triplets, and smoking alter/ego/similarity effects. Covariates
are centered automatically; use `center=false` only when uncentered values are intended.

## Fit and check convergence

```julia
result = fit_siena(data, effects; rng=MersenneTwister(1),
                   algorithm=SienaAlgorithm(verbose=false))
println(result)
@assert result.converged
```

Estimation uses Robbins–Monro updates, capped Newton refinement, and a separate final
simulation batch. A successful return under the defaults requires every convergence
|t-ratio| below 0.1 and overall `tconv_max` below 0.25, plus an identifiable derivative
matrix. These diagnostics measure simulated moment discrepancies; they are different
from the Wald z-statistics testing coefficients.

An unconverged fit throws `SienaConvergenceError`. Its `result` field preserves the
full diagnostic result. For a deliberately short exploratory run, explicitly set
`allow_unconverged=true`; the returned result then emits a warning even when
`verbose=false`. Do not interpret that diagnostic result as a validated estimate.

A generator is consumed by a fit. To reproduce an analysis, construct a fresh seeded
RNG as above. `siena07 === fit_siena`; both names call the same function.

## Inspect estimates and uncertainty

```julia
coef(result)
stderror(result)
vcov(result)
confint(result)
coeftable(result)
result.t_ratios
result.tconv_max
result.condition_number
```

The parameter order is listed in `result.parameter_names`: free rates first, then
objective parameters. `coeftable` reports Wald tests for objective parameters; tests
of a positive basic rate against zero are left undefined. No likelihood is evaluated,
so likelihood, AIC and BIC are unavailable for this Method-of-Moments fit.

The raw derivative and phase-3 statistic covariance are retained for diagnosis:

```julia
size(result.derivative_matrix)
size(result.phase3_cov)
result.n_refinements
result.n_simulations_run
```

Standard errors use `D⁻¹ Σ D⁻ᵀ`. No fixed ridge is added. Singular or severely
ill-conditioned derivatives produce undefined standard errors and fail convergence.

## Goodness of fit

The result saves an independent copy of its data and effects:

```julia
report = gof(result; n_sim=100, rng=Xoshiro(2))
println(report)
triads = gof(result, TriadCensus(:friendship); n_sim=100, rng=Xoshiro(3))
```

The default report compares indegree and outdegree distributions for networks and
category distributions for behaviors. Select additional statistics based on the
scientific question. See [Goodness of Fit](guide/gof.md).

## Extending the analysis

For joint network/behavior models, add a `DependentBehavior` with one integer vector
per wave and include its shape and influence effects. For conditional estimation,
use `SienaAlgorithm(conditional=true, condvar=:friendship)`; basic rates for the
conditioning variable are fixed only on the fit's own effects copy and recovered
from stopping times. Conditional derivatives use finite differences.

Missing ties are not handled by the estimator. The Networks bridge rejects masked
dyads unless an explicit face-value conversion policy is supplied. Structural codes
10 and 11 mean known, fixed zero/one values, and are not missing-data codes.

RSiena parity is tested for 34 deterministic effect targets and one fitted s50 model,
not the entire effects catalogue. Network selection based on an evolving behavior
is not provided by `get_effects`; an observed covariate is not a substitute for
that changing behavior state. Conditional fitting is implemented, but the fitted
RSiena reference validates the unconditional model above. Maximum likelihood, Bayesian estimation,
interaction effects, and forcing/initiative network model types remain unavailable.
See [Model Estimation](guide/estimation.md), [Effects](@ref), and [Data Preparation](@ref) for details.
