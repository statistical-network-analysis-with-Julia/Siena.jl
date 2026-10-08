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
<p><a href="/NetworkCore.jl/dev/">NetworkCore.jl</a> supplies the panel and the network-to-Siena bridge. The examples below assume the environment is already prepared.</p>
```

## Using NetworkCore.jl objects

`NetworkCore.load_dataset(:s50)` returns fresh friendship networks and alcohol/smoking
matrices. `DependentNetwork` accepts network objects directly; matrix waves are also
accepted. The bridge is part of Siena's ordinary source and preserves directedness,
node sets, self-loop policy and one-/two-mode structure.

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
```

[`get_effects`](@ref) already includes RSiena's default effects, as `getEffects()`
does: one basic rate per period, `outdegree` (density) and `recip` (naming them again
changes nothing). The six objective parameters model density, reciprocity, transitive
triplets, and smoking alter/ego/similarity effects. Covariates
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

An unconverged fit is returned with a warning, as RSiena does (the warning is
emitted even when `verbose=false`), with `result.converged == false`; do not
interpret it as a validated estimate. `SienaAlgorithm(allow_unconverged=false)`
throws `SienaConvergenceError` instead, whose `result` field holds the full
diagnostic result.

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
per wave (its `linear` and `quad` shape effects are included by default, as in RSiena)
and include its influence effects; the network's covariate effects
(`egoX`, `altX`, `simX`, …) are also available for the behaviour itself
(`egoalcohol`, `simalcohol`, …), which models selection on the co-evolving behaviour.
With two dependent variables the estimation is unconditional by default; use
`SienaAlgorithm(conditional=true, condvar=:friendship)` to condition on one of them.
An undirected relation is a `DependentNetwork(...; directed=false)`.

Missing ties are not handled by the estimator. The NetworkCore bridge rejects masked
dyads unless an explicit face-value conversion policy is supplied. Structural codes
10 and 11 mean known, fixed zero/one values, and are not missing-data codes.

The target statistic of every effect offered under an RSiena name is checked against
RSiena, and five fitted s50 models (unconditional, conditional with a score test,
undirected, co-evolution, GWESP with an interaction) are compared with RSiena fits.
Interactions (`include_interaction!`) and time-heterogeneity tests
(`siena_time_test`) follow RSiena. Maximum likelihood, Bayesian estimation,
multi-group data and initiative/pairwise network model types remain unavailable. See [Model Estimation](guide/estimation.md),
[Effects](@ref), and [Data Preparation](@ref) for details.
