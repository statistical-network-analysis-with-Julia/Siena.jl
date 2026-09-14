# Siena.jl

Model how a network changes between observed waves. Siena.jl fits stochastic
actor-oriented models (SAOMs) to network panels, with optional behavior variables
and actor or dyad covariates. Actors receive opportunities to change a tie or
behavior; simulated changes connect the observed waves.

**Start here:** [Getting started](getting_started.md) ·
[Prepare panel data](guide/data.md) · [Choose effects](guide/effects.md) ·
[Estimation API](api/estimation.md)

## Fit the bundled friendship panel

This example uses three friendship waves for 50 actors and smoking measured at
wave 1. It fits the unconditional model used by the package's RSiena fit fixture.
Simulation is seeded; estimation takes longer than a descriptive network measure.

```@raw html
<p>Use Julia <strong>1.12+</strong> and the <a href="/getting-started/">workspace installation guide</a> for the current <strong>0.2.0 development version, unreleased</strong>. The examples assume that environment is already prepared.</p>
```

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

fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false))
@assert fit.converged
coeftable(fit)
(fit.t_ratios, fit.tconv_max)
```

The model includes reciprocity, transitive triplets and associations with the
observed smoking covariate. These estimates alone do not establish peer influence
or a causal effect of smoking.

## Scope and checks

| Need | Current support |
|---|---|
| Network panels | One-mode and two-mode networks, structural fixed ties, covariates and composition changes; see [data preparation](guide/data.md). |
| Estimation | Method of Moments with stochastic approximation, capped Newton refinement and a fresh final simulation batch. Conditional estimation is also implemented; its validation scope differs from the unconditional example. |
| Uncertainty and fit | Raw derivative and phase-3 covariance, convergence diagnostics and simulation-based [goodness of fit](guide/gof.md). No likelihood, AIC or BIC is supplied for Method of Moments. |
| RSiena correspondence | Reference checks cover 34 deterministic effect targets and one fitted s50 model. This is a subset of RSiena, with [effect-specific limitations](guide/effects.md). |

A fit must pass the final convergence checks: individual absolute t-ratios below
0.1, `tconv_max < 0.25`, and an identifiable derivative. Failure raises
`SienaConvergenceError`; its `result` holds the diagnostics. Explicit
`allow_unconverged=true` is for diagnosis and still warns. A successful return is
followed by substantive model checking, including goodness of fit.

Missing ties are unsupported by estimation; structural zeros and ones are known
constraints, not missing observations. Selection based on an evolving behavior is
not supplied by the default effects table. Maximum likelihood, Bayesian estimation,
interaction effects and RSiena forcing/initiative model types remain unavailable.
See the [estimation guide](guide/estimation.md) before extending the example.

```@raw html
<p>For a single observed network, use <a href="/SNA.jl/dev/">SNA.jl</a> for descriptive analysis or <a href="/ERGM.jl/dev/">ERGM.jl</a> for a network model. For individual timestamped interactions, start with <a href="/REM.jl/dev/">REM.jl</a>.</p>
```

Package citation: [CITATION.bib](https://github.com/statistical-network-analysis-with-Julia/Siena.jl/blob/main/CITATION.bib).
