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
wave 1. It fits the model of the package's RSiena fit fixtures; with one dependent
network the estimation is conditional on the observed change, RSiena's default.
Simulation is seeded; estimation takes longer than a descriptive network measure.

```@raw html
<p>Use Julia <strong>1.12+</strong> and the <a href="/getting-started/">workspace installation guide</a> for the current <strong>0.2.0 development version, unreleased</strong>. The examples assume that environment is already prepared.</p>
```

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

fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false))
@assert fit.converged
coeftable(fit)
(fit.t_ratios, fit.tconv_max)
```

The model includes reciprocity, transitive triplets and associations with the
observed smoking covariate. As RSiena's `getEffects()`, [`get_effects`](@ref)
already includes the basic rates, `outdegree` (RSiena's `density`) and `recip`, so
naming them again changes nothing. These estimates alone do not establish peer influence
or a causal effect of smoking.

## Scope and checks

| Need | Current support |
|---|---|
| Network panels | Directed, undirected (RSiena's model type 2) and two-mode networks, behaviour co-evolution (with selection on the behaviour), structural fixed ties, covariates and whole-wave composition changes; see [data preparation](guide/data.md). |
| Estimation | Method of Moments with stochastic approximation, capped Newton refinement and a fresh final simulation batch; conditional (the default for one dependent variable, as in RSiena) or unconditional; interactions, score-type tests of fixed effects and time-heterogeneity tests. |
| Uncertainty and fit | Score-function derivative, phase-3 covariance, convergence diagnostics and simulation-based [goodness of fit](guide/gof.md) with an exact Monte-Carlo test. No likelihood, AIC or BIC is supplied for Method of Moments. |
| RSiena correspondence | The default effects of `getEffects()`, the target statistic of every effect offered under an RSiena name, interaction targets, five fitted s50 models (unconditional, conditional with a score test, undirected, co-evolution, GWESP with an interaction) and the time test are checked against RSiena 1.6.6. RSiena features that are absent are listed in the README and the [effects concordance](guide/effects.md). |

A fit must pass the final convergence checks: individual absolute t-ratios below
0.1, `tconv_max < 0.25`, and an identifiable derivative. An unconverged fit is
returned with a warning, as RSiena does, and flagged by `show` and
`approximations`; its estimates must not be reported
(`allow_unconverged=false` throws `SienaConvergenceError` instead). A successful
return is followed by substantive model checking, including goodness of fit.

Missing ties are unsupported by estimation; structural zeros and ones are known
constraints, not missing observations. Maximum likelihood, Bayesian estimation,
multi-group data and RSiena's initiative/pairwise model types remain unavailable.
See the [estimation guide](guide/estimation.md) before extending the example.

```@raw html
<p>For a single observed network, use <a href="/SNA.jl/dev/">SNA.jl</a> for descriptive analysis or <a href="/ERGM.jl/dev/">ERGM.jl</a> for a network model. For individual timestamped interactions, start with <a href="/REM.jl/dev/">REM.jl</a>.</p>
```

Package citation: [CITATION.bib](https://github.com/statistical-network-analysis-with-Julia/Siena.jl/blob/main/CITATION.bib).
