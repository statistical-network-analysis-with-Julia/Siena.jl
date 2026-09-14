# Goodness of Fit

A converged fit solves the selected moment equations. Goodness of fit asks whether
it also reproduces scientifically relevant features such as degree distributions
and triad counts.

## Fit an example model

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

## Shared result interface

`gof(result)` uses the independent data snapshot stored on the fit. It returns the
shared `Networks.GOFResult`. Defaults are in/outdegree distributions for every network
and category distributions for every behavior variable.

```julia
report = gof(result; n_sim=100, rng=Xoshiro(2))
println(report)
selected = gof(result, [IndegreeDistribution(:friendship), TriadCensus(:friendship)];
               n_sim=100, rng=Xoshiro(3))
```

Supply `n_sim` and a fresh seeded generator for reproducible simulation. The older
`n_sims` and `seed` keywords are deprecated compatibility shims. At least two draws
are required to estimate a covariance. A larger sample gives finer Monte Carlo
resolution.

## Available statistics

| Statistic | What it compares |
|:--|:--|
| `IndegreeDistribution(variable)` | Counts by incoming degree |
| `OutdegreeDistribution(variable)` | Counts by outgoing degree |
| `TriadCensus(variable)` | Directed triad census |
| `GeodesicDistribution(variable)` | Shortest-path distance counts |
| `BehaviorDistribution(variable)` | Counts by behavior category |

The model's `model_type` restriction is respected during GOF simulation. Periods
start at their observed start waves; GOF compares simulated and observed final-wave
statistics. The result is not a free-running forecast across all waves.

## RSiena-style detail

`siena_gof` and the convenience functions return a `SienaGOFResult` with the observed
counts, simulated count matrix, labels, per-level p-values, Mahalanobis distance and
overall p-value. Existing methods accepting data explicitly remain available.

```julia
detail = siena_gof(result, result.data, IndegreeDistribution(:friendship);
                   n_sim=100, rng=Xoshiro(4))
detail.labels
detail.observed
size(detail.simulated)
detail.p_values
detail.mahalanobis
detail.p_overall
```

Per-level p-values use the shared two-sided rank-tail convention:
`min(1, 2 * (1 + min(number ≤ observed, number ≥ observed)) / (N + 1))`.
Ties count in both tails; the finite-simulation correction prevents zero p-values.
This is identical to `Networks.mc_pvalue`.

The overall p-value is the upper-tail Monte Carlo proportion of simulated Mahalanobis
distances at least as large as the observed distance, with the same `+1` correction.
The statistic covariance is regularized for this descriptive distance because
frequency tables have linear constraints. This GOF regularization is separate from
the unregularized derivative used for fitted-parameter standard errors.

These are simulation diagnostics, not exact finite-sample tests: the reference mean
and covariance are themselves estimated from simulations, and parameter-estimation
uncertainty is not included. Individual bins are also dependent. Inspect the simulated
distributions and substantively important discrepancies instead of interpreting each
bin as an independent hypothesis test.

```julia
converted = GOFResult(detail)
println(converted)
```

## Choosing checks

Degree distributions diagnose heterogeneity. A triad census can reveal transitivity
or reciprocity patterns missed by the fitted effects. Behavior distributions check
shape and category support. Choose relevant statistics before comparing alternative
models, and first resolve convergence failures. A high simulation p-value does not
prove that the model is correct.
