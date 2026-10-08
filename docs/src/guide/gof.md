# Goodness of Fit

A converged fit solves the selected moment equations. Goodness of fit asks whether
it also reproduces scientifically relevant features such as degree distributions
and triad counts.

## Fit an example model

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

## Shared result interface

`gof(result)` is the preferred entry point. It uses the independent data snapshot
stored on the fit and returns the shared `NetworkCore.GOFResult`. Defaults are
in/outdegree distributions for every network and category distributions for every
behavior variable.

```julia
report = gof(result; n_sim=100, rng=Xoshiro(2))
println(report)
selected = gof(result, [IndegreeDistribution(:friendship), TriadCensus(:friendship)];
               n_sim=100, rng=Xoshiro(3))
```

Supply `n_sim` and a fresh seeded generator for reproducible simulation. At least
two draws are required to estimate a covariance. A larger sample gives finer Monte Carlo
resolution.

## Available statistics

| Statistic | What it compares | Default levels |
|:--|:--|:--|
| `IndegreeDistribution(variable)` | Actors with indegree ≤ k | `0:8`, cumulative (RSiena) |
| `OutdegreeDistribution(variable)` | Actors with outdegree ≤ k | `0:8`, cumulative (RSiena) |
| `TriadCensus(variable)` | Directed triad census | the 16 classes |
| `GeodesicDistribution(variable)` | Ordered pairs at distance ≤ k | `1:5`, cumulative (RSiena) |
| `BehaviorDistribution(variable)` | Actors with behaviour ≤ value | the behaviour's range, cumulative |

The levels are fixed in advance and never derived from the observed network
(`levls=` and `cumulative=false` change them). Before 0.2 the degree levels were
`0:maximum(observed)`, which dropped simulated mass above the observed maximum and
made the test depend on the data.

The model's `model_type` restriction and conditioning are respected during GOF
simulation. Every period starts at its observed start wave, and the statistic at the
end of every period is summed over the periods (RSiena's `join=TRUE`); pass
`period=m` to compare one period. The result is not a free-running forecast across
all waves.

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
This is identical to `NetworkCore.mc_pvalue`.

The overall p-value is a Monte-Carlo test of the Mahalanobis distance. The mean and
covariance are computed from the observed vector **pooled** with the `N` simulated
ones (a pseudo-inverse handles the linear constraints of frequency tables), and the
observed distance is ranked among all `N + 1` distances:
`p = (1 + #{simulated distance ≥ observed}) / (N + 1)`. Under the fitted model the
`N + 1` vectors are exchangeable, so this is an exact Monte-Carlo test: a correct
model is rejected at the 5 % level about 5 % of the time (a simulation testset checks
it). `sienaGOF` uses the simulations alone for the mean and covariance; with
data-derived levels that combination rejected a correct model 15.7 % of the time.

The uncertainty of the estimated parameters is not included, which makes the test
slightly conservative. Individual bins are dependent: inspect the simulated
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
