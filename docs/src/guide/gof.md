# Goodness of Fit

This guide covers how to assess whether an estimated SAOM adequately reproduces key features of the observed network. Goodness of fit (GOF) testing compares network statistics from the observed data to distributions of the same statistics from simulated networks.

## Overview

Even if a model converges, it may not capture all important features of the network. GOF assessment works by:

1. Simulating many networks from the estimated model
2. Computing auxiliary statistics (not used in estimation) for each simulation
3. Comparing the observed statistics to the simulated distribution
4. Computing p-values to test whether the observed values are plausible under the model

A good model should reproduce network features that were not explicitly modeled.

## Available GOF Statistics

| Statistic | Type | Description |
|-----------|------|-------------|
| [`IndegreeDistribution`](@ref) | Network | Distribution of actors' indegrees |
| [`OutdegreeDistribution`](@ref) | Network | Distribution of actors' outdegrees |
| [`TriadCensus`](@ref) | Network | Counts of triad types (mutual, asymmetric, null) |
| [`GeodesicDistribution`](@ref) | Network | Distribution of shortest path lengths |
| [`BehaviorDistribution`](@ref) | Behavior | Distribution of behavior values |

## Running GOF Tests

### Basic Usage

```julia
# After estimation
result = siena07(data, effects; algorithm=siena_algorithm(seed=42))

# GOF for indegree distribution
gof_in = siena_gof(result, data, IndegreeDistribution(:friendship);
    n_sims=100, seed=42)
println(gof_in)
```

### Convenience Functions

Siena.jl provides convenience functions for common GOF statistics:

```julia
# Indegree distribution
gof_in = siena_gof_indegree(result, data, :friendship; n_sims=100, seed=42)

# Outdegree distribution
gof_out = siena_gof_outdegree(result, data, :friendship; n_sims=100, seed=42)

# Triad census
gof_triad = siena_gof_triad(result, data, :friendship; n_sims=100, seed=42)

# Behavior distribution
gof_beh = siena_gof_behavior(result, data, :drinking; n_sims=100, seed=42)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `SienaResult` | Required | Estimation result |
| `data` | `SienaData` | Required | The data object |
| `statistic` | `AbstractGOFStatistic` | Required | The GOF statistic |
| `n_sims` | `Int` | 100 | Number of simulations |
| `seed` | `Int` or `nothing` | nothing | Random seed |

### Choosing Number of Simulations

| n_sims | Use Case |
|--------|----------|
| 50 | Quick exploratory check |
| 100 | Standard analysis |
| 200-500 | Publication-quality results |

More simulations provide more precise p-values but increase computation time linearly.

## Interpreting GOF Results

### The GOFResult Object

The [`GOFResult`](@ref) contains:

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `AbstractGOFStatistic` | The GOF statistic used |
| `labels` | `Vector` | Labels for each level/category |
| `observed` | `Vector{Int}` | Observed statistic values |
| `simulated` | `Matrix{Int}` | Simulated values (n_sims x n_levels) |
| `p_values` | `Vector{Float64}` | Level-specific p-values |
| `mahalanobis` | `Float64` | Mahalanobis distance |
| `p_overall` | `Float64` | Overall p-value |

### Displaying Results

```julia
println(gof_in)
```

Output:

```text
Goodness of Fit: IndegreeDistribution
Overall p-value: 0.4523

Level-specific results:
  0         : obs=5, sim=4.8 (1.2), p=0.850
  1         : obs=12, sim=11.3 (2.1), p=0.720
  2         : obs=15, sim=14.7 (2.5), p=0.890
  3         : obs=10, sim=10.9 (2.0), p=0.640
  4         : obs=5, sim=5.3 (1.5), p=0.810
  5         : obs=3, sim=3.0 (1.1), p=0.950
```

### Reading the Output

For each level (e.g., indegree value):
- **obs**: Observed count in the real network
- **sim**: Mean count across simulations (with standard deviation)
- **p**: Two-sided p-value (proportion of simulations as or more extreme)

### Overall Assessment

| Overall p-value | Interpretation |
|-----------------|----------------|
| > 0.10 | Good fit -- model reproduces this feature well |
| 0.05 - 0.10 | Marginal -- model may not fully capture this feature |
| < 0.05 | Poor fit -- model fails to reproduce this feature |

### What to Check

A comprehensive GOF assessment should test multiple statistics:

```julia
# Check all standard GOF statistics
gof_in = siena_gof_indegree(result, data, :friendship; n_sims=100, seed=42)
gof_out = siena_gof_outdegree(result, data, :friendship; n_sims=100, seed=42)
gof_triad = siena_gof_triad(result, data, :friendship; n_sims=100, seed=42)

println("Indegree GOF p-value:  ", round(gof_in.p_overall, digits=3))
println("Outdegree GOF p-value: ", round(gof_out.p_overall, digits=3))
println("Triad GOF p-value:     ", round(gof_triad.p_overall, digits=3))
```

## Iterative Model Building with GOF

GOF results guide model improvement. If a particular feature is not well reproduced, add effects that capture that feature:

### Step 1: Estimate a Basic Model

```julia
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])

result1 = siena07(data, effects; algorithm=siena_algorithm(seed=42))
```

### Step 2: Check GOF

```julia
gof_triad = siena_gof_triad(result1, data, :friendship; n_sims=100, seed=42)
println("Triad p-value: ", gof_triad.p_overall)
# If p < 0.05, the model does not reproduce triadic structure
```

### Step 3: Add Effects to Improve Fit

```julia
# Poor triad fit suggests adding transitivity
include_effects!(effects, :friendship, [:transTrip])

result2 = siena07(data, effects; algorithm=siena_algorithm(seed=42))
```

### Step 4: Re-Check GOF

```julia
gof_triad2 = siena_gof_triad(result2, data, :friendship; n_sims=100, seed=42)
println("Triad p-value (after adding transitivity): ", gof_triad2.p_overall)
# Should be improved
```

### Common GOF Problems and Solutions

| Poor Fit on... | Likely Missing Effect |
|----------------|---------------------|
| Indegree distribution | `IndegreePopularityEffect` (inPop or inPopSqrt) |
| Outdegree distribution | `OutdegreeActivityEffect` (outAct or outActSqrt) |
| Triad census | `TransitiveTripletsEffect` or `GWESPEffect` |
| Geodesic distances | Transitivity effects; possibly `NbrDist2Effect` |
| Behavior distribution | `QuadraticShapeEffect`; influence effects |

## GOF for Behavior Models

When modeling behavior co-evolution, check both network and behavior GOF:

```julia
# Network GOF
gof_in = siena_gof_indegree(result, data, :friendship; n_sims=100, seed=42)
gof_out = siena_gof_outdegree(result, data, :friendship; n_sims=100, seed=42)

# Behavior GOF
gof_beh = siena_gof_behavior(result, data, :drinking; n_sims=100, seed=42)

println("Network indegree GOF:  p = ", round(gof_in.p_overall, digits=3))
println("Network outdegree GOF: p = ", round(gof_out.p_overall, digits=3))
println("Behavior GOF:          p = ", round(gof_beh.p_overall, digits=3))
```

## Advanced GOF Options

### Custom Degree Levels

Specify which degree values to test:

```julia
# Only test indegrees 0-10
gof = siena_gof(result, data,
    IndegreeDistribution(:friendship; levls=collect(0:10));
    n_sims=100, seed=42)
```

### Geodesic Distribution

Test the distribution of shortest path lengths:

```julia
gof_geo = siena_gof(result, data,
    GeodesicDistribution(:friendship; max_dist=5);
    n_sims=100, seed=42)
println(gof_geo)
```

### Accessing Simulated Distributions

For custom analysis or visualization:

```julia
gof = siena_gof_indegree(result, data, :friendship; n_sims=100, seed=42)

# Observed values
obs = gof.observed

# Simulated values (100 x n_levels matrix)
sim = gof.simulated

# Mean and SD of simulated distribution
sim_mean = vec(mean(gof.simulated, dims=1))
sim_sd = vec(std(gof.simulated, dims=1))

# Level-specific p-values
p_vals = gof.p_values
```

## Complete GOF Example

```julia
using Siena
using Random
using Statistics

Random.seed!(42)

# Setup data (abbreviated)
n = 30
data = siena_data()
add_nodeset!(data, NodeSet(n))
waves = [rand(0:1, n, n) for _ in 1:3]
for w in waves; w[diagind(w)] .= 0; end
add_dependent!(data, DependentNetwork(:net, waves))

# Estimate model
effects = get_effects(data)
include_effects!(effects, :net, [:outdegree, :recip, :transTrip])
result = siena07(data, effects; algorithm=siena_algorithm(seed=42, verbose=false))

# Comprehensive GOF assessment
if result.converged
    println("=== Goodness of Fit Assessment ===\n")

    gof_in = siena_gof_indegree(result, data, :net; n_sims=100, seed=42)
    gof_out = siena_gof_outdegree(result, data, :net; n_sims=100, seed=42)
    gof_triad = siena_gof_triad(result, data, :net; n_sims=100, seed=42)

    println("Indegree distribution:  p = $(round(gof_in.p_overall, digits=3))")
    println("Outdegree distribution: p = $(round(gof_out.p_overall, digits=3))")
    println("Triad census:           p = $(round(gof_triad.p_overall, digits=3))")

    all_good = all(g -> g.p_overall > 0.05, [gof_in, gof_out, gof_triad])
    println("\nOverall assessment: ", all_good ? "Model fits well" : "Consider adding effects")
else
    println("Model did not converge -- fix convergence before GOF")
end
```

## Best Practices

1. **Always check GOF** after estimation, even if the model converges
2. **Test multiple statistics**: indegree, outdegree, and triad census at minimum
3. **Use sufficient simulations**: at least 100, preferably 200+ for publication
4. **Set a random seed**: for reproducible GOF results
5. **Do not over-fit**: adding too many effects to improve GOF can lead to unstable estimation
6. **Report GOF p-values** in publications alongside parameter estimates
7. **Iterate**: use GOF results to guide model improvement, then re-check
8. **Check behavior GOF** separately when modeling co-evolution
