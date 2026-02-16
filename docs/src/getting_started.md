# Getting Started

This tutorial walks through a complete SAOM analysis using Siena.jl, from data preparation to model interpretation.

## Installation

Install Siena.jl from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Statistical-network-analysis-with-Julia/Siena.jl")
```

## Basic Workflow

The typical Siena.jl workflow consists of five steps:

1. **Prepare data** - Create node sets, dependent variables, and covariates
2. **Define effects** - Choose which social mechanisms to model
3. **Configure algorithm** - Set estimation parameters
4. **Estimate model** - Run `siena07` for Method of Moments estimation
5. **Interpret results** - Assess convergence, significance, and goodness of fit

## Step 1: Prepare Data

SAOM analysis requires panel data: networks (and optionally behavior) observed at two or more time points.

### Create a Node Set

```julia
using Siena

# 50 actors
nodes = NodeSet(50)

# With named actors
nodes = NodeSet(5; names=["Alice", "Bob", "Carol", "David", "Eve"])
```

### Define a Dependent Network

Network data consists of adjacency matrices at each observation wave:

```julia
# Three waves of a 50-actor directed network
n = 50
wave1 = zeros(Int, n, n)
wave2 = zeros(Int, n, n)
wave3 = zeros(Int, n, n)

# Populate with observed ties (1 = tie present, 0 = absent)
# In practice, load from data files
using Random
Random.seed!(42)
for w in [wave1, wave2, wave3]
    for i in 1:n, j in 1:n
        i != j && rand() < 0.1 && (w[i,j] = 1)
    end
end

# Create dependent network variable
friendship = DependentNetwork(:friendship, [wave1, wave2, wave3])
println("Waves: ", n_waves(friendship))    # 3
println("Actors: ", n_actors(friendship))  # 50
```

### Add Covariates

Actor-level covariates can be constant or varying across waves:

```julia
# Constant covariate (same value at all waves)
gender = constant_covariate(:gender, rand(0:1, n))

# Varying covariate (different values at each wave)
performance = varying_covariate(:performance, [rand(n), rand(n), rand(n)])
```

### Assemble the Data Object

```julia
data = siena_data()
add_nodeset!(data, nodes)
add_dependent!(data, friendship)
add_covariate!(data, gender)
add_covariate!(data, performance)

println(data)  # SienaData(nodesets=1, dependents=1, covariates=2, waves=3)
```

## Step 2: Define Effects

Effects specify the social mechanisms you want to test. Start by creating an effects object from the data:

```julia
effects = get_effects(data)
println(effects)  # SienaEffects(N total, M included)
```

By default, only rate effects are included. Add structural effects by their short names:

```julia
# Basic structural effects
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])
```

### Viewing Available Effects

Use `effects_table` to see all available effects:

```julia
df = effects_table(effects)
println(df)
```

This shows each effect's name, short name, type, variable, and inclusion status.

### Adding Covariate Effects

If covariates are present, covariate effects are automatically available:

```julia
# Ego and alter effects for gender
include_effects!(effects, :friendship, [Symbol("egogender"), Symbol("altgender")])

# Homophily: same gender
include_effects!(effects, :friendship, [Symbol("samegender")])
```

### Common Effect Combinations

| Model Type | Effects to Include |
|------------|-------------------|
| **Baseline** | `:outdegree`, `:recip` |
| **Transitivity** | `:transTrip` or `:transTies` or `:gwesp` |
| **Popularity/Activity** | `:inPop`, `:inPopSqrt`, `:outAct`, `:outActSqrt` |
| **Homophily** | `:sameX`, `:simX`, `:egoX`, `:altX` |
| **Structural balance** | `:balance` |
| **Cycles** | `:cycle3` |

## Step 3: Configure Algorithm

The estimation algorithm has several tuning parameters:

```julia
algorithm = siena_algorithm(
    n_subphases = 4,           # Number of subphases in phase 2
    phase1_iterations = 50,    # Iterations for rough estimation
    phase3_iterations = 1000,  # Iterations for SE estimation
    initial_gain = 0.2,        # Starting gain parameter
    convergence_threshold = 0.25,  # Max t-ratio for convergence
    seed = 42,                 # Random seed for reproducibility
    verbose = true             # Print progress
)
```

### Default Settings

For most analyses, the defaults work well:

```julia
algorithm = siena_algorithm(seed=42)
```

### Key Parameters

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|----------------|
| `n_subphases` | Subphases in phase 2 | 4 | Increase to 6-8 for difficult models |
| `phase1_iterations` | Phase 1 iterations | 50 | Rarely needs changing |
| `phase3_iterations` | Phase 3 iterations | 1000 | Increase for more precise SEs |
| `initial_gain` | Starting gain | 0.2 | Decrease if estimation is unstable |
| `convergence_threshold` | Convergence criterion | 0.25 | Use 0.1 for strict convergence |
| `seed` | Random seed | nothing | Always set for reproducibility |

## Step 4: Estimate Model

Run the main estimation function:

```julia
result = siena07(data, effects; algorithm=algorithm)
```

During estimation, you will see output for each phase:

```text
Starting SAOM estimation
Number of parameters: 3
Target statistics computed

--- Phase 1 ---
  Iteration 10, max deviation: 12.34
  Iteration 20, max deviation: 5.67
  ...

--- Phase 2 ---
  Subphase 1
  Subphase 2
  ...

--- Phase 3 ---
  Iteration 100 / 1000
  ...

--- Results ---
Converged: true
Max t-ratio: 0.08
```

## Step 5: Interpret Results

### Viewing Results

```julia
println(result)
```

Output:

```text
SAOM Estimation Results
=======================
Converged: true
Iterations: 1250

Parameter Estimates:
--------------------
outdegree           -2.1234 (0.1456) *
recip                1.8901 (0.2345) *
transTrip            0.3456 (0.0789) *

Rate Parameters:
----------------
Rate friendship period 1:   5.4321
Rate friendship period 2:   6.7890
```

### Accessing Results Programmatically

```julia
# Coefficient vector
coef(result)

# Standard errors
stderror(result)

# Covariance matrix
vcov(result)

# Confidence intervals (95%)
ci = confint(result)

# Custom confidence level
ci90 = confint(result; level=0.90)
```

### Interpreting Coefficients

Coefficients represent the weight of each effect in the objective function:

| Coefficient | Interpretation |
|-------------|----------------|
| beta > 0 | Effect increases probability of tie formation / behavior increase |
| beta < 0 | Effect decreases probability of tie formation / behavior increase |
| beta = 0 | No effect |

**Example interpretations:**

- `outdegree = -2.1` -- Negative density effect indicates a general tendency against forming ties (networks are typically sparse)
- `recip = 1.9` -- Strong positive reciprocity: actors prefer to reciprocate ties
- `transTrip = 0.35` -- Positive transitivity: friends of friends tend to become friends

### Checking Convergence

Convergence is assessed via t-ratios (deviation / standard deviation):

```julia
if result.converged
    println("Model converged successfully")
    println("Max t-ratio: ", maximum(abs.(result.t_ratios)))
else
    println("WARNING: Model did not converge")
    println("t-ratios: ", result.t_ratios)
end
```

**Convergence guidelines:**

| Max |t-ratio| | Assessment |
|-----------------|------------|
| < 0.1 | Excellent convergence |
| 0.1 - 0.2 | Adequate convergence |
| 0.2 - 0.25 | Borderline -- consider re-running |
| > 0.25 | Not converged -- do not interpret results |

### Goodness of Fit

After estimation, assess whether the model reproduces key network features:

```julia
# GOF for indegree distribution
gof_in = siena_gof_indegree(result, data, :friendship; n_sims=100, seed=42)
println(gof_in)

# GOF for outdegree distribution
gof_out = siena_gof_outdegree(result, data, :friendship; n_sims=100, seed=42)

# GOF for triad census
gof_triad = siena_gof_triad(result, data, :friendship; n_sims=100, seed=42)
```

A high overall p-value (> 0.05) indicates the model fits well for that statistic.

## Complete Example

```julia
using Siena
using Random

Random.seed!(123)

# === Data Preparation ===
n = 30
data = siena_data()
add_nodeset!(data, NodeSet(n))

# Generate 3-wave network with clear structure
waves = Matrix{Int}[]
for w in 1:3
    net = zeros(Int, n, n)
    for i in 1:n, j in 1:n
        i != j && rand() < 0.08 + 0.02*w && (net[i,j] = 1)
    end
    push!(waves, net)
end
add_dependent!(data, DependentNetwork(:net, waves))

# Add a covariate
group = constant_covariate(:group, rand(1:3, n))
add_covariate!(data, group)

# === Effects ===
effects = get_effects(data)
include_effects!(effects, :net, [:outdegree, :recip, :transTrip])
include_effects!(effects, :net, [Symbol("samegroup")])

# === Estimation ===
alg = siena_algorithm(seed=42, phase3_iterations=500, verbose=false)
result = siena07(data, effects; algorithm=alg)

# === Results ===
println(result)
println("\nConfidence intervals:")
ci = confint(result)
for (i, entry) in enumerate(get_objective_effects(result.effects))
    if !entry.fix
        println("  $(entry.shortname): [$(round(ci[i,1], digits=3)), $(round(ci[i,2], digits=3))]")
    end
end

# === Goodness of Fit ===
gof = siena_gof_indegree(result, data, :net; n_sims=50, seed=42)
println("\n", gof)
```

## Working with Behavior Variables

SAOMs can jointly model network and behavior dynamics (co-evolution):

```julia
using Siena

n = 40
data = siena_data()
add_nodeset!(data, NodeSet(n))

# Network data
waves = [rand(0:1, n, n) for _ in 1:3]
for w in waves; w[diagind(w)] .= 0; end
add_dependent!(data, DependentNetwork(:friendship, waves))

# Behavior data (integer values, e.g., substance use on 1-5 scale)
beh_waves = [rand(1:5, n) for _ in 1:3]
add_dependent!(data, DependentBehavior(:drinking, beh_waves))

# Get effects for both variables
effects = get_effects(data)

# Network effects
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])

# Behavior effects: shape and influence
include_effects!(effects, :drinking, [:linear, :quad])
include_effects!(effects, :drinking, [Symbol("avAltfriendship")])

# Estimate
result = siena07(data, effects; algorithm=siena_algorithm(seed=42))
println(result)
```

### Interpreting Co-Evolution Results

In co-evolution models, you can distinguish:

- **Selection effects** (network part): Do actors form ties based on behavioral similarity?
  - Use `SimilarityEffect`, `SameEffect` in the network model
- **Influence effects** (behavior part): Does network position affect behavior change?
  - Use `AverageAlterEffect`, `TotalAlterEffect` in the behavior model

## Best Practices

1. **Start simple**: Begin with basic structural effects (outdegree, reciprocity) before adding complexity
2. **Check convergence**: Always verify `result.converged == true` with t-ratios < 0.1
3. **Set random seed**: Use `seed` in `siena_algorithm()` for reproducible results
4. **Re-run if needed**: If convergence is borderline, re-run with the previous estimates as starting values
5. **Assess GOF**: Test indegree, outdegree, and triad census distributions after estimation
6. **Iterative model building**: Add effects one at a time and check convergence after each addition
7. **Sufficient observations**: At least 20 actors and 2 waves; 3+ waves preferred
8. **Avoid redundant effects**: Do not include highly correlated effects simultaneously
9. **Center covariates**: Covariates are centered by default; this improves estimation stability
10. **Monitor rate parameters**: Rate parameters should be positive and reasonable (typically 2-20)

## Next Steps

- Learn about [Data Preparation](guide/data.md) for detailed data handling
- Explore all available [Effects](guide/effects.md)
- Understand [Model Estimation](guide/estimation.md) in depth
- Assess model fit with [Goodness of Fit](guide/gof.md) testing
