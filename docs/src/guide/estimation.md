# Model Estimation

Siena.jl estimates SAOM parameters using the Method of Moments with Robbins-Monro stochastic approximation. This guide covers the estimation algorithm, configuration, and interpretation of results.

## Overview

The estimation process matches simulated network statistics to observed ones:

1. **Simulate** the SAOM forward from wave 1 using current parameter values
2. **Compare** simulated statistics to observed statistics at the final wave
3. **Update** parameters to reduce the discrepancy
4. **Repeat** until convergence

This is fundamentally different from maximum likelihood estimation: rather than optimizing a likelihood function directly, the Method of Moments finds parameters such that the expected value of the simulated statistics equals the observed statistics.

## The Three Phases

Estimation proceeds in three distinct phases:

### Phase 1: Initial Rough Estimation

- Uses an identity matrix as the derivative approximation
- Applies decreasing gain sequence: $a_k = a_0 / k$
- Purpose: get parameters into the right ballpark
- Typically 50 iterations

### Phase 2: Refinement with Subphases

- Estimates the derivative matrix $D = \partial E[s(\theta)] / \partial \theta$
- Uses Newton-Raphson-style updates: $\theta_{k+1} = \theta_k - a_k D^{-1} (s_{sim} - s_{obs})$
- Multiple subphases with decreasing gain
- Purpose: refine parameter estimates
- Typically 4 subphases of 25 iterations each

### Phase 3: Final Estimation

- Parameters are held fixed
- Collects a large number of simulations (typically 1000)
- Computes the covariance of simulated statistics ($\Sigma$)
- Re-estimates the derivative matrix ($D$)
- Computes standard errors via $\text{Var}(\hat\theta) \approx D^{-1} \Sigma D^{-\top}$
- Purpose: compute standard errors and assess convergence

## Configuring the Algorithm

### The SienaAlgorithm Object

Use [`siena_algorithm`](@ref) to create a configuration:

```julia
algorithm = siena_algorithm(
    n_subphases = 4,
    phase1_iterations = 50,
    phase3_iterations = 1000,
    initial_gain = 0.2,
    min_gain = 0.0005,
    max_iterations = 50,
    convergence_threshold = 0.25,
    seed = 42,
    model_type = :standard,
    conditional = false,
    n_simulations = 1,
    verbose = true
)
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_subphases` | 4 | Number of subphases in phase 2 |
| `phase1_iterations` | 50 | Number of iterations in phase 1 |
| `phase3_iterations` | 1000 | Number of simulations in phase 3 |
| `initial_gain` | 0.2 | Starting gain parameter $a_0$ |
| `min_gain` | 0.0005 | Minimum gain value |
| `max_iterations` | 50 | Maximum iterations per phase/subphase |
| `convergence_threshold` | 0.25 | Maximum t-ratio for convergence |
| `seed` | nothing | Random seed for reproducibility |
| `model_type` | `:standard` | `:standard`, `:behavioronly`, or `:networkonly` |
| `conditional` | false | Use conditional estimation |
| `n_simulations` | 1 | Simulations per iteration |
| `verbose` | true | Print progress during estimation |

### Model Types

| Type | Description | Use When |
|------|-------------|----------|
| `:standard` | Joint network and behavior model | Co-evolution analysis |
| `:networkonly` | Network model only | No behavior variables |
| `:behavioronly` | Behavior model only | Network is exogenous |

### Tuning for Difficult Models

For models that fail to converge with default settings:

```julia
# More subphases and iterations
algorithm = siena_algorithm(
    n_subphases = 8,           # More refinement
    phase1_iterations = 100,    # Longer warm-up
    phase3_iterations = 2000,   # More precise SEs
    initial_gain = 0.1,         # Smaller steps (more stable)
    convergence_threshold = 0.1, # Stricter convergence
    seed = 42
)
```

## Running the Estimation

### Basic Usage

```julia
result = siena07(data, effects)
```

### With Custom Algorithm

```julia
algorithm = siena_algorithm(seed=42, verbose=true)
result = siena07(data, effects; algorithm=algorithm)
```

### Monitoring Progress

With `verbose=true`, you see output for each phase:

```text
Starting SAOM estimation
Number of parameters: 5
Target statistics computed

--- Phase 1 ---
  Iteration 10, max deviation: 45.23
  Iteration 20, max deviation: 12.67
  Iteration 30, max deviation: 5.89
  Iteration 40, max deviation: 3.12
  Iteration 50, max deviation: 1.45

--- Phase 2 ---
  Subphase 1
  Subphase 2
  Subphase 3
  Subphase 4

--- Phase 3 ---
  Iteration 100 / 1000
  Iteration 200 / 1000
  ...
  Iteration 1000 / 1000

--- Results ---
Converged: true
Max t-ratio: 0.087
```

## Understanding Results

### The SienaResult Object

The [`SienaResult`](@ref) contains:

| Field | Type | Description |
|-------|------|-------------|
| `effects` | `SienaEffects` | The effects specification |
| `estimates` | `Vector{Float64}` | Parameter estimates |
| `standard_errors` | `Vector{Float64}` | Standard errors |
| `t_ratios` | `Vector{Float64}` | Convergence t-ratios |
| `covariance` | `Matrix{Float64}` | Parameter covariance matrix |
| `converged` | `Bool` | Whether estimation converged |
| `n_iterations` | `Int` | Total iterations used |
| `rate_estimates` | `Dict{Symbol, Vector{Float64}}` | Rate estimates per period |

### Displaying Results

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
outdegree           -2.4567 (0.1234) *
recip                1.8901 (0.2345) *
transTrip            0.3456 (0.0789) *
samegender           0.2134 (0.0956) *
simage               0.1567 (0.0678) *

Rate Parameters:
----------------
Rate friendship period 1:   5.4321
Rate friendship period 2:   6.7890
```

### Accessor Functions

```julia
# Parameter estimates
coef(result)        # Vector{Float64}

# Standard errors
stderror(result)    # Vector{Float64}

# Covariance matrix
vcov(result)        # Matrix{Float64}

# Confidence intervals (default 95%)
confint(result)                # Matrix with lower and upper columns
confint(result; level=0.99)    # 99% confidence intervals
```

### Interpreting Coefficients

SAOM coefficients represent weights in the objective function. They are not directly comparable to regression coefficients, but their sign and significance are interpretable:

| Sign | Meaning |
|------|---------|
| Positive | Effect increases probability of the change |
| Negative | Effect decreases probability of the change |
| Not significant | No evidence for this mechanism |

**Example interpretations for network effects:**

- `outdegree = -2.5` -- Strong negative density: actors avoid forming ties (network is sparse)
- `recip = 1.9` -- Strong reciprocity: actors strongly prefer mutual ties
- `transTrip = 0.35` -- Moderate transitivity: friends of friends tend to become friends
- `samegender = 0.21` -- Gender homophily: actors prefer same-gender ties

**Example interpretations for behavior effects:**

- `linear = 0.3` -- General tendency toward higher behavior values
- `quad = -0.5` -- Negative quadratic: actors avoid extreme values
- `avAlt = 1.2` -- Strong peer influence: actors adopt alters' average behavior

### Significance Testing

Parameters are tested using their t-statistic (estimate / standard error):

```julia
estimates = coef(result)
ses = stderror(result)

for (i, entry) in enumerate(get_objective_effects(result.effects))
    if !entry.fix
        t_stat = estimates[i] / ses[i]
        sig = abs(t_stat) > 1.96 ? "significant" : "not significant"
        println("$(entry.shortname): $(round(estimates[i], digits=3)) ",
                "(SE=$(round(ses[i], digits=3)), t=$(round(t_stat, digits=2))) - $sig")
    end
end
```

## Convergence Assessment

### What Are t-Ratios?

Convergence t-ratios measure how well the model reproduces observed statistics. They are computed as:

$$t_k = \frac{\bar{s}_k^{sim} - s_k^{obs}}{\text{sd}(s_k^{sim})}$$

where $\bar{s}_k^{sim}$ is the mean simulated statistic, $s_k^{obs}$ is the observed statistic, and $\text{sd}(s_k^{sim})$ is the standard deviation of simulated statistics.

### Convergence Criteria

| Max |t-ratio| | Assessment | Action |
|-----------------|------------|--------|
| < 0.1 | Excellent | Proceed with interpretation |
| 0.1 - 0.2 | Good | Results are reliable |
| 0.2 - 0.25 | Adequate | Consider re-running |
| > 0.25 | Not converged | Do not interpret; re-run with different settings |

### Checking Convergence

```julia
if result.converged
    max_t = maximum(abs.(result.t_ratios))
    println("Converged with max t-ratio: $(round(max_t, digits=3))")
else
    println("NOT CONVERGED")
    println("t-ratios: ", round.(result.t_ratios, digits=3))
end
```

### Improving Convergence

If the model does not converge:

1. **Simplify the model**: Remove effects with large standard errors

```julia
# Start with minimal model
include_effects!(effects, :net, [:outdegree, :recip])
result1 = siena07(data, effects; algorithm=algorithm)

# Add effects one at a time
include_effects!(effects, :net, [:transTrip])
result2 = siena07(data, effects; algorithm=algorithm)
```

2. **Increase iterations**:

```julia
algorithm = siena_algorithm(
    n_subphases = 8,
    phase3_iterations = 2000,
    seed = 42
)
```

3. **Use previous estimates as starting values**:

```julia
# After first run
prev_estimates = coef(result)

# Set initial values for next run
obj_effects = get_objective_effects(effects)
for (i, entry) in enumerate(obj_effects)
    if !entry.fix && i <= length(prev_estimates)
        entry.initial_value = prev_estimates[i]
    end
end

# Re-run
result2 = siena07(data, effects; algorithm=algorithm)
```

4. **Decrease initial gain**:

```julia
algorithm = siena_algorithm(initial_gain=0.05, seed=42)
```

## Confidence Intervals

### Standard Confidence Intervals

```julia
# 95% confidence intervals
ci = confint(result)

obj_effects = get_objective_effects(result.effects)
for (i, entry) in enumerate(obj_effects)
    if !entry.fix
        println("$(entry.shortname): [$(round(ci[i,1], digits=3)), $(round(ci[i,2], digits=3))]")
    end
end
```

### Custom Confidence Levels

```julia
# 99% confidence intervals
ci99 = confint(result; level=0.99)

# 90% confidence intervals
ci90 = confint(result; level=0.90)
```

## Common Issues and Solutions

### Issue: Very Large or Very Small Rate Parameters

Rate parameters should typically be between 1 and 30. Values outside this range suggest problems:

| Rate Value | Meaning | Solution |
|-----------|---------|----------|
| < 1 | Very few changes expected | Check data: is there enough change between waves? |
| 1 - 30 | Normal range | No action needed |
| > 30 | Very many changes expected | Check data: are waves too far apart? |

### Issue: Perfect Prediction (Separation)

If a covariate perfectly predicts tie presence/absence, the coefficient diverges:

```julia
# Check for extreme estimates
for (i, entry) in enumerate(get_objective_effects(result.effects))
    if !entry.fix && abs(coef(result)[i]) > 10
        println("WARNING: Possibly separated effect: $(entry.shortname)")
    end
end
```

**Solution**: Remove the problematic effect or recode the covariate.

### Issue: Multicollinearity

Correlated effects can inflate standard errors and cause instability:

```julia
# Check correlation of estimates
C = vcov(result)
n = size(C, 1)
for i in 1:n, j in (i+1):n
    corr = C[i,j] / sqrt(C[i,i] * C[j,j])
    if abs(corr) > 0.8
        println("WARNING: High correlation between parameters $i and $j: $(round(corr, digits=2))")
    end
end
```

**Solution**: Remove one of the correlated effects.

### Issue: Non-Convergence Due to Model Complexity

Complex models with many effects are harder to estimate:

**Solution**: Use stepwise model building:

```julia
# Step 1: Basic model
include_effects!(effects, :net, [:outdegree, :recip])
result1 = siena07(data, effects; algorithm=siena_algorithm(seed=42))

# Step 2: Add transitivity (only if step 1 converges)
if result1.converged
    include_effects!(effects, :net, [:transTrip])
    result2 = siena07(data, effects; algorithm=siena_algorithm(seed=42))
end

# Step 3: Add covariates (only if step 2 converges)
if result2.converged
    include_effects!(effects, :net, [Symbol("samegender")])
    result3 = siena07(data, effects; algorithm=siena_algorithm(seed=42))
end
```

## Advanced Topics

### Conditional Estimation

Conditional estimation fixes the number of network changes to the observed value:

```julia
algorithm = siena_algorithm(conditional=true, seed=42)
result = siena07(data, effects; algorithm=algorithm)
```

This is sometimes useful for small networks but is generally less recommended than unconditional estimation.

### The Derivative Matrix

The derivative matrix $D$ captures how simulated statistics change with parameters. It is estimated via finite differences:

$$D_{kl} \approx \frac{E[s_k(\theta + \epsilon e_l)] - E[s_k(\theta)]}{\epsilon}$$

A well-conditioned $D$ matrix is essential for stable estimation. If $D$ is nearly singular, the algorithm adds regularization.

### Simulation-Based Inference

You can use the estimated model to simulate networks:

```julia
# Simulate from the estimated model
state, results = simulate_saom(data, result.effects, coef(result); seed=42)

# Inspect the simulated final network
sim_network = state.networks[:friendship]
println("Simulated density: ", sum(sim_network) / (size(sim_network, 1) * (size(sim_network, 1) - 1)))
```

## Best Practices

1. **Always set a random seed** for reproducible results
2. **Check convergence** before interpreting parameters (all t-ratios < 0.1)
3. **Use stepwise model building** -- add effects one at a time
4. **Include outdegree** (density) as a baseline network effect
5. **Include linear shape** as a baseline behavior effect
6. **Monitor rate parameters** -- they should be positive and reasonable
7. **Re-run non-converged models** with previous estimates as starting values
8. **Increase phase 3 iterations** for publication-quality standard errors (2000+)
9. **Compare across seeds** to verify stability of results
10. **Report convergence t-ratios** alongside parameter estimates in publications
