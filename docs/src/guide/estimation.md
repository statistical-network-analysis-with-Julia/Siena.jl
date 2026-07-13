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
- Re-estimates the derivative matrix ($D$), by default with the
  score-function estimator (Schweinberger–Snijders), which accumulates
  trajectory scores over all phase-3 simulations
- Computes standard errors via $\text{Var}(\hat\theta) \approx D^{-1} \Sigma D^{-\top}$
- Purpose: compute standard errors and assess convergence

Phase 2 additionally applies Polyak–Ruppert averaging over each subphase, so
the estimates carry substantially less Monte-Carlo noise at the same
simulation budget. Phase-3 simulations run on threads with pre-drawn
per-simulation seeds, so results are bitwise identical regardless of
`JULIA_NUM_THREADS`.

## Configuring the Algorithm

The examples on this page share a small synthetic three-wave data set:

```julia
using Siena, Random

rng = Xoshiro(42)
n = 15
w1 = zeros(Int, n, n)
for i in 1:n, j in 1:n
    i != j && rand(rng) < 0.1 && (w1[i, j] = 1)
end
w2 = copy(w1); w3 = copy(w2)
for w in (w2, w3), _ in 1:20
    i, j = rand(rng, 1:n), rand(rng, 1:n)
    i == j && continue
    w[i, j] = 1 - w[i, j]
end

data = siena_data()
add_nodeset!(data, NodeSet(n))
add_dependent!(data, DependentNetwork(:net, [w1, w2, w3]))
add_covariate!(data, ConstantCovariate(:gender, Float64.(rand(rng, 0:1, n))))

effects = get_effects(data)
include_effects!(effects, :net, [:outdegree, :recip])
```

### The SienaAlgorithm Object

Use [`siena_algorithm`](@ref) to create a configuration:

```julia
algorithm = siena_algorithm(
    n_subphases = 4,
    phase1_iterations = 50,
    phase3_iterations = 1000,
    initial_gain = 0.2,
    min_gain = 0.0005,
    max_iterations = nothing,
    convergence_threshold = 0.1,
    seed = 42,
    model_type = :standard,
    conditional = false,
    n_simulations = 1,
    parallel = true,
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
| `max_iterations` | `nothing` | Budget on the total number of phase-1/phase-2 iterations (`nothing`: no budget) |
| `convergence_threshold` | 0.1 | Maximum per-parameter t-ratio for convergence |
| `overall_convergence_threshold` | 0.25 | Maximum overall convergence ratio (`tconv.max`) |
| `seed` | nothing | Random seed for reproducibility |
| `model_type` | `:standard` | Which dependent variables co-evolve: `:standard`, `:networkonly` or `:behavioronly` |
| `conditional` | false | Use conditional estimation |
| `condvar` | nothing | Conditioning variable for conditional estimation |
| `n_simulations` | 1 | Simulations averaged into each Robbins-Monro iteration |
| `parallel` | true | Run the independent simulations (phase 3, derivatives) multi-threaded |
| `derivative_method` | `:score` | `:score` (score-function) or `:finite_difference` |
| `derivative_sims` | 30 | Simulations per finite-difference derivative estimate |
| `verbose` | true | Print progress during estimation |

### Model Types

`model_type` selects which dependent variables take ministeps. The variables it
leaves out are *frozen*: they stay in the simulation state at their period-start
values and are still read by the effects of the simulated variables (a network
effect may depend on a frozen behavior, and vice versa), but they never change.
Because a frozen variable's moments are then constant, its own rate and objective
effects are not identified and are dropped from the estimated parameter vector.

| Type | Simulated | Frozen | Use When |
|------|-----------|--------|----------|
| `:standard` | every dependent variable | — | Co-evolution analysis |
| `:networkonly` | the network variables | the behavior variables | Behavior is exogenous |
| `:behavioronly` | the behavior variables | the networks | Network is exogenous |

```julia
# Network dynamics with the behavior held fixed: only :net effects are estimated,
# but network effects may still read :beh.
algorithm = siena_algorithm(model_type = :networkonly)
```

Use [`simulated_variables`](@ref) to see which variables a `model_type` simulates
for a given data set. `condvar` (conditional estimation) must be one of them.

!!! warning "Not RSiena's `modelType`"
    RSiena's `modelType` selects the *kind of network model* driving a ministep
    (1 standard actor-oriented, 2 forcing, 3 initiative, ...), not a subset of the
    dependent variables. Those forcing/initiative model types are **not implemented**
    in Siena.jl — every ministep is the standard actor-oriented one — so a Siena.jl
    `model_type` value never corresponds to an RSiena `modelType` value.

### Tuning for Difficult Models

For models that fail to converge with default settings:

```julia
# More subphases and iterations
algorithm = siena_algorithm(
    n_subphases = 8,           # More refinement
    phase1_iterations = 100,    # Longer warm-up
    phase3_iterations = 2000,   # More precise SEs
    initial_gain = 0.1,         # Smaller steps (more stable)
    seed = 42
)
```

## Running the Estimation

### Basic Usage

[`fit_siena`](@ref) is the primary estimation entry point;
[`siena07`](@ref) is kept as the RSiena-faithful alias and accepts the same
arguments:

```julia
result = fit_siena(data, effects)
result = siena07(data, effects)    # identical
```

### With Custom Algorithm

```julia
algorithm = siena_algorithm(seed=42, verbose=true)
result = fit_siena(data, effects; algorithm=algorithm)
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
Max |t-ratio|: 0.087
Overall max convergence ratio: 0.152
```

## Understanding Results

### The SienaResult Object

The [`SienaResult`](@ref) contains:

| Field | Type | Description |
|-------|------|-------------|
| `effects` | `SienaEffects` | The effects specification |
| `parameter_names` | `Vector{String}` | Labels for the estimate vector |
| `estimates` | `Vector{Float64}` | Parameter estimates |
| `standard_errors` | `Vector{Float64}` | Standard errors |
| `t_ratios` | `Vector{Float64}` | Convergence t-ratios |
| `covariance` | `Matrix{Float64}` | Parameter covariance matrix |
| `converged` | `Bool` | Whether estimation converged |
| `tconv_max` | `Float64` | Overall maximum convergence ratio (RSiena's `tconv.max`) |
| `diverged` | `Bool` | Whether any estimate hit the parameter clamp |
| `n_iterations` | `Int` | Total iterations used |
| `rate_estimates` | `Dict{Symbol, Vector{Float64}}` | Rate estimates per period |
| `targets` | `Vector{Float64}` | Observed target statistics |
| `simulated_means` | `Vector{Float64}` | Mean phase-3 simulated statistics |

### Displaying Results

```julia
println(result)
```

Output:

```text
SAOM Estimation Results
=======================
Converged: true (max |t-ratio| = 0.093, overall max convergence ratio = 0.184)
Iterations: 1250

Rate Parameters:
----------------
Rate friendship (period 1)     5.4321 (0.9163)
Rate friendship (period 2)     6.7890 (1.0963)

Objective Function Parameters:
------------------------------
outdegree                     -2.4567 (0.1234) *
recip                          1.8901 (0.2345) *
transTrip                      0.3456 (0.0789) *
samegender                     0.2134 (0.0956) *
simage                         0.1567 (0.0678) *
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

Parameters are tested using their t-statistic (estimate / standard error). The
estimate vector contains the free rate parameters first, then the objective
parameters; `parameter_names` gives the matching labels:

```julia
estimates = coef(result)
ses = stderror(result)

for (name, est, se) in zip(result.parameter_names, estimates, ses)
    t_stat = est / se
    sig = abs(t_stat) > 1.96 ? "significant" : "not significant"
    println("$name: $(round(est, digits=3)) ",
            "(SE=$(round(se, digits=3)), t=$(round(t_stat, digits=2))) - $sig")
end
```

## Convergence Assessment

### What Are t-Ratios?

Convergence t-ratios measure how well the model reproduces observed statistics. They are computed as:

$$t_k = \frac{\bar{s}_k^{sim} - s_k^{obs}}{\text{sd}(s_k^{sim})}$$

where $\bar{s}_k^{sim}$ is the mean simulated statistic, $s_k^{obs}$ is the observed statistic, and $\text{sd}(s_k^{sim})$ is the standard deviation of simulated statistics.

In addition to the per-parameter t-ratios, Siena.jl computes the **overall
maximum convergence ratio** (RSiena's `tconv.max`),
$\sqrt{\bar{e}^\top \Sigma^{-1} \bar{e}}$ for the mean deviation vector
$\bar{e}$ — the largest t-ratio over all linear combinations of the
deviations. It is reported as `result.tconv_max` and printed with the result.

### Convergence Criteria

Following RSiena's published standard, `result.converged` is `true` only when
**every** per-parameter |t-ratio| is below `convergence_threshold` (default
0.1) *and* `tconv_max` is below `overall_convergence_threshold` (default
0.25).

| Max |t-ratio| | Assessment | Action |
|-----------------|------------|--------|
| < 0.1 | Converged | Proceed with interpretation (if `tconv_max` < 0.25) |
| 0.1 - 0.25 | Close | Re-run using the current estimates as starting values |
| > 0.25 | Not converged | Do not interpret; re-run with different settings |

To reproduce the laxer pre-0.2 gate (single max |t| < 0.25), pass
`siena_algorithm(convergence_threshold=0.25)` and ignore `tconv_max`.

### Checking Convergence

```julia
if result.converged
    max_t = maximum(abs.(result.t_ratios))
    println("Converged with max t-ratio: $(round(max_t, digits=3))")
    println("tconv.max: $(round(result.tconv_max, digits=3))")
else
    println("NOT CONVERGED")
    println("t-ratios: ", round.(result.t_ratios, digits=3))
end

# Divergence (estimates clamped at the parameter bounds) is flagged separately
result.diverged && println("WARNING: divergence detected")
```

### Improving Convergence

If the model does not converge:

1. **Simplify the model**: Remove effects with large standard errors

```julia
# Start with minimal model
include_effects!(effects, :net, [:outdegree, :recip])
result1 = fit_siena(data, effects; algorithm=algorithm)

# Add effects one at a time
include_effects!(effects, :net, [:transTrip])
result2 = fit_siena(data, effects; algorithm=algorithm)
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
# The estimate vector must come from a run with the SAME effects object:
# coef(...) is aligned with the free entries of the parameter map
# (rates first, then objective effects)
result_a = fit_siena(data, effects; algorithm=algorithm)
prev_estimates = coef(result_a)

pm = build_param_map(effects)
for (i, entry) in enumerate(pm.free)
    entry.initial_value = prev_estimates[i]
end

# Re-run, starting from the previous estimates
result_b = fit_siena(data, effects; algorithm=algorithm)
```

4. **Decrease initial gain**:

```julia
algorithm = siena_algorithm(initial_gain=0.05, seed=42)
```

## Confidence Intervals

### Standard Confidence Intervals

```julia
# 95% confidence intervals (rows aligned with result.parameter_names)
ci = confint(result)

for (i, name) in enumerate(result.parameter_names)
    println("$name: [$(round(ci[i,1], digits=3)), $(round(ci[i,2], digits=3))]")
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

If a covariate perfectly predicts tie presence/absence, the coefficient
diverges. Objective parameters are clamped at ±10 (rates at [0.05, 1000]);
hitting the clamp sets `result.diverged = true` and emits a warning:

```julia
# Divergence is surfaced explicitly
result.diverged && println("WARNING: estimates hit the parameter clamp")

# Check for extreme estimates
for (name, est) in zip(result.parameter_names, coef(result))
    if abs(est) >= 8
        println("WARNING: Possibly separated effect: $name")
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
result1 = fit_siena(data, effects; algorithm=siena_algorithm(seed=42))

# Step 2: Add transitivity (only if step 1 converges)
if result1.converged
    include_effects!(effects, :net, [:transTrip])
    result2 = fit_siena(data, effects; algorithm=siena_algorithm(seed=42))
end

# Step 3: Add covariates (only if step 2 converges)
if result2.converged
    include_effects!(effects, :net, [Symbol("samegender")])
    result3 = fit_siena(data, effects; algorithm=siena_algorithm(seed=42))
end
```

## Advanced Topics

### Conditional Estimation

Conditional estimation (RSiena's `cond=TRUE`) conditions on the observed
amount of change of one dependent variable: every simulated period runs
until that variable's distance from the period-start observation reaches
the observed distance, instead of until time 1 (Snijders 2001):

```julia
algorithm = siena_algorithm(conditional=true, seed=42)
result = fit_siena(data, effects; algorithm=algorithm)
```

The conditioned variable's basic rate parameters leave the moment
equations (they are determined by the conditioning) and are estimated
afterwards from the phase-3 stopping times; they are reported in
`result.rate_estimates`. With several dependent variables, name the
conditioning variable via `condvar` (RSiena's `condvarno`):

<!-- skip-check -->
```julia
algorithm = siena_algorithm(conditional=true, condvar=:friendship)
```

Conditional estimation always uses the finite-difference derivative
estimator (the trajectory score function assumes time-1 termination).

### The Derivative Matrix

The derivative matrix $D$ captures how simulated statistics change with
parameters. By default (`derivative_method=:score`) it is estimated with the
score-function estimator (Schweinberger & Snijders):

$$\hat{D} = \widehat{\text{cov}}(s, J)$$

where $J$ is the trajectory score function, accumulated over all phase-3
simulations by a [`ScoreAccumulator`](@ref) — no extra simulations are
needed, and the estimate is far less noisy than finite differences.

With `derivative_method=:finite_difference` (and always under conditional
estimation, whose stopping rule invalidates the trajectory score), $D$ is
instead estimated from `derivative_sims` simulations with common random
numbers:

$$D_{kl} \approx \frac{E[s_k(\theta + \epsilon e_l)] - E[s_k(\theta)]}{\epsilon}$$

A well-conditioned $D$ matrix is essential for stable estimation. If $D$ is nearly singular, the algorithm adds regularization.

### Simulation-Based Inference

You can use the estimated model to simulate networks:

```julia
# Simulate from the estimated model
state, results = simulate_saom(data, result.effects, coef(result); seed=42)

# Inspect the simulated final network
sim_network = state.networks[:net]
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
