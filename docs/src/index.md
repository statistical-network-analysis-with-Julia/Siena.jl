# Siena.jl

*Stochastic Actor-Oriented Models for Julia*

A Julia package for statistical analysis of longitudinal network data using Stochastic Actor-Oriented Models (SAOM).

## Overview

Siena.jl is a port of [RSiena](https://github.com/stocnet/rsiena), providing tools for modeling the co-evolution of networks and behavior over time. The package implements the full SAOM framework including 150+ effects, three-phase stochastic approximation estimation, and goodness-of-fit assessment.

### What is a SAOM?

A Stochastic Actor-Oriented Model treats network change as a continuous-time Markov chain where individual actors make decisions about their outgoing ties and behavior. Between two observed network panels, the model assumes that:

1. Actors get opportunities to change ties or behavior at random moments (governed by a **rate function**)
2. When an actor gets an opportunity, they choose the change that maximizes their **objective function** (plus randomness)
3. Changes happen one at a time (micro-steps), building up to the observed macro-level change

This actor-centered perspective is natural for modeling social processes where individuals make choices about whom to befriend, how to behave, and how to respond to their social environment.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Panel Data** | Network and behavior observed at 2+ discrete time points (waves) |
| **Dependent Variable** | The network or behavior variable being modeled (changes between waves) |
| **Effect** | A function capturing a specific social mechanism (e.g., reciprocity, homophily) |
| **Rate Function** | Controls how often actors get opportunities to change |
| **Objective Function** | Determines which changes actors prefer, given current network state |
| **Method of Moments** | Estimation approach matching simulated to observed statistics |

### Applications

SAOMs are widely used in:

- **Social network dynamics**: Understanding how friendship networks evolve
- **Peer influence studies**: Modeling how behavior spreads through networks
- **Organizational networks**: Analyzing advice, collaboration, and knowledge-sharing ties
- **School networks**: Studying friendship formation and substance use among adolescents
- **Political networks**: Modeling alliance formation and policy adoption
- **Health behavior**: Understanding diffusion of health behaviors through social ties

## Features

- **150+ effects**: Comprehensive library of structural, covariate, behavior, rate, and two-mode effects
- **Co-evolution modeling**: Joint modeling of network and behavior dynamics
- **Three-phase estimation**: Robbins-Monro stochastic approximation with derivative estimation
- **Goodness of fit**: Assessment via indegree, outdegree, triad census, geodesic, and behavior distributions
- **RSiena-compatible API**: Familiar function names (`siena07`, `get_effects`, `include_effects!`)
- **Flexible data structures**: Support for one-mode, two-mode, and multivariate networks
- **Composition change**: Handle actors entering or leaving the network

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Statistical-network-analysis-with-Julia/Siena.jl")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/Siena.jl")
```

## Quick Start

```julia
using Siena

# Step 1: Create data
data = siena_data()
add_nodeset!(data, NodeSet(50))

# Network observed at 3 time points (waves)
wave1 = rand(0:1, 50, 50)
wave2 = rand(0:1, 50, 50)
wave3 = rand(0:1, 50, 50)
for w in [wave1, wave2, wave3]
    w[diagind(w)] .= 0  # No self-loops
end

add_dependent!(data, DependentNetwork(:friendship, [wave1, wave2, wave3]))

# Step 2: Define effects
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])

# Step 3: Configure algorithm and estimate
algorithm = siena_algorithm(seed=42, phase3_iterations=500)
result = siena07(data, effects; algorithm=algorithm)

# Step 4: Interpret results
println(result)
ci = confint(result)
```

## Choosing Effects

| Research Question | Recommended Effects |
|-------------------|--------------------|
| Basic network structure | [`OutdegreeEffect`](@ref), [`ReciprocityEffect`](@ref) |
| Transitivity / clustering | [`TransitiveTripletsEffect`](@ref), [`TransitiveTiesEffect`](@ref), [`GWESPEffect`](@ref) |
| Popularity / activity | [`IndegreePopularityEffect`](@ref), [`OutdegreeActivityEffect`](@ref) |
| Homophily (categorical) | [`SameEffect`](@ref), [`SameXRecipEffect`](@ref) |
| Homophily (continuous) | [`SimilarityEffect`](@ref), [`EgoEffect`](@ref), [`AlterEffect`](@ref) |
| Peer influence on behavior | [`AverageAlterEffect`](@ref), [`AverageSimilarityEffect`](@ref) |
| Behavior shape | [`LinearShapeEffect`](@ref), [`QuadraticShapeEffect`](@ref) |
| Two-mode networks | [`FourCyclesEffect`](@ref), [`TwoModeOutdegreeEffect`](@ref) |

## Documentation

```@contents
Pages = [
    "getting_started.md",
    "guide/data.md",
    "guide/effects.md",
    "guide/estimation.md",
    "guide/gof.md",
    "api/types.md",
    "api/effects.md",
    "api/estimation.md",
]
Depth = 2
```

## Theoretical Background

### The Continuous-Time Markov Chain Model

SAOMs model network evolution as a continuous-time Markov chain. Between observation moments $t_m$ and $t_{m+1}$, the network evolves through micro-steps. At each micro-step:

1. An actor $i$ is selected with probability proportional to the **rate function**:

$$\lambda_i(\theta, x) = \rho_m \cdot \prod_k \exp(\alpha_k v_{ik})$$

where $\rho_m$ is the basic rate parameter for period $m$, $\alpha_k$ are rate effect parameters, and $v_{ik}$ are actor-level covariates or network properties.

2. The selected actor chooses to change the tie to alter $j$ (or make no change) according to the **objective function**:

$$P(x_{ij} \to 1 - x_{ij}) = \frac{\exp(f_i(\beta, x^{(ij)}))}{\sum_{h} \exp(f_i(\beta, x^{(ih)}))}$$

where $f_i$ is the objective function:

$$f_i(\beta, x) = \sum_k \beta_k s_{ik}(x)$$

and $s_{ik}(x)$ are the effect statistics evaluated at the network configuration $x$.

### Method of Moments Estimation

Parameters are estimated by matching simulated network statistics to observed ones. The estimation proceeds in three phases:

1. **Phase 1** (rough estimation): Initial parameter updates using identity derivative matrix
2. **Phase 2** (refinement): Multiple subphases with decreasing gain and estimated derivative matrix $\hat{D} = \partial E[s] / \partial \theta$
3. **Phase 3** (final): Fixed parameters, collecting simulations for standard error computation via $\text{Var}(\hat\theta) \approx D^{-1} \Sigma D^{-\top}$

Convergence is assessed using t-ratios: the ratio of the deviation between observed and simulated statistics to the standard deviation of the simulated statistics. All t-ratios should be below 0.1 in absolute value for good convergence.

## References

1. Snijders, T.A.B. (2001). The statistical evaluation of social network dynamics. *Sociological Methodology*, 31(1), 361-395.

2. Snijders, T.A.B., van de Bunt, G.G., Steglich, C.E.G. (2010). Introduction to stochastic actor-based models for network dynamics. *Social Networks*, 32(1), 44-60.

3. Ripley, R.M., Snijders, T.A.B., Boda, Z., Voros, A., Preciado, P. (2022). Manual for RSiena. University of Oxford, Department of Statistics; Nuffield College.

4. Steglich, C., Snijders, T.A.B., Pearson, M. (2010). Dynamic networks and behavior: Separating selection from influence. *Sociological Methodology*, 40(1), 329-393.

5. Snijders, T.A.B. (2005). Models for longitudinal network data. In P.J. Carrington, J. Scott, S. Wasserman (Eds.), *Models and Methods in Social Network Analysis* (pp. 215-247). Cambridge University Press.
