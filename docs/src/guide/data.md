# Data Preparation

This guide covers how to prepare data for SAOM analysis in Siena.jl. The data layer mirrors the RSiena API with Julia-idiomatic types.

## Overview

A SAOM analysis requires:

1. A **node set** defining the actors
2. One or more **dependent variables** (networks and/or behaviors) observed at multiple waves
3. Optionally, **covariates** (actor-level or dyadic, constant or varying)

All data is assembled into a [`SienaData`](@ref) container.

## Node Sets

A [`NodeSet`](@ref) defines the set of actors in the analysis:

```julia
using Siena

# Basic: 50 actors with auto-generated names ("1", "2", ...)
nodes = NodeSet(50)

# With custom names
nodes = NodeSet(5; names=["Alice", "Bob", "Carol", "David", "Eve"])

# With a custom identifier (useful for two-mode networks)
students = NodeSet(30; id=:students)
teachers = NodeSet(10; id=:teachers)
```

### Node Set Properties

```julia
nodes = NodeSet(50)

length(nodes)   # 50
nodes.n         # 50
nodes.id        # :actors (default)
nodes.names     # ["1", "2", ..., "50"]
```

### Using the Convenience Constructor

```julia
nodes = siena_nodeset(50; id=:actors)
```

## Dependent Network Variables

A [`DependentNetwork`](@ref) represents a network observed at multiple time points. Each observation is an adjacency matrix where entry `(i, j) = 1` indicates a tie from actor `i` to actor `j`.

### Creating from Adjacency Matrices

```julia
n = 30

# Create adjacency matrices for 3 waves
wave1 = zeros(Int, n, n)
wave2 = zeros(Int, n, n)
wave3 = zeros(Int, n, n)

# Populate with observed data
# (In practice, load from files or DataFrames)
wave1[1, 2] = 1  # Actor 1 -> Actor 2 at wave 1
wave1[2, 1] = 1  # Actor 2 -> Actor 1 at wave 1
wave1[1, 3] = 1  # Actor 1 -> Actor 3 at wave 1

# Create the dependent network
friendship = DependentNetwork(:friendship, [wave1, wave2, wave3])
```

### Network Properties

```julia
n_waves(friendship)   # 3
n_actors(friendship)  # 30
friendship.name       # :friendship
friendship.directed   # true (default)
friendship.type       # :onemode (default)
```

### Network Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `Symbol` | Required | Variable name |
| `networks` | `Vector{Matrix{Int}}` | Required | Adjacency matrices |
| `type` | `Symbol` | `:onemode` | `:onemode`, `:twomode`, or `:bipartite` |
| `directed` | `Bool` | `true` | Whether the network is directed |
| `allow_self_loops` | `Bool` | `false` | Whether self-ties are allowed |
| `nodeset1` | `Symbol` | `:actors` | ID of the row node set |
| `nodeset2` | `Symbol` | `nothing` | ID of the column node set (bipartite) |

### Two-Mode Networks

For bipartite networks connecting two different types of nodes:

```julia
# 20 students connected to 10 clubs
n_students = 20
n_clubs = 10

# Affiliation matrices (student x club)
aff1 = rand(0:1, n_students, n_clubs)
aff2 = rand(0:1, n_students, n_clubs)

membership = DependentNetwork(:membership, [aff1, aff2];
    type=:twomode,
    nodeset1=:students,
    nodeset2=:clubs
)
```

### Using the Convenience Constructor

```julia
net = siena_dependent(:friendship, [wave1, wave2, wave3])
```

## Dependent Behavior Variables

A [`DependentBehavior`](@ref) represents an ordinal or integer-valued behavior observed across waves. Each observation is a vector of values for all actors.

### Creating Behavior Variables

```julia
n = 30

# Behavior values at each wave (e.g., substance use on 1-5 scale)
beh_wave1 = rand(1:5, n)
beh_wave2 = rand(1:5, n)
beh_wave3 = rand(1:5, n)

drinking = DependentBehavior(:drinking, [beh_wave1, beh_wave2, beh_wave3])
```

### Behavior Properties

```julia
n_waves(drinking)    # 3
n_actors(drinking)   # 30
drinking.name        # :drinking
drinking.min_val     # 1 (automatically detected)
drinking.max_val     # 5 (automatically detected)
```

### Specifying Range Explicitly

```julia
# Override auto-detected range
exercise = DependentBehavior(:exercise, [beh_wave1, beh_wave2, beh_wave3];
    min_val=0, max_val=7
)
```

### Using the Convenience Constructor

```julia
beh = siena_dependent(:drinking, [beh_wave1, beh_wave2, beh_wave3])
```

## Covariates

Covariates are predictor variables that are not modeled as dependent. They can be actor-level or dyadic, and constant or varying across waves.

### Constant Covariates

A [`ConstantCovariate`](@ref) has the same value for each actor across all waves:

```julia
# Gender (binary)
gender = constant_covariate(:gender, [0, 1, 0, 1, 1, 0, 0, 1, 1, 0])

# Age (continuous)
age = constant_covariate(:age, [15.2, 16.1, 15.8, 14.9, 16.3, 15.5, 14.7, 15.9, 16.0, 15.1])
```

By default, covariates are centered (mean-subtracted). To disable centering:

```julia
# Uncentered
gender_raw = ConstantCovariate(:gender_raw, [0, 1, 0, 1, 1]; center=false)
```

### Varying Covariates

A [`VaryingCovariate`](@ref) has different values at each wave:

```julia
# GPA changing over 3 waves
gpa = varying_covariate(:gpa, [
    [3.5, 3.2, 3.8, 3.1, 3.6],  # Wave 1
    [3.6, 3.1, 3.9, 3.3, 3.5],  # Wave 2
    [3.4, 3.3, 3.7, 3.2, 3.7],  # Wave 3
])
```

### Constant Dyadic Covariates

A [`ConstantDyadCovariate`](@ref) is a matrix of values for each pair of actors:

```julia
n = 10

# Geographical distance between actors
distances = rand(n, n)
distances = (distances + distances') / 2  # Symmetrize
geo = constant_dyad_covariate(:distance, distances)
```

### Varying Dyadic Covariates

A [`VaryingDyadCovariate`](@ref) changes across waves:

```julia
n = 10

# Shared class membership at each wave
shared1 = rand(0:1, n, n)
shared2 = rand(0:1, n, n)
shared3 = rand(0:1, n, n)

shared_class = varying_dyad_covariate(:shared_class, [shared1, shared2, shared3])
```

### Covariate Properties

| Type | Fields | Centering |
|------|--------|-----------|
| `ConstantCovariate` | `values::Vector{Float64}` | Mean-centered by default |
| `VaryingCovariate` | `values::Vector{Vector{Float64}}` | Overall mean-centered |
| `ConstantDyadCovariate` | `values::Matrix{Float64}` | Mean-centered by default |
| `VaryingDyadCovariate` | `values::Vector{Matrix{Float64}}` | Overall mean-centered |

All covariates store:
- `name::Symbol` -- Covariate identifier
- `centered::Bool` -- Whether centering was applied
- `mean::Float64` -- The mean used for centering

## Assembling the SienaData Object

The [`SienaData`](@ref) container holds all data components:

```julia
# Create empty container
data = siena_data()

# Add node set (required)
add_nodeset!(data, NodeSet(30))

# Add dependent variables (at least one required)
add_dependent!(data, friendship)
add_dependent!(data, drinking)  # Optional second dependent

# Add covariates (optional)
add_covariate!(data, gender)
add_covariate!(data, age)
add_covariate!(data, geo)

println(data)
# SienaData(nodesets=1, dependents=2, covariates=3, waves=3)
```

### Validation

The `SienaData` object validates consistency:

```julia
# All dependent variables must have the same number of waves
net_2wave = DependentNetwork(:net2, [wave1, wave2])        # 2 waves
beh_3wave = DependentBehavior(:beh3, [beh_wave1, beh_wave2, beh_wave3])  # 3 waves

data = siena_data()
add_nodeset!(data, NodeSet(30))
add_dependent!(data, net_2wave)
# add_dependent!(data, beh_3wave)  # ERROR: Number of waves must be consistent
```

### Accessing Data Components

```julia
data.nodesets        # Dict{Symbol, NodeSet}
data.dependents      # Dict{Symbol, AbstractDependent}
data.covariates      # Dict{Symbol, AbstractCovariate}
data.n_waves         # Number of observation waves
```

## Network State

The [`NetworkState`](@ref) represents the current state of networks and behaviors during simulation. It is primarily used internally but can be useful for inspection:

```julia
state = NetworkState()
initialize!(state, data, 1)  # Initialize from wave 1

# Access current network
net = state.networks[:friendship]  # Matrix{Int}

# Access current behavior
beh = state.behaviors[:drinking]   # Vector{Int}

# Current simulation time
state.time  # Float64
```

### Initializing from Different Waves

```julia
# Start from wave 1 (for forward simulation)
initialize!(state, data, 1)

# Start from wave 2 (for period 2 simulation)
initialize!(state, data, 2)
```

## Composition Change

If actors join or leave the network between waves, use [`CompositionChange`](@ref):

```julia
cc = CompositionChange()

# Actor 5 leaves after wave 1
add_change!(cc, 5, 1, :leave)

# Actor 31 joins at wave 2
add_change!(cc, 31, 2, :join)

# Attach to data
data.composition_change = cc
```

## Data Validation Tips

### Network Data

1. **Square matrices**: One-mode networks must have square adjacency matrices
2. **No self-loops**: Diagonal entries should be 0 (unless `allow_self_loops=true`)
3. **Binary values**: Entries should be 0 or 1
4. **Consistent size**: All wave matrices must have the same dimensions

```julia
# Validate network data
for (w, net) in enumerate(waves)
    @assert size(net, 1) == size(net, 2) "Wave $w: matrix not square"
    @assert all(net[diagind(net)] .== 0) "Wave $w: self-loops present"
    @assert all(x -> x in (0, 1), net) "Wave $w: non-binary values"
end
```

### Behavior Data

1. **Integer values**: Behavior values must be integers
2. **Consistent range**: Values should span a reasonable range (typically 1-5 or 1-7)
3. **Sufficient variation**: At least some change between waves is needed

```julia
# Check behavior variation
for w in 2:length(beh_waves)
    changes = sum(beh_waves[w] .!= beh_waves[w-1])
    println("Wave $w: $changes actors changed behavior")
end
```

### Covariate Data

1. **Matching dimensions**: Covariate vector length must equal number of actors
2. **No excessive missing data**: NaN values may cause issues
3. **Centering**: Covariates are centered by default for numerical stability

```julia
# Check for NaN values
@assert !any(isnan, gender.values) "Covariate has NaN values"

# Verify dimensions match
@assert length(gender.values) == n_actors(friendship)
```

## Complete Data Setup Example

```julia
using Siena
using Random

Random.seed!(42)

# Parameters
n = 40
n_waves_total = 3

# === Generate synthetic data ===

# Network waves with some structure
function generate_network(n, density)
    net = zeros(Int, n, n)
    for i in 1:n, j in 1:n
        if i != j && rand() < density
            net[i, j] = 1
        end
    end
    return net
end

waves = [generate_network(n, 0.1) for _ in 1:n_waves_total]

# Behavior waves (ordinal 1-5)
beh_waves = [rand(1:5, n) for _ in 1:n_waves_total]

# Covariates
gender_vals = rand(0:1, n)
age_vals = randn(n) .* 2 .+ 16

# === Assemble ===
data = siena_data()
add_nodeset!(data, NodeSet(n))
add_dependent!(data, DependentNetwork(:net, waves))
add_dependent!(data, DependentBehavior(:beh, beh_waves))
add_covariate!(data, constant_covariate(:gender, gender_vals))
add_covariate!(data, constant_covariate(:age, age_vals))

println(data)
println("Network density wave 1: ", sum(waves[1]) / (n * (n-1)))
println("Behavior range: ", minimum(vcat(beh_waves...)), "-", maximum(vcat(beh_waves...)))
```
