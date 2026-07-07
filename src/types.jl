"""
Core types for Stochastic Actor-Oriented Models (SAOM).
"""

#==============================================================================#
# Node Sets
#==============================================================================#

"""
    NodeSet

Represents a set of actors/nodes in the network.

# Fields
- `n::Int`: Number of nodes
- `names::Vector{String}`: Optional node names
- `id::Symbol`: Identifier for the node set
"""
struct NodeSet
    n::Int
    names::Vector{String}
    id::Symbol

    function NodeSet(n::Int; names::Vector{String}=String[], id::Symbol=:actors)
        if !isempty(names) && length(names) != n
            throw(ArgumentError("Length of names must match n"))
        end
        new(n, isempty(names) ? ["$i" for i in 1:n] : names, id)
    end
end

Base.length(ns::NodeSet) = ns.n
Base.show(io::IO, ns::NodeSet) = print(io, "NodeSet(:$(ns.id), n=$(ns.n))")

#==============================================================================#
# Dependent Variables
#==============================================================================#

"""
    AbstractDependent

Abstract type for dependent variables in SAOM.
"""
abstract type AbstractDependent end

"""
    DependentNetwork

A dependent network variable observed at multiple time points.

# Fields
- `name::Symbol`: Variable name
- `networks::Vector{Matrix{Int}}`: Network adjacency matrices at each observation
- `type::Symbol`: Network type (:onemode, :twomode, :bipartite)
- `directed::Bool`: Whether the network is directed
- `allow_self_loops::Bool`: Whether self-loops are allowed
- `nodeset1::Symbol`: ID of the first node set
- `nodeset2::Union{Symbol, Nothing}`: ID of the second node set (for bipartite)
"""
mutable struct DependentNetwork <: AbstractDependent
    name::Symbol
    networks::Vector{Matrix{Int}}
    type::Symbol
    directed::Bool
    allow_self_loops::Bool
    nodeset1::Symbol
    nodeset2::Union{Symbol, Nothing}

    function DependentNetwork(
        name::Symbol,
        networks::Vector{<:AbstractMatrix{<:Integer}};
        type::Symbol=:onemode,
        directed::Bool=true,
        allow_self_loops::Bool=false,
        nodeset1::Symbol=:actors,
        nodeset2::Union{Symbol, Nothing}=nothing
    )
        # Validate
        if isempty(networks)
            throw(ArgumentError("At least one network observation required"))
        end
        if type == :onemode && size(networks[1], 1) != size(networks[1], 2)
            throw(ArgumentError("One-mode networks must be square"))
        end
        # Convert to Int matrices
        int_networks = [Matrix{Int}(net) for net in networks]
        new(name, int_networks, type, directed, allow_self_loops, nodeset1, nodeset2)
    end
end

"""
    n_waves(dn::DependentNetwork)

Return the number of observation waves.
"""
n_waves(dn::DependentNetwork) = length(dn.networks)

"""
    n_actors(dn::DependentNetwork)

Return the number of actors (rows) in the network.
"""
n_actors(dn::DependentNetwork) = size(dn.networks[1], 1)

"""
    DependentBehavior

A dependent behavioral variable observed at multiple time points.

# Fields
- `name::Symbol`: Variable name
- `values::Vector{Vector{Int}}`: Behavior values at each observation
- `min_val::Int`: Minimum allowed value
- `max_val::Int`: Maximum allowed value
- `nodeset::Symbol`: ID of the node set
- `mean_val::Float64`: Overall mean of the observed values (used for centering, as in RSiena)
- `sim_mean::Float64`: Mean pairwise similarity of the observed values over waves
  1..M-1 (RSiena's ^sim)
"""
mutable struct DependentBehavior <: AbstractDependent
    name::Symbol
    values::Vector{Vector{Int}}
    min_val::Int
    max_val::Int
    nodeset::Symbol
    mean_val::Float64
    sim_mean::Float64

    function DependentBehavior(
        name::Symbol,
        values::Vector{<:AbstractVector{<:Integer}};
        min_val::Union{Int, Nothing}=nothing,
        max_val::Union{Int, Nothing}=nothing,
        nodeset::Symbol=:actors
    )
        if isempty(values)
            throw(ArgumentError("At least one observation required"))
        end
        # Determine range from data if not specified
        all_vals = vcat(values...)
        actual_min = minimum(all_vals)
        actual_max = maximum(all_vals)
        min_v = isnothing(min_val) ? actual_min : min_val
        max_v = isnothing(max_val) ? actual_max : max_val

        int_values = [Vector{Int}(v) for v in values]
        mean_v = mean(all_vals)
        rng_v = max_v - min_v
        # RSiena computes the similarity mean over the waves that serve as period
        # starting points (1..M-1)
        sim_waves = int_values[1:max(length(int_values) - 1, 1)]
        sim_m = rng_v == 0 ? 1.0 :
            mean(1.0 - abs(v[i] - v[j]) / rng_v
                 for v in sim_waves for i in eachindex(v) for j in eachindex(v) if i != j)
        new(name, int_values, min_v, max_v, nodeset, mean_v, sim_m)
    end
end

n_waves(db::DependentBehavior) = length(db.values)
n_actors(db::DependentBehavior) = length(db.values[1])

#==============================================================================#
# Covariates
#==============================================================================#

"""
    AbstractCovariate

Abstract type for covariates.
"""
abstract type AbstractCovariate end

# Mean pairwise similarity 1 - |v_i - v_j|/r over ordered pairs i != j
# (RSiena's ^sim, used to center similarity effects).
function _similarity_mean(values::Vector{Float64}, r::Float64)
    ok = findall(!isnan, values)
    (length(ok) < 2 || r == 0) && return 1.0
    total = 0.0
    for i in ok, j in ok
        i == j && continue
        total += 1.0 - abs(values[i] - values[j]) / r
    end
    return total / (length(ok) * (length(ok) - 1))
end

function _similarity_mean(values::Vector{Float64})
    ok = filter(!isnan, values)
    length(ok) < 2 && return 1.0
    lo, hi = extrema(ok)
    return _similarity_mean(values, hi - lo)
end

"""
    ConstantCovariate

A covariate that is constant across all waves.

# Fields
- `name::Symbol`: Covariate name
- `values::Vector{Float64}`: Values for each actor
- `nodeset::Symbol`: ID of the node set
- `centered::Bool`: Whether values are centered
- `mean::Float64`: Mean value (for centering)
"""
struct ConstantCovariate <: AbstractCovariate
    name::Symbol
    values::Vector{Float64}
    nodeset::Symbol
    centered::Bool
    mean::Float64
    sim_mean::Float64

    function ConstantCovariate(name::Symbol, values::AbstractVector{<:Real};
                               nodeset::Symbol=:actors, center::Bool=true)
        fvals = Float64.(values)
        m = mean(fvals)
        centered_vals = center ? fvals .- m : fvals
        new(name, centered_vals, nodeset, center, m, _similarity_mean(centered_vals))
    end
end

"""
    VaryingCovariate

A covariate that varies across waves.

# Fields
- `name::Symbol`: Covariate name
- `values::Vector{Vector{Float64}}`: Values for each actor at each wave
- `nodeset::Symbol`: ID of the node set
- `centered::Bool`: Whether values are centered
- `mean::Float64`: Overall mean value
"""
struct VaryingCovariate <: AbstractCovariate
    name::Symbol
    values::Vector{Vector{Float64}}
    nodeset::Symbol
    centered::Bool
    mean::Float64
    sim_mean::Float64

    function VaryingCovariate(name::Symbol, values::Vector{<:AbstractVector{<:Real}};
                              nodeset::Symbol=:actors, center::Bool=true)
        fvals = [Float64.(v) for v in values]
        m = mean(vcat(fvals...))
        centered_vals = center ? [v .- m for v in fvals] : fvals
        all_vals = filter(!isnan, vcat(centered_vals...))
        r = length(all_vals) < 2 ? 0.0 : maximum(all_vals) - minimum(all_vals)
        sim_m = mean(_similarity_mean(v, r) for v in centered_vals)
        new(name, centered_vals, nodeset, center, m, sim_m)
    end
end

"""
    ConstantDyadCovariate

A dyadic covariate that is constant across waves.

# Fields
- `name::Symbol`: Covariate name
- `values::Matrix{Float64}`: Values for each dyad
- `nodeset1::Symbol`: ID of row node set
- `nodeset2::Symbol`: ID of column node set
- `centered::Bool`: Whether values are centered
- `mean::Float64`: Mean value
"""
struct ConstantDyadCovariate <: AbstractCovariate
    name::Symbol
    values::Matrix{Float64}
    nodeset1::Symbol
    nodeset2::Symbol
    centered::Bool
    mean::Float64

    function ConstantDyadCovariate(name::Symbol, values::AbstractMatrix{<:Real};
                                   nodeset1::Symbol=:actors, nodeset2::Symbol=:actors,
                                   center::Bool=true)
        fvals = Float64.(values)
        m = mean(fvals)
        centered_vals = center ? fvals .- m : fvals
        new(name, centered_vals, nodeset1, nodeset2, center, m)
    end
end

"""
    VaryingDyadCovariate

A dyadic covariate that varies across waves.

# Fields
- `name::Symbol`: Covariate name
- `values::Vector{Matrix{Float64}}`: Values for each dyad at each wave
- `nodeset1::Symbol`: ID of row node set
- `nodeset2::Symbol`: ID of column node set
- `centered::Bool`: Whether values are centered
- `mean::Float64`: Overall mean value
"""
struct VaryingDyadCovariate <: AbstractCovariate
    name::Symbol
    values::Vector{Matrix{Float64}}
    nodeset1::Symbol
    nodeset2::Symbol
    centered::Bool
    mean::Float64

    function VaryingDyadCovariate(name::Symbol, values::Vector{<:AbstractMatrix{<:Real}};
                                  nodeset1::Symbol=:actors, nodeset2::Symbol=:actors,
                                  center::Bool=true)
        fvals = [Float64.(v) for v in values]
        m = mean(vcat([vec(v) for v in fvals]...))
        centered_vals = center ? [v .- m for v in fvals] : fvals
        new(name, centered_vals, nodeset1, nodeset2, center, m)
    end
end

#==============================================================================#
# Composition Change
#==============================================================================#

"""
    CompositionChange

Tracks changes in network composition (actors joining/leaving).

# Fields
- `changes::Vector{Tuple{Int, Int, Symbol}}`: (actor, wave, action) tuples
  where action is :join or :leave
"""
struct CompositionChange
    changes::Vector{Tuple{Int, Int, Symbol}}

    function CompositionChange(changes::Vector{Tuple{Int, Int, Symbol}}=Tuple{Int, Int, Symbol}[])
        for (_, _, action) in changes
            if action ∉ (:join, :leave)
                throw(ArgumentError("Action must be :join or :leave"))
            end
        end
        new(changes)
    end
end

"""
    add_change!(cc::CompositionChange, actor::Int, wave::Int, action::Symbol)

Add a composition change event.
"""
function add_change!(cc::CompositionChange, actor::Int, wave::Int, action::Symbol)
    if action ∉ (:join, :leave)
        throw(ArgumentError("Action must be :join or :leave"))
    end
    push!(cc.changes, (actor, wave, action))
end

#==============================================================================#
# Siena Data Container
#==============================================================================#

"""
    SienaData

Container for all data needed for SAOM estimation.

# Fields
- `nodesets::Dict{Symbol, NodeSet}`: Named node sets
- `dependents::Dict{Symbol, AbstractDependent}`: Dependent variables
- `covariates::Dict{Symbol, AbstractCovariate}`: Covariates
- `composition_change::Union{CompositionChange, Nothing}`: Composition changes
- `n_waves::Int`: Number of observation waves
"""
mutable struct SienaData
    nodesets::Dict{Symbol, NodeSet}
    dependents::Dict{Symbol, AbstractDependent}
    covariates::Dict{Symbol, AbstractCovariate}
    composition_change::Union{CompositionChange, Nothing}
    n_waves::Int

    function SienaData()
        new(
            Dict{Symbol, NodeSet}(),
            Dict{Symbol, AbstractDependent}(),
            Dict{Symbol, AbstractCovariate}(),
            nothing,
            0
        )
    end
end

"""
    add_nodeset!(data::SienaData, ns::NodeSet)

Add a node set to the data.
"""
function add_nodeset!(data::SienaData, ns::NodeSet)
    data.nodesets[ns.id] = ns
    data
end

"""
    add_dependent!(data::SienaData, dep::AbstractDependent)

Add a dependent variable to the data.
"""
function add_dependent!(data::SienaData, dep::AbstractDependent)
    nw = n_waves(dep)
    if data.n_waves == 0
        data.n_waves = nw
    elseif data.n_waves != nw
        throw(ArgumentError("Number of waves must be consistent (expected $(data.n_waves), got $nw)"))
    end
    data.dependents[dep.name] = dep
    data
end

"""
    add_covariate!(data::SienaData, cov::AbstractCovariate)

Add a covariate to the data.
"""
function add_covariate!(data::SienaData, cov::AbstractCovariate)
    data.covariates[cov.name] = cov
    data
end

function Base.show(io::IO, data::SienaData)
    print(io, "SienaData(")
    print(io, "nodesets=$(length(data.nodesets)), ")
    print(io, "dependents=$(length(data.dependents)), ")
    print(io, "covariates=$(length(data.covariates)), ")
    print(io, "waves=$(data.n_waves))")
end

#==============================================================================#
# Network State (for simulation)
#==============================================================================#

"""
    NetworkState

Mutable state of networks and behaviors during simulation.

# Fields
- `networks::Dict{Symbol, Matrix{Int}}`: Current network states
- `behaviors::Dict{Symbol, Vector{Int}}`: Current behavior states
- `time::Float64`: Current simulation time within period
- `period::Int`: Current period (index of the starting wave); used to select the
  values of varying covariates
"""
mutable struct NetworkState
    networks::Dict{Symbol, Matrix{Int}}
    behaviors::Dict{Symbol, Vector{Int}}
    time::Float64
    period::Int

    function NetworkState()
        new(Dict{Symbol, Matrix{Int}}(), Dict{Symbol, Vector{Int}}(), 0.0, 1)
    end
end

"""
    initialize!(state::NetworkState, data::SienaData, wave::Int; period::Int=wave)

Initialize network state from data at a given wave. `period` sets the period used
for varying-covariate lookups (defaults to the wave itself; pass the starting wave
when initializing at the *end* observation of a period).
"""
function initialize!(state::NetworkState, data::SienaData, wave::Int; period::Int=wave)
    state.time = 0.0
    state.period = min(period, max(data.n_waves - 1, 1))
    for (name, dep) in data.dependents
        if dep isa DependentNetwork
            state.networks[name] = copy(dep.networks[wave])
        elseif dep isa DependentBehavior
            state.behaviors[name] = copy(dep.values[wave])
        end
    end
    state
end

"""
    snapshot(state::NetworkState)

Return an independent copy of the current state.
"""
function snapshot(state::NetworkState)
    s = NetworkState()
    for (k, v) in state.networks
        s.networks[k] = copy(v)
    end
    for (k, v) in state.behaviors
        s.behaviors[k] = copy(v)
    end
    s.time = state.time
    s.period = state.period
    s
end
