# Network.jl integration for Siena.jl, loaded automatically when both Siena
# and Network are in the environment (package extension).
#
# Bridges the ecosystem's `Network` type into Siena's panel-data types so the
# statnet-style workflow (describe with SNA.jl -> model dynamics with Siena.jl)
# needs no manual matrix wrangling: a `Vector` of `Network` objects (one per
# observation wave) converts directly into a `DependentNetwork`, and single
# networks / network panels convert into dyadic covariates.
module SienaNetworkExt

using Siena
import Network
using Network: as_matrix, is_directed, is_two_mode, nv, get_vertex_attribute

# `Network.Network` (the struct shares the module's name) or a
# `BipartiteNetwork` wrapper; `as_matrix`/`nv`/`is_directed` work on both.
const AnyNetwork = Network.AbstractNetwork

# The underlying `Network.Network` carrying the attribute dictionaries.
_core(net::Network.Network) = net
_core(net::Network.BipartiteNetwork) = net.network

_allow_loops(net::AnyNetwork) = _core(net).loops

# Number of mode-1 vertices for two-mode networks, `nothing` for one-mode.
_mode1(net::Network.Network) = net.bipartite
_mode1(net::Network.BipartiteNetwork) = net.n_mode1

# Vertex names (the `:vertex_names` attribute), or `nothing` if unset.
function _vertex_names(net::AnyNetwork)
    d = get_vertex_attribute(_core(net), :vertex_names)
    return isempty(d) ? nothing : d
end

"""
    _validate_panel(networks) -> (directed, two_mode)

Validate that a vector of networks forms a coherent panel: at least one wave,
identical node sets (same vertex count, same mode-1 size for two-mode
networks, and matching `:vertex_names` where present), and consistent
directedness. Returns the shared directedness and two-mode flags.
"""
function _validate_panel(networks::Vector{<:AnyNetwork})
    isempty(networks) && throw(ArgumentError(
        "at least one network observation required"))
    ref = networks[1]
    n = nv(ref)
    dir = is_directed(ref)
    two = is_two_mode(ref)
    names_ref = _vertex_names(ref)
    for (w, net) in enumerate(networks)
        w == 1 && continue
        nv(net) == n || throw(ArgumentError(
            "all waves must share the same node set: wave $w has $(nv(net)) " *
            "vertices but wave 1 has $n"))
        is_directed(net) == dir || throw(ArgumentError(
            "all waves must share the same directedness: wave $w is " *
            "$(is_directed(net) ? "directed" : "undirected") but wave 1 is " *
            "$(dir ? "directed" : "undirected")"))
        is_two_mode(net) == two || throw(ArgumentError(
            "all waves must be one-mode or all two-mode: wave $w differs from wave 1"))
        if two && _mode1(net) != _mode1(ref)
            throw(ArgumentError(
                "all waves must share the same mode-1 size: wave $w has " *
                "$(_mode1(net)) mode-1 vertices but wave 1 has $(_mode1(ref))"))
        end
        names_w = _vertex_names(net)
        if names_ref !== nothing && names_w !== nothing && names_w != names_ref
            throw(ArgumentError(
                "all waves must share the same node set: vertex names of " *
                "wave $w differ from wave 1"))
        end
    end
    return dir, two
end

# Binary adjacency (or two-mode incidence) matrix as Matrix{Int}.
_int_matrix(net::AnyNetwork) = Matrix{Int}(as_matrix(net))

"""
    DependentNetwork(name::Symbol, networks::Vector{<:AbstractNetwork}; kwargs...)

Build a dependent network variable directly from a vector of `Network.Network`
(or `BipartiteNetwork`) observations, one per wave. Requires Network.jl to be
loaded (this method lives in the `SienaNetworkExt` package extension).

Adjacency matrices are extracted with `Network.as_matrix`; undirected networks
yield symmetric matrices. Directedness, self-loop allowance, and one-/two-mode
type are taken from the networks themselves, and the waves are validated to
share the same node set (vertex count, mode sizes, and `:vertex_names` where
present) and directedness.

# Keyword Arguments
- `type::Union{Symbol,Nothing}=nothing`: override the network type
  (defaults to `:onemode`, or `:twomode` for bipartite networks)
- `nodeset1::Symbol=:actors`: ID of the (first) node set
- `nodeset2`: ID of the second node set (defaults to `:mode2` for two-mode
  networks, `nothing` otherwise)

# Example
```julia
using Network, Siena

waves = [network(20) for _ in 1:3]
# ... add_edge!(waves[w], i, j) ...
dep = DependentNetwork(:friendship, waves)
```
"""
function Siena.DependentNetwork(
    name::Symbol,
    networks::Vector{<:AnyNetwork};
    type::Union{Symbol, Nothing}=nothing,
    nodeset1::Symbol=:actors,
    nodeset2::Union{Symbol, Nothing}=nothing
)
    dir, two = _validate_panel(networks)
    return DependentNetwork(
        name,
        [_int_matrix(net) for net in networks];
        type=isnothing(type) ? (two ? :twomode : :onemode) : type,
        directed=dir,
        allow_self_loops=_allow_loops(networks[1]),
        nodeset1=nodeset1,
        nodeset2=isnothing(nodeset2) ? (two ? :mode2 : nothing) : nodeset2
    )
end

"""
    siena_dependent(name::Symbol, networks::Vector{<:AbstractNetwork}; kwargs...)

Create a dependent network variable from a vector of `Network` observations
(requires Network.jl; see `DependentNetwork`).
"""
Siena.siena_dependent(name::Symbol, networks::Vector{<:AnyNetwork}; kwargs...) =
    DependentNetwork(name, networks; kwargs...)

"""
    ConstantDyadCovariate(name::Symbol, net::AbstractNetwork;
                          attr=nothing, kwargs...)

Build a constant dyadic covariate from a single `Network` (requires
Network.jl). The dyad values are `Network.as_matrix(net; attr=attr)`: binary
adjacency by default, or an edge attribute's values when `attr` is given
(absent edges contribute 0). Remaining keyword arguments (`nodeset1`,
`nodeset2`, `center`) are passed through.

# Example
```julia
prox = ConstantDyadCovariate(:proximity, office_net; attr=:distance)
```
"""
Siena.ConstantDyadCovariate(name::Symbol, net::AnyNetwork;
                            attr::Union{Symbol, Nothing}=nothing, kwargs...) =
    ConstantDyadCovariate(name, as_matrix(net; attr=attr); kwargs...)

"""
    VaryingDyadCovariate(name::Symbol, networks::Vector{<:AbstractNetwork};
                         attr=nothing, kwargs...)

Build a wave-varying dyadic covariate from a vector of `Network` observations
(requires Network.jl). Each wave's dyad values are
`Network.as_matrix(net; attr=attr)`. The networks are validated to share the
same node set and directedness across waves.
"""
function Siena.VaryingDyadCovariate(name::Symbol, networks::Vector{<:AnyNetwork};
                                    attr::Union{Symbol, Nothing}=nothing, kwargs...)
    _validate_panel(networks)
    return VaryingDyadCovariate(name, [as_matrix(net; attr=attr) for net in networks];
                                kwargs...)
end

# RSiena-style convenience constructors (coDyadCovar / varDyadCovar analogues)
Siena.constant_dyad_covariate(name::Symbol, net::AnyNetwork; kwargs...) =
    ConstantDyadCovariate(name, net; kwargs...)
Siena.varying_dyad_covariate(name::Symbol, networks::Vector{<:AnyNetwork}; kwargs...) =
    VaryingDyadCovariate(name, networks; kwargs...)

end # module
