# NetworkCore.jl integration for Siena.jl, loaded automatically when both Siena
# and NetworkCore are loaded (ordinary source integration).
#
# Bridges the ecosystem's `Network` type into Siena's panel-data types so the
# statnet-style workflow (describe with SNA.jl -> model dynamics with Siena.jl)
# needs no manual matrix wrangling: a `Vector` of `Network` objects (one per
# observation wave) converts directly into a `DependentNetwork`, and single
# networks / network panels convert into dyadic covariates.
"""
    Siena.NetworkIntegration

The NetworkCore.jl bridge: constructors of [`DependentNetwork`](@ref),
[`ConstantDyadCovariate`](@ref) and [`VaryingDyadCovariate`](@ref) (and their
RSiena-style aliases) from `Network`/`BipartiteNetwork` waves, honouring the
ecosystem's missing-data and conversion contracts. Loaded with Siena; nothing in it
needs to be called by its module name.

# Example
```julia
using Siena, NetworkCore
waves = load_dataset(:s50).friendship
dep = DependentNetwork(:friendship, waves)        # through Siena.NetworkIntegration
n_waves(dep)                                      # 3
```
"""
module NetworkIntegration

using ..Siena
import NetworkCore
using NetworkCore: Network, as_matrix, is_directed, is_two_mode, nv, get_vertex_attribute
using NetworkCore: ConversionReport, record_drop!, require_observed, n_missing_dyads

# A `Network` or a `BipartiteNetwork` wrapper; `as_matrix`/`nv`/`is_directed`
# work on both.
const AnyNetwork = NetworkCore.AbstractNetwork

# The underlying `Network` carrying the attribute dictionaries.
_core(net::Network) = net
_core(net::NetworkCore.BipartiteNetwork) = net.network

_allow_loops(net::AnyNetwork) = _core(net).loops

# Number of mode-1 vertices for two-mode networks, `nothing` for one-mode.
_mode1(net::Network) = net.bipartite
_mode1(net::NetworkCore.BipartiteNetwork) = net.n_mode1

# Vertex names (the `:vertex_names` attribute), or `nothing` if unset.
function _vertex_names(net::AnyNetwork)
    d = get_vertex_attribute(_core(net), :vertex_names)
    return isempty(d) ? nothing : d
end

"""
    _validate_panel(networks) -> (directed, two_mode)

Validate that a vector of networks forms a coherent panel: at least one wave,
identical node sets (same vertex count, same mode-1 size for two-mode
networks, and matching `:vertex_names` where present), consistent
directedness and a consistent `loops` flag (as the `DynamicNetworks` panel
constructor requires). Returns the shared directedness and two-mode flags.
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
        _allow_loops(net) == _allow_loops(ref) || throw(ArgumentError(
            "all waves must share the same loops flag: wave $w has " *
            "loops=$(_allow_loops(net)) but wave 1 has loops=$(_allow_loops(ref)); " *
            "a dependent network either allows self-ties in every wave or in none"))
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

# A two-mode dependent network holds ties from mode 1 (the actors, who make the
# choices) to mode 2: RSiena's bipartite variable. A directed two-mode `Network`
# can also carry arcs from mode 2 to mode 1, which that variable cannot represent
# (`as_matrix` would drop them from the incidence matrix), so they are refused
# rather than lost. An undirected two-mode network has no such arcs.
function _check_twomode_direction(networks)
    for (w, net) in enumerate(networks)
        is_directed(net) || continue
        k = _mode1(net)::Int
        A = as_matrix(net; expand_bipartite=true)
        n_rev = count(!iszero, view(A, (k + 1):size(A, 1), 1:k))
        n_rev == 0 || throw(ArgumentError(
            "wave $w is a directed two-mode network with $n_rev arc(s) from mode 2 " *
            "to mode 1; a Siena two-mode dependent network holds only ties from " *
            "the mode-1 actors to mode-2 nodes, so these arcs cannot be represented"))
    end
    return nothing
end

# The missing-dyad guard for every Network → Siena path.
#
# Siena's matrices carry a *structural* mask (structural zeros/ones: ties that
# are determined, not free), which is a different claim from NetworkCore.jl's
# missing-dyad mask (ties that are UNOBSERVED). Coding an unobserved dyad as a
# structural zero would tell the estimator the tie is known to be impossible,
# which is exactly the plausible-but-wrong output the ecosystem contract
# forbids. Siena.jl has no missing-data machinery, so a masked network is
# rejected by default; `missing=:face` is the explicit, auditable opt-in.
function _siena_guard(networks, policy::Symbol, context::AbstractString)
    for net in networks
        require_observed(_core(net), policy; context=context)
    end
    rep = ConversionReport(:Network, :SienaMatrix)
    n_mask = sum(n_missing_dyads(_core(net)) for net in networks; init=0)
    n_mask > 0 && record_drop!(rep, :missing_dyads,
        "$n_mask masked dyad(s) across the waves were written at face value " *
        "under missing=:face; Siena's structural mask records determined ties, " *
        "not unobserved ones, so the fact that they are unobserved is lost")
    record_drop!(rep, :attributes,
        "vertex/edge/network attributes are not carried into a Siena matrix; " *
        "add them as Siena covariates (constant_covariate, ConstantDyadCovariate, ...)")
    return rep
end

"""
    DependentNetwork(name::Symbol, networks::Vector{<:AbstractNetwork};
                     missing=:error, report=false, kwargs...)

Build a dependent network variable directly from a vector of `Network`
(or `BipartiteNetwork`) observations, one per wave. Requires NetworkCore.jl to be
loaded (NetworkCore is a hard dependency of Siena).

Adjacency matrices are extracted with `NetworkCore.as_matrix`; undirected networks
yield symmetric matrices. Directedness, self-loop allowance, and one-/two-mode
type are taken from the networks themselves, and the waves are validated to
share the same node set (vertex count, mode sizes, and `:vertex_names` where
present), directedness and `loops` flag.

A **two-mode** network (`network(n; bipartite=n₁)` or a `BipartiteNetwork`)
becomes RSiena's two-mode dependent variable (`type=:twomode`): the waves are
the `n₁ × (n - n₁)` incidence matrices, and the ties run from the mode-1 actors,
who make the choices, to the mode-2 nodes. Whether the `Network` is stored as
directed or undirected does not matter, except that a directed one carrying an
arc from mode 2 to mode 1 is refused (the variable cannot hold it). Add the
second node set to the `SienaData` with `add_nodeset!(data, NodeSet(n - n₁;
name=:mode2))` (or the `nodeset2` you pass).

# Conversion invariants

Preserved: directedness (one-mode), self-loop allowance, one-/two-mode type,
node-set sizes, and the binary tie values of every wave.

A network with a **missing-dyad mask is rejected** (`missing=:error`, the
default). Siena's own per-wave mask records *structural* zeros/ones — ties that
are determined — which is a different claim from "unobserved", so an
unobserved dyad has no faithful Siena encoding and must not be silently written
as a 0. Pass `missing=:face` to write the recorded face values anyway.

Vertex/edge/network attributes are dropped (add them as Siena covariates);
`report=true` returns `(dep, ::NetworkCore.ConversionReport)` saying so.

# Keyword Arguments
- `type::Union{Symbol,Nothing}=nothing`: override the network type
  (defaults to `:onemode`, or `:twomode` for bipartite networks)
- `nodeset1::Symbol=:actors`: ID of the (first) node set
- `nodeset2`: ID of the second node set (defaults to `:mode2` for two-mode
  networks, `nothing` otherwise)
- `missing::Symbol=:error`: missing-dyad policy (`:error` or `:face`)
- `report::Bool=false`: also return a `NetworkCore.ConversionReport`

# Example
```julia
using NetworkCore, Siena

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
    nodeset2::Union{Symbol, Nothing}=nothing,
    missing::Symbol=:error,
    report::Bool=false,
    allow_only::Bool=true
)
    dir, two = _validate_panel(networks)
    rep = _siena_guard(networks, missing, "DependentNetwork(::Vector{<:AbstractNetwork})")
    kind = isnothing(type) ? (two ? :twomode : :onemode) : type
    (kind == :onemode) == !two || throw(ArgumentError(
        "type=:$kind does not match the networks, which are " *
        "$(two ? "two-mode" : "one-mode"); leave `type` unset to take it from them"))
    two && _check_twomode_direction(networks)
    dep = DependentNetwork(
        name,
        [_int_matrix(net) for net in networks];
        type=kind,
        # A two-mode variable is always directed: actor (mode 1) -> mode-2 node.
        directed=two ? true : dir,
        allow_self_loops=two ? false : _allow_loops(networks[1]),
        nodeset1=nodeset1,
        nodeset2=isnothing(nodeset2) ? (two ? :mode2 : nothing) : nodeset2,
        allow_only=allow_only
    )
    return report ? (dep, rep) : dep
end

"""
    siena_dependent(name::Symbol, networks::Vector{<:AbstractNetwork}; kwargs...)

Create a dependent network variable from a vector of `Network` observations
(requires NetworkCore.jl; see `DependentNetwork`, including its missing-dyad
policy).
"""
Siena.siena_dependent(name::Symbol, networks::Vector{<:AnyNetwork}; kwargs...) =
    DependentNetwork(name, networks; kwargs...)

"""
    ConstantDyadCovariate(name::Symbol, net::AbstractNetwork;
                          attr=nothing, missing=:error, report=false, kwargs...)

Build a constant dyadic covariate from a single `Network` (requires
NetworkCore.jl). The dyad values are `NetworkCore.as_matrix(net; attr=attr)`: binary
adjacency by default, or an edge attribute's values when `attr` is given
(absent edges contribute 0). Remaining keyword arguments (`nodeset1`,
`nodeset2`, `center`) are passed through.

A dyadic covariate matrix has no missing-dyad slot, so a masked network is
**rejected** by default; pass `missing=:face` to use the recorded face values,
and `report=true` for a `NetworkCore.ConversionReport`.

# Example
```julia
prox = ConstantDyadCovariate(:proximity, office_net; attr=:distance)
```
"""
function Siena.ConstantDyadCovariate(name::Symbol, net::AnyNetwork;
                                     attr::Union{Symbol, Nothing}=nothing,
                                     missing::Symbol=:error,
                                     report::Bool=false, kwargs...)
    rep = _siena_guard([net], missing, "ConstantDyadCovariate(::AbstractNetwork)")
    cov = ConstantDyadCovariate(name, as_matrix(net; attr=attr); kwargs...)
    return report ? (cov, rep) : cov
end

"""
    VaryingDyadCovariate(name::Symbol, networks::Vector{<:AbstractNetwork};
                         attr=nothing, missing=:error, report=false, kwargs...)

Build a wave-varying dyadic covariate from a vector of `Network` observations
(requires NetworkCore.jl). Each wave's dyad values are
`NetworkCore.as_matrix(net; attr=attr)`. The networks are validated to share the
same node set and directedness across waves. Masked networks are **rejected**
by default; see `ConstantDyadCovariate`.
"""
function Siena.VaryingDyadCovariate(name::Symbol, networks::Vector{<:AnyNetwork};
                                    attr::Union{Symbol, Nothing}=nothing,
                                    missing::Symbol=:error,
                                    report::Bool=false, kwargs...)
    _validate_panel(networks)
    rep = _siena_guard(networks, missing,
                       "VaryingDyadCovariate(::Vector{<:AbstractNetwork})")
    cov = VaryingDyadCovariate(name, [as_matrix(net; attr=attr) for net in networks];
                               kwargs...)
    return report ? (cov, rep) : cov
end

# RSiena-style convenience constructors (coDyadCovar / varDyadCovar analogues)
Siena.constant_dyad_covariate(name::Symbol, net::AnyNetwork; kwargs...) =
    ConstantDyadCovariate(name, net; kwargs...)
Siena.varying_dyad_covariate(name::Symbol, networks::Vector{<:AnyNetwork}; kwargs...) =
    VaryingDyadCovariate(name, networks; kwargs...)

end # module
