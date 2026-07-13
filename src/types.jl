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
  (0/1 face values; structural codes are decoded on construction)
- `type::Symbol`: Network type (:onemode, :twomode, :bipartite)
- `directed::Bool`: Whether the network is directed
- `allow_self_loops::Bool`: Whether self-loops are allowed
- `nodeset1::Symbol`: ID of the first node set
- `nodeset2::Union{Symbol, Nothing}`: ID of the second node set (for bipartite)
- `structural::Vector{BitMatrix}`: Per-wave masks of structurally determined
  dyads (see below); empty when the data contain no structural codes

# Structural zeros and ones (RSiena 10/11 coding)
Adjacency matrices may contain RSiena-style structural codes alongside 0/1:
by default `10` marks a **structural zero** (the tie is structurally absent
— impossible, e.g. between actors who never met) and `11` a **structural
one** (the tie is structurally present — forced, e.g. formal ties). The
codes are configurable via the `structural_zero`/`structural_one` keywords.
On construction each coded entry is decoded to its determined value (10 → 0,
11 → 1) in `networks`, and its position is recorded in the per-wave
`structural` mask. Any entry other than 0, 1, or the two codes throws an
`ArgumentError`.

Structurally determined dyads behave as in RSiena's first-order semantics:
they are excluded from the candidate sets of ministep simulation for the
period whose *start* wave marks them (an actor can never toggle them), they
are excluded from the target and simulated moment statistics, and they do
not count toward observed change in rate statistics.

!!! warning "Structural status that changes between waves"
    Only the period-start mask is used. When a dyad's structural status *changes*
    from one wave to the next, RSiena applies a further correction to the
    statistics that is not implemented here, so results on such data are not
    numerically equivalent to RSiena's. Data whose structural masks are the same
    in every wave — the common case — are unaffected.

# Interoperability
When Networks.jl is loaded, the `SienaNetworkExt` package extension adds a
constructor taking a `Vector` of `Network` objects (one per wave); adjacency
matrices are extracted with `Networks.as_matrix`, directedness and self-loop
settings are taken from the networks, and the waves are validated to share
the same node set. See the extension's docstring for details.
"""
mutable struct DependentNetwork <: AbstractDependent
    name::Symbol
    networks::Vector{Matrix{Int}}
    type::Symbol
    directed::Bool
    allow_self_loops::Bool
    nodeset1::Symbol
    nodeset2::Union{Symbol, Nothing}
    structural::Vector{BitMatrix}

    function DependentNetwork(
        name::Symbol,
        networks::Vector{<:AbstractMatrix{<:Integer}};
        type::Symbol=:onemode,
        directed::Bool=true,
        allow_self_loops::Bool=false,
        nodeset1::Symbol=:actors,
        nodeset2::Union{Symbol, Nothing}=nothing,
        structural_zero::Int=10,
        structural_one::Int=11
    )
        # Validate
        if isempty(networks)
            throw(ArgumentError("At least one network observation required"))
        end
        if type == :onemode && size(networks[1], 1) != size(networks[1], 2)
            throw(ArgumentError("One-mode networks must be square"))
        end
        if structural_zero == structural_one ||
           structural_zero in (0, 1) || structural_one in (0, 1)
            throw(ArgumentError("structural codes must be distinct and different " *
                                "from the tie values 0 and 1 (got " *
                                "structural_zero=$structural_zero, " *
                                "structural_one=$structural_one)"))
        end
        # Convert to Int matrices, validate codes, and decode structural
        # entries (structural_zero -> 0, structural_one -> 1) into per-wave
        # masks of structurally determined dyads
        int_networks = [Matrix{Int}(net) for net in networks]
        masks = [falses(size(m)) for m in int_networks]
        any_structural = false
        for (w, m) in enumerate(int_networks)
            for idx in eachindex(m)
                v = m[idx]
                if v == structural_zero
                    m[idx] = 0
                    masks[w][idx] = true
                    any_structural = true
                elseif v == structural_one
                    m[idx] = 1
                    masks[w][idx] = true
                    any_structural = true
                elseif v != 0 && v != 1
                    throw(ArgumentError("invalid tie value $v in wave $w of " *
                                        "dependent network :$name: entries must be " *
                                        "0, 1, $structural_zero (structural zero), " *
                                        "or $structural_one (structural one)"))
                end
            end
        end
        structural = any_structural ? masks : BitMatrix[]
        new(name, int_networks, type, directed, allow_self_loops, nodeset1,
            nodeset2, structural)
    end
end

"""
    has_structural(dep::DependentNetwork) -> Bool

Whether the dependent network contains any structurally determined dyads
(structural zeros/ones; see [`DependentNetwork`](@ref)).
"""
has_structural(dep::DependentNetwork) = !isempty(dep.structural)

"""
    is_structural_dyad(dep::DependentNetwork, wave::Int, i::Int, j::Int) -> Bool

Whether the dyad `(i, j)` is structurally determined (structural zero or
one) at observation `wave`. Its determined face value is
`dep.networks[wave][i, j]`.
"""
is_structural_dyad(dep::DependentNetwork, wave::Int, i::Int, j::Int) =
    !isempty(dep.structural) && dep.structural[wave][i, j]

"""
    n_structural_dyads(dep::DependentNetwork, wave::Int) -> Int

Number of structurally determined dyads at observation `wave`.
"""
n_structural_dyads(dep::DependentNetwork, wave::Int) =
    isempty(dep.structural) ? 0 : count(dep.structural[wave])

# Structural mask relevant to a simulation state: the mask of the state's
# period-start wave, or `nothing` when the variable has no structural dyads.
@inline function _structural_mask(dep::DependentNetwork, period::Int)
    isempty(dep.structural) && return nothing
    return dep.structural[period]
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

# Validation
A composition-change sequence describes a state machine per actor, and an
inconsistent sequence would silently produce a wrong presence pattern (and hence
wrong moment statistics), so it is rejected on construction:

- `action` must be `:join` or `:leave`;
- `actor` and `wave` must be positive (their upper bounds depend on the data and are
  checked by [`add_composition_change!`](@ref), which knows the number of actors and
  waves);
- an actor cannot have two events at the same wave;
- an actor's events must alternate: joining an actor that has already joined, or
  removing one that has already left, is contradictory and throws.

An actor's first event fixes its initial state: a first `:join` at wave `w` means the
actor is absent before `w`, a first `:leave` at wave `w` means it is present before
`w` (see [`is_present`](@ref)).
"""
struct CompositionChange
    changes::Vector{Tuple{Int, Int, Symbol}}

    function CompositionChange(changes::Vector{Tuple{Int, Int, Symbol}}=Tuple{Int, Int, Symbol}[])
        for (actor, wave, action) in changes
            _validate_change_event(actor, wave, action)
        end
        for actor in unique(a for (a, _, _) in changes)
            _validate_actor_history(actor, [(w, act) for (a, w, act) in changes
                                            if a == actor])
        end
        new(changes)
    end
end

# One event, in isolation: valid action and positive actor/wave. The upper bounds
# need the data (see `add_composition_change!`).
function _validate_change_event(actor::Int, wave::Int, action::Symbol)
    action ∈ (:join, :leave) ||
        throw(ArgumentError("composition change: action must be :join or :leave, " *
                            "got :$action (actor $actor, wave $wave)"))
    actor >= 1 ||
        throw(ArgumentError("composition change: actor must be >= 1, got $actor " *
                            "(:$action at wave $wave)"))
    wave >= 1 ||
        throw(ArgumentError("composition change: wave must be >= 1, got $wave " *
                            "(:$action of actor $actor)"))
    return nothing
end

# One actor's event history, as (wave, action) pairs: at most one event per wave, and
# the events must alternate join/leave — an actor cannot join while already present or
# leave while already absent.
function _validate_actor_history(actor::Int, events::Vector{Tuple{Int, Symbol}})
    sorted = sort(events; by=first)
    for k in 2:length(sorted)
        (w_prev, a_prev) = sorted[k - 1]
        (w, a) = sorted[k]
        w == w_prev &&
            throw(ArgumentError("composition change: actor $actor has two events at " *
                                "wave $w (:$a_prev and :$a); an actor can join or " *
                                "leave at most once per wave"))
        a == a_prev &&
            throw(ArgumentError("composition change: actor $actor is set to :$a at " *
                                "wave $w_prev and again at wave $w with no " *
                                "intervening :$(a == :join ? :leave : :join); join " *
                                "and leave events must alternate"))
    end
    return nothing
end

"""
    add_change!(cc::CompositionChange, actor::Int, wave::Int, action::Symbol)

Add a composition change event, validating it against the actor's existing history
(see [`CompositionChange`](@ref)). A contradictory or duplicate event throws and
leaves `cc` unchanged.
"""
function add_change!(cc::CompositionChange, actor::Int, wave::Int, action::Symbol)
    _validate_change_event(actor, wave, action)
    history = [(w, act) for (a, w, act) in cc.changes if a == actor]
    push!(history, (wave, action))
    # Validate before mutating, so a rejected event leaves `cc` untouched.
    _validate_actor_history(actor, history)
    push!(cc.changes, (actor, wave, action))
    return cc
end

"""
    is_present(cc::CompositionChange, actor::Int, wave::Int) -> Bool

Whether an actor is present at an observation wave. An actor with a `:join`
event at wave `w` is present from wave `w` onward (and absent before its
first event); a `:leave` event at wave `w` makes the actor absent from wave
`w` onward. Actors without composition-change events are always present.
"""
function is_present(cc::CompositionChange, actor::Int, wave::Int)
    events = [(w, action) for (a, w, action) in cc.changes if a == actor]
    isempty(events) && return true
    sort!(events; by=first)
    present = events[1][2] != :join   # joiners start absent, leavers start present
    for (w, action) in events
        w <= wave || break
        present = action == :join
    end
    return present
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

"""
    add_composition_change!(data::SienaData, cc::CompositionChange)

Attach composition-change information (actors joining/leaving; see
[`CompositionChange`](@ref)) to the data. The counterpart of supplying a
`sienaCompositionChange` object in RSiena.

During estimation, actors are treated with RSiena's method-of-moments
composition-change semantics: an actor contributes to a period only when
present at both of its endpoint waves. Absent actors get no ministep
opportunities, their dyads are excluded from the candidate sets, and their
rows/columns are excluded from the target and simulated moment statistics
and from the observed change (rate) distances.

The data fix the ranges the events must fall in, so this is where they are checked:
every actor must be an actor of the data (`1:n_actors`) and every wave an observation
wave (`1:n_waves`). An out-of-range actor or wave throws — silently ignoring it would
mean estimating a model whose composition differs from the one that was requested.
The internal consistency of the sequence itself is checked earlier, by
[`CompositionChange`](@ref)/[`add_change!`](@ref).
"""
function add_composition_change!(data::SienaData, cc::CompositionChange)
    n = _actor_count(data)
    n_w = data.n_waves
    n_w >= 1 || throw(ArgumentError(
        "cannot attach composition change: the data have no observation waves yet " *
        "(add the dependent variables first)"))
    for (actor, wave, action) in cc.changes
        actor <= n || throw(ArgumentError(
            "composition change: actor $actor (:$action at wave $wave) is out of " *
            "range — the data have $n actors"))
        wave <= n_w || throw(ArgumentError(
            "composition change: wave $wave (:$action of actor $actor) is out of " *
            "range — the data have $n_w observation waves"))
    end
    data.composition_change = cc
    data
end

# Number of actors in the primary node set.
function _actor_count(data::SienaData)
    haskey(data.nodesets, :actors) && return length(data.nodesets[:actors])
    for dep in values(data.dependents)
        return n_actors(dep)
    end
    throw(ArgumentError("cannot determine the number of actors: add a node set " *
                        "or a dependent variable first"))
end

# Per-period activity mask from the composition changes: `active[i]` is true iff
# actor i is present at both endpoint waves of the period. Returns `nothing` when
# the data have no composition changes.
function _activity_mask(data::SienaData, period::Int)
    cc = data.composition_change
    (cc === nothing || isempty(cc.changes)) && return nothing
    n = _actor_count(data)
    active = trues(n)
    for i in 1:n
        active[i] = is_present(cc, i, period) && is_present(cc, i, period + 1)
    end
    return active
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
    StateNetwork <: AbstractMatrix{Int}

Simulation-state representation of one network variable: a compact
`Matrix{Int8}` of the 0/1 tie values (structural codes are decoded on data
construction, so the state never holds values beyond 0/1) plus incrementally
maintained out-/indegree vectors.

Indexing reads and writes behave exactly like a 0/1 `Matrix{Int}`; every
`setindex!` updates the degree vectors, so `_row_sum`/`_col_sum` (the degree
lookups of the effect hot loops) are O(1) instead of O(n) scans. Writing a
value other than 0 or 1 throws.
"""
struct StateNetwork <: AbstractMatrix{Int}
    m::Matrix{Int8}
    outdeg::Vector{Int}
    indeg::Vector{Int}
end

function StateNetwork(m::AbstractMatrix{<:Integer})
    all(v -> v == 0 || v == 1, m) ||
        throw(ArgumentError("state networks hold 0/1 tie values only"))
    m8 = Matrix{Int8}(m)
    return StateNetwork(m8, vec(sum(Int, m8, dims=2)), vec(sum(Int, m8, dims=1)))
end

Base.convert(::Type{StateNetwork}, m::AbstractMatrix) = StateNetwork(m)

Base.size(sn::StateNetwork) = size(sn.m)
Base.IndexStyle(::Type{StateNetwork}) = IndexCartesian()

Base.@propagate_inbounds Base.getindex(sn::StateNetwork, i::Int, j::Int) =
    Int(sn.m[i, j])

Base.@propagate_inbounds function Base.setindex!(sn::StateNetwork, v, i::Int, j::Int)
    (v == 0 || v == 1) ||
        throw(ArgumentError("state networks hold 0/1 tie values only (got $v)"))
    b = Int8(v)
    old = sn.m[i, j]
    if b != old
        sn.m[i, j] = b
        d = Int(b) - Int(old)
        sn.outdeg[i] += d
        sn.indeg[j] += d
    end
    return sn
end

# Note: no `Base.copy` override — generic AbstractArray `copy` yields a plain
# `Matrix{Int}`, which is what callers extracting a wave matrix expect.
_copy_state_network(sn::StateNetwork) =
    StateNetwork(copy(sn.m), copy(sn.outdeg), copy(sn.indeg))

"""
    NetworkState

Mutable state of networks and behaviors during simulation.

# Fields
- `networks::Dict{Symbol, StateNetwork}`: Current network states (bit-packed
  0/1 matrices with incrementally maintained degree vectors; assigning a plain
  0/1 integer matrix converts automatically — see [`StateNetwork`](@ref))
- `behaviors::Dict{Symbol, Vector{Int}}`: Current behavior states
- `time::Float64`: Current simulation time within period
- `period::Int`: Current period (index of the starting wave); used to select the
  values of varying covariates
- `active::Union{Nothing, BitVector}`: Per-actor activity mask of the current
  period when the data have composition changes (`nothing` otherwise); inactive
  actors take no ministeps and their dyads are not candidates
"""
mutable struct NetworkState
    networks::Dict{Symbol, StateNetwork}
    behaviors::Dict{Symbol, Vector{Int}}
    time::Float64
    period::Int
    active::Union{Nothing, BitVector}

    function NetworkState()
        new(Dict{Symbol, StateNetwork}(), Dict{Symbol, Vector{Int}}(), 0.0, 1,
            nothing)
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
    state.active = _activity_mask(data, state.period)
    for (name, dep) in data.dependents
        if dep isa DependentNetwork
            state.networks[name] = StateNetwork(dep.networks[wave])
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
        s.networks[k] = _copy_state_network(v)
    end
    for (k, v) in state.behaviors
        s.behaviors[k] = copy(v)
    end
    s.time = state.time
    s.period = state.period
    s.active = state.active === nothing ? nothing : copy(state.active)
    s
end
