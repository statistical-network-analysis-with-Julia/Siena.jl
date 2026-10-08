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
# Example
```julia
using Siena
ns = NodeSet(3; names=["Ann", "Bob", "Cat"])
length(ns), ns.names[2]        # (3, "Bob")
```
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
# Example
```julia
using Siena
DependentBehavior(:mood, [[1, 2, 3], [2, 2, 3]]) isa AbstractDependent  # true
```
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
- `allow_only::Bool`: Whether up-only and down-only periods restrict the
  simulation (RSiena's `allowOnly`; default `true`, see below)
- `uponly::Vector{Bool}`, `downonly::Vector{Bool}`: Per period, whether the observed
  network only gains ties (only loses ties) between its two waves; all `false`
  when `allow_only=false`

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

# Up-only and down-only periods
As in RSiena (`sienaDependent(allowOnly = TRUE)`, the default), a period in which
the observed network only gains ties is *up-only*: in that period an actor can
create ties but never drop one. A period that only loses ties is *down-only*, and
actors can only drop ties. Structurally determined dyads and the diagonal are
ignored when the periods are classified. When **every** period is up-only (or every
period is down-only), [`get_effects`](@ref) does not offer the `outdegree` (density)
effect, as RSiena's `getEffects` does not. Pass `allow_only=false` to simulate every
period without the restriction (RSiena's `allowOnly = FALSE`).

# Undirected (symmetric) networks
`directed=false` declares a non-directed one-mode network (RSiena: a symmetric
`sienaDependent`). Every wave must be a symmetric matrix (and so must its
structural codes); a non-symmetric wave throws an `ArgumentError`, and a two-mode
network cannot be undirected. Such a network evolves under RSiena's default model
for non-directed networks, model type 2 (*unilateral initiative and reciprocal
confirmation*, the "forcing" model): the actor with the opportunity chooses an alter
using its objective function, and the tie `i–j` is created or dissolved in both
directions at once. Only the effects RSiena defines for symmetric networks are
available (see [`get_effects`](@ref)); their target statistics follow RSiena's
conventions for symmetric data (e.g. the degree/density statistic counts each edge
once).

# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 0 0 0]
w2 = [0 1 1; 0 0 1; 0 0 0]
dep = DependentNetwork(:advice, [w1, w2])
n_actors(dep), n_waves(dep)            # (3, 2)

u1 = [0 1 0; 1 0 0; 0 0 0]            # undirected: symmetric waves
u2 = [0 1 1; 1 0 0; 1 0 0]
DependentNetwork(:friends, [u1, u2]; directed=false).directed   # false
```

# Interoperability
The built-in NetworkCore.jl integration adds a
constructor taking a `Vector` of `Network` objects (one per wave); adjacency
matrices are extracted with `NetworkCore.as_matrix`, directedness and self-loop
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
    allow_only::Bool
    uponly::Vector{Bool}
    downonly::Vector{Bool}

    function DependentNetwork(
        name::Symbol,
        networks::Vector{<:AbstractMatrix{<:Real}};
        type::Symbol=:onemode,
        directed::Bool=true,
        allow_self_loops::Bool=false,
        nodeset1::Symbol=:actors,
        nodeset2::Union{Symbol, Nothing}=nothing,
        structural_zero::Int=10,
        structural_one::Int=11,
        allow_only::Bool=true
    )
        # Validate
        if isempty(networks)
            throw(ArgumentError("At least one network observation required"))
        end
        # Every type other than :onemode is treated as two-mode, so a misspelt type
        # must not get through silently.
        type in (:onemode, :twomode, :bipartite) || throw(ArgumentError(
            "dependent network :$name: type must be :onemode, :twomode or " *
            ":bipartite (a synonym of :twomode), got :$type"))
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
        for (w, net) in enumerate(networks)
            all(isinteger, net) || throw(ArgumentError(
                "invalid tie value in wave $w of dependent network :$name: entries " *
                "must be 0, 1, $structural_zero (structural zero), or " *
                "$structural_one (structural one)"))
        end
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
        if !directed
            type == :onemode || throw(ArgumentError(
                "dependent network :$name: a two-mode network cannot be undirected " *
                "(directed=false requires type=:onemode)"))
            for (w, m) in enumerate(int_networks)
                issymmetric(m) || throw(ArgumentError(
                    "dependent network :$name is declared undirected (directed=false) " *
                    "but wave $w is not a symmetric matrix; symmetrize it (e.g. " *
                    "max.(x, x')) or declare the network directed"))
                issymmetric(masks[w]) || throw(ArgumentError(
                    "dependent network :$name is declared undirected (directed=false) " *
                    "but the structural codes of wave $w are not symmetric"))
            end
        end
        structural = any_structural ? masks : BitMatrix[]
        up, down = allow_only ? _network_only_periods(int_networks, structural,
                                                      type == :onemode) :
                   (fill(false, length(int_networks) - 1),
                    fill(false, length(int_networks) - 1))
        new(name, int_networks, type, directed, allow_self_loops, nodeset1,
            nodeset2, structural, allow_only, up, down)
    end
end

"""
    has_structural(dep::DependentNetwork) -> Bool

Whether the dependent network contains any structurally determined dyads
(structural zeros/ones; see [`DependentNetwork`](@ref)).
# Example
```julia
using Siena
has_structural(DependentNetwork(:net, [[0 10; 1 0], [0 10; 0 0]]))   # true: 10 = structural zero
```
"""
has_structural(dep::DependentNetwork) = !isempty(dep.structural)

"""
    is_structural_dyad(dep::DependentNetwork, wave::Int, i::Int, j::Int) -> Bool

Whether the dyad `(i, j)` is structurally determined (structural zero or
one) at observation `wave`. Its determined face value is
`dep.networks[wave][i, j]`.
# Example
```julia
using Siena
dep = DependentNetwork(:net, [[0 11; 0 0], [0 11; 1 0]])
is_structural_dyad(dep, 1, 1, 2)       # true: 11 = structural one
```
"""
is_structural_dyad(dep::DependentNetwork, wave::Int, i::Int, j::Int) =
    !isempty(dep.structural) && dep.structural[wave][i, j]

"""
    n_structural_dyads(dep::DependentNetwork, wave::Int) -> Int

Number of structurally determined dyads at observation `wave`.
# Example
```julia
using Siena
n_structural_dyads(DependentNetwork(:net, [[0 10; 11 0], [0 10; 11 0]]), 1)    # 2
```
"""
n_structural_dyads(dep::DependentNetwork, wave::Int) =
    isempty(dep.structural) ? 0 : count(dep.structural[wave])

# The kind of network variable, in RSiena's effect-group terms: `:directed`
# (nonSymmetric), `:undirected` (symmetric) or `:twomode` (bipartite).
_network_kind(dep::DependentNetwork) =
    dep.type == :onemode ? (dep.directed ? :directed : :undirected) : :twomode

_is_undirected(dep::DependentNetwork) = dep.type == :onemode && !dep.directed

# Structural mask relevant to a simulation state: the mask of the state's
# period-start wave, or `nothing` when the variable has no structural dyads.
@inline function _structural_mask(dep::DependentNetwork, period::Int)
    isempty(dep.structural) && return nothing
    return dep.structural[period]
end

# RSiena's up-only/down-only classification of each period of a network panel
# (sienaDataCreate): the diagonal of a one-mode network and the dyads that are
# structurally determined at either wave are ignored. A period without any change
# is both up-only and down-only.
function _network_only_periods(waves::Vector{Matrix{Int}}, masks::Vector{BitMatrix},
                               onemode::Bool)
    n_p = length(waves) - 1
    up = fill(false, n_p)
    down = fill(false, n_p)
    for p in 1:n_p
        a, b = waves[p], waves[p + 1]
        gains = losses = false
        for j in axes(a, 2), i in axes(a, 1)
            onemode && i == j && continue
            !isempty(masks) && (masks[p][i, j] || masks[p + 1][i, j]) && continue
            d = b[i, j] - a[i, j]
            gains |= d > 0
            losses |= d < 0
        end
        up[p] = !losses
        down[p] = !gains
    end
    return up, down
end

# Per period, the same classification for a behaviour (values only rise / only fall).
function _behavior_only_periods(values::Vector{Vector{Int}})
    n_p = length(values) - 1
    up = [all(values[p + 1] .>= values[p]) for p in 1:n_p]
    down = [all(values[p + 1] .<= values[p]) for p in 1:n_p]
    return up, down
end

# Whether the simulation of `period` may only increase (`_up_only`) or only decrease
# (`_down_only`) the variable. Periods beyond the observed ones are unrestricted.
@inline _up_only(dep, period::Int) =
    dep.allow_only && 1 <= period <= length(dep.uponly) && @inbounds dep.uponly[period]
@inline _down_only(dep, period::Int) =
    dep.allow_only && 1 <= period <= length(dep.downonly) && @inbounds dep.downonly[period]

# Every period only goes up, or every period only goes down: RSiena then leaves the
# density (network) or linear shape (behaviour) effect out of the model.
_all_one_direction(dep) = dep.allow_only && !isempty(dep.uponly) &&
                          (all(dep.uponly) || all(dep.downonly))

"""
    n_waves(dn::DependentNetwork)

Return the number of observation waves.
# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]
w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
n_waves(data.dependents[:advice])        # 2
```
"""
n_waves(dn::DependentNetwork) = length(dn.networks)

"""
    n_actors(dn::DependentNetwork)

Return the number of actors (rows) in the network.
# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]
w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
n_actors(data.dependents[:advice])       # 3
```
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
- `allow_only::Bool`: Whether up-only and down-only periods restrict the
  simulation (RSiena's `allowOnly`; default `true`)
- `uponly::Vector{Bool}`, `downonly::Vector{Bool}`: Per period, whether no observed
  value falls (no observed value rises) between the period's two waves; all `false`
  when `allow_only=false`

As in RSiena, in an up-only period actors can only raise their value (or keep it),
and in a down-only period only lower it. When every period is up-only (or every
period is down-only), [`get_effects`](@ref) does not offer the `linear` shape effect,
as RSiena's `getEffects` does not. Pass `allow_only=false` to lift the restriction.
# Example
```julia
using Siena
dep = DependentBehavior(:mood, [[1, 2, 3, 2], [2, 2, 3, 1]])
(dep.min_val, dep.max_val, dep.mean_val)       # (1, 3, 2.0)
```
"""
mutable struct DependentBehavior <: AbstractDependent
    name::Symbol
    values::Vector{Vector{Int}}
    min_val::Int
    max_val::Int
    nodeset::Symbol
    mean_val::Float64
    sim_mean::Float64
    allow_only::Bool
    uponly::Vector{Bool}
    downonly::Vector{Bool}

    function DependentBehavior(
        name::Symbol,
        values::Vector{<:AbstractVector{<:Integer}};
        min_val::Union{Int, Nothing}=nothing,
        max_val::Union{Int, Nothing}=nothing,
        nodeset::Symbol=:actors,
        allow_only::Bool=true
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
        up, down = allow_only ? _behavior_only_periods(int_values) :
                   (fill(false, length(int_values) - 1), fill(false, length(int_values) - 1))
        new(name, int_values, min_v, max_v, nodeset, mean_v, sim_m, allow_only, up, down)
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
# Example
```julia
using Siena
ConstantCovariate(:age, [20, 30, 40]) isa AbstractCovariate     # true
```
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

# Range of a set of covariate values (the denominator of RSiena's similarity
# scores). Computed ONCE, at construction: the similarity effects read it on every
# candidate dyad of every ministep.
function _value_range(vals)
    ok = filter(!isnan, collect(vals))
    return isempty(ok) ? 0.0 : maximum(ok) - minimum(ok)
end

_is_na(v) = v === missing || (v isa AbstractFloat && isnan(v))

# Missing covariate values (`missing` or `NaN`) under the `missing=` policy.
# `:error` (the default) refuses them: a single NaN used to turn the whole centred
# covariate into NaN and the fit then failed deep inside LAPACK. `:mean` imputes
# the mean of the observed values -- RSiena's default rule ("missing covariate
# data are replaced by the variable's global mean") -- and the number of imputed
# values is recorded on the covariate and listed by `approximations(fit)`.
function _impute_missing(name::Symbol, vals::AbstractArray, policy::Symbol,
                         kind::AbstractString)
    policy in (:error, :mean) || throw(ArgumentError(
        "covariate :$name: missing must be :error or :mean, got :$policy"))
    na = findall(_is_na, vals)
    out = Array{Float64}(undef, size(vals))
    isempty(na) && (out .= vals; return out, 0)
    if policy === :error
        throw(ArgumentError(
            "$kind :$name has $(length(na)) missing value(s) (missing or NaN). " *
            "Siena.jl does not model missing covariate values; pass " *
            "missing=:mean to impute the mean of the observed values (RSiena's " *
            "default rule), or impute them yourself"))
    end
    observed = [Float64(v) for v in vals if !_is_na(v)]
    isempty(observed) && throw(ArgumentError(
        "$kind :$name has no observed values; nothing to impute from"))
    m = mean(observed)
    for idx in eachindex(vals)
        out[idx] = _is_na(vals[idx]) ? m : Float64(vals[idx])
    end
    return out, length(na)
end

"""
    ConstantCovariate(name::Symbol, values::AbstractVector; nodeset=:actors,
                      center=true, missing=:error)

A covariate that is constant across all waves (RSiena's `coCovar`).

# Fields
- `name::Symbol`: Covariate name
- `values::Vector{Float64}`: Values for each actor (centred when `center=true`)
- `nodeset::Symbol`: ID of the node set
- `centered::Bool`: Whether values are centered
- `mean::Float64`: Mean value (for centering)
- `sim_mean::Float64`: Mean pairwise similarity (RSiena's ``\\widehat{sim}``)
- `range::Float64`: Range of the values (the similarity denominator)
- `n_imputed::Int`: Number of missing values imputed under `missing=:mean`

Missing values (`missing` or `NaN`) are refused by default; `missing=:mean`
replaces them by the mean of the observed values, RSiena's default rule. (RSiena
additionally leaves such actors out of the target statistics; Siena.jl does not,
which `approximations(fit)` records for a fit using an imputed covariate.)

# Example
```julia
using Siena
smoke = ConstantCovariate(:smoke, [1, 2, 2, 3])
smoke.values                          # centred: [-1.0, 0.0, 0.0, 1.0]
ConstantCovariate(:age, [20.0, NaN, 30.0]; missing=:mean).n_imputed   # 1
```
"""
struct ConstantCovariate <: AbstractCovariate
    name::Symbol
    values::Vector{Float64}
    nodeset::Symbol
    centered::Bool
    mean::Float64
    sim_mean::Float64
    range::Float64
    n_imputed::Int

    function ConstantCovariate(name::Symbol, values::AbstractVector;
                               nodeset::Symbol=:actors, center::Bool=true,
                               missing::Symbol=:error)
        all(v -> v isa Real || v === Base.missing, values) || throw(ArgumentError(
            "covariate :$name: values must be real numbers (or missing)"))
        fvals, n_imp = _impute_missing(name, values, missing, "covariate")
        m = mean(fvals)
        centered_vals = center ? fvals .- m : fvals
        new(name, centered_vals, nodeset, center, m, _similarity_mean(centered_vals),
            _value_range(centered_vals), n_imp)
    end
end

"""
    VaryingCovariate(name::Symbol, values::Vector{<:AbstractVector}; nodeset=:actors,
                     center=true, missing=:error)

A covariate that varies across waves (RSiena's `varCovar`); `values[w]` holds the
actor values used in period `w`.

# Fields
- `name::Symbol`: Covariate name
- `values::Vector{Vector{Float64}}`: Values for each actor at each wave
- `nodeset::Symbol`: ID of the node set
- `centered::Bool`: Whether values are centered
- `mean::Float64`: Overall mean value
- `sim_mean::Float64`: Mean pairwise similarity, averaged over the waves
- `range::Float64`: Range over all waves (the similarity denominator)
- `n_imputed::Int`: Number of missing values imputed under `missing=:mean`

Missing values are handled as for [`ConstantCovariate`](@ref) (imputation uses the
global mean over all waves, as RSiena does).

# Example
```julia
using Siena
VaryingCovariate(:mood, [[1, 2, 3], [2, 2, 4]]).mean     # 2.333…
```
"""
struct VaryingCovariate <: AbstractCovariate
    name::Symbol
    values::Vector{Vector{Float64}}
    nodeset::Symbol
    centered::Bool
    mean::Float64
    sim_mean::Float64
    range::Float64
    n_imputed::Int

    function VaryingCovariate(name::Symbol, values::Vector{<:AbstractVector};
                              nodeset::Symbol=:actors, center::Bool=true,
                              missing::Symbol=:error)
        isempty(values) && throw(ArgumentError("covariate :$name: no waves given"))
        lens = length.(values)
        all(==(lens[1]), lens) || throw(ArgumentError(
            "covariate :$name: every wave must give one value per actor"))
        all(v -> v isa Real || v === Base.missing, Iterators.flatten(values)) ||
            throw(ArgumentError("covariate :$name: values must be real numbers (or missing)"))
        mat = reduce(hcat, [collect(v) for v in values])
        fmat, n_imp = _impute_missing(name, mat, missing, "covariate")
        fvals = [fmat[:, w] for w in axes(fmat, 2)]
        m = mean(fmat)
        centered_vals = center ? [v .- m for v in fvals] : fvals
        r = _value_range(Iterators.flatten(centered_vals))
        sim_m = mean(_similarity_mean(v, r) for v in centered_vals)
        new(name, centered_vals, nodeset, center, m, sim_m, r, n_imp)
    end
end

# Mean of a dyadic covariate. For a one-mode (square, same node set) covariate the
# diagonal is excluded, as in RSiena: a self-tie is never a candidate, so including
# the diagonal shifted the centring constant and hence the outdegree coefficient.
function _dyad_mean(mats, one_mode::Bool)
    total = 0.0
    count = 0
    for M in mats, j in axes(M, 2), i in axes(M, 1)
        one_mode && i == j && continue
        total += M[i, j]
        count += 1
    end
    return count == 0 ? 0.0 : total / count
end

_one_mode_dyadic(M::AbstractMatrix, nodeset1::Symbol, nodeset2::Symbol) =
    nodeset1 == nodeset2 && size(M, 1) == size(M, 2)

"""
    ConstantDyadCovariate(name::Symbol, values::AbstractMatrix; nodeset1=:actors,
                          nodeset2=:actors, center=true, missing=:error)

A dyadic covariate that is constant across waves (RSiena's `coDyadCovar`).

# Fields
- `name::Symbol`: Covariate name
- `values::Matrix{Float64}`: Values for each dyad (centred when `center=true`)
- `nodeset1::Symbol`: ID of row node set
- `nodeset2::Symbol`: ID of column node set
- `centered::Bool`: Whether values are centered
- `mean::Float64`: Mean value (over the off-diagonal entries for a one-mode
  covariate, as in RSiena)
- `n_imputed::Int`: Number of missing values imputed under `missing=:mean`

# Example
```julia
using Siena
W = [0 1 2; 1 0 1; 2 1 0]
ConstantDyadCovariate(:distance, W).mean       # 4/3: the diagonal is excluded
```
"""
struct ConstantDyadCovariate <: AbstractCovariate
    name::Symbol
    values::Matrix{Float64}
    nodeset1::Symbol
    nodeset2::Symbol
    centered::Bool
    mean::Float64
    n_imputed::Int

    function ConstantDyadCovariate(name::Symbol, values::AbstractMatrix;
                                   nodeset1::Symbol=:actors, nodeset2::Symbol=:actors,
                                   center::Bool=true, missing::Symbol=:error)
        all(v -> v isa Real || v === Base.missing, values) || throw(ArgumentError(
            "dyadic covariate :$name: values must be real numbers (or missing)"))
        one_mode = _one_mode_dyadic(values, nodeset1, nodeset2)
        if one_mode
            # The diagonal is never used; do not let it trip the missing check.
            values = [i == j ? 0.0 : values[i, j] for i in axes(values, 1),
                      j in axes(values, 2)]
        end
        fvals, n_imp = _impute_missing(name, values, missing, "dyadic covariate")
        m = _dyad_mean((fvals,), one_mode)
        centered_vals = center ? fvals .- m : fvals
        new(name, centered_vals, nodeset1, nodeset2, center, m, n_imp)
    end
end

"""
    VaryingDyadCovariate(name::Symbol, values::Vector{<:AbstractMatrix};
                         nodeset1=:actors, nodeset2=:actors, center=true,
                         missing=:error)

A dyadic covariate that varies across waves (RSiena's `varDyadCovar`).

# Fields
- `name::Symbol`: Covariate name
- `values::Vector{Matrix{Float64}}`: Values for each dyad at each wave
- `nodeset1::Symbol`: ID of row node set
- `nodeset2::Symbol`: ID of column node set
- `centered::Bool`: Whether values are centered
- `mean::Float64`: Overall mean value (off-diagonal for one-mode covariates)
- `n_imputed::Int`: Number of missing values imputed under `missing=:mean`

# Example
```julia
using Siena
W1 = [0 1; 1 0]; W2 = [0 2; 2 0]
VaryingDyadCovariate(:contact, [W1, W2]).mean   # 1.5
```
"""
struct VaryingDyadCovariate <: AbstractCovariate
    name::Symbol
    values::Vector{Matrix{Float64}}
    nodeset1::Symbol
    nodeset2::Symbol
    centered::Bool
    mean::Float64
    n_imputed::Int

    function VaryingDyadCovariate(name::Symbol, values::Vector{<:AbstractMatrix};
                                  nodeset1::Symbol=:actors, nodeset2::Symbol=:actors,
                                  center::Bool=true, missing::Symbol=:error)
        isempty(values) && throw(ArgumentError("dyadic covariate :$name: no waves given"))
        all(v -> v isa Real || v === Base.missing, Iterators.flatten(values)) ||
            throw(ArgumentError("dyadic covariate :$name: values must be real numbers (or missing)"))
        one_mode = _one_mode_dyadic(values[1], nodeset1, nodeset2)
        if one_mode
            values = [[i == j ? 0.0 : V[i, j] for i in axes(V, 1), j in axes(V, 2)]
                      for V in values]
        end
        stacked = cat(values...; dims=3)
        fstack, n_imp = _impute_missing(name, stacked, missing, "dyadic covariate")
        fvals = [fstack[:, :, w] for w in axes(fstack, 3)]
        m = _dyad_mean(fvals, one_mode)
        centered_vals = center ? [v .- m for v in fvals] : fvals
        new(name, centered_vals, nodeset1, nodeset2, center, m, n_imp)
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
# Example
```julia
using Siena
cc = CompositionChange([(4, 2, :join), (7, 3, :leave)])
is_present(cc, 4, 1), is_present(cc, 4, 2)      # (false, true)
```
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
# Example
```julia
using Siena
cc = CompositionChange()
add_change!(cc, 3, 2, :leave)
is_present(cc, 3, 2)          # false
```
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
# Example
```julia
using Siena
cc = CompositionChange([(2, 3, :leave)])
is_present(cc, 2, 2), is_present(cc, 2, 3)      # (true, false)
```
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
# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]
w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
data.n_waves, length(data.dependents)       # (2, 1)
```
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
# Example
```julia
using Siena
data = siena_data()
add_nodeset!(data, NodeSet(3))
length(data.nodesets[:actors])     # 3
```
"""
function add_nodeset!(data::SienaData, ns::NodeSet)
    data.nodesets[ns.id] = ns
    data
end

"""
    add_dependent!(data::SienaData, dep::AbstractDependent)

Add a dependent variable to the data.
# Example
```julia
using Siena
data = siena_data()
add_dependent!(data, DependentBehavior(:mood, [[1, 2, 3], [2, 2, 3]]))
data.n_waves      # 2
```
"""
function add_dependent!(data::SienaData, dep::AbstractDependent)
    nw = n_waves(dep)
    if data.n_waves == 0
        data.n_waves = nw
    elseif data.n_waves != nw
        throw(ArgumentError("Number of waves must be consistent (expected $(data.n_waves), got $nw)"))
    end
    haskey(data.covariates, dep.name) && throw(ArgumentError(
        "the name :$(dep.name) is already used by a covariate; dependent variables " *
        "and covariates need distinct names"))
    data.dependents[dep.name] = dep
    data
end

"""
    add_covariate!(data::SienaData, cov::AbstractCovariate)

Add a covariate to the data.
# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]
w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40]))
haskey(data.covariates, :age)        # true
```
"""
function add_covariate!(data::SienaData, cov::AbstractCovariate)
    haskey(data.dependents, cov.name) && throw(ArgumentError(
        "the name :$(cov.name) is already used by a dependent variable; covariates " *
        "and dependent variables need distinct names (a dependent behaviour can " *
        "itself be used as a covariate of network effects)"))
    data.covariates[cov.name] = cov
    data
end

"""
    add_composition_change!(data::SienaData, cc::CompositionChange)

Attach composition-change information (actors joining/leaving; see
[`CompositionChange`](@ref)) to the data. The counterpart of supplying a
`sienaCompositionChange` object in RSiena.

During estimation an actor contributes to a period only when present at both of
its endpoint waves. Absent actors get no ministep opportunities, their dyads are
excluded from the candidate sets, and their rows/columns are excluded from the
target and simulated moment statistics and from the observed change (rate)
distances. This is a whole-period (listwise) approximation, **not** RSiena's
treatment, which places joining and leaving at times within the period (Huisman &
Snijders 2003): an actor who leaves during a period is dropped from that whole
period here, also as an alter of the others. `approximations(fit)` records it.

The data fix the ranges the events must fall in, so this is where they are checked:
every actor must be an actor of the data (`1:n_actors`) and every wave an observation
wave (`1:n_waves`). An out-of-range actor or wave throws — silently ignoring it would
mean estimating a model whose composition differs from the one that was requested.
The internal consistency of the sequence itself is checked earlier, by
[`CompositionChange`](@ref)/[`add_change!`](@ref).
# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]
w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
add_composition_change!(data, CompositionChange([(3, 2, :join)]))
is_present(data.composition_change, 3, 1)        # false
```
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
maintained out-/indegree vectors and sorted outgoing/incoming adjacency lists.

Indexing reads and writes behave exactly like a 0/1 `Matrix{Int}`; every
`setindex!` updates the degree vectors, so `_row_sum`/`_col_sum` (the degree
lookups of the effect hot loops) are O(1) instead of O(n) scans. Writing a
value other than 0 or 1 throws.
# Example
```julia
using Siena
sn = StateNetwork([0 1 1; 0 0 1; 0 0 0])
sn[2, 1] = 1
sn.outdeg, sn.indeg       # ([2, 2, 0], [1, 1, 2])
```
"""
struct StateNetwork <: AbstractMatrix{Int}
    m::Matrix{Int8}
    outdeg::Vector{Int}
    indeg::Vector{Int}
    outneighbors::Vector{Vector{Int}}
    inneighbors::Vector{Vector{Int}}
end

function StateNetwork(m::AbstractMatrix{<:Integer})
    all(v -> v == 0 || v == 1, m) ||
        throw(ArgumentError("state networks hold 0/1 tie values only"))
    m8 = Matrix{Int8}(m)
    outgoing = [findall(!iszero, view(m8, i, :)) for i in axes(m8, 1)]
    incoming = [findall(!iszero, view(m8, :, j)) for j in axes(m8, 2)]
    return StateNetwork(m8, length.(outgoing), length.(incoming), outgoing, incoming)
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
        out = sn.outneighbors[i]
        inc = sn.inneighbors[j]
        if b == 1
            insert!(out, searchsortedfirst(out, j), j)
            insert!(inc, searchsortedfirst(inc, i), i)
        else
            deleteat!(out, searchsortedfirst(out, j))
            deleteat!(inc, searchsortedfirst(inc, i))
        end
    end
    return sn
end

# Note: no `Base.copy` override — generic AbstractArray `copy` yields a plain
# `Matrix{Int}`, which is what callers extracting a wave matrix expect.
_copy_state_network(sn::StateNetwork) =
    StateNetwork(copy(sn.m), copy(sn.outdeg), copy(sn.indeg),
                 copy.(sn.outneighbors), copy.(sn.inneighbors))

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
# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]
w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
state = initialize!(NetworkState(), data, 1)
state.networks[:advice] == w1        # true
```
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
# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]
w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
state = initialize!(NetworkState(), data, 2; period=1)
state.period, state.networks[:advice] == w2      # (1, true)
```
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
# Example
```julia
using Siena
w1 = [0 1 0; 0 0 1; 1 0 0]
w2 = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w1, w2]))
state = initialize!(NetworkState(), data, 1)
copy_ = snapshot(state)
copy_.networks[:advice] !== state.networks[:advice]      # independent copy
```
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
