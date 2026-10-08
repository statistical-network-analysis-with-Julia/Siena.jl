"""
Goodness of fit assessment for SAOM.

The auxiliary statistics, their default levels and the joining over periods follow
RSiena's `sienaGOF`; the overall test pools the observed vector with the simulated
ones (see [`siena_gof`](@ref)).
"""

#==============================================================================#
# GOF Statistics
#==============================================================================#

"""
    AbstractGOFStatistic

Abstract type for goodness of fit statistics.

# Example
```julia
using Siena
IndegreeDistribution(:friendship) isa AbstractGOFStatistic     # true
```
"""
abstract type AbstractGOFStatistic end

"""
    IndegreeDistribution(variable; levls=0:8, cumulative=true) <: AbstractGOFStatistic

Indegree distribution, as RSiena's `sienaGOF` auxiliary statistic
`IndegreeDistribution`: for every level ``k`` in `levls`, the number of actors with
indegree ``\\le k`` (`cumulative=true`, RSiena's default) or ``= k``.

The levels are FIXED in advance and never derived from the observed network. Before
0.2 they defaulted to `0:maximum(observed indegree)`, which dropped simulated mass
above the observed maximum and made the test statistic depend on the data; with the
in-sample covariance the test then rejected a correct model 15.7 % of the time at
the 5 % level.

# Example
```julia
using Siena
IndegreeDistribution(:friendship).levls          # 0:8
IndegreeDistribution(:friendship; levls=0:4, cumulative=false)
```
"""
struct IndegreeDistribution <: AbstractGOFStatistic
    variable::Symbol
    levls::Vector{Int}
    cumulative::Bool

    IndegreeDistribution(variable::Symbol; levls::AbstractVector{<:Integer}=0:8,
                         cumulative::Bool=true) =
        new(variable, _check_levels(levls), cumulative)
end

"""
    OutdegreeDistribution(variable; levls=0:8, cumulative=true) <: AbstractGOFStatistic

Outdegree distribution, as RSiena's `OutdegreeDistribution` (see
[`IndegreeDistribution`](@ref) for the levels).

# Example
```julia
using Siena
OutdegreeDistribution(:friendship; levls=0:10).levls     # 0:10
```
"""
struct OutdegreeDistribution <: AbstractGOFStatistic
    variable::Symbol
    levls::Vector{Int}
    cumulative::Bool

    OutdegreeDistribution(variable::Symbol; levls::AbstractVector{<:Integer}=0:8,
                          cumulative::Bool=true) =
        new(variable, _check_levels(levls), cumulative)
end

function _check_levels(levls)
    isempty(levls) && throw(ArgumentError("levls must not be empty"))
    issorted(levls) && allunique(levls) ||
        throw(ArgumentError("levls must be strictly increasing"))
    return collect(Int, levls)
end

"""
    TriadCensus(variable) <: AbstractGOFStatistic

Full 16-type Davis–Leinhardt triad census (003, 012, 102, 021D, 021U, 021C, 111D,
111U, 030T, 030C, 201, 120D, 120U, 120C, 210, 300), as RSiena's `TriadCensus`.

# Example
```julia
using Siena
TriadCensus(:friendship).variable                # :friendship
```
"""
struct TriadCensus <: AbstractGOFStatistic
    variable::Symbol
end

"""
    GeodesicDistribution(variable; max_dist=5, cumulative=true) <: AbstractGOFStatistic

Geodesic (shortest path) distribution, as RSiena's `GeodesicDistribution`: for
``k = 1, …, `` `max_dist`, the number of ordered pairs at distance ``\\le k``
(`cumulative=true`, RSiena's default) or ``= k``; the non-cumulative form adds the
pairs that are farther apart or unreachable.

# Example
```julia
using Siena
GeodesicDistribution(:friendship; max_dist=4).max_dist      # 4
```
"""
struct GeodesicDistribution <: AbstractGOFStatistic
    variable::Symbol
    max_dist::Int
    cumulative::Bool

    function GeodesicDistribution(variable::Symbol; max_dist::Int=5,
                                  cumulative::Bool=true)
        max_dist >= 1 || throw(ArgumentError("max_dist must be >= 1"))
        new(variable, max_dist, cumulative)
    end
end

"""
    BehaviorDistribution(variable; levls=nothing, cumulative=true) <: AbstractGOFStatistic

Behaviour distribution, as RSiena's `BehaviorDistribution`: for every level of the
behaviour's range (or of `levls`), the number of actors with a value ``\\le`` that
level (`cumulative=true`) or ``=`` it.

# Example
```julia
using Siena
BehaviorDistribution(:alcohol; cumulative=false).cumulative      # false
```
"""
struct BehaviorDistribution <: AbstractGOFStatistic
    variable::Symbol
    levls::Union{Vector{Int}, Nothing}
    cumulative::Bool

    BehaviorDistribution(variable::Symbol;
                         levls::Union{AbstractVector{<:Integer}, Nothing}=nothing,
                         cumulative::Bool=true) =
        new(variable, levls === nothing ? nothing : _check_levels(levls), cumulative)
end

#==============================================================================#
# Statistic Computation
#==============================================================================#

function _level_counts(values::AbstractVector{<:Integer}, levls::Vector{Int},
                       cumulative::Bool)
    return [cumulative ? count(<=(l), values) : count(==(l), values) for l in levls]
end

_level_labels(levls::Vector{Int}, cumulative::Bool) =
    [cumulative ? "≤$l" : string(l) for l in levls]

"""
    compute_gof_statistic(stat::AbstractGOFStatistic, state::NetworkState,
                          data::SienaData) -> (labels, counts)

Evaluate an auxiliary GOF statistic on one network/behaviour state: the labels of
its components and the integer count of each.

# Example
```julia
using Siena
w = [0 1 1; 0 0 1; 0 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:advice, [w, w]))
state = initialize!(NetworkState(), data, 1)
compute_gof_statistic(IndegreeDistribution(:advice; levls=0:2, cumulative=false),
                      state, data)            # (["0", "1", "2"], [1, 1, 1])
```
"""
function compute_gof_statistic(stat::IndegreeDistribution, state::NetworkState,
                              data::SienaData)
    net = state.networks[stat.variable]
    indegrees = [_col_sum(net, j) for j in 1:size(net, 2)]
    return _level_labels(stat.levls, stat.cumulative),
           _level_counts(indegrees, stat.levls, stat.cumulative)
end

function compute_gof_statistic(stat::OutdegreeDistribution, state::NetworkState,
                              data::SienaData)
    net = state.networks[stat.variable]
    outdegrees = [_row_sum(net, i) for i in 1:size(net, 1)]
    return _level_labels(stat.levls, stat.cumulative),
           _level_counts(outdegrees, stat.levls, stat.cumulative)
end

const TRIAD_LABELS = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U",
                      "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]

# Classify the directed triad {a, b, c} into one of the 16 Davis–Leinhardt M-A-N
# classes (1-based index into TRIAD_LABELS).
function _triad_type(net::AbstractMatrix{Int}, a::Int, b::Int, c::Int)
    mutual = 0
    asym_arcs = Tuple{Int, Int}[]
    mutual_pair = (0, 0)

    for (i, j) in ((a, b), (a, c), (b, c))
        y_ij = net[i, j] == 1
        y_ji = net[j, i] == 1
        if y_ij && y_ji
            mutual += 1
            mutual_pair = (i, j)
        elseif y_ij
            push!(asym_arcs, (i, j))
        elseif y_ji
            push!(asym_arcs, (j, i))
        end
    end

    A = length(asym_arcs)

    if mutual == 3
        return 16                     # 300
    elseif mutual == 2
        return A == 1 ? 15 : 11       # 210 : 201
    elseif mutual == 1
        if A == 0
            return 3                  # 102
        elseif A == 1
            # 111D: A<->B<-C (arc points into the mutual pair)
            # 111U: A<->B->C (arc points out of the mutual pair)
            _, d = asym_arcs[1]
            return (d == mutual_pair[1] || d == mutual_pair[2]) ? 7 : 8
        else  # A == 2
            s1, d1 = asym_arcs[1]
            s2, d2 = asym_arcs[2]
            s1 == s2 && return 12     # 120D (common source)
            d1 == d2 && return 13     # 120U (common sink)
            return 14                 # 120C (chain)
        end
    else  # mutual == 0
        if A == 0
            return 1                  # 003
        elseif A == 1
            return 2                  # 012
        elseif A == 2
            s1, d1 = asym_arcs[1]
            s2, d2 = asym_arcs[2]
            s1 == s2 && return 4      # 021D (out-star)
            d1 == d2 && return 5      # 021U (in-star)
            return 6                  # 021C (path)
        else  # A == 3
            # Cyclic if every vertex is the source of exactly one arc
            sources = (asym_arcs[1][1], asym_arcs[2][1], asym_arcs[3][1])
            return allunique(sources) ? 10 : 9   # 030C : 030T
        end
    end
end

function compute_gof_statistic(stat::TriadCensus, state::NetworkState,
                              data::SienaData)
    net = state.networks[stat.variable]
    n = size(net, 1)
    counts = zeros(Int, 16)
    for i in 1:n, j in (i+1):n, k in (j+1):n
        counts[_triad_type(net, i, j, k)] += 1
    end
    return copy(TRIAD_LABELS), counts
end

function compute_gof_statistic(stat::GeodesicDistribution, state::NetworkState,
                              data::SienaData)
    net = state.networks[stat.variable]
    n = size(net, 1)
    dist_counts = zeros(Int, stat.max_dist)
    beyond = 0
    distances = fill(-1, n)
    queue = Int[]
    for i in 1:n
        fill!(distances, -1)
        distances[i] = 0
        empty!(queue)
        push!(queue, i)
        head = 1
        while head <= length(queue)
            curr = queue[head]
            head += 1
            distances[curr] >= stat.max_dist && continue
            for j in 1:n
                if net[curr, j] == 1 && distances[j] == -1
                    distances[j] = distances[curr] + 1
                    push!(queue, j)
                end
            end
        end
        for j in 1:n
            i == j && continue
            d = distances[j]
            if d == -1
                beyond += 1
            else
                dist_counts[d] += 1
            end
        end
    end
    if stat.cumulative
        return ["≤$k" for k in 1:stat.max_dist], cumsum(dist_counts)
    end
    return vcat(string.(1:stat.max_dist), [">$(stat.max_dist) or unreachable"]),
           vcat(dist_counts, [beyond])
end

function compute_gof_statistic(stat::BehaviorDistribution, state::NetworkState,
                              data::SienaData)
    beh = state.behaviors[stat.variable]
    dep = data.dependents[stat.variable]::DependentBehavior
    levls = stat.levls === nothing ? collect(dep.min_val:dep.max_val) : stat.levls
    return _level_labels(levls, stat.cumulative),
           _level_counts(beh, levls, stat.cumulative)
end

#==============================================================================#
# GOF Result
#==============================================================================#

"""
    SienaGOFResult{S<:AbstractGOFStatistic}

Result of a [`siena_gof`](@ref) goodness of fit assessment for ONE statistic
(RSiena-style, keeping the Mahalanobis machinery). Convertible to the
ecosystem-wide `GOFResult` (from NetworkCore.jl) via `GOFResult(result)`, which is
also what the shared [`gof`](@ref) generic returns; display goes through the
shared GOF table.

# Fields
- `statistic::S`: The GOF statistic used
- `labels::Vector{String}`: Labels of the statistic's components
- `observed::Vector{Int}`: Observed values (summed over `periods`)
- `simulated::Matrix{Int}`: Simulated values (n_sim × n_components)
- `p_values::Vector{Float64}`: Monte-Carlo two-sided p-values per component, with
  the `(1 + k)/(N + 1)` estimator (never exactly 0)
- `mahalanobis::Float64`: Mahalanobis distance of the observed vector
- `p_overall::Float64`: Monte-Carlo p-value of the Mahalanobis distance
- `periods::Vector{Int}`: The periods whose end states were compared (all of them,
  joined, by default — RSiena's `join=TRUE`)

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
detail = siena_gof(fit, data, OutdegreeDistribution(:friendship); n_sim=50,
                   rng=Xoshiro(2))
detail.periods             # [1, 2]: both periods, joined
```
"""
struct SienaGOFResult{S<:AbstractGOFStatistic}
    statistic::S
    labels::Vector{String}
    observed::Vector{Int}
    simulated::Matrix{Int}
    p_values::Vector{Float64}
    mahalanobis::Float64
    p_overall::Float64
    periods::Vector{Int}
end

# Table heading of a GOF statistic in the shared GOFResult display.
_gof_statistic_name(s::IndegreeDistribution) = "indegree distribution ($(s.variable))"
_gof_statistic_name(s::OutdegreeDistribution) = "outdegree distribution ($(s.variable))"
_gof_statistic_name(s::TriadCensus) = "triad census ($(s.variable))"
_gof_statistic_name(s::GeodesicDistribution) = "geodesic distribution ($(s.variable))"
_gof_statistic_name(s::BehaviorDistribution) = "behavior distribution ($(s.variable))"
_gof_statistic_name(s::AbstractGOFStatistic) = string(typeof(s).name.name)

# One SienaGOFResult -> the shared per-statistic GOF container.
_gof_statistic(r::SienaGOFResult) =
    GOFStatistic(_gof_statistic_name(r.statistic), r.labels,
                 Float64.(r.observed), Float64.(r.simulated); p_values=r.p_values)

"""
    GOFResult(result::SienaGOFResult; model="SAOM") -> GOFResult

Convert an RSiena-style [`SienaGOFResult`](@ref) to the ecosystem-wide
`GOFResult` (from NetworkCore.jl), carrying the Monte-Carlo p-value of the
Mahalanobis distance as the overall p-value.

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
detail = siena_gof(fit, data, TriadCensus(:friendship); n_sim=20, rng=Xoshiro(4))
GOFResult(detail).p_overall == detail.p_overall      # true
```
"""
GOFResult(result::SienaGOFResult; model::AbstractString="SAOM") =
    GOFResult([_gof_statistic(result)]; model=model, p_overall=result.p_overall)

function Base.show(io::IO, result::SienaGOFResult)
    show(io, GOFResult(result))
    @printf(io, "Mahalanobis distance: %.3f\n", result.mahalanobis)
end

#==============================================================================#
# Main GOF Function
#==============================================================================#

# Overall Monte-Carlo test of the observed vector against the simulated ones. The
# mean and covariance are computed from the observed vector POOLED with the
# simulations, and the observed distance is ranked among all N + 1 distances: under
# the fitted model the N + 1 vectors are exchangeable, so the p-value
# (1 + #{simulated distance >= observed}) / (N + 1) is an exact Monte-Carlo p-value.
# A Moore-Penrose inverse handles the linear constraints of frequency tables
# (RSiena's ginv); no ridge is added.
function _pooled_mahalanobis_test(observed::AbstractVector{<:Real},
                                  simulated::AbstractMatrix{<:Real})
    X = vcat(permutedims(observed), simulated)
    center = vec(mean(X, dims=1))
    C = cov(X)
    Cinv = pinv(Symmetric(C); rtol=sqrt(eps(Float64)) * size(C, 1))
    dist(x) = (d = x .- center; sqrt(max(0.0, dot(d, Cinv * d))))
    obs_dist = dist(observed)
    sim_dists = [dist(view(simulated, s, :)) for s in axes(simulated, 1)]
    # Tolerance for exact ties of distances computed through a pseudo-inverse.
    tol = 1e-10 * max(1.0, obs_dist)
    p = (1 + count(>=(obs_dist - tol), sim_dists)) / (size(simulated, 1) + 1)
    return obs_dist, p
end

# The GOF engine, on a model given by its effects and full parameter vector.
function _gof_core(data::SienaData, effects::SienaEffects, θ::Vector{Float64},
                   statistic::AbstractGOFStatistic; n_sim::Int, rng::AbstractRNG,
                   variables=nothing, condvar=nothing,
                   period::Union{Nothing, Integer}=nothing)
    n_sim >= 2 || throw(ArgumentError("n_sim must be >= 2 for GOF covariance"))
    n_periods = data.n_waves - 1
    periods = period === nothing ? collect(1:n_periods) : [Int(period)]
    all(p -> 1 <= p <= n_periods, periods) || throw(ArgumentError(
        "period must be between 1 and $n_periods, got $period"))
    validate_effects(data, effects)
    cond_targets = condvar === nothing ? nothing :
        [_observed_distance(data, condvar, p) for p in 1:n_periods]

    # Observed statistic at the end of every compared period (joined: summed).
    labels = String[]
    observed = Int[]
    for p in periods
        st = initialize!(NetworkState(), data, p + 1; period=p)
        lab, cnt = compute_gof_statistic(statistic, st, data)
        labels = string.(lab)
        observed = isempty(observed) ? copy(cnt) : observed .+ cnt
    end

    simulated = zeros(Int, n_sim, length(observed))
    seeds = [rand(rng, 1:10^8) for _ in 1:n_sim]
    for s in 1:n_sim
        _, results = simulate_saom(data, effects, θ; rng=MersenneTwister(seeds[s]),
                                   variables=variables, condvar=condvar,
                                   cond_targets=cond_targets, validate=false)
        for p in periods
            _, cnt = compute_gof_statistic(statistic, results[p].final_state, data)
            simulated[s, :] .+= cnt
        end
    end

    # The shared two-sided rank-tail Monte Carlo convention (including ties and
    # the finite-simulation +1 correction), identical to NetworkCore.GOFStatistic.
    p_values = [mc_pvalue(view(simulated, :, i), observed[i]) for i in eachindex(observed)]
    obs_dist, p_overall = _pooled_mahalanobis_test(Float64.(observed),
                                                   Float64.(simulated))
    return SienaGOFResult(statistic, labels, observed, simulated, p_values, obs_dist,
                          p_overall, periods)
end

"""
    siena_gof(result::SienaResult, data::SienaData, statistic::AbstractGOFStatistic;
             n_sim::Int=100, rng::AbstractRNG=Random.default_rng(), period=nothing)

Goodness of fit of an estimated model on one auxiliary statistic — RSiena's
`sienaGOF`. `n_sim` trajectories are simulated from the estimates (each period
starting from its observed wave, conditionally on the observed amount of change when
the fit was conditional, and with the fit's `model_type` restriction), the statistic
is evaluated at the end of every period, and — RSiena's `join=TRUE` — summed over
the periods; pass `period=m` to compare period `m` only.

The overall p-value is a Monte-Carlo test of the Mahalanobis distance. Unlike
`sienaGOF`, which centres and scales the distances with the mean and covariance of
the simulations alone, the observed vector is pooled with the simulated ones, so
under the fitted model the observed distance is exchangeable with the simulated
ones and ``p = (1 + \\#\\{d_s \\ge d_{obs}\\})/(N + 1)`` is an exact Monte-Carlo
p-value (a pseudo-inverse handles the linear constraints of the counts). With the
default fixed cumulative degree levels the test holds its size: a correct model is
rejected about 5 % of the time at the 5 % level (pinned by a simulation testset).
The estimation uncertainty of the parameters is not accounted for, which makes the
test slightly conservative, as RSiena's.

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
detail = siena_gof(fit, data, IndegreeDistribution(:friendship); n_sim=50,
                   rng=Xoshiro(2))
0 < detail.p_overall <= 1
```
"""
function siena_gof(result::SienaResult, data::SienaData, statistic::AbstractGOFStatistic;
                  n_sim::Int=100, rng::AbstractRNG=Random.default_rng(),
                  period::Union{Nothing, Integer}=nothing)
    # The GOF simulations reproduce the fitted model, including its `model_type`
    # restriction (a dependent variable frozen during estimation is frozen here
    # too) and its conditioning.
    sim_vars = result.model_type == :standard ? nothing :
               simulated_variables(data, result.model_type)
    return _gof_core(data, result.effects, result.estimates, statistic;
                     n_sim=n_sim, rng=rng, variables=sim_vars,
                     condvar=result.conditional ? result.condvar : nothing,
                     period=period)
end

#==============================================================================#
# Shared `gof` generic (NetworkCore.jl)
#==============================================================================#

"""
    gof(result::SienaResult; statistics=nothing, n_sim=100, rng=Random.default_rng(), period=nothing)
    gof(result::SienaResult, statistic_or_statistics; kwargs...)
    gof(result::SienaResult, data::SienaData, statistic_or_statistics; kwargs...)

Goodness-of-fit assessment of an estimated SAOM — **the preferred GOF entry point**,
a method of the shared `NetworkCore.gof` generic returning the ecosystem-wide
`GOFResult` (one table per statistic). It runs [`siena_gof`](@ref) (which returns
the RSiena-style detail) for each statistic.

Without `statistics`, every network is assessed on its in- and outdegree
distributions (RSiena's default levels `0:8`, cumulative) and every behaviour on its
value distribution, using the data stored on the fit. With a single statistic the
overall p-value is the Monte-Carlo p-value of the Mahalanobis distance.

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
report = gof(fit; n_sim=50, rng=Xoshiro(2))                 # in- and outdegrees
triads = gof(fit, TriadCensus(:friendship); n_sim=50, rng=Xoshiro(3))
```
"""
gof(result::SienaResult, data::SienaData, statistic::AbstractGOFStatistic;
    kwargs...) = GOFResult(siena_gof(result, data, statistic; kwargs...))

function gof(result::SienaResult, data::SienaData,
             statistics::AbstractVector{<:AbstractGOFStatistic};
             n_sim::Int=100, rng::AbstractRNG=Random.default_rng(),
             period::Union{Nothing, Integer}=nothing)
    isempty(statistics) && throw(ArgumentError("gof requires at least one GOF statistic"))
    results = [siena_gof(result, data, stat; n_sim=n_sim, rng=rng, period=period)
               for stat in statistics]
    length(results) == 1 && return GOFResult(results[1])
    return GOFResult([_gof_statistic(r) for r in results]; model="SAOM")
end

# Infer meaningful defaults for every fitted dependent variable.
function _default_gof_statistics(result::SienaResult)
    out = AbstractGOFStatistic[]
    for variable in sort!(collect(keys(result.data.dependents)); by=String)
        dep = result.data.dependents[variable]
        if dep isa DependentNetwork
            push!(out, IndegreeDistribution(variable), OutdegreeDistribution(variable))
        elseif dep isa DependentBehavior
            push!(out, BehaviorDistribution(variable))
        end
    end
    return out
end

function gof(result::SienaResult; statistics=nothing, kwargs...)
    stats = statistics === nothing ? _default_gof_statistics(result) : statistics
    return gof(result, result.data, stats; kwargs...)
end

gof(result::SienaResult, statistic::AbstractGOFStatistic; kwargs...) =
    gof(result, result.data, statistic; kwargs...)
gof(result::SienaResult, statistics::AbstractVector{<:AbstractGOFStatistic}; kwargs...) =
    gof(result, result.data, statistics; kwargs...)

#==============================================================================#
# Convenience Functions
#==============================================================================#

"""
    siena_gof_indegree(result::SienaResult, data::SienaData, variable::Symbol;
                      kwargs...)

Shorthand for `siena_gof(result, data, IndegreeDistribution(variable); kwargs...)`
(prefer [`gof`](@ref)).

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
siena_gof_indegree(fit, data, :friendship; n_sim=20, rng=Xoshiro(1)).p_overall
```
"""
siena_gof_indegree(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, IndegreeDistribution(variable); kwargs...)

"""
    siena_gof_outdegree(result::SienaResult, data::SienaData, variable::Symbol;
                       kwargs...)

Shorthand for `siena_gof(result, data, OutdegreeDistribution(variable); kwargs...)`
(prefer [`gof`](@ref)).

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
siena_gof_outdegree(fit, data, :friendship; n_sim=20, rng=Xoshiro(1)).p_overall
```
"""
siena_gof_outdegree(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, OutdegreeDistribution(variable); kwargs...)

"""
    siena_gof_triad(result::SienaResult, data::SienaData, variable::Symbol;
                   kwargs...)

Shorthand for `siena_gof(result, data, TriadCensus(variable); kwargs...)` (prefer
[`gof`](@ref)).

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
siena_gof_triad(fit, data, :friendship; n_sim=20, rng=Xoshiro(1)).p_overall
```
"""
siena_gof_triad(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, TriadCensus(variable); kwargs...)

"""
    siena_gof_behavior(result::SienaResult, data::SienaData, variable::Symbol;
                      kwargs...)

Shorthand for `siena_gof(result, data, BehaviorDistribution(variable); kwargs...)`
(prefer [`gof`](@ref)).

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
add_dependent!(data, DependentBehavior(:alcohol, [s50.alcohol[:, w] for w in 1:3]))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip])
include_effects!(effects, :alcohol, [:linear, :quad])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=200))
siena_gof_behavior(fit, data, :alcohol; n_sim=20, rng=Xoshiro(1)).p_overall
```
"""
siena_gof_behavior(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, BehaviorDistribution(variable); kwargs...)
