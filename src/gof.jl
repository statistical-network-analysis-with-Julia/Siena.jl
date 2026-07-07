"""
Goodness of fit assessment for SAOM.
"""

#==============================================================================#
# GOF Statistics
#==============================================================================#

"""
    AbstractGOFStatistic

Abstract type for goodness of fit statistics.
"""
abstract type AbstractGOFStatistic end

"""
    IndegreeDistribution <: AbstractGOFStatistic

Indegree distribution statistic.
"""
struct IndegreeDistribution <: AbstractGOFStatistic
    variable::Symbol
    levls::Union{Vector{Int}, Nothing}

    IndegreeDistribution(variable::Symbol; levls::Union{Vector{Int}, Nothing}=nothing) =
        new(variable, levls)
end

"""
    OutdegreeDistribution <: AbstractGOFStatistic

Outdegree distribution statistic.
"""
struct OutdegreeDistribution <: AbstractGOFStatistic
    variable::Symbol
    levls::Union{Vector{Int}, Nothing}

    OutdegreeDistribution(variable::Symbol; levls::Union{Vector{Int}, Nothing}=nothing) =
        new(variable, levls)
end

"""
    TriadCensus <: AbstractGOFStatistic

Full 16-type Davis–Leinhardt triad census (003, 012, 102, 021D, 021U, 021C, 111D,
111U, 030T, 030C, 201, 120D, 120U, 120C, 210, 300).
"""
struct TriadCensus <: AbstractGOFStatistic
    variable::Symbol
end

"""
    GeodesicDistribution <: AbstractGOFStatistic

Geodesic (shortest path) distribution.
"""
struct GeodesicDistribution <: AbstractGOFStatistic
    variable::Symbol
    max_dist::Int

    GeodesicDistribution(variable::Symbol; max_dist::Int=5) =
        new(variable, max_dist)
end

"""
    BehaviorDistribution <: AbstractGOFStatistic

Behavior value distribution.
"""
struct BehaviorDistribution <: AbstractGOFStatistic
    variable::Symbol
end

# Fix the degree levels of a statistic so that observed and simulated counts are
# computed on the same support.
_with_levels(stat::IndegreeDistribution, levls::Vector{Int}) =
    IndegreeDistribution(stat.variable; levls=levls)
_with_levels(stat::OutdegreeDistribution, levls::Vector{Int}) =
    OutdegreeDistribution(stat.variable; levls=levls)
_with_levels(stat::AbstractGOFStatistic, ::Vector) = stat

#==============================================================================#
# Statistic Computation
#==============================================================================#

function _degree_counts(degrees::Vector{Int}, levls::Vector{Int})
    counts = zeros(Int, length(levls))
    for d in degrees
        idx = findfirst(==(d), levls)
        isnothing(idx) || (counts[idx] += 1)
    end
    return counts
end

"""
    compute_gof_statistic(stat::IndegreeDistribution, state::NetworkState,
                         data::SienaData)

Compute indegree distribution. Returns `(levels, counts)`.
"""
function compute_gof_statistic(stat::IndegreeDistribution, state::NetworkState,
                              data::SienaData)
    net = state.networks[stat.variable]
    n = size(net, 1)
    indegrees = [_col_sum(net, j) for j in 1:n]
    levls = isnothing(stat.levls) ? collect(0:maximum(indegrees)) : stat.levls
    return levls, _degree_counts(indegrees, levls)
end

"""
    compute_gof_statistic(stat::OutdegreeDistribution, state::NetworkState,
                         data::SienaData)

Compute outdegree distribution. Returns `(levels, counts)`.
"""
function compute_gof_statistic(stat::OutdegreeDistribution, state::NetworkState,
                              data::SienaData)
    net = state.networks[stat.variable]
    n = size(net, 1)
    outdegrees = [_row_sum(net, i) for i in 1:n]
    levls = isnothing(stat.levls) ? collect(0:maximum(outdegrees)) : stat.levls
    return levls, _degree_counts(outdegrees, levls)
end

const TRIAD_LABELS = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U",
                      "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]

# Classify the directed triad {a, b, c} into one of the 16 Davis–Leinhardt M-A-N
# classes (1-based index into TRIAD_LABELS).
function _triad_type(net::Matrix{Int}, a::Int, b::Int, c::Int)
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

"""
    compute_gof_statistic(stat::TriadCensus, state::NetworkState, data::SienaData)

Compute the full 16-type Davis–Leinhardt triad census. Returns `(labels, counts)`.
"""
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

"""
    compute_gof_statistic(stat::GeodesicDistribution, state::NetworkState,
                         data::SienaData)

Compute geodesic (shortest path) distribution. Returns `(labels, counts)` where the
labels are the distances 1..max_dist plus "unreachable" (which includes distances
beyond max_dist).
"""
function compute_gof_statistic(stat::GeodesicDistribution, state::NetworkState,
                              data::SienaData)
    net = state.networks[stat.variable]
    n = size(net, 1)

    dist_counts = zeros(Int, stat.max_dist)
    beyond = 0

    for i in 1:n
        # BFS from node i
        distances = fill(-1, n)
        distances[i] = 0
        queue = [i]

        while !isempty(queue)
            curr = popfirst!(queue)
            for j in 1:n
                if net[curr, j] == 1 && distances[j] == -1
                    distances[j] = distances[curr] + 1
                    if distances[j] < stat.max_dist
                        push!(queue, j)
                    end
                end
            end
        end

        for j in 1:n
            i == j && continue
            if distances[j] == -1
                beyond += 1
            elseif distances[j] <= stat.max_dist
                dist_counts[distances[j]] += 1
            end
        end
    end

    labels = vcat(string.(1:stat.max_dist), ["unreachable"])
    counts = vcat(dist_counts, [beyond])

    return labels, counts
end

"""
    compute_gof_statistic(stat::BehaviorDistribution, state::NetworkState,
                         data::SienaData)

Compute behavior value distribution. Returns `(levels, counts)`.
"""
function compute_gof_statistic(stat::BehaviorDistribution, state::NetworkState,
                              data::SienaData)
    beh = state.behaviors[stat.variable]
    dep = data.dependents[stat.variable]::DependentBehavior

    levls = collect(dep.min_val:dep.max_val)
    counts = zeros(Int, length(levls))

    for v in beh
        idx = v - dep.min_val + 1
        if 1 <= idx <= length(counts)
            counts[idx] += 1
        end
    end

    return levls, counts
end

#==============================================================================#
# GOF Result
#==============================================================================#

"""
    GOFResult

Result of goodness of fit assessment.

# Fields
- `statistic::AbstractGOFStatistic`: The GOF statistic used
- `labels::Vector`: Labels of the statistic's levels
- `observed::Vector{Int}`: Observed counts
- `simulated::Matrix{Int}`: Simulated counts (n_sims × n_levels)
- `p_values::Vector{Float64}`: Monte-Carlo two-sided p-values per level
- `mahalanobis::Float64`: Mahalanobis distance of the observed vector
- `p_overall::Float64`: Monte-Carlo p-value of the Mahalanobis distance
"""
struct GOFResult
    statistic::AbstractGOFStatistic
    labels::Vector
    observed::Vector{Int}
    simulated::Matrix{Int}
    p_values::Vector{Float64}
    mahalanobis::Float64
    p_overall::Float64
end

function Base.show(io::IO, result::GOFResult)
    println(io, "Goodness of Fit: $(typeof(result.statistic).name.name)")
    println(io, "Overall p-value: $(round(result.p_overall, digits=4)) " *
                "(Mahalanobis distance $(round(result.mahalanobis, digits=3)))")
    println(io)
    println(io, "Level-specific results:")
    for (i, label) in enumerate(result.labels)
        obs = result.observed[i]
        sim_mean = mean(result.simulated[:, i])
        sim_sd = std(result.simulated[:, i])
        p = result.p_values[i]
        @printf(io, "  %-12s: obs=%d, sim=%.1f (%.1f), p=%.3f\n",
                string(label), obs, sim_mean, sim_sd, p)
    end
end

#==============================================================================#
# Main GOF Function
#==============================================================================#

# Mahalanobis distance with a regularized covariance.
function _mahalanobis(x::Vector{Float64}, center::Vector{Float64}, C::Matrix{Float64})
    diff = x .- center
    return sqrt(max(0.0, dot(diff, C \ diff)))
end

"""
    siena_gof(result::SienaResult, data::SienaData, statistic::AbstractGOFStatistic;
             n_sims::Int=100, seed::Union{Int, Nothing}=nothing)

Assess goodness of fit for an estimated model, comparing the statistic at the final
wave of the simulations with the observed final wave. The overall p-value is the
Monte-Carlo proportion of simulations whose Mahalanobis distance (with respect to the
simulated distribution) is at least the observed one, as in RSiena's `sienaGOF`.
"""
function siena_gof(result::SienaResult, data::SienaData, statistic::AbstractGOFStatistic;
                  n_sims::Int=100, seed::Union{Int, Nothing}=nothing)

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    # Observed statistic at the final wave, and level-resolved statistic so that all
    # simulated counts align with the observed support
    state_obs = NetworkState()
    initialize!(state_obs, data, data.n_waves; period=data.n_waves - 1)
    labels, observed = compute_gof_statistic(statistic, state_obs, data)
    stat = _with_levels(statistic, labels isa Vector{Int} ? labels : Int[])

    n_levels = length(observed)
    simulated = zeros(Int, n_sims, n_levels)

    for s in 1:n_sims
        state_sim, _ = simulate_saom(data, result.effects, result.estimates;
                                     seed=rand(rng, 1:10^8))
        _, sim_counts = compute_gof_statistic(stat, state_sim, data)
        simulated[s, :] = sim_counts
    end

    # Per-level Monte-Carlo two-sided p-values
    p_values = Float64[]
    for i in 1:n_levels
        sim_col = simulated[:, i]
        n_extreme = sum(abs.(sim_col .- mean(sim_col)) .>= abs(observed[i] - mean(sim_col)))
        push!(p_values, n_extreme / n_sims)
    end

    # Overall Monte-Carlo Mahalanobis test
    sim_f = Float64.(simulated)
    center = vec(mean(sim_f, dims=1))
    C = cov(sim_f)
    C += (1e-6 + 0.01 * mean(diag(C))) * I  # regularize for invertibility

    obs_dist = _mahalanobis(Float64.(observed), center, C)
    sim_dists = [_mahalanobis(sim_f[s, :], center, C) for s in 1:n_sims]
    p_overall = (1 + count(>=(obs_dist), sim_dists)) / (n_sims + 1)

    return GOFResult(statistic, labels, observed, simulated, p_values, obs_dist, p_overall)
end

#==============================================================================#
# Convenience Functions
#==============================================================================#

"""
    siena_gof_indegree(result::SienaResult, data::SienaData, variable::Symbol;
                      kwargs...)

Assess GOF for indegree distribution.
"""
siena_gof_indegree(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, IndegreeDistribution(variable); kwargs...)

"""
    siena_gof_outdegree(result::SienaResult, data::SienaData, variable::Symbol;
                       kwargs...)

Assess GOF for outdegree distribution.
"""
siena_gof_outdegree(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, OutdegreeDistribution(variable); kwargs...)

"""
    siena_gof_triad(result::SienaResult, data::SienaData, variable::Symbol;
                   kwargs...)

Assess GOF for the triad census.
"""
siena_gof_triad(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, TriadCensus(variable); kwargs...)

"""
    siena_gof_behavior(result::SienaResult, data::SienaData, variable::Symbol;
                      kwargs...)

Assess GOF for behavior distribution.
"""
siena_gof_behavior(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, BehaviorDistribution(variable); kwargs...)
