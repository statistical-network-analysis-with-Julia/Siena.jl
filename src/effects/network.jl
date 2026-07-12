"""
Network effects for the SAOM evaluation function.

Each effect implements `evaluate_actor` (the actor's evaluation-function component
``s_{ki}(x)``, following the RSiena manual §12) and, where a cheap exact closed form
exists, `compute_contribution` (the add-direction change statistic
``s_{ki}(x^{+ij}) - s_{ki}(x^{-ij})``). Effects without a closed form fall back to the
generic toggle-based contribution in `effects/base.jl`; the test suite verifies every
closed form against that fallback.
"""

#==============================================================================#
# Helper Functions
#==============================================================================#

function _get_covariate_value(cov::AbstractCovariate, actor::Int, wave::Int)
    if cov isa ConstantCovariate
        return cov.values[actor]
    elseif cov isa VaryingCovariate
        w = clamp(wave, 1, length(cov.values))
        return cov.values[w][actor]
    end
    return 0.0
end

function _get_covariate_range(cov::AbstractCovariate)
    if cov isa ConstantCovariate
        vals = filter(!isnan, cov.values)
        return isempty(vals) ? 1.0 : maximum(vals) - minimum(vals)
    elseif cov isa VaryingCovariate
        all_vals = filter(!isnan, vcat(cov.values...))
        return isempty(all_vals) ? 1.0 : maximum(all_vals) - minimum(all_vals)
    end
    return 1.0
end

function _get_covariate_sim_mean(cov::AbstractCovariate)
    (cov isa ConstantCovariate || cov isa VaryingCovariate) && return cov.sim_mean
    return 0.0
end

function _get_dyad_covariate_value(cov::AbstractCovariate, i::Int, j::Int, wave::Int)
    if cov isa ConstantDyadCovariate
        return cov.values[i, j]
    elseif cov isa VaryingDyadCovariate
        w = clamp(wave, 1, length(cov.values))
        return cov.values[w][i, j]
    end
    return 0.0
end

# Centered similarity between actors i and j on a covariate (RSiena's sim_ij - ^sim).
function _centered_similarity(cov::AbstractCovariate, i::Int, j::Int, wave::Int)
    v1 = _get_covariate_value(cov, i, wave)
    v2 = _get_covariate_value(cov, j, wave)
    r = _get_covariate_range(cov)
    sim = r > 0 ? 1.0 - abs(v1 - v2) / r : 1.0
    return sim - _get_covariate_sim_mean(cov)
end

# Degree lookups. Simulation states store networks as `StateNetwork`s with
# incrementally maintained degree vectors, so the hot-loop lookups are O(1);
# the generic methods keep plain-matrix callers working.
_row_sum(net::StateNetwork, i::Int) = net.outdeg[i]
_col_sum(net::StateNetwork, j::Int) = net.indeg[j]
_row_sum(net::AbstractMatrix{Int}, i::Int) = @views sum(net[i, :])
_col_sum(net::AbstractMatrix{Int}, j::Int) = @views sum(net[:, j])

#==============================================================================#
# Basic Structural Effects
#==============================================================================#

"""
    OutdegreeEffect <: NetworkEffect

Basic outdegree effect (density): ``s_i = x_{i+}``. RSiena: density
"""
struct OutdegreeEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::OutdegreeEffect) = :density
effect_type(::OutdegreeEffect) = :eval
target_variable(e::OutdegreeEffect) = e.variable

function evaluate_actor(e::OutdegreeEffect, state::NetworkState, data::SienaData, actor::Int)
    return Float64(_row_sum(state.networks[e.variable], actor))
end

function compute_contribution(e::OutdegreeEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return 1.0
end

"""
    ReciprocityEffect <: NetworkEffect

Reciprocity effect: ``s_i = \\sum_j x_{ij} x_{ji}``. RSiena: recip
"""
struct ReciprocityEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::ReciprocityEffect) = :recip
effect_type(::ReciprocityEffect) = :eval
target_variable(e::ReciprocityEffect) = e.variable

function evaluate_actor(e::ReciprocityEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        j == actor && continue
        count += net[actor, j] * net[j, actor]
    end
    return Float64(count)
end

function compute_contribution(e::ReciprocityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    return Float64(net[alter, actor])
end

#==============================================================================#
# Triadic Effects
#==============================================================================#

"""
    TransitiveTripletsEffect <: NetworkEffect

Transitive triplets: ``s_i = \\sum_{j \\ne h} x_{ij} x_{ih} x_{jh}``. RSiena: transTrip
"""
struct TransitiveTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveTripletsEffect) = :transTrip
effect_type(::TransitiveTripletsEffect) = :eval
target_variable(e::TransitiveTripletsEffect) = e.variable

function evaluate_actor(e::TransitiveTripletsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[j, h] == 1
                count += 1
            end
        end
    end
    return Float64(count)
end

function compute_contribution(e::TransitiveTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for h in 1:n
        (h == actor || h == alter) && continue
        if net[actor, h] == 1
            count += net[alter, h] + net[h, alter]
        end
    end
    return Float64(count)
end

"""
    TransitiveTiesEffect <: NetworkEffect

Transitive ties: ``s_i = \\sum_j x_{ij} \\, I(\\exists h: x_{ih} x_{hj} = 1)``.
RSiena: transTies
"""
struct TransitiveTiesEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveTiesEffect) = :transTies
effect_type(::TransitiveTiesEffect) = :eval
target_variable(e::TransitiveTiesEffect) = e.variable

function evaluate_actor(e::TransitiveTiesEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[h, j] == 1
                count += 1
                break
            end
        end
    end
    return Float64(count)
end

# Alias: TransitiveTriadsEffect is the same as TransitiveTiesEffect
const TransitiveTriadsEffect = TransitiveTiesEffect

"""
    TransitiveMediatedTripletsEffect <: NetworkEffect

Transitive mediated triplets (ego in the mediating position):
``s_i = \\sum_{j \\ne h} x_{ji} x_{ih} x_{jh}``. RSiena: transMedTrip
"""
struct TransitiveMediatedTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveMediatedTripletsEffect) = :transMedTrip
effect_type(::TransitiveMediatedTripletsEffect) = :eval
target_variable(e::TransitiveMediatedTripletsEffect) = e.variable

function evaluate_actor(e::TransitiveMediatedTripletsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[j, actor] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[j, h] == 1
                count += 1
            end
        end
    end
    return Float64(count)
end

function compute_contribution(e::TransitiveMediatedTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for h in 1:n
        (h == actor || h == alter) && continue
        if net[h, actor] == 1 && net[h, alter] == 1
            count += 1
        end
    end
    return Float64(count)
end

"""
    TransitiveRecipTripletsEffect <: NetworkEffect

Transitive reciprocated triplets:
``s_i = \\sum_{j \\ne h} x_{ij} x_{ji} x_{ih} x_{hj}``. RSiena: transRecTrip
"""
struct TransitiveRecipTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveRecipTripletsEffect) = :transRecTrip
effect_type(::TransitiveRecipTripletsEffect) = :eval
target_variable(e::TransitiveRecipTripletsEffect) = e.variable

function evaluate_actor(e::TransitiveRecipTripletsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0 || net[j, actor] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[h, j] == 1
                count += 1
            end
        end
    end
    return Float64(count)
end

function compute_contribution(e::TransitiveRecipTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    # Toggled tie as the i -> j side of the mutual dyad
    if net[alter, actor] == 1
        for h in 1:n
            (h == actor || h == alter) && continue
            if net[actor, h] == 1 && net[h, alter] == 1
                count += 1.0
            end
        end
    end
    # Toggled tie as the two-path leg i -> h with h = alter
    for h in 1:n
        (h == actor || h == alter) && continue
        if net[actor, h] == 1 && net[h, actor] == 1 && net[alter, h] == 1
            count += 1.0
        end
    end
    return count
end

"""
    CyclicTripletsEffect <: NetworkEffect

Three-cycles: ``s_i = \\sum_{j \\ne h} x_{ij} x_{jh} x_{hi}``. RSiena: cycle3
"""
struct CyclicTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::CyclicTripletsEffect) = :cycle3
effect_type(::CyclicTripletsEffect) = :eval
target_variable(e::CyclicTripletsEffect) = e.variable

function evaluate_actor(e::CyclicTripletsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[j, h] == 1 && net[h, actor] == 1
                count += 1
            end
        end
    end
    return Float64(count)
end

function compute_contribution(e::CyclicTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for h in 1:n
        (h == actor || h == alter) && continue
        if net[alter, h] == 1 && net[h, actor] == 1
            count += 1
        end
    end
    return Float64(count)
end

# RSiena's cycle3 target statistic counts each 3-cycle once (sum_i s_i counts it
# three times, once per member).
function compute_statistic(e::CyclicTripletsEffect, state::NetworkState, data::SienaData)
    n = size(state.networks[e.variable], 1)
    return sum(evaluate_actor(e, state, data, i) for i in 1:n) / 3.0
end

"""
    BalanceEffect <: NetworkEffect

Structural balance (similarity of outgoing tie patterns with alters):
``s_i = \\frac{1}{n-2} \\sum_j x_{ij} \\sum_{h \\ne i,j} (1 - |x_{ih} - x_{jh}|)``.

Note: RSiena subtracts a data-dependent constant ``b_0``; here the sum is normalized
by ``n-2`` instead. RSiena: balance
"""
struct BalanceEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::BalanceEffect) = :balance
effect_type(::BalanceEffect) = :eval
target_variable(e::BalanceEffect) = e.variable

function evaluate_actor(e::BalanceEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    n <= 2 && return 0.0
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            total += 1.0 - abs(net[actor, h] - net[j, h])
        end
    end
    return total / (n - 2)
end

"""
    BetweennessEffect <: NetworkEffect

Betweenness (brokerage): ``s_i = \\sum_{j \\ne h} x_{hi} x_{ij} (1 - x_{hj})``.
RSiena: between
"""
struct BetweennessEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::BetweennessEffect) = :between
effect_type(::BetweennessEffect) = :eval
target_variable(e::BetweennessEffect) = e.variable

function evaluate_actor(e::BetweennessEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for h in 1:n
        (h == actor || net[h, actor] == 0) && continue
        for j in 1:n
            (j == actor || j == h) && continue
            if net[actor, j] == 1 && net[h, j] == 0
                count += 1
            end
        end
    end
    return Float64(count)
end

function compute_contribution(e::BetweennessEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for h in 1:n
        (h == actor || h == alter) && continue
        if net[h, actor] == 1 && net[h, alter] == 0
            count += 1
        end
    end
    return Float64(count)
end

"""
    NbrDist2Effect <: NetworkEffect

Number of actors at distance exactly 2:
``s_i = \\#\\{j \\ne i : x_{ij} = 0, \\exists h: x_{ih} x_{hj} = 1\\}``. RSiena: nbrDist2
"""
struct NbrDist2Effect <: NetworkEffect
    variable::Symbol
end

effect_name(::NbrDist2Effect) = :nbrDist2
effect_type(::NbrDist2Effect) = :eval
target_variable(e::NbrDist2Effect) = e.variable

function evaluate_actor(e::NbrDist2Effect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 1) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[h, j] == 1
                count += 1
                break
            end
        end
    end
    return Float64(count)
end

"""
    DenseTriadsEffect <: NetworkEffect

Dense triads (tie-weighted, following RSiena):
``s_i = \\sum_j x_{ij} \\#\\{h \\ne i,j : \\text{triad } (i,j,h) \\text{ has} \\ge c
\\text{ arcs}\\}`` with ``c = 5``. RSiena: denseTriads
"""
struct DenseTriadsEffect <: NetworkEffect
    variable::Symbol
    c::Int
    DenseTriadsEffect(variable::Symbol; c::Int=5) = new(variable, c)
end

effect_name(::DenseTriadsEffect) = :denseTriads
effect_type(::DenseTriadsEffect) = :eval
target_variable(e::DenseTriadsEffect) = e.variable

function evaluate_actor(e::DenseTriadsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            ties = net[actor, j] + net[j, actor] + net[actor, h] + net[h, actor] +
                   net[j, h] + net[h, j]
            ties >= e.c && (count += 1)
        end
    end
    return Float64(count)
end

"""
    SharedInEffect <: NetworkEffect

Ties to alters with shared in-neighbors:
``s_i = \\sum_j x_{ij} \\#\\{h \\ne i,j : x_{hi} x_{hj} = 1\\}``.
"""
struct SharedInEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::SharedInEffect) = :sharedIn
effect_type(::SharedInEffect) = :eval
target_variable(e::SharedInEffect) = e.variable

function evaluate_actor(e::SharedInEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[h, actor] == 1 && net[h, j] == 1
                count += 1
            end
        end
    end
    return Float64(count)
end

function compute_contribution(e::SharedInEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for h in 1:n
        (h == actor || h == alter) && continue
        if net[h, actor] == 1 && net[h, alter] == 1
            count += 1
        end
    end
    return Float64(count)
end

"""
    SharedOutEffect <: NetworkEffect

Ties to alters with shared out-neighbors:
``s_i = \\sum_j x_{ij} \\#\\{h \\ne i,j : x_{ih} x_{jh} = 1\\}``.
"""
struct SharedOutEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::SharedOutEffect) = :sharedOut
effect_type(::SharedOutEffect) = :eval
target_variable(e::SharedOutEffect) = e.variable

function evaluate_actor(e::SharedOutEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[j, h] == 1
                count += 1
            end
        end
    end
    return Float64(count)
end

#==============================================================================#
# Degree-Based Effects
#==============================================================================#

"""
    IndegreePopularityEffect <: NetworkEffect

Indegree popularity: ``s_i = \\sum_j x_{ij} f(x_{+j})`` with ``f`` the identity or
square root. RSiena: inPop, inPopSqrt
"""
struct IndegreePopularityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    IndegreePopularityEffect(variable::Symbol; sqrt::Bool=false) = new(variable, sqrt)
end

effect_name(e::IndegreePopularityEffect) = e.sqrt ? :inPopSqrt : :inPop
effect_type(::IndegreePopularityEffect) = :eval
target_variable(e::IndegreePopularityEffect) = e.variable

function evaluate_actor(e::IndegreePopularityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        indeg = _col_sum(net, j)
        total += e.sqrt ? sqrt(Float64(indeg)) : Float64(indeg)
    end
    return total
end

function compute_contribution(e::IndegreePopularityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    indeg = _col_sum(net, alter) - net[actor, alter]  # alter's indegree without the tie
    d = Float64(indeg + 1)                            # ... and with it
    return e.sqrt ? sqrt(d) : d
end

"""
    OutdegreePopularityEffect <: NetworkEffect

Outdegree popularity: ``s_i = \\sum_j x_{ij} f(x_{j+})``. RSiena: outPop, outPopSqrt
"""
struct OutdegreePopularityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    OutdegreePopularityEffect(variable::Symbol; sqrt::Bool=false) = new(variable, sqrt)
end

effect_name(e::OutdegreePopularityEffect) = e.sqrt ? :outPopSqrt : :outPop
effect_type(::OutdegreePopularityEffect) = :eval
target_variable(e::OutdegreePopularityEffect) = e.variable

function evaluate_actor(e::OutdegreePopularityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        outdeg = _row_sum(net, j)
        total += e.sqrt ? sqrt(Float64(outdeg)) : Float64(outdeg)
    end
    return total
end

function compute_contribution(e::OutdegreePopularityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = Float64(_row_sum(net, alter))
    return e.sqrt ? sqrt(outdeg) : outdeg
end

"""
    IndegreeActivityEffect <: NetworkEffect

Indegree activity: ``s_i = x_{i+} f(x_{+i})``. RSiena: inAct, inActSqrt
"""
struct IndegreeActivityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    IndegreeActivityEffect(variable::Symbol; sqrt::Bool=false) = new(variable, sqrt)
end

effect_name(e::IndegreeActivityEffect) = e.sqrt ? :inActSqrt : :inAct
effect_type(::IndegreeActivityEffect) = :eval
target_variable(e::IndegreeActivityEffect) = e.variable

function evaluate_actor(e::IndegreeActivityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    indeg = Float64(_col_sum(net, actor))
    outdeg = Float64(_row_sum(net, actor))
    return outdeg * (e.sqrt ? sqrt(indeg) : indeg)
end

function compute_contribution(e::IndegreeActivityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    indeg = Float64(_col_sum(net, actor))
    return e.sqrt ? sqrt(indeg) : indeg
end

"""
    OutdegreeActivityEffect <: NetworkEffect

Outdegree activity: ``s_i = x_{i+} f(x_{i+})`` (i.e. ``x_{i+}^2`` or
``x_{i+}^{3/2}``). RSiena: outAct, outActSqrt
"""
struct OutdegreeActivityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    OutdegreeActivityEffect(variable::Symbol; sqrt::Bool=false) = new(variable, sqrt)
end

effect_name(e::OutdegreeActivityEffect) = e.sqrt ? :outActSqrt : :outAct
effect_type(::OutdegreeActivityEffect) = :eval
target_variable(e::OutdegreeActivityEffect) = e.variable

_outact_value(e::OutdegreeActivityEffect, d::Float64) = e.sqrt ? d^1.5 : d^2

function evaluate_actor(e::OutdegreeActivityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    return _outact_value(e, Float64(_row_sum(net, actor)))
end

function compute_contribution(e::OutdegreeActivityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    d = Float64(_row_sum(net, actor) - net[actor, alter])  # outdegree without the tie
    return _outact_value(e, d + 1) - _outact_value(e, d)
end

"""
    OutdegreeTruncEffect <: NetworkEffect

Truncated outdegree: ``s_i = \\min(x_{i+}, c)``. RSiena: outTrunc
"""
struct OutdegreeTruncEffect <: NetworkEffect
    variable::Symbol
    c::Int
    OutdegreeTruncEffect(variable::Symbol; c::Int=1) = new(variable, c)
end

effect_name(::OutdegreeTruncEffect) = :outTrunc
effect_type(::OutdegreeTruncEffect) = :eval
target_variable(e::OutdegreeTruncEffect) = e.variable

function evaluate_actor(e::OutdegreeTruncEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    return Float64(min(_row_sum(net, actor), e.c))
end

function compute_contribution(e::OutdegreeTruncEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    d = _row_sum(net, actor) - net[actor, alter]
    return d < e.c ? 1.0 : 0.0
end

"""
    IndegreeTruncEffect <: NetworkEffect

Truncated indegree popularity: ``s_i = \\sum_j x_{ij} \\min(x_{+j}, c)``.
"""
struct IndegreeTruncEffect <: NetworkEffect
    variable::Symbol
    c::Int
    IndegreeTruncEffect(variable::Symbol; c::Int=1) = new(variable, c)
end

effect_name(::IndegreeTruncEffect) = :inTrunc
effect_type(::IndegreeTruncEffect) = :eval
target_variable(e::IndegreeTruncEffect) = e.variable

function evaluate_actor(e::IndegreeTruncEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += min(_col_sum(net, j), e.c)
    end
    return Float64(total)
end

function compute_contribution(e::IndegreeTruncEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    d = _col_sum(net, alter) - net[actor, alter]
    return Float64(min(d + 1, e.c))
end

"""
    DegreeAssortativityEffect <: NetworkEffect

Degree assortativity: ``s_i = \\sum_j x_{ij} (d_i + d_j)`` where ``d`` is the total
(in + out) degree. RSiena: degPlus
"""
struct DegreeAssortativityEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::DegreeAssortativityEffect) = :degPlus
effect_type(::DegreeAssortativityEffect) = :eval
target_variable(e::DegreeAssortativityEffect) = e.variable

function evaluate_actor(e::DegreeAssortativityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    deg_i = _row_sum(net, actor) + _col_sum(net, actor)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += deg_i + _row_sum(net, j) + _col_sum(net, j)
    end
    return total
end

#==============================================================================#
# Isolate Effects
#==============================================================================#

"""
    IsolateEffect <: NetworkEffect

Isolate effect: ``s_i = I(x_{i+} = 0 \\text{ and } x_{+i} = 0)``. RSiena: isolate
"""
struct IsolateEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::IsolateEffect) = :isolate
effect_type(::IsolateEffect) = :eval
target_variable(e::IsolateEffect) = e.variable

function evaluate_actor(e::IsolateEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    return (_row_sum(net, actor) == 0 && _col_sum(net, actor) == 0) ? 1.0 : 0.0
end

function compute_contribution(e::IsolateEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = _row_sum(net, actor) - net[actor, alter]
    indeg = _col_sum(net, actor)
    return (outdeg == 0 && indeg == 0) ? -1.0 : 0.0
end

"""
    IsolateNetEffect <: NetworkEffect

Network isolate (total isolate, as in RSiena):
``s_i = I(x_{i+} = 0 \\text{ and } x_{+i} = 0)``. RSiena: isolateNet
"""
struct IsolateNetEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::IsolateNetEffect) = :isolateNet
effect_type(::IsolateNetEffect) = :eval
target_variable(e::IsolateNetEffect) = e.variable

function evaluate_actor(e::IsolateNetEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    return (_row_sum(net, actor) == 0 && _col_sum(net, actor) == 0) ? 1.0 : 0.0
end

function compute_contribution(e::IsolateNetEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = _row_sum(net, actor) - net[actor, alter]
    indeg = _col_sum(net, actor)
    return (outdeg == 0 && indeg == 0) ? -1.0 : 0.0
end

"""
    OutIsolateEffect <: NetworkEffect

Out-isolate: ``s_i = I(x_{i+} = 0)`` (same definition as [`IsolateNetEffect`](@ref)).
RSiena: outIsolate
"""
struct OutIsolateEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::OutIsolateEffect) = :outIsolate
effect_type(::OutIsolateEffect) = :eval
target_variable(e::OutIsolateEffect) = e.variable

function evaluate_actor(e::OutIsolateEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    return _row_sum(net, actor) == 0 ? 1.0 : 0.0
end

function compute_contribution(e::OutIsolateEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = _row_sum(net, actor) - net[actor, alter]
    return outdeg == 0 ? -1.0 : 0.0
end

"""
    InIsolateEffect <: NetworkEffect

Ties to in-isolates: ``s_i = \\sum_j x_{ij} I(x_{+j} - x_{ij} = 0)`` — ties to alters
whose only incoming tie is from ego.
"""
struct InIsolateEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::InIsolateEffect) = :inIsolate
effect_type(::InIsolateEffect) = :eval
target_variable(e::InIsolateEffect) = e.variable

function evaluate_actor(e::InIsolateEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        if _col_sum(net, j) - net[actor, j] == 0
            count += 1
        end
    end
    return Float64(count)
end

function compute_contribution(e::InIsolateEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    indeg = _col_sum(net, alter) - net[actor, alter]
    return indeg == 0 ? 1.0 : 0.0
end

#==============================================================================#
# GWESP / Shared Partner Effects
#==============================================================================#

# GWESP weight: exp(alpha) * (1 - (1 - exp(-alpha))^sp)
_gwesp_weight(alpha::Float64, sp::Int) =
    sp == 0 ? 0.0 : exp(alpha) * (1.0 - (1.0 - exp(-alpha))^sp)

"""
    GWESPEffect <: NetworkEffect

Geometrically weighted edgewise shared partners, forward-forward (two-path closure,
the GWESP analogue of transTrip): shared partners of the tie ``i \\to j`` are actors
``h`` with ``x_{ih} = x_{hj} = 1``. RSiena: gwespFF
"""
struct GWESPEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWESPEffect(variable::Symbol; alpha::Float64=log(2.0)) = new(variable, alpha)
end

effect_name(::GWESPEffect) = :gwespFF
effect_type(::GWESPEffect) = :eval
target_variable(e::GWESPEffect) = e.variable

function _esp_count(net::AbstractMatrix{Int}, i::Int, j::Int, ego_out::Bool, alter_out::Bool)
    n = size(net, 1)
    sp = 0
    for h in 1:n
        (h == i || h == j) && continue
        ego_tie = ego_out ? net[i, h] : net[h, i]
        alter_tie = alter_out ? net[j, h] : net[h, j]
        if ego_tie == 1 && alter_tie == 1
            sp += 1
        end
    end
    return sp
end

function evaluate_actor(e::GWESPEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        # FF: i -> h and h -> j
        total += _gwesp_weight(e.alpha, _esp_count(net, actor, j, true, false))
    end
    return total
end

"""
    GWESPBackwardEffect <: NetworkEffect

GWESP backward-backward: shared partners of the tie ``i \\to j`` are actors ``h``
with ``x_{hi} = x_{jh} = 1``. RSiena: gwespBB
"""
struct GWESPBackwardEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWESPBackwardEffect(variable::Symbol; alpha::Float64=log(2.0)) = new(variable, alpha)
end

effect_name(::GWESPBackwardEffect) = :gwespBB
effect_type(::GWESPBackwardEffect) = :eval
target_variable(e::GWESPBackwardEffect) = e.variable

function evaluate_actor(e::GWESPBackwardEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        # BB: h -> i and j -> h
        total += _gwesp_weight(e.alpha, _esp_count(net, actor, j, false, true))
    end
    return total
end

"""
    GWESPMixedEffect <: NetworkEffect

GWESP forward-backward: shared partners of the tie ``i \\to j`` are actors ``h``
with ``x_{ih} = x_{jh} = 1`` (shared out-alters). RSiena: gwespFB
"""
struct GWESPMixedEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWESPMixedEffect(variable::Symbol; alpha::Float64=log(2.0)) = new(variable, alpha)
end

effect_name(::GWESPMixedEffect) = :gwespFB
effect_type(::GWESPMixedEffect) = :eval
target_variable(e::GWESPMixedEffect) = e.variable

function evaluate_actor(e::GWESPMixedEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        # FB: i -> h and j -> h
        total += _gwesp_weight(e.alpha, _esp_count(net, actor, j, true, true))
    end
    return total
end

"""
    GWDSPEffect <: NetworkEffect

Geometrically weighted dyadwise shared partners (two-paths, regardless of the direct
tie): ``s_i = \\sum_{j \\ne i} w(\\#\\{h : x_{ih} x_{hj} = 1\\})``. RSiena: gwdspFF
"""
struct GWDSPEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWDSPEffect(variable::Symbol; alpha::Float64=log(2.0)) = new(variable, alpha)
end

effect_name(::GWDSPEffect) = :gwdspFF
effect_type(::GWDSPEffect) = :eval
target_variable(e::GWDSPEffect) = e.variable

function evaluate_actor(e::GWDSPEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        j == actor && continue
        total += _gwesp_weight(e.alpha, _esp_count(net, actor, j, true, false))
    end
    return total
end

#==============================================================================#
# Covariate Effects
#==============================================================================#

"""
    EgoEffect <: NetworkEffect

Ego covariate effect: ``s_i = v_i x_{i+}``. RSiena: egoX
"""
struct EgoEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoEffect) = :egoX
effect_type(::EgoEffect) = :eval
target_variable(e::EgoEffect) = e.variable
interaction_with(e::EgoEffect) = e.covariate

function evaluate_actor(e::EgoEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    v = _get_covariate_value(data.covariates[e.covariate], actor, state.period)
    return v * _row_sum(net, actor)
end

function compute_contribution(e::EgoEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _get_covariate_value(data.covariates[e.covariate], actor, state.period)
end

"""
    EgoSqEffect <: NetworkEffect

Squared ego covariate effect: ``s_i = v_i^2 x_{i+}``. RSiena: egoSqX
"""
struct EgoSqEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoSqEffect) = :egoSqX
effect_type(::EgoSqEffect) = :eval
target_variable(e::EgoSqEffect) = e.variable
interaction_with(e::EgoSqEffect) = e.covariate

function evaluate_actor(e::EgoSqEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    v = _get_covariate_value(data.covariates[e.covariate], actor, state.period)
    return v^2 * _row_sum(net, actor)
end

function compute_contribution(e::EgoSqEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    v = _get_covariate_value(data.covariates[e.covariate], actor, state.period)
    return v^2
end

"""
    AlterEffect <: NetworkEffect

Alter covariate effect: ``s_i = \\sum_j x_{ij} v_j``. RSiena: altX
"""
struct AlterEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::AlterEffect) = :altX
effect_type(::AlterEffect) = :eval
target_variable(e::AlterEffect) = e.variable
interaction_with(e::AlterEffect) = e.covariate

function evaluate_actor(e::AlterEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _get_covariate_value(cov, j, state.period)
    end
    return total
end

function compute_contribution(e::AlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _get_covariate_value(data.covariates[e.covariate], alter, state.period)
end

"""
    AlterSqEffect <: NetworkEffect

Squared alter covariate effect: ``s_i = \\sum_j x_{ij} v_j^2``. RSiena: altSqX
"""
struct AlterSqEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::AlterSqEffect) = :altSqX
effect_type(::AlterSqEffect) = :eval
target_variable(e::AlterSqEffect) = e.variable
interaction_with(e::AlterSqEffect) = e.covariate

function evaluate_actor(e::AlterSqEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _get_covariate_value(cov, j, state.period)^2
    end
    return total
end

function compute_contribution(e::AlterSqEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    v = _get_covariate_value(data.covariates[e.covariate], alter, state.period)
    return v^2
end

"""
    SimilarityEffect <: NetworkEffect

Covariate similarity: ``s_i = \\sum_j x_{ij} (\\text{sim}_{ij} - \\widehat{sim})``
with ``\\text{sim}_{ij} = 1 - |v_i - v_j|/r_V`` and ``\\widehat{sim}`` the observed
mean similarity (as in RSiena). RSiena: simX
"""
struct SimilarityEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SimilarityEffect) = :simX
effect_type(::SimilarityEffect) = :eval
target_variable(e::SimilarityEffect) = e.variable
interaction_with(e::SimilarityEffect) = e.covariate

function evaluate_actor(e::SimilarityEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _centered_similarity(cov, actor, j, state.period)
    end
    return total
end

function compute_contribution(e::SimilarityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _centered_similarity(data.covariates[e.covariate], actor, alter, state.period)
end

"""
    SameEffect <: NetworkEffect

Same covariate value: ``s_i = \\sum_j x_{ij} I(v_i = v_j)``. RSiena: sameX
"""
struct SameEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SameEffect) = :sameX
effect_type(::SameEffect) = :eval
target_variable(e::SameEffect) = e.variable
interaction_with(e::SameEffect) = e.covariate

function evaluate_actor(e::SameEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    v1 = _get_covariate_value(cov, actor, state.period)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        v1 == _get_covariate_value(cov, j, state.period) && (count += 1)
    end
    return Float64(count)
end

function compute_contribution(e::SameEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    v1 = _get_covariate_value(cov, actor, state.period)
    v2 = _get_covariate_value(cov, alter, state.period)
    return v1 == v2 ? 1.0 : 0.0
end

"""
    DifferenceEffect <: NetworkEffect

Difference effect: ``s_i = \\sum_j x_{ij} (v_i - v_j)``. RSiena: diffX
"""
struct DifferenceEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::DifferenceEffect) = :diffX
effect_type(::DifferenceEffect) = :eval
target_variable(e::DifferenceEffect) = e.variable
interaction_with(e::DifferenceEffect) = e.covariate

function evaluate_actor(e::DifferenceEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    v1 = _get_covariate_value(cov, actor, state.period)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += v1 - _get_covariate_value(cov, j, state.period)
    end
    return total
end

function compute_contribution(e::DifferenceEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    return _get_covariate_value(cov, actor, state.period) -
           _get_covariate_value(cov, alter, state.period)
end

"""
    DifferenceSqEffect <: NetworkEffect

Squared difference effect: ``s_i = \\sum_j x_{ij} (v_i - v_j)^2``. RSiena: diffSqX
"""
struct DifferenceSqEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::DifferenceSqEffect) = :diffSqX
effect_type(::DifferenceSqEffect) = :eval
target_variable(e::DifferenceSqEffect) = e.variable
interaction_with(e::DifferenceSqEffect) = e.covariate

function evaluate_actor(e::DifferenceSqEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    v1 = _get_covariate_value(cov, actor, state.period)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += (v1 - _get_covariate_value(cov, j, state.period))^2
    end
    return total
end

function compute_contribution(e::DifferenceSqEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    d = _get_covariate_value(cov, actor, state.period) -
        _get_covariate_value(cov, alter, state.period)
    return d^2
end

"""
    AbsDifferenceEffect <: NetworkEffect

Absolute difference effect: ``s_i = \\sum_j x_{ij} |v_i - v_j|``. RSiena: absDiffX
"""
struct AbsDifferenceEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::AbsDifferenceEffect) = :absDiffX
effect_type(::AbsDifferenceEffect) = :eval
target_variable(e::AbsDifferenceEffect) = e.variable
interaction_with(e::AbsDifferenceEffect) = e.covariate

function evaluate_actor(e::AbsDifferenceEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    v1 = _get_covariate_value(cov, actor, state.period)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += abs(v1 - _get_covariate_value(cov, j, state.period))
    end
    return total
end

function compute_contribution(e::AbsDifferenceEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    return abs(_get_covariate_value(cov, actor, state.period) -
               _get_covariate_value(cov, alter, state.period))
end

"""
    HigherEffect <: NetworkEffect

Ego higher than alter: ``s_i = \\sum_j x_{ij} I(v_i > v_j)``. RSiena: higher
"""
struct HigherEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::HigherEffect) = :higher
effect_type(::HigherEffect) = :eval
target_variable(e::HigherEffect) = e.variable
interaction_with(e::HigherEffect) = e.covariate

function evaluate_actor(e::HigherEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    v1 = _get_covariate_value(cov, actor, state.period)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        v1 > _get_covariate_value(cov, j, state.period) && (count += 1)
    end
    return Float64(count)
end

function compute_contribution(e::HigherEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    return _get_covariate_value(cov, actor, state.period) >
           _get_covariate_value(cov, alter, state.period) ? 1.0 : 0.0
end

"""
    EgoTimesAlterEffect <: NetworkEffect

Ego × alter interaction: ``s_i = \\sum_j x_{ij} v_i v_j``. RSiena: egoXaltX
"""
struct EgoTimesAlterEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoTimesAlterEffect) = :egoXaltX
effect_type(::EgoTimesAlterEffect) = :eval
target_variable(e::EgoTimesAlterEffect) = e.variable
interaction_with(e::EgoTimesAlterEffect) = e.covariate

function evaluate_actor(e::EgoTimesAlterEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    v1 = _get_covariate_value(cov, actor, state.period)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += v1 * _get_covariate_value(cov, j, state.period)
    end
    return total
end

function compute_contribution(e::EgoTimesAlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    return _get_covariate_value(cov, actor, state.period) *
           _get_covariate_value(cov, alter, state.period)
end

"""
    EgoPlusAlterEffect <: NetworkEffect

Ego + alter sum effect: ``s_i = \\sum_j x_{ij} (v_i + v_j)``. RSiena: egoPlusAltX
"""
struct EgoPlusAlterEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoPlusAlterEffect) = :egoPlusAltX
effect_type(::EgoPlusAlterEffect) = :eval
target_variable(e::EgoPlusAlterEffect) = e.variable
interaction_with(e::EgoPlusAlterEffect) = e.covariate

function evaluate_actor(e::EgoPlusAlterEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    v1 = _get_covariate_value(cov, actor, state.period)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += v1 + _get_covariate_value(cov, j, state.period)
    end
    return total
end

function compute_contribution(e::EgoPlusAlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    return _get_covariate_value(cov, actor, state.period) +
           _get_covariate_value(cov, alter, state.period)
end

"""
    DyadCovariateEffect <: NetworkEffect

Dyadic covariate effect: ``s_i = \\sum_j x_{ij} w_{ij}``. RSiena: X
"""
struct DyadCovariateEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::DyadCovariateEffect) = :X
effect_type(::DyadCovariateEffect) = :eval
target_variable(e::DyadCovariateEffect) = e.variable
interaction_with(e::DyadCovariateEffect) = e.covariate

function evaluate_actor(e::DyadCovariateEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 2)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _get_dyad_covariate_value(cov, actor, j, state.period)
    end
    return total
end

function compute_contribution(e::DyadCovariateEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _get_dyad_covariate_value(data.covariates[e.covariate], actor, alter, state.period)
end

"""
    SameXRecipEffect <: NetworkEffect

Same covariate × reciprocity: ``s_i = \\sum_j x_{ij} x_{ji} I(v_i = v_j)``.
RSiena: sameXRecip
"""
struct SameXRecipEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SameXRecipEffect) = :sameXRecip
effect_type(::SameXRecipEffect) = :eval
target_variable(e::SameXRecipEffect) = e.variable
interaction_with(e::SameXRecipEffect) = e.covariate

function evaluate_actor(e::SameXRecipEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    v1 = _get_covariate_value(cov, actor, state.period)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0 || net[j, actor] == 0) && continue
        v1 == _get_covariate_value(cov, j, state.period) && (count += 1)
    end
    return Float64(count)
end

function compute_contribution(e::SameXRecipEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    net[alter, actor] == 0 && return 0.0
    cov = data.covariates[e.covariate]
    v1 = _get_covariate_value(cov, actor, state.period)
    v2 = _get_covariate_value(cov, alter, state.period)
    return v1 == v2 ? 1.0 : 0.0
end

"""
    SimXRecipEffect <: NetworkEffect

Centered similarity × reciprocity:
``s_i = \\sum_j x_{ij} x_{ji} (\\text{sim}_{ij} - \\widehat{sim})``. RSiena: simXRecip
"""
struct SimXRecipEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SimXRecipEffect) = :simXRecip
effect_type(::SimXRecipEffect) = :eval
target_variable(e::SimXRecipEffect) = e.variable
interaction_with(e::SimXRecipEffect) = e.covariate

function evaluate_actor(e::SimXRecipEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0 || net[j, actor] == 0) && continue
        total += _centered_similarity(cov, actor, j, state.period)
    end
    return total
end

function compute_contribution(e::SimXRecipEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    net[alter, actor] == 0 && return 0.0
    return _centered_similarity(data.covariates[e.covariate], actor, alter, state.period)
end

"""
    SimXTransTripEffect <: NetworkEffect

Centered similarity × transitive triplets:
``s_i = \\sum_j x_{ij} (\\text{sim}_{ij} - \\widehat{sim}) \\, \\#\\{h : x_{ih} x_{hj} = 1\\}``.
RSiena: simXTransTrip
"""
struct SimXTransTripEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SimXTransTripEffect) = :simXTransTrip
effect_type(::SimXTransTripEffect) = :eval
target_variable(e::SimXTransTripEffect) = e.variable
interaction_with(e::SimXTransTripEffect) = e.covariate

function evaluate_actor(e::SimXTransTripEffect, state::NetworkState, data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n = size(net, 1)
    total = 0.0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        tp = 0
        for h in 1:n
            (h == actor || h == j) && continue
            if net[actor, h] == 1 && net[h, j] == 1
                tp += 1
            end
        end
        tp > 0 && (total += _centered_similarity(cov, actor, j, state.period) * tp)
    end
    return total
end

#==============================================================================#
# Endowment/Creation Effects
#==============================================================================#

"""
    EndowmentEffect <: NetworkEffect

Wrapper for an endowment effect: the wrapped effect's change statistic enters the
objective function only for tie dissolution (handled in `compute_objective`).
Not yet supported in Method-of-Moments estimation.
"""
struct EndowmentEffect{E<:NetworkEffect} <: NetworkEffect
    base_effect::E
end

effect_name(e::EndowmentEffect) = Symbol(string(effect_name(e.base_effect)), "Endow")
effect_type(::EndowmentEffect) = :endow
target_variable(e::EndowmentEffect) = target_variable(e.base_effect)
interaction_with(e::EndowmentEffect) = interaction_with(e.base_effect)

function compute_contribution(e::EndowmentEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return compute_contribution(e.base_effect, state, data, actor, alter)
end

function compute_statistic(e::EndowmentEffect, state::NetworkState, data::SienaData)
    throw(ArgumentError("endowment effects are not yet supported in Method-of-Moments " *
                        "estimation (effect $(effect_name(e)))"))
end

"""
    CreationEffect <: NetworkEffect

Wrapper for a creation effect: the wrapped effect's change statistic enters the
objective function only for tie creation (handled in `compute_objective`).
Not yet supported in Method-of-Moments estimation.
"""
struct CreationEffect{E<:NetworkEffect} <: NetworkEffect
    base_effect::E
end

effect_name(e::CreationEffect) = Symbol(string(effect_name(e.base_effect)), "Create")
effect_type(::CreationEffect) = :creation
target_variable(e::CreationEffect) = target_variable(e.base_effect)
interaction_with(e::CreationEffect) = interaction_with(e.base_effect)

function compute_contribution(e::CreationEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return compute_contribution(e.base_effect, state, data, actor, alter)
end

function compute_statistic(e::CreationEffect, state::NetworkState, data::SienaData)
    throw(ArgumentError("creation effects are not yet supported in Method-of-Moments " *
                        "estimation (effect $(effect_name(e)))"))
end

#==============================================================================#
# Multiplex Effects
#==============================================================================#

"""
    CrossNetworkReciprocityEffect <: NetworkEffect

Reciprocity from another network: ``s_i = \\sum_j x_{ij} z_{ji}``. RSiena: crprodRecip
"""
struct CrossNetworkReciprocityEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkReciprocityEffect) = :crprodRecip
effect_type(::CrossNetworkReciprocityEffect) = :eval
target_variable(e::CrossNetworkReciprocityEffect) = e.variable
interaction_with(e::CrossNetworkReciprocityEffect) = e.other_network

function evaluate_actor(e::CrossNetworkReciprocityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    other = state.networks[e.other_network]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        count += other[j, actor]
    end
    return Float64(count)
end

function compute_contribution(e::CrossNetworkReciprocityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(state.networks[e.other_network][alter, actor])
end

"""
    CrossNetworkActivityEffect <: NetworkEffect

Ego's outdegree in another network: ``s_i = \\sum_j x_{ij} z_{i+}``. RSiena: crprodAct
"""
struct CrossNetworkActivityEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkActivityEffect) = :crprodAct
effect_type(::CrossNetworkActivityEffect) = :eval
target_variable(e::CrossNetworkActivityEffect) = e.variable
interaction_with(e::CrossNetworkActivityEffect) = e.other_network

function evaluate_actor(e::CrossNetworkActivityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    other = state.networks[e.other_network]
    return Float64(_row_sum(net, actor) * _row_sum(other, actor))
end

function compute_contribution(e::CrossNetworkActivityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(_row_sum(state.networks[e.other_network], actor))
end

"""
    CrossNetworkPopularityEffect <: NetworkEffect

Alter's indegree in another network: ``s_i = \\sum_j x_{ij} z_{+j}``. RSiena: crprodPop
"""
struct CrossNetworkPopularityEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkPopularityEffect) = :crprodPop
effect_type(::CrossNetworkPopularityEffect) = :eval
target_variable(e::CrossNetworkPopularityEffect) = e.variable
interaction_with(e::CrossNetworkPopularityEffect) = e.other_network

function evaluate_actor(e::CrossNetworkPopularityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    other = state.networks[e.other_network]
    n = size(net, 1)
    total = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        total += _col_sum(other, j)
    end
    return Float64(total)
end

function compute_contribution(e::CrossNetworkPopularityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(_col_sum(state.networks[e.other_network], alter))
end

"""
    CrossNetworkTiesEffect <: NetworkEffect

Tie in another network: ``s_i = \\sum_j x_{ij} z_{ij}``. RSiena: crprod
"""
struct CrossNetworkTiesEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkTiesEffect) = :crprod
effect_type(::CrossNetworkTiesEffect) = :eval
target_variable(e::CrossNetworkTiesEffect) = e.variable
interaction_with(e::CrossNetworkTiesEffect) = e.other_network

function evaluate_actor(e::CrossNetworkTiesEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    other = state.networks[e.other_network]
    n = size(net, 1)
    count = 0
    for j in 1:n
        (j == actor || net[actor, j] == 0) && continue
        count += other[actor, j]
    end
    return Float64(count)
end

function compute_contribution(e::CrossNetworkTiesEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(state.networks[e.other_network][actor, alter])
end
