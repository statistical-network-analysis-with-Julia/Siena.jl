"""
Two-mode (bipartite) network effects for SAOM.

Two-mode networks connect actors (rows) to events/affiliations (columns) rather than
to other actors. `evaluate_actor` gives the actor's evaluation component over its
event ties; contributions use closed forms where exact and the generic toggle
fallback otherwise.
"""

#==============================================================================#
# Abstract Type
#==============================================================================#

"""
    TwoModeEffect <: NetworkEffect

Abstract type for two-mode network effects.
# Example
```julia
using Siena
FourCyclesEffect(:attends) isa TwoModeEffect      # true
```
"""
abstract type TwoModeEffect <: NetworkEffect end

# Number of shared events between actors i and o, optionally excluding one event.
function _shared_events(net::AbstractMatrix{Int}, i::Int, o::Int; exclude::Int=0)
    shared = 0
    for e in 1:size(net, 2)
        e == exclude && continue
        shared += net[i, e] * net[o, e]
    end
    return shared
end

#==============================================================================#
# Basic Two-Mode Effects
#==============================================================================#

"""
    TwoModeOutdegreeEffect <: TwoModeEffect

Outdegree (number of events attended): ``s_i = x_{i+}``.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(TwoModeOutdegreeEffect(:attends), state, data)
```
"""
struct TwoModeOutdegreeEffect <: TwoModeEffect
    variable::Symbol
end

effect_name(::TwoModeOutdegreeEffect) = :outdegree2
effect_type(::TwoModeOutdegreeEffect) = :eval
target_variable(e::TwoModeOutdegreeEffect) = e.variable

function evaluate_actor(e::TwoModeOutdegreeEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    return Float64(_row_sum(state.networks[e.variable], actor))
end

function compute_contribution(e::TwoModeOutdegreeEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    return 1.0
end

"""
    TwoModeIndegreeEffect <: TwoModeEffect

Event popularity: ``s_i = \\sum_e x_{ie} f(x_{+e})`` with ``f`` identity or sqrt.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(TwoModeIndegreeEffect(:attends), state, data)
```
"""
struct TwoModeIndegreeEffect <: TwoModeEffect
    variable::Symbol
    sqrt::Bool
end

TwoModeIndegreeEffect(var::Symbol; sqrt::Bool=false) =
    TwoModeIndegreeEffect(var, sqrt)

effect_name(e::TwoModeIndegreeEffect) = e.sqrt ? :indegreeSqrt2 : :indegree2
effect_type(::TwoModeIndegreeEffect) = :eval
target_variable(e::TwoModeIndegreeEffect) = e.variable

function evaluate_actor(e::TwoModeIndegreeEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    total = 0.0
    for ev in 1:size(net, 2)
        net[actor, ev] == 0 && continue
        indeg = Float64(_col_sum(net, ev))
        total += e.sqrt ? sqrt(indeg) : indeg
    end
    return total
end

function compute_contribution(e::TwoModeIndegreeEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    d = Float64(_col_sum(net, event) - net[actor, event] + 1)  # indegree with the tie
    return e.sqrt ? sqrt(d) : d
end

#==============================================================================#
# Shared Partners Effects (Two-Mode)
#==============================================================================#

"""
    FourCyclesEffect <: TwoModeEffect

Four-cycles of a two-mode network: ``s_i = \\sum_{o \\ne i} \\binom{s_{io}}{2}``
where ``s_{io}`` is the number of shared events of actors ``i`` and ``o``; the
change statistic of a tie is the number of four-cycles it closes. RSiena: `cycle4`
(internal parameter 1) for bipartite networks; the target statistic is the number
of four-cycles (half the actor sum), as RSiena's. Its short name was `fourCycles`
before 0.2 (still accepted by `include_effects!`).

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(FourCyclesEffect(:attends), state, data)
```
"""
struct FourCyclesEffect <: TwoModeEffect
    variable::Symbol
end

effect_name(::FourCyclesEffect) = :cycle4

# Every four-cycle has two actors: the actor sum counts it twice.
compute_statistic(e::FourCyclesEffect, state::NetworkState, data::SienaData) =
    _actor_sum(e, state, data) / 2
effect_type(::FourCyclesEffect) = :eval
target_variable(e::FourCyclesEffect) = e.variable

function evaluate_actor(e::FourCyclesEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    total = 0.0
    for o in 1:size(net, 1)
        o == actor && continue
        s = _shared_events(net, actor, o)
        total += s * (s - 1) / 2
    end
    return total
end

function compute_contribution(e::FourCyclesEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    # Adding the tie i-event turns each pre-existing shared event with a co-attendee
    # into a new four-cycle: binom(s+1, 2) - binom(s, 2) = s.
    total = 0.0
    for o in 1:size(net, 1)
        (o == actor || net[o, event] == 0) && continue
        total += _shared_events(net, actor, o; exclude=event)
    end
    return total
end

"""
    SharedEventsEffect <: TwoModeEffect

Total shared events with other actors: ``s_i = \\sum_{o \\ne i} f(s_{io})`` with
``f`` identity or sqrt.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(SharedEventsEffect(:attends), state, data)
```
"""
struct SharedEventsEffect <: TwoModeEffect
    variable::Symbol
    sqrt::Bool
end

SharedEventsEffect(var::Symbol; sqrt::Bool=false) =
    SharedEventsEffect(var, sqrt)

effect_name(e::SharedEventsEffect) = e.sqrt ? :sharedEventsSqrt : :sharedEvents
effect_type(::SharedEventsEffect) = :eval
target_variable(e::SharedEventsEffect) = e.variable

function evaluate_actor(e::SharedEventsEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    total = 0.0
    for o in 1:size(net, 1)
        o == actor && continue
        s = _shared_events(net, actor, o)
        total += e.sqrt ? sqrt(Float64(s)) : Float64(s)
    end
    return total
end

function compute_contribution(e::SharedEventsEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    total = 0.0
    for o in 1:size(net, 1)
        (o == actor || net[o, event] == 0) && continue
        s = _shared_events(net, actor, o; exclude=event)
        total += e.sqrt ? sqrt(Float64(s + 1)) - sqrt(Float64(s)) : 1.0
    end
    return total
end

"""
    GWESPTwoModeEffect <: TwoModeEffect

Geometrically weighted shared events:
``s_i = \\sum_{o \\ne i} (1 - (1 - e^{-\\alpha})^{s_{io}})``.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(GWESPTwoModeEffect(:attends), state, data)
```
"""
struct GWESPTwoModeEffect <: TwoModeEffect
    variable::Symbol
    α::Float64  # Decay parameter
end

GWESPTwoModeEffect(var::Symbol; α::Float64=0.69) =
    GWESPTwoModeEffect(var, α)

effect_name(::GWESPTwoModeEffect) = :gwesp2
effect_type(::GWESPTwoModeEffect) = :eval
target_variable(e::GWESPTwoModeEffect) = e.variable

function evaluate_actor(e::GWESPTwoModeEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    total = 0.0
    for o in 1:size(net, 1)
        o == actor && continue
        s = _shared_events(net, actor, o)
        s > 0 && (total += 1.0 - (1.0 - exp(-e.α))^s)
    end
    return total
end

#==============================================================================#
# Covariate Effects (Two-Mode)
#==============================================================================#

"""
    TwoModeEgoEffect <: TwoModeEffect

Actor covariate effect on event attendance: ``s_i = v_i x_{i+}``.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(TwoModeEgoEffect(:attends, :age), state, data)
```
"""
struct TwoModeEgoEffect <: TwoModeEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(e::TwoModeEgoEffect) = Symbol("ego2$(e.covariate)")
effect_type(::TwoModeEgoEffect) = :eval
target_variable(e::TwoModeEgoEffect) = e.variable
interaction_with(e::TwoModeEgoEffect) = e.covariate

function evaluate_actor(e::TwoModeEgoEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    v = _get_covariate_value(data.covariates[e.covariate], actor, state.period)
    return v * _row_sum(net, actor)
end

function compute_contribution(e::TwoModeEgoEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    return _get_covariate_value(data.covariates[e.covariate], actor, state.period)
end

"""
    TwoModeEventEffect <: TwoModeEffect

Event attribute effect: ``s_i = \\sum_e x_{ie} w_{ie}`` using a dyadic covariate
whose rows are actors and columns are events.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(TwoModeEventEffect(:attends, :fee), state, data)
```
"""
struct TwoModeEventEffect <: TwoModeEffect
    variable::Symbol
    event_covariate::Symbol  # Should be a dyadic covariate
end

effect_name(e::TwoModeEventEffect) = Symbol("event2$(e.event_covariate)")
effect_type(::TwoModeEventEffect) = :eval
target_variable(e::TwoModeEventEffect) = e.variable
interaction_with(e::TwoModeEventEffect) = e.event_covariate

function evaluate_actor(e::TwoModeEventEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.event_covariate]
    total = 0.0
    for ev in 1:size(net, 2)
        net[actor, ev] == 0 && continue
        total += _get_dyad_covariate_value(cov, actor, ev, state.period)
    end
    return total
end

function compute_contribution(e::TwoModeEventEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    return _get_dyad_covariate_value(data.covariates[e.event_covariate], actor, event,
                                     state.period)
end

"""
    TwoModeSameEffect <: TwoModeEffect

Attendance with same-covariate others:
``s_i = \\sum_e x_{ie} \\#\\{o \\ne i : x_{oe} = 1, |v_i - v_o| < 0.5\\}``.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(TwoModeSameEffect(:attends, :age), state, data)
```
"""
struct TwoModeSameEffect <: TwoModeEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(e::TwoModeSameEffect) = Symbol("same2$(e.covariate)")
effect_type(::TwoModeSameEffect) = :eval
target_variable(e::TwoModeSameEffect) = e.variable
interaction_with(e::TwoModeSameEffect) = e.covariate

function evaluate_actor(e::TwoModeSameEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    ego_val = _get_covariate_value(cov, actor, state.period)
    total = 0
    for ev in 1:size(net, 2)
        net[actor, ev] == 0 && continue
        for o in 1:size(net, 1)
            (o == actor || net[o, ev] == 0) && continue
            if abs(ego_val - _get_covariate_value(cov, o, state.period)) < 0.5
                total += 1
            end
        end
    end
    return Float64(total)
end

function compute_contribution(e::TwoModeSameEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    ego_val = _get_covariate_value(cov, actor, state.period)
    count = 0
    for o in 1:size(net, 1)
        (o == actor || net[o, event] == 0) && continue
        if abs(ego_val - _get_covariate_value(cov, o, state.period)) < 0.5
            count += 1
        end
    end
    return Float64(count)
end

"""
    TwoModeSimilarityEffect <: TwoModeEffect

Covariate similarity with co-attendees:
``s_i = \\sum_e x_{ie} \\sum_{o \\ne i} x_{oe} \\, \\text{sim}_{io}``.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(TwoModeSimilarityEffect(:attends, :age), state, data)
```
"""
struct TwoModeSimilarityEffect <: TwoModeEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(e::TwoModeSimilarityEffect) = Symbol("sim2$(e.covariate)")
effect_type(::TwoModeSimilarityEffect) = :eval
target_variable(e::TwoModeSimilarityEffect) = e.variable
interaction_with(e::TwoModeSimilarityEffect) = e.covariate

function _twomode_similarity(cov::AbstractCovariate, i::Int, o::Int, wave::Int)
    r = max(_get_covariate_range(cov), 1e-10)
    return 1.0 - abs(_get_covariate_value(cov, i, wave) -
                     _get_covariate_value(cov, o, wave)) / r
end

function evaluate_actor(e::TwoModeSimilarityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    total = 0.0
    for ev in 1:size(net, 2)
        net[actor, ev] == 0 && continue
        for o in 1:size(net, 1)
            (o == actor || net[o, ev] == 0) && continue
            total += _twomode_similarity(cov, actor, o, state.period)
        end
    end
    return total
end

function compute_contribution(e::TwoModeSimilarityEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    total = 0.0
    for o in 1:size(net, 1)
        (o == actor || net[o, event] == 0) && continue
        total += _twomode_similarity(cov, actor, o, state.period)
    end
    return total
end

#==============================================================================#
# Degree-Based Effects (Two-Mode)
#==============================================================================#

"""
    TwoModeActivityEffect <: TwoModeEffect

Co-attendee activity: ``s_i = \\sum_e x_{ie} \\sum_{o \\ne i} x_{oe} f(x_{o+})``.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(TwoModeActivityEffect(:attends), state, data)
```
"""
struct TwoModeActivityEffect <: TwoModeEffect
    variable::Symbol
    sqrt::Bool
end

TwoModeActivityEffect(var::Symbol; sqrt::Bool=false) =
    TwoModeActivityEffect(var, sqrt)

effect_name(e::TwoModeActivityEffect) = e.sqrt ? :activitySqrt2 : :activity2
effect_type(::TwoModeActivityEffect) = :eval
target_variable(e::TwoModeActivityEffect) = e.variable

function evaluate_actor(e::TwoModeActivityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    total = 0.0
    for ev in 1:size(net, 2)
        net[actor, ev] == 0 && continue
        for o in 1:size(net, 1)
            (o == actor || net[o, ev] == 0) && continue
            outdeg = Float64(_row_sum(net, o))
            total += e.sqrt ? sqrt(outdeg) : outdeg
        end
    end
    return total
end

function compute_contribution(e::TwoModeActivityEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    total = 0.0
    for o in 1:size(net, 1)
        (o == actor || net[o, event] == 0) && continue
        outdeg = Float64(_row_sum(net, o))
        total += e.sqrt ? sqrt(outdeg) : outdeg
    end
    return total
end

"""
    TwoModePopularityAltEffect <: TwoModeEffect

Actors with high outdegree attend popular events:
``s_i = x_{i+} \\sum_e x_{ie} (x_{+e} - 1)``.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(TwoModePopularityAltEffect(:attends), state, data)
```
"""
struct TwoModePopularityAltEffect <: TwoModeEffect
    variable::Symbol
end

effect_name(::TwoModePopularityAltEffect) = :popAlt2
effect_type(::TwoModePopularityAltEffect) = :eval
target_variable(e::TwoModePopularityAltEffect) = e.variable

function evaluate_actor(e::TwoModePopularityAltEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    outdeg = _row_sum(net, actor)
    total = 0
    for ev in 1:size(net, 2)
        net[actor, ev] == 0 && continue
        total += _col_sum(net, ev) - 1
    end
    return Float64(outdeg * total)
end

#==============================================================================#
# Closure Effects (Two-Mode)
#==============================================================================#

"""
    TwoModeTransitiveClosureEffect <: TwoModeEffect

Closure of four-paths: ``s_i = \\sum_{o \\ne i} s_{io} (s_{io} - 1)`` (twice the
four-cycle count).

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(TwoModeTransitiveClosureEffect(:attends), state, data)
```
"""
struct TwoModeTransitiveClosureEffect <: TwoModeEffect
    variable::Symbol
end

effect_name(::TwoModeTransitiveClosureEffect) = :transClosure2
effect_type(::TwoModeTransitiveClosureEffect) = :eval
target_variable(e::TwoModeTransitiveClosureEffect) = e.variable

function evaluate_actor(e::TwoModeTransitiveClosureEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    total = 0
    for o in 1:size(net, 1)
        o == actor && continue
        s = _shared_events(net, actor, o)
        total += s * (s - 1)
    end
    return Float64(total)
end

"""
    TwoModeActorAssortativityEffect <: TwoModeEffect

Outdegree assortativity with co-attendees:
``s_i = \\sum_e x_{ie} \\sum_{o \\ne i} x_{oe} \\, x_{i+} x_{o+}``.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(TwoModeActorAssortativityEffect(:attends), state, data)
```
"""
struct TwoModeActorAssortativityEffect <: TwoModeEffect
    variable::Symbol
end

effect_name(::TwoModeActorAssortativityEffect) = :actAssort2
effect_type(::TwoModeActorAssortativityEffect) = :eval
target_variable(e::TwoModeActorAssortativityEffect) = e.variable

function evaluate_actor(e::TwoModeActorAssortativityEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    ego_outdeg = _row_sum(net, actor)
    total = 0
    for ev in 1:size(net, 2)
        net[actor, ev] == 0 && continue
        for o in 1:size(net, 1)
            (o == actor || net[o, ev] == 0) && continue
            total += ego_outdeg * _row_sum(net, o)
        end
    end
    return Float64(total)
end

#==============================================================================#
# Between/Within Effects (Two-Mode with Settings)
#==============================================================================#

"""
    TwoModeWithinEffect <: TwoModeEffect

Attendance of events within the actor's own setting:
``s_i = \\sum_e x_{ie} I(|v_i - w_{ie}| < 0.5)`` where ``v`` is the actor setting
covariate and ``w`` is a dyadic covariate whose ``(i, e)`` entry gives the setting of
event ``e`` (typically constant across rows). Both are compared as given: the
covariates' centring is undone, so actor and event settings use the same codes. A
Siena.jl effect with no RSiena counterpart.

# Example
```julia
using Siena
x1 = [1 0 1 0; 0 1 1 0; 1 1 0 0; 0 0 1 1; 1 0 0 1]
x2 = [1 1 1 0; 0 1 1 0; 1 1 0 1; 0 0 1 1; 0 0 1 1]
data = siena_data()
add_nodeset!(data, NodeSet(5))
add_nodeset!(data, NodeSet(4; id=:events))
add_dependent!(data, DependentNetwork(:attends, [x1, x2]; type=:twomode, nodeset2=:events))
add_covariate!(data, ConstantCovariate(:age, [20, 30, 40, 50, 60]))
add_covariate!(data, ConstantCovariate(:club, [1, 1, 2, 2, 1]))
add_covariate!(data, ConstantDyadCovariate(:fee, [Float64(e) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
add_covariate!(data, ConstantDyadCovariate(:eventclub, [Float64(1 + (e > 2)) for i in 1:5, e in 1:4];
                                           nodeset2=:events))
state = initialize!(NetworkState(), data, 2; period=1)
compute_statistic(TwoModeWithinEffect(:attends, :club, :eventclub), state, data)
```
"""
struct TwoModeWithinEffect <: TwoModeEffect
    variable::Symbol
    setting::Symbol  # Covariate indicating setting for each actor
    event_setting::Symbol  # Dyadic covariate indicating event settings
end

effect_name(e::TwoModeWithinEffect) = Symbol("within2$(e.setting)")
effect_type(::TwoModeWithinEffect) = :eval
target_variable(e::TwoModeWithinEffect) = e.variable
interaction_with(e::TwoModeWithinEffect) = e.setting

function _within_setting(e::TwoModeWithinEffect, data::SienaData, actor::Int,
                         event::Int, wave::Int)
    v = _raw_covariate_value(data.covariates[e.setting], actor, wave)
    w = _raw_dyad_covariate_value(data.covariates[e.event_setting], actor, event, wave)
    return abs(v - w) < 0.5 ? 1.0 : 0.0
end

function evaluate_actor(e::TwoModeWithinEffect, state::NetworkState,
                        data::SienaData, actor::Int)
    net = state.networks[e.variable]
    total = 0.0
    for ev in 1:size(net, 2)
        net[actor, ev] == 0 && continue
        total += _within_setting(e, data, actor, ev, state.period)
    end
    return total
end

function compute_contribution(e::TwoModeWithinEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    return _within_setting(e, data, actor, event, state.period)
end
