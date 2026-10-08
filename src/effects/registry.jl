"""
Which effects exist for which dependent variable, and the effects table that
`get_effects` builds from that.

RSiena groups its effects by the kind of dependent variable they apply to
(`nonSymmetricObjective`, `symmetricObjective`, `bipartiteObjective`,
`behaviorOneModeObjective`, ...). The same grouping decides here what
[`get_effects`](@ref) offers and what [`fit_siena`](@ref) accepts: an effect
included for a variable of a kind it is not defined for is refused with an
`ArgumentError` instead of being computed on the wrong index set (a one-mode
effect on a two-mode network used to read events as actors, or throw a
`BoundsError`).
"""

#==============================================================================#
# Network kinds an effect is defined for
#==============================================================================#

# Kinds of the network an effect reads: `:directed` (one-mode, non-symmetric),
# `:undirected` (one-mode, symmetric) and `:twomode` (bipartite). For network effects
# this is the target network; for behaviour and rate effects it is the network the
# effect refers to (its `network` field).
_supported_kinds(::NetworkEffect) = (:directed,)
_supported_kinds(::TwoModeEffect) = (:twomode,)
_supported_kinds(e::InteractionEffect) =
    Tuple(intersect(map(_supported_kinds, e.components)...))
_supported_kinds(e::BehaviorProductEffect) =
    Tuple(intersect(map(_supported_kinds, e.components)...))
_supported_kinds(e::EndowmentEffect) = _supported_kinds(e.base_effect)
_supported_kinds(e::CreationEffect) = _supported_kinds(e.base_effect)
_supported_kinds(::BehaviorEffect) = (:directed, :undirected)
_supported_kinds(::RateEffect) = (:directed, :undirected)

# Network effects RSiena also defines for symmetric networks (symmetricObjective /
# covarSymmetricObjective / dyadObjective), with the same change statistic.
for T in (TransitiveTiesEffect, BetweennessEffect, NbrDist2Effect,
          IsolateNetEffect, GWESPEffect,
          AlterEffect, AlterSqEffect, SimilarityEffect,
          SameEffect, DifferenceEffect, DifferenceSqEffect, AbsDifferenceEffect,
          HigherEffect, EgoTimesAlterEffect, EgoPlusAlterEffect)
    @eval _supported_kinds(::$T) = (:directed, :undirected)
end
# Effects RSiena also defines for two-mode (bipartite) networks, whose
# implementation only uses ego's row and the alters' columns.
for T in (OutdegreeEffect, IndegreePopularityEffect, OutdegreeActivityEffect,
          OutdegreeTruncEffect, OutIsolateEffect, EgoEffect, EgoSqEffect,
          DyadCovariateEffect)
    @eval _supported_kinds(::$T) = (:directed, :undirected, :twomode)
end
for T in (OutdegreeLogRateEffect, OutdegreeInvRateEffect)
    @eval _supported_kinds(::$T) = (:directed, :undirected, :twomode)
end
# Defined for symmetric networks only.
_supported_kinds(::TransitiveTriadsEffect) = (:undirected,)
_supported_kinds(::DegreeAssortativityEffect) = (:undirected,)

# Behaviour and rate effects that use in-degrees or reciprocity are only defined
# for directed networks; the out-degree ones also for two-mode networks.
for T in (IndegreeEffect, AverageInAlterEffect, AverageRecipAlterEffect,
          TotalInAlterEffect, RecipDegreeEffect,
          IndegreeRateEffect, IndegreeLogRateEffect, IndegreeInvRateEffect,
          RecipDegreeRateEffect)
    @eval _supported_kinds(::$T) = (:directed,)
end
_supported_kinds(::BehaviorOutdegreeEffect) = (:directed, :undirected, :twomode)
_supported_kinds(::OutdegreeRateEffect) = (:directed, :undirected, :twomode)

_kind_phrase(k::Symbol) = k === :directed ? "directed one-mode" :
                          k === :undirected ? "undirected (symmetric) one-mode" :
                          "two-mode (bipartite)"

# Effects whose `covariate` may also name a co-evolving dependent behaviour.
const _BEHAVIOR_AS_COVARIATE = Union{EgoEffect, EgoSqEffect, AlterEffect,
    AlterSqEffect, SimilarityEffect, SameEffect, DifferenceEffect, DifferenceSqEffect,
    AbsDifferenceEffect, HigherEffect, EgoTimesAlterEffect, EgoPlusAlterEffect,
    SameXRecipEffect, SimXRecipEffect, SimXTransTripEffect}

#==============================================================================#
# Validation of a model against its data
#==============================================================================#

_label(entry::EffectEntry) = "$(entry.shortname) ($(effect_name(entry.effect)))"

function _require_network(data::SienaData, name::Symbol, entry::EffectEntry,
                          role::AbstractString)
    dep = get(data.dependents, name, nothing)
    dep isa DependentNetwork || throw(ArgumentError(
        "effect $(_label(entry)) refers to $role :$name, which is not a dependent " *
        "network of the data"))
    return dep
end

function _check_kind(entry::EffectEntry, dep::DependentNetwork)
    kind = _network_kind(dep)
    kinds = _supported_kinds(entry.effect)
    kind in kinds || throw(ArgumentError(
        "effect $(_label(entry)) is defined for " *
        join(_kind_phrase.(kinds), " and ") * " networks, but :$(dep.name) is a " *
        "$(_kind_phrase(kind)) network. Use get_effects(data) to see the effects " *
        "available for it."))
    return nothing
end

function _check_actor_covariate(data::SienaData, entry::EffectEntry, name::Symbol;
                                behaviour_ok::Bool)
    cov = get(data.covariates, name, nothing)
    if cov === nothing
        if behaviour_ok && get(data.dependents, name, nothing) isa DependentBehavior
            return nothing
        end
        throw(ArgumentError("effect $(_label(entry)) refers to :$name, which is " *
                            "not an actor covariate" *
                            (behaviour_ok ? " or dependent behaviour" : "") *
                            " of the data"))
    end
    cov isa Union{ConstantCovariate, VaryingCovariate} || throw(ArgumentError(
        "effect $(_label(entry)) needs an actor covariate, but :$name is a " *
        "dyadic covariate"))
    return nothing
end

"""
    validate_effects(data::SienaData, effects::SienaEffects) -> Nothing

Check that every *included* effect is defined for the variables it refers to, and
throw an `ArgumentError` naming the first one that is not. [`fit_siena`](@ref),
[`simulate_saom`](@ref) and the GOF functions call it, so a model that does not
make sense for its data is refused rather than computed on the wrong index set:

- network effects must target a dependent network of a kind they are defined for
  (directed / undirected / two-mode, following RSiena's effect groups — e.g.
  `recip` and `transTrip` only exist for directed networks, `transTriads` only for
  undirected ones, and one-mode effects never apply to a two-mode network);
- behaviour effects must target a dependent behaviour, and the network they read
  must be of a supported kind;
- covariates and other dependent variables the effect reads must exist and be of
  the right type (actor vs dyadic covariate, matching dimensions).

# Example
```julia
using Siena
u1 = [0 1 0; 1 0 1; 0 1 0]
u2 = [0 1 1; 1 0 0; 1 0 0]
data = siena_data()
add_dependent!(data, DependentNetwork(:friends, [u1, u2]; directed=false))
effects = get_effects(data)                      # includes density (outdegree)
include_effects!(effects, :friends, [:density, :transTriads])
validate_effects(data, effects)          # passes: both are symmetric-network effects
```
"""
function validate_effects(data::SienaData, effects::SienaEffects)
    for entry in effects.effects
        entry.include || continue
        _validate_entry(data, entry)
    end
    return nothing
end

function _validate_entry(data::SienaData, entry::EffectEntry)
    eff = entry.effect
    if eff isa Union{InteractionEffect, BehaviorProductEffect}
        # An interaction is valid when each component is, and (for networks) when
        # RSiena's interaction rules hold for this kind of network.
        for c in eff.components
            _validate_entry(data, EffectEntry(c; shortname=entry.shortname,
                                              name=entry.name, include=true))
        end
        if eff isa InteractionEffect
            dep = get(data.dependents, eff.variable, nothing)
            dep isa DependentNetwork || throw(ArgumentError(
                "interaction $(_label(entry)) targets :$(eff.variable), which is " *
                "not a dependent network"))
            _check_network_interaction(eff.components, _network_kind(dep))
        end
        return nothing
    end
    v = target_variable(eff)
    dep = get(data.dependents, v, nothing)
    dep === nothing && throw(ArgumentError(
        "effect $(_label(entry)) targets :$v, which is not a dependent variable of " *
        "the data"))
    if eff isa NetworkEffect
        dep isa DependentNetwork || throw(ArgumentError(
            "network effect $(_label(entry)) targets :$v, which is a behaviour"))
        _check_kind(entry, dep)
        base = eff isa Union{EndowmentEffect, CreationEffect} ? eff.base_effect : eff
        if hasfield(typeof(base), :covariate)
            name = getfield(base, :covariate)
            if base isa DyadCovariateEffect
                cov = get(data.covariates, name, nothing)
                cov isa Union{ConstantDyadCovariate, VaryingDyadCovariate} ||
                    throw(ArgumentError("effect $(_label(entry)) needs a dyadic " *
                                        "covariate, but :$name is not one"))
                sz = cov isa ConstantDyadCovariate ? size(cov.values) :
                     size(cov.values[1])
                sz == size(dep.networks[1]) || throw(ArgumentError(
                    "dyadic covariate :$name is $(sz[1])×$(sz[2]) but network :$v " *
                    "is $(size(dep.networks[1], 1))×$(size(dep.networks[1], 2))"))
            elseif base isa TwoModeEffect
                # two-mode covariate effects read actor or actor×event covariates;
                # the accessors refuse a covariate of the wrong type
                haskey(data.covariates, name) || throw(ArgumentError(
                    "effect $(_label(entry)) refers to :$name, which is not a " *
                    "covariate of the data"))
            else
                _check_actor_covariate(data, entry, name;
                                       behaviour_ok=base isa _BEHAVIOR_AS_COVARIATE)
            end
        end
        if hasfield(typeof(base), :other_network)
            other = _require_network(data, getfield(base, :other_network), entry,
                                     "the other network")
            size(other.networks[1]) == size(dep.networks[1]) &&
                _network_kind(other) == _network_kind(dep) || throw(ArgumentError(
                "effect $(_label(entry)): networks :$v and :$(other.name) must " *
                "have the same node sets and kind"))
        end
    elseif eff isa BehaviorEffect
        dep isa DependentBehavior || throw(ArgumentError(
            "behaviour effect $(_label(entry)) targets :$v, which is a network"))
        if hasfield(typeof(eff), :network)
            net = _require_network(data, getfield(eff, :network), entry, "network")
            _check_kind(entry, net)
        end
        if hasfield(typeof(eff), :covariate)
            _check_actor_covariate(data, entry, getfield(eff, :covariate);
                                   behaviour_ok=false)
        end
        for f in (:other_behavior,)
            if hasfield(typeof(eff), f)
                o = getfield(eff, f)
                get(data.dependents, o, nothing) isa DependentBehavior ||
                    throw(ArgumentError("effect $(_label(entry)) refers to :$o, " *
                                        "which is not a dependent behaviour"))
            end
        end
    elseif eff isa RateEffect
        if hasfield(typeof(eff), :network)
            net = _require_network(data, getfield(eff, :network), entry, "network")
            _check_kind(entry, net)
        end
        for f in (:covariate, :setting)
            hasfield(typeof(eff), f) &&
                _check_actor_covariate(data, entry, getfield(eff, f); behaviour_ok=false)
        end
        if hasfield(typeof(eff), :behavior)
            b = getfield(eff, :behavior)
            get(data.dependents, b, nothing) isa DependentBehavior ||
                throw(ArgumentError("effect $(_label(entry)) refers to :$b, which is " *
                                    "not a dependent behaviour"))
        end
    end
    return nothing
end

#==============================================================================#
# The effects table built by get_effects
#==============================================================================#

_entry(eff, shortname, name; kw...) =
    EffectEntry(eff; name=String(name), shortname=String(shortname), kw...)

# Actor covariates and the dependent behaviours that network effects may condition
# on, in a deterministic order.
_actor_covariates(data::SienaData) =
    sort!([k for (k, c) in data.covariates
           if c isa Union{ConstantCovariate, VaryingCovariate}]; by=String)
_dyadic_covariates(data::SienaData) =
    sort!([k for (k, c) in data.covariates
           if c isa Union{ConstantDyadCovariate, VaryingDyadCovariate}]; by=String)
_dependents_of(data::SienaData, T) =
    sort!([k for (k, d) in data.dependents if d isa T]; by=String)

function _network_entries!(out::Vector{EffectEntry}, data::SienaData, v::Symbol,
                           dep::DependentNetwork)
    kind = _network_kind(dep)
    n = size(dep.networks[1], 1)
    if kind === :twomode
        # RSiena's bipartite effects ...
        push!(out,
            _entry(OutdegreeEffect(v), "outdegree", "outdegree (density)"),
            _entry(FourCyclesEffect(v), "cycle4", "4 cycles (#)"),
            _entry(IndegreePopularityEffect(v), "inPop", "indegree - popularity"),
            _entry(IndegreePopularityEffect(v; sqrt=true), "inPopSqrt",
                   "indegree - popularity (sqrt)"),
            _entry(OutdegreeActivityEffect(v), "outAct", "outdegree - activity"),
            _entry(OutdegreeActivityEffect(v; sqrt=true), "outActSqrt",
                   "outdegree - activity (sqrt)"),
            _entry(OutdegreeTruncEffect(v), "outTrunc", "outdegree-trunc(#)"),
            _entry(OutIsolateEffect(v), "outIso", "out-isolate"))
        # ... and Siena.jl's own two-mode effects (no RSiena counterpart).
        push!(out,
            _entry(SharedEventsEffect(v), "sharedEvents", "shared events (Siena.jl)"),
            _entry(SharedEventsEffect(v; sqrt=true), "sharedEventsSqrt",
                   "shared events (sqrt) (Siena.jl)"),
            _entry(GWESPTwoModeEffect(v), "gwesp2", "GW shared events (Siena.jl)"),
            _entry(TwoModeActivityEffect(v), "activity2",
                   "co-attendee activity (Siena.jl)"),
            _entry(TwoModeActivityEffect(v; sqrt=true), "activitySqrt2",
                   "co-attendee activity (sqrt) (Siena.jl)"),
            _entry(TwoModePopularityAltEffect(v), "popAlt2",
                   "popular events by active actors (Siena.jl)"),
            _entry(TwoModeTransitiveClosureEffect(v), "transClosure2",
                   "four-path closure (Siena.jl)"),
            _entry(TwoModeActorAssortativityEffect(v), "actAssort2",
                   "co-attendee assortativity (Siena.jl)"))
        attrs2 = vcat(_actor_covariates(data),
                      [b for b in _dependents_of(data, DependentBehavior)
                       if n_actors(data.dependents[b]) == n])
        for c in attrs2
            push!(out,
                _entry(EgoEffect(v, c), "ego$c", "$c ego"),
                _entry(EgoSqEffect(v, c), "egoSq$c", "$c squared ego"))
        end
        for c in _actor_covariates(data)
            push!(out,
                _entry(TwoModeSameEffect(v, c), "same2$c",
                       "same $c co-attendance (Siena.jl)"),
                _entry(TwoModeSimilarityEffect(v, c), "sim2$c",
                       "$c similarity co-attendance (Siena.jl)"))
        end
        for c in _dyadic_covariates(data)
            cov = data.covariates[c]
            sz = cov isa ConstantDyadCovariate ? size(cov.values) : size(cov.values[1])
            sz == size(dep.networks[1]) &&
                push!(out, _entry(DyadCovariateEffect(v, c), "dyad$c", "$c (dyadic)"))
        end
        return out
    end

    directed = kind === :directed
    push!(out, _entry(OutdegreeEffect(v), "outdegree",
                      directed ? "outdegree (density)" : "degree (density)"))
    if directed
        push!(out,
            _entry(ReciprocityEffect(v), "recip", "reciprocity"),
            _entry(TransitiveTripletsEffect(v), "transTrip", "transitive triplets"),
            _entry(TransitiveMediatedTripletsEffect(v), "transMedTrip",
                   "transitive mediated triplets"),
            _entry(TransitiveRecipTripletsEffect(v), "transRecTrip",
                   "transitive recipr. triplets"),
            _entry(CyclicTripletsEffect(v), "cycle3", "3-cycles"))
    else
        push!(out, _entry(TransitiveTriadsEffect(v), "transTriads", "transitive triads"))
    end
    push!(out,
        _entry(TransitiveTiesEffect(v), "transTies", "transitive ties"),
        _entry(BetweennessEffect(v), "between", "betweenness"),
        _entry(NbrDist2Effect(v), "nbrDist2",
               directed ? "number of actors at dist 2" : "number of actor pairs at dist 2"))
    if directed
        push!(out,
            _entry(DenseTriadsEffect(v), "denseTriads", "dense triads"),
            _entry(BalanceSimpleEffect(v), "balanceSimple",
                   "balance (Siena.jl simplified; not RSiena's balance)"),
            _entry(SharedInEffect(v), "sharedInNbrs",
                   "ties to alters with shared in-neighbours (Siena.jl)"),
            _entry(SharedOutEffect(v), "sharedOutNbrs",
                   "ties to alters with shared out-neighbours (Siena.jl)"),
            _entry(GWESPEffect(v), "gwespFF", "GWESP I -> K -> J (#)"),
            _entry(GWESPBackwardEffect(v), "gwespBB", "GWESP I <- K <- J (#)"),
            _entry(GWESPMixedEffect(v), "gwespFB", "GWESP I -> K <- J (#)"),
            _entry(GWDSPEffect(v), "gwdspFF", "GWDSP I -> K -> J (#)"),
            _entry(IndegreePopularityEffect(v), "inPop", "indegree - popularity"),
            _entry(IndegreePopularityEffect(v; sqrt=true), "inPopSqrt",
                   "indegree - popularity (sqrt)"),
            _entry(OutdegreePopularityEffect(v), "outPop", "outdegree - popularity(#)"),
            _entry(OutdegreePopularityEffect(v; sqrt=true), "outPopSqrt",
                   "outdegree - popularity (sqrt)"),
            _entry(IndegreeActivityEffect(v), "inAct", "indegree - activity(#)"),
            _entry(IndegreeActivityEffect(v; sqrt=true), "inActSqrt",
                   "indegree - activity (sqrt)"),
            _entry(OutdegreeActivityEffect(v), "outAct", "outdegree - activity"),
            _entry(OutdegreeActivityEffect(v; sqrt=true), "outActSqrt",
                   "outdegree - activity (sqrt)"),
            _entry(OutdegreeTruncEffect(v), "outTrunc", "outdegree-trunc(#)"),
            _entry(IndegreeTruncEffect(v), "inTrunc", "truncated indegree popularity (Siena.jl)"),
            _entry(IsolateNetEffect(v), "isolateNet", "network-isolate"),
            _entry(OutIsolateEffect(v), "outIso", "out-isolate"),
            _entry(InIsolateEffect(v), "inIsolate", "ties to in-isolates (Siena.jl)"))
    else
        push!(out,
            _entry(GWESPEffect(v), "gwesp", "GWESP (#)"),
            _entry(IndegreePopularityEffect(v), "inPop", "degree of alter"),
            _entry(IndegreePopularityEffect(v; sqrt=true), "inPopSqrt",
                   "sqrt degree of alter"),
            _entry(OutdegreeActivityEffect(v), "outAct", "degree of ego"),
            _entry(OutdegreeActivityEffect(v; sqrt=true), "outActSqrt", "degree^(1.5)"),
            _entry(DegreeAssortativityEffect(v), "degPlus", "degree act+pop"),
            _entry(OutdegreeTruncEffect(v), "outTrunc", "outdegree-trunc(#)"),
            _entry(IsolateNetEffect(v), "isolateNet", "network-isolate"),
            _entry(OutIsolateEffect(v), "outIso", "out-isolate"))
    end

    # Actor covariates, and co-evolving dependent behaviours (network selection on
    # a behaviour), with RSiena's covariate effects.
    attrs = vcat(_actor_covariates(data),
                 [b for b in _dependents_of(data, DependentBehavior)
                  if n_actors(data.dependents[b]) == n])
    for c in attrs
        push!(out,
            _entry(EgoEffect(v, c), "ego$c", "$c ego"),
            _entry(AlterEffect(v, c), "alt$c", "$c alter"),
            _entry(SimilarityEffect(v, c), "sim$c", "$c similarity"),
            _entry(SameEffect(v, c), "same$c", "same $c"),
            _entry(EgoSqEffect(v, c), "egoSq$c", "$c squared ego"),
            _entry(AlterSqEffect(v, c), "altSq$c", "$c squared alter"),
            _entry(DifferenceEffect(v, c), "diff$c", "$c difference"),
            _entry(DifferenceSqEffect(v, c), "diffSq$c", "$c squared difference"),
            _entry(AbsDifferenceEffect(v, c), "absDiff$c", "$c absolute difference"),
            _entry(HigherEffect(v, c), "higher$c", "higher $c"),
            _entry(EgoTimesAlterEffect(v, c), "ego$(c)alt$c", "$c ego x $c alter"),
            _entry(EgoPlusAlterEffect(v, c), "egoPlusAlt$c", "$c ego and alter"))
        if directed
            push!(out,
                _entry(SameXRecipEffect(v, c), "same$(c)Recip", "same $c x reciprocity"),
                _entry(SimXRecipEffect(v, c), "simRecip$c", "$c similarity x reciprocity"),
                _entry(SimXTransTripEffect(v, c), "sim$(c)TransTrip",
                       "$c similarity x transitive triplets"))
        end
    end
    for c in _dyadic_covariates(data)
        cov = data.covariates[c]
        sz = cov isa ConstantDyadCovariate ? size(cov.values) : size(cov.values[1])
        sz == size(dep.networks[1]) &&
            push!(out, _entry(DyadCovariateEffect(v, c), "dyad$c", "$c (dyadic)"))
    end
    # Other dependent networks on the same node set (multiplex effects).
    for w in _dependents_of(data, DependentNetwork)
        w == v && continue
        other = data.dependents[w]
        (_network_kind(other) == :directed && directed &&
         size(other.networks[1]) == size(dep.networks[1])) || continue
        push!(out,
            _entry(CrossNetworkTiesEffect(v, w), "crprod$w", "$w (tie in other network)"),
            _entry(CrossNetworkReciprocityEffect(v, w), "crprodRecip$w",
                   "reciprocity with $w"),
            _entry(CrossNetworkActivityEffect(v, w), "crprodAct$w",
                   "$w outdegree activity (Siena.jl)"),
            _entry(CrossNetworkPopularityEffect(v, w), "crprodPop$w",
                   "$w indegree popularity (Siena.jl)"))
    end
    return out
end

function _behavior_entries!(out::Vector{EffectEntry}, data::SienaData, v::Symbol,
                            dep::DependentBehavior)
    push!(out,
        _entry(LinearShapeEffect(v), "linear", "$v linear shape"),
        _entry(QuadraticShapeEffect(v), "quad", "$v quadratic shape"),
        _entry(CubicShapeEffect(v), "cubic", "$v cubic shape (Siena.jl)"))
    for w in _dependents_of(data, DependentNetwork)
        net = data.dependents[w]
        size(net.networks[1], 1) == n_actors(dep) || continue
        kind = _network_kind(net)
        if kind === :twomode
            push!(out, _entry(BehaviorOutdegreeEffect(v, w), "outdeg$w", "$v outdegree"))
            continue
        end
        push!(out,
            _entry(AverageAlterEffect(v, w), "avAlt$w", "$v average alter"),
            _entry(AverageSimilarityEffect(v, w), "avSim$w", "$v average similarity"),
            _entry(TotalAlterEffect(v, w), "totAlt$w", "$v total alter"),
            _entry(TotalSimilarityEffect(v, w), "totSim$w", "$v total similarity"),
            _entry(AverageAlterDist2Effect(v, w), "avAltDist2$w", "$v av. alter dist. 2"),
            _entry(BehaviorOutdegreeEffect(v, w), "outdeg$w",
                   kind === :directed ? "$v outdegree" : "$v degree"),
            _entry(BehaviorIsolateEffect(v, w), "behIsolate$w",
                   "$v total isolate (Siena.jl)"),
            _entry(AverageAttHigherSimpleEffect(v, w), "avAttHigherSimple$w",
                   "$v share of higher alters (Siena.jl simplified)"),
            _entry(AverageAttLowerSimpleEffect(v, w), "avAttLowerSimple$w",
                   "$v share of lower alters (Siena.jl simplified)"),
            _entry(FeedbackEffect(v, w), "simProd$w",
                   "$v product of similarities (Siena.jl)"))
        if kind === :directed
            push!(out,
                _entry(AverageInAlterEffect(v, w), "avInAlt$w", "$v average in-alter"),
                _entry(AverageRecipAlterEffect(v, w), "avRecAlt$w",
                       "$v average reciprocated alter"),
                _entry(TotalInAlterEffect(v, w), "totInAlt$w", "$v total in-alter"),
                _entry(IndegreeEffect(v, w), "indeg$w", "$v indegree"),
                _entry(RecipDegreeEffect(v, w), "recipDeg$w", "$v reciprocated degree"))
        end
    end
    for c in _actor_covariates(data)
        push!(out,
            _entry(BehaviorCovariateEffect(v, c), "effFrom$c", "$v: effect from $c"),
            _entry(CovariateInteractionEffect(v, c), "covInt$c",
                   "$v squared x $c (Siena.jl)"))
    end
    for b in _dependents_of(data, DependentBehavior)
        b == v && continue
        n_actors(data.dependents[b]) == n_actors(dep) || continue
        push!(out, _entry(BehaviorInteractionEffect(v, b), "behBeh$b",
                          "$v x $b (Siena.jl)"))
    end
    return out
end

function _rate_entries!(out::Vector{EffectEntry}, data::SienaData, v::Symbol,
                        dep::AbstractDependent)
    for p in 1:(data.n_waves - 1)
        push!(out, EffectEntry(BasicRateEffect(v, p);
                               name="Rate $v (period $p)", shortname="rate$p",
                               include=true,
                               initial_value=default_basic_rate(data, v, p)))
    end
    if dep isa DependentNetwork
        kind = _network_kind(dep)
        push!(out,
            _entry(OutdegreeRateEffect(v, v, 1), "outRate", "outdegree effect on rate $v"))
        push!(out,
            _entry(OutdegreeLogRateEffect(v, v, 1), "outRateLog",
                   "outdegree effect on rate $v (log)"),
            _entry(OutdegreeInvRateEffect(v, v, 1), "outRateInv",
                   "inverse outdegree effect on rate $v"))
        if kind === :directed
            push!(out,
                _entry(IndegreeRateEffect(v, v, 1), "inRate", "indegree effect on rate $v"),
                _entry(IndegreeLogRateEffect(v, v, 1), "inRateLog",
                       "indegree effect on rate $v (log)"),
                _entry(IndegreeInvRateEffect(v, v, 1), "inRateInv",
                       "inverse indegree effect on rate $v"),
                _entry(RecipDegreeRateEffect(v, v, 1), "recipRate",
                       "reciprocity effect on rate $v"),
                _entry(OutdegreeSqRateEffect(v, v, 1), "outRateSq",
                       "squared outdegree effect on rate $v (Siena.jl)"))
        end
    else
        for w in _dependents_of(data, DependentNetwork)
            net = data.dependents[w]
            size(net.networks[1], 1) == n_actors(dep) || continue
            push!(out, _entry(OutdegreeRateEffect(v, w, 1), "outRate$w",
                              "$w outdegree effect on rate $v"))
            if _network_kind(net) === :directed
                push!(out,
                    _entry(IndegreeRateEffect(v, w, 1), "inRate$w",
                           "$w indegree effect on rate $v"),
                    _entry(RecipDegreeRateEffect(v, w, 1), "recipRate$w",
                           "$w reciprocity effect on rate $v"))
            end
        end
    end
    for c in _actor_covariates(data)
        push!(out, _entry(CovariateRateEffect(v, c, 1), "Rate$c", "effect $c on rate $v"))
    end
    return out
end
