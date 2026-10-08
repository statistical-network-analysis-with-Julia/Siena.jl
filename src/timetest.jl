"""
Time-heterogeneity tests — RSiena's `sienaTimeTest` (Lospinoso et al. 2011).
"""

"""
    SienaTimeTest

Result of [`siena_time_test`](@ref): score-type tests of time heterogeneity.

# Fields
- `effects::Vector{String}`: The tested (base) effects
- `periods::Vector{Int}`: The periods with a dummy (all but the first)
- `chisq::Float64`, `df::Int`, `p_value::Float64`: Joint test of all dummies
- `effect_chisq`, `effect_df`, `effect_p_values`: Per effect, the joint test of its
  dummies (given the estimated parameters)
- `period_chisq`, `period_df`, `period_p_values`: Per period, the joint test of
  that period's dummies; for period 1 (the reference) the test of the summed dummies
  of every effect, as in RSiena
- `dummy_names::Vector{String}`: `"(*)Dummy<period>:<effect>"`, effect by effect
- `dummy_z`, `dummy_p_values`: One-sided score statistics (approximately standard
  normal under homogeneity) of each dummy and their two-sided p-values
- `one_step::Vector{Float64}`: One-step estimates of the dummy parameters
- `excluded::Vector{String}`: Effects left out because their dummies are collinear
  with the model (e.g. an effect that is itself period-specific)

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=300))
tt = siena_time_test(fit)
tt.df                    # 3 effects x 1 dummy (period 2)
```
"""
struct SienaTimeTest
    effects::Vector{String}
    periods::Vector{Int}
    chisq::Float64
    df::Int
    p_value::Float64
    effect_chisq::Vector{Float64}
    effect_df::Vector{Int}
    effect_p_values::Vector{Float64}
    period_chisq::Vector{Float64}
    period_df::Vector{Int}
    period_p_values::Vector{Float64}
    dummy_names::Vector{String}
    dummy_z::Vector{Float64}
    dummy_p_values::Vector{Float64}
    one_step::Vector{Float64}
    excluded::Vector{String}
end

# RSiena's transformed.scoreTest: score test of A * g given the nuisance rows.
function _transformed_score(D, Σ, g, A::AbstractMatrix, nuisance::Vector{Int})
    A1 = zeros(length(nuisance), size(D, 1))
    for (r, k) in enumerate(nuisance)
        A1[r, k] = 1.0
    end
    AA = vcat(A1, A)
    q = size(A, 1)
    n = length(nuisance)
    chisq, _ = _score_statistic(AA * D * AA', AA * Σ * AA', AA * g, collect(1:n),
                                collect((n + 1):(n + q)))
    return chisq
end

_chisq_p(c, df) = (isfinite(c) && df > 0) ? ccdf(Chisq(df), c) : NaN

"""
    siena_time_test(result::SienaResult; effects=nothing, condition=false)

Test the fitted model for time heterogeneity — RSiena's `sienaTimeTest`
(Lospinoso, Schweinberger, Snijders & Ripley 2011). For every estimated objective or
non-basic rate effect and every period after the first, the model is extended by the
interaction of the effect with a dummy for that period; the dummies are *not*
estimated but tested at zero with the score-type test of Schweinberger (2012), from
the per-period statistics and scores of the fit's phase-3 simulations (no new
simulations are run). Basic rate parameters are already period-specific and are not
tested.

`effects` restricts the tested effects (short names, e.g. `["recip"]`, or positions
in `result.parameter_names`). With `condition=true` each individual dummy test also
conditions on the other dummies, as RSiena's `condition=TRUE`. Effects whose dummies
are linearly dependent on the model are excluded (and listed in `excluded`).

Needs at least two periods (three waves). A significant joint test says that some
parameter differs between periods; RSiena's remedy, period-dummy effects
(`sienaTimeFix`), is not implemented — fit the periods separately instead.

# Example
```julia
using Siena, NetworkCore, Random
s50 = load_dataset(:s50)
data = siena_data()
add_dependent!(data, DependentNetwork(:friendship, s50.friendship))
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])
fit = fit_siena(data, effects; rng=MersenneTwister(1),
                algorithm=SienaAlgorithm(verbose=false, phase3_iterations=300))
tt = siena_time_test(fit)
tt.p_value               # joint test of time heterogeneity
tt.effect_p_values       # one test per effect
```
"""
function siena_time_test(result::SienaResult; effects=nothing, condition::Bool=false)
    pstats = result.phase3_period_stats
    pscores = result.phase3_period_scores
    n_sims, n_par, n_periods = size(pstats)
    n_periods >= 2 || throw(ArgumentError(
        "siena_time_test needs at least two periods (three waves) to test for " *
        "heterogeneity across time"))
    pm = build_param_map(result.effects)
    # RSiena: basic rates are neither tested nor part of the null model here.
    base_all = [k for (k, e) in enumerate(pm.free) if !(e.effect isa BasicRateEffect)]
    isempty(base_all) && throw(ArgumentError("no effects available to test"))
    names_all = result.parameter_names
    tested = if effects === nothing
        copy(base_all)
    else
        sel = Int[]
        for ef in effects
            k = ef isa Integer ? Int(ef) : findfirst(==(String(ef)), names_all)
            (k === nothing || !(1 <= k <= n_par)) && throw(ArgumentError(
                "effect $(repr(ef)) is not an estimated parameter of the fit; " *
                "estimated: $(names_all)"))
            pm.free[k].effect isa BasicRateEffect && throw(ArgumentError(
                "siena time tests are inappropriate for basic rates"))
            push!(sel, k)
        end
        sel
    end
    nb = length(base_all)
    # Moment deviations per simulation and period for the base effects.
    G0 = pstats[:, base_all, :] .- reshape(result.period_targets[base_all, :], 1, nb,
                                           n_periods)
    S0 = pscores[:, base_all, :]

    function build(tested_now)
        dummies = Tuple{Int, Int}[]            # (position in base_all, period)
        for k in tested_now, p in 2:n_periods
            push!(dummies, (findfirst(==(k), base_all), p))
        end
        nd = length(dummies)
        G = zeros(n_sims, nb + nd, n_periods)
        S = zeros(n_sims, nb + nd, n_periods)
        G[:, 1:nb, :] = G0
        S[:, 1:nb, :] = S0
        for (d, (b, p)) in enumerate(dummies)
            G[:, nb + d, p] = G0[:, b, p]
            S[:, nb + d, p] = S0[:, b, p]
        end
        D = _score_derivative(G, S)
        total = dropdims(sum(G; dims=3); dims=3)
        return dummies, D, cov(total), vec(mean(total; dims=1))
    end

    excluded = Int[]
    dummies, D, Σ, g = build(tested)
    if rank(Σ) < size(Σ, 1)
        # RSiena's automatic exclusion: drop the effects whose dummies are linearly
        # dependent on the base effects.
        for k in copy(tested)
            _, _, Σk, _ = build([k])
            rank(Σk) < size(Σk, 1) && push!(excluded, k)
        end
        tested = setdiff(tested, excluded)
        isempty(tested) && throw(ArgumentError(
            "every tested effect's time dummies are linearly dependent on the " *
            "model; nothing can be tested"))
        dummies, D, Σ, g = build(tested)
        rank(Σ) < size(Σ, 1) && throw(ArgumentError(
            "the time dummies are linearly dependent; rerun siena_time_test with " *
            "a smaller set of `effects`"))
    end
    nd = length(dummies)
    null = collect(1:nb)
    dum = collect((nb + 1):(nb + nd))

    chisq, _ = _score_statistic(D, Σ, g, null, dum)
    # individual dummies
    z = zeros(nd)
    for d in 1:nd
        nuisance = condition ? vcat(null, setdiff(dum, nb + d)) : null
        _, z[d] = _score_statistic(D, Σ, g, nuisance, [nb + d])
    end
    one_step = try
        -(D \ g)[dum]
    catch err
        err isa Union{SingularException, LinearAlgebra.LAPACKException} || rethrow()
        fill(NaN, nd)
    end
    # per effect
    eff_chisq = Float64[]; eff_df = Int[]
    for k in tested
        b = findfirst(==(k), base_all)
        rows = [nb + d for (d, (bb, _)) in enumerate(dummies) if bb == b]
        nuisance = condition ? vcat(null, setdiff(dum, rows)) : null
        c, _ = _score_statistic(D, Σ, g, nuisance, rows)
        push!(eff_chisq, c); push!(eff_df, length(rows))
    end
    # per period (period 1: the summed dummies of each effect, as RSiena)
    per_chisq = zeros(n_periods); per_df = zeros(Int, n_periods)
    A = zeros(length(tested), nb + nd)
    for (r, k) in enumerate(tested)
        b = findfirst(==(k), base_all)
        for (d, (bb, _)) in enumerate(dummies)
            bb == b && (A[r, nb + d] = 1.0)
        end
    end
    per_chisq[1] = _transformed_score(D, Σ, g, A, null)
    per_df[1] = length(tested)
    for p in 2:n_periods
        rows = [nb + d for (d, (_, pp)) in enumerate(dummies) if pp == p]
        c, _ = _score_statistic(D, Σ, g, null, rows)
        per_chisq[p] = c; per_df[p] = length(rows)
    end
    dummy_names = ["(*)Dummy$(p):$(names_all[base_all[b]])" for (b, p) in dummies]
    return SienaTimeTest(names_all[tested], collect(2:n_periods), chisq, nd,
                         _chisq_p(chisq, nd), eff_chisq, eff_df,
                         _chisq_p.(eff_chisq, eff_df), per_chisq, per_df,
                         _chisq_p.(per_chisq, per_df), dummy_names, z,
                         2 .* ccdf.(Normal(), abs.(z)), one_step, names_all[excluded])
end

function Base.show(io::IO, t::SienaTimeTest)
    println(io, "Time-heterogeneity score-type tests (sienaTimeTest)")
    @printf(io, "Joint test: chi-squared = %.2f, df = %d, p = %s\n", t.chisq, t.df,
            format_pvalue(t.p_value))
    println(io, "\nEffect-wise joint tests:")
    for k in eachindex(t.effects)
        @printf(io, "  %-28s chi-squared = %7.2f, df = %d, p = %s\n", t.effects[k],
                t.effect_chisq[k], t.effect_df[k], format_pvalue(t.effect_p_values[k]))
    end
    println(io, "\nPeriod-wise joint tests:")
    for p in eachindex(t.period_chisq)
        @printf(io, "  Period %-21d chi-squared = %7.2f, df = %d, p = %s\n", p,
                t.period_chisq[p], t.period_df[p], format_pvalue(t.period_p_values[p]))
    end
    println(io, "\nIndividual dummies (one-sided score statistic, one-step estimate):")
    for d in eachindex(t.dummy_names)
        @printf(io, "  %-34s z = %6.2f, p = %s, one-step = %7.3f\n", t.dummy_names[d],
                t.dummy_z[d], format_pvalue(t.dummy_p_values[d]), t.one_step[d])
    end
    isempty(t.excluded) ||
        println(io, "\nExcluded (dummies collinear with the model): ",
                join(t.excluded, ", "))
end
