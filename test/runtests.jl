using Siena
using Test
using LinearAlgebra
using Random
using Statistics
using StatsBase   # co-loading must not break the StatsAPI verbs (coef, stderror, ...)
using DataFrames
using Networks     # activates the SienaNetworkExt package extension

"Generic toggle-based change statistic, computed independently of the package fallback."
function brute_network_contribution(eff, state, data, i, j)
    net = state.networks[Siena.target_variable(eff)]
    old = net[i, j]
    net[i, j] = 1
    with_tie = evaluate_actor(eff, state, data, i)
    net[i, j] = 0
    without_tie = evaluate_actor(eff, state, data, i)
    net[i, j] = old
    return with_tie - without_tie
end

function brute_behavior_contribution(eff, state, data, i, dir)
    beh = state.behaviors[Siena.target_variable(eff)]
    old = beh[i]
    base = evaluate_actor(eff, state, data, i)
    beh[i] = old + dir
    changed = evaluate_actor(eff, state, data, i)
    beh[i] = old
    return changed - base
end

const DATA_DIR = joinpath(@__DIR__, "data")
function read_int_matrix(f)
    rows = [parse.(Int, split(line, ',')) for line in eachline(joinpath(DATA_DIR, f))]
    return permutedims(reduce(hcat, rows))
end

@testset "Siena.jl" begin

    @testset "NodeSet" begin
        ns = NodeSet(10)
        @test length(ns) == 10
        @test ns.id == :actors

        ns_named = NodeSet(3; names=["Alice", "Bob", "Carol"], id=:students)
        @test length(ns_named) == 3
        @test ns_named.names == ["Alice", "Bob", "Carol"]
        @test ns_named.id == :students
    end

    @testset "DependentNetwork" begin
        Random.seed!(42)
        n = 20
        networks = [zeros(Int, n, n) for _ in 1:3]
        for w in 1:3, i in 1:n, j in 1:n
            if i != j && rand() < 0.1 + 0.05 * w
                networks[w][i, j] = 1
            end
        end

        dep = DependentNetwork(:friendship, networks)
        @test n_waves(dep) == 3
        @test n_actors(dep) == 20
        @test dep.type == :onemode
        @test dep.directed == true
    end

    @testset "DependentBehavior" begin
        beh = DependentBehavior(:drinking, [[1, 2, 3], [2, 2, 4], [3, 2, 5]])
        @test n_waves(beh) == 3
        @test n_actors(beh) == 3
        @test beh.min_val == 1
        @test beh.max_val == 5
        # Mean over all observations
        @test beh.mean_val ≈ mean([1, 2, 3, 2, 2, 4, 3, 2, 5])
        # Similarity mean over waves 1..M-1 (RSiena convention)
        r = 4
        sims = Float64[]
        for v in ([1, 2, 3], [2, 2, 4])
            for i in 1:3, j in 1:3
                i != j && push!(sims, 1 - abs(v[i] - v[j]) / r)
            end
        end
        @test beh.sim_mean ≈ mean(sims)
    end

    @testset "Covariates" begin
        vals = randn(20)
        cov = ConstantCovariate(:age, vals)
        @test cov.centered == true
        @test abs(mean(cov.values)) < 1e-10  # Should be centered
        @test 0.0 <= cov.sim_mean <= 1.0

        vals_v = [randn(20) for _ in 1:3]
        cov_v = VaryingCovariate(:income, vals_v)
        @test length(cov_v.values) == 3

        mat = randn(20, 20)
        dcov = ConstantDyadCovariate(:distance, mat)
        @test size(dcov.values) == (20, 20)
    end

    @testset "SienaData & NetworkState" begin
        data = siena_data()
        add_nodeset!(data, NodeSet(20))
        networks = [rand(0:1, 20, 20) for _ in 1:3]
        for net in networks, i in 1:20
            net[i, i] = 0
        end
        add_dependent!(data, DependentNetwork(:friendship, networks))
        add_covariate!(data, ConstantCovariate(:gender, rand(0:1, 20)))

        @test data.n_waves == 3
        @test length(data.dependents) == 1
        @test length(data.covariates) == 1

        state = NetworkState()
        initialize!(state, data, 2)
        @test state.networks[:friendship] == networks[2]
        @test state.time == 0.0
        @test state.period == 2

        initialize!(state, data, 3; period=2)
        @test state.period == 2
        @test state.networks[:friendship] == networks[3]

        snap = snapshot(state)
        snap.networks[:friendship][1, 2] = 1 - snap.networks[:friendship][1, 2]
        @test snap.networks[:friendship] != state.networks[:friendship]
        @test snap.period == state.period
    end

    @testset "StateNetwork state representation" begin
        m = [0 1 0; 1 0 1; 0 0 0]
        sn = Siena.StateNetwork(m)
        # reads like a 0/1 Matrix{Int}
        @test sn == m
        @test sn[1, 2] == 1 && sn[1, 3] == 0
        @test eltype(sn) == Int
        @test size(sn) == (3, 3)
        # incrementally maintained degrees
        @test Siena._row_sum(sn, 2) == 2
        @test Siena._col_sum(sn, 1) == 1
        sn[3, 1] = 1
        @test Siena._row_sum(sn, 3) == 1
        @test Siena._col_sum(sn, 1) == 2
        sn[3, 1] = 1              # no-op write leaves degrees unchanged
        @test Siena._col_sum(sn, 1) == 2
        sn[1, 2] = 0
        @test Siena._row_sum(sn, 1) == 0
        @test Siena._col_sum(sn, 2) == 0
        # degrees always match full scans
        @test [Siena._row_sum(sn, i) for i in 1:3] == vec(sum(Matrix(sn), dims=2))
        @test [Siena._col_sum(sn, j) for j in 1:3] == vec(sum(Matrix(sn), dims=1))
        # copy extracts a plain Matrix{Int} (wave-matrix semantics)
        c = copy(sn)
        @test c isa Matrix{Int}
        @test c == sn
        # only 0/1 values are representable
        @test_throws ArgumentError sn[1, 2] = 10
        @test_throws ArgumentError Siena.StateNetwork([0 10; 0 0])

        # states store networks as StateNetwork and stay consistent under
        # simulation toggles
        data = siena_data()
        add_nodeset!(data, NodeSet(3))
        add_dependent!(data, DependentNetwork(:net, [m, copy(m)]))
        state = NetworkState()
        initialize!(state, data, 1)
        @test state.networks[:net] isa Siena.StateNetwork
        @test state.networks[:net] == m
        # dict assignment of a plain matrix converts
        state.networks[:net] = [0 0 1; 0 0 0; 1 1 0]
        @test state.networks[:net] isa Siena.StateNetwork
        @test Siena._row_sum(state.networks[:net], 3) == 2
    end

    @testset "Network effect contributions match evaluate_actor (brute force)" begin
        Random.seed!(1)
        n = 8
        data = siena_data()
        add_nodeset!(data, NodeSet(n))
        nets = [zeros(Int, n, n) for _ in 1:2]
        for w in 1:2, i in 1:n, j in 1:n
            i != j && rand() < 0.35 && (nets[w][i, j] = 1)
        end
        add_dependent!(data, DependentNetwork(:net, nets))
        add_covariate!(data, ConstantCovariate(:age, randn(n)))
        add_covariate!(data, ConstantCovariate(:grp, rand(0:1, n)))
        add_covariate!(data, ConstantDyadCovariate(:dist, randn(n, n)))
        other = [Int(rand() < 0.3 && i != j) for i in 1:n, j in 1:n]
        add_dependent!(data, DependentNetwork(:net2, [other, other]))

        state = NetworkState()
        initialize!(state, data, 1)

        effects_to_check = [
            OutdegreeEffect(:net), ReciprocityEffect(:net),
            TransitiveTripletsEffect(:net), TransitiveTiesEffect(:net),
            TransitiveMediatedTripletsEffect(:net), TransitiveRecipTripletsEffect(:net),
            CyclicTripletsEffect(:net), BalanceSimpleEffect(:net), BetweennessEffect(:net),
            NbrDist2Effect(:net), DenseTriadsEffect(:net), SharedInEffect(:net),
            SharedOutEffect(:net),
            IndegreePopularityEffect(:net), IndegreePopularityEffect(:net; sqrt=true),
            OutdegreePopularityEffect(:net), OutdegreePopularityEffect(:net; sqrt=true),
            IndegreeActivityEffect(:net), IndegreeActivityEffect(:net; sqrt=true),
            OutdegreeActivityEffect(:net), OutdegreeActivityEffect(:net; sqrt=true),
            OutdegreeTruncEffect(:net; c=2), IndegreeTruncEffect(:net; c=2),
            DegreeAssortativityEffect(:net),
            IsolateEffect(:net), IsolateNetEffect(:net), OutIsolateEffect(:net),
            InIsolateEffect(:net),
            GWESPEffect(:net), GWESPBackwardEffect(:net), GWESPMixedEffect(:net),
            GWDSPEffect(:net),
            EgoEffect(:net, :age), EgoSqEffect(:net, :age), AlterEffect(:net, :age),
            AlterSqEffect(:net, :age), SimilarityEffect(:net, :age), SameEffect(:net, :grp),
            DifferenceEffect(:net, :age), DifferenceSqEffect(:net, :age),
            AbsDifferenceEffect(:net, :age), HigherEffect(:net, :age),
            EgoTimesAlterEffect(:net, :age), EgoPlusAlterEffect(:net, :age),
            DyadCovariateEffect(:net, :dist), SameXRecipEffect(:net, :grp),
            SimXRecipEffect(:net, :age), SimXTransTripEffect(:net, :age),
            CrossNetworkReciprocityEffect(:net, :net2), CrossNetworkActivityEffect(:net, :net2),
            CrossNetworkPopularityEffect(:net, :net2), CrossNetworkTiesEffect(:net, :net2),
        ]

        for eff in effects_to_check
            ok = true
            for i in 1:n, j in 1:n
                i == j && continue
                got = compute_contribution(eff, state, data, i, j)
                want = brute_network_contribution(eff, state, data, i, j)
                ok &= isapprox(got, want; atol=1e-10)
            end
            @test ok

            # Statistic is the sum of the actor components (cycle3 counts each
            # cycle once, so it is a third of the actor sum)
            s = compute_statistic(eff, state, data)
            s2 = sum(evaluate_actor(eff, state, data, i) for i in 1:n)
            if eff isa CyclicTripletsEffect
                @test s ≈ s2 / 3
            else
                @test s ≈ s2
            end
        end

        # State unchanged by all the toggling
        @test state.networks[:net] == nets[1]
    end

    @testset "Hand-computed statistics" begin
        # 1 -> 2, 2 -> 1, 1 -> 3, 2 -> 3 on 4 actors
        net = zeros(Int, 4, 4)
        net[1, 2] = net[2, 1] = net[1, 3] = net[2, 3] = 1
        data = siena_data()
        add_nodeset!(data, NodeSet(4))
        add_dependent!(data, DependentNetwork(:net, [net, net]))
        state = NetworkState()
        initialize!(state, data, 1)

        @test compute_statistic(OutdegreeEffect(:net), state, data) == 4.0
        @test compute_statistic(ReciprocityEffect(:net), state, data) == 2.0
        # transitive triplets: 1->2->3 & 1->3; 2->1->3 & 2->3
        @test compute_statistic(TransitiveTripletsEffect(:net), state, data) == 2.0
        # no cyclic triplets
        @test compute_statistic(CyclicTripletsEffect(:net), state, data) == 0.0
        # inPop: sum_ij x_ij * indeg(j) = indeg^2 summed = 1^2 + 1^2 + 2^2
        @test compute_statistic(IndegreePopularityEffect(:net), state, data) == 6.0
        # outAct: outdeg^2 summed = 4 + 4 + 0 + 0
        @test compute_statistic(OutdegreeActivityEffect(:net), state, data) == 8.0
        # actor 4 is a total isolate; actor 3 has incoming ties only
        @test compute_statistic(IsolateNetEffect(:net), state, data) == 1.0
        @test compute_statistic(OutIsolateEffect(:net), state, data) == 2.0
    end

    @testset "Behavior effects" begin
        Random.seed!(3)
        n = 8
        data = siena_data()
        add_nodeset!(data, NodeSet(n))
        net = [Int(rand() < 0.35 && i != j) for i in 1:n, j in 1:n]
        add_dependent!(data, DependentNetwork(:net, [net, net]))
        z = [rand(1:5, n), rand(1:5, n)]
        add_dependent!(data, DependentBehavior(:beh, z))
        add_dependent!(data, DependentBehavior(:beh2, [rand(1:3, n), rand(1:3, n)]))
        add_covariate!(data, ConstantCovariate(:age, randn(n)))

        state = NetworkState()
        initialize!(state, data, 1)

        beh_effects = [
            LinearShapeEffect(:beh), QuadraticShapeEffect(:beh), CubicShapeEffect(:beh),
            AverageAlterEffect(:beh, :net), AverageSimilarityEffect(:beh, :net),
            AverageInAlterEffect(:beh, :net), AverageRecipAlterEffect(:beh, :net),
            AverageAttHigherSimpleEffect(:beh, :net), AverageAttLowerSimpleEffect(:beh, :net),
            TotalAlterEffect(:beh, :net), TotalSimilarityEffect(:beh, :net),
            TotalInAlterEffect(:beh, :net), AverageAlterDist2Effect(:beh, :net),
            IndegreeEffect(:beh, :net), BehaviorOutdegreeEffect(:beh, :net),
            RecipDegreeEffect(:beh, :net), BehaviorCovariateEffect(:beh, :age),
            CovariateInteractionEffect(:beh, :age), BehaviorInteractionEffect(:beh, :beh2),
            BehaviorSimilarityEffect(:beh, :beh2, :net), ThresholdEffect(:beh, 3),
            PropThresholdEffect(:beh, :net, 0.5), BehaviorIsolateEffect(:beh, :net),
            FeedbackEffect(:beh, :net), MainBehaviorEffect(:beh),
        ]

        for eff in beh_effects
            ok = true
            for i in 1:n, dir in (-1, 1)
                got = compute_contribution(eff, state, data, i, dir)
                want = brute_behavior_contribution(eff, state, data, i, dir)
                ok &= isapprox(got, want; atol=1e-10)
            end
            @test ok
            # no-change option is exactly 0
            @test compute_contribution(eff, state, data, 1, 0) == 0.0
        end

        # Linear shape delta is exactly the direction
        dep = data.dependents[:beh]
        lin = LinearShapeEffect(:beh)
        @test compute_contribution(lin, state, data, 1, 1) ≈ 1.0
        # Quadratic delta: (z+1 - mean)^2 - (z - mean)^2
        quad = QuadraticShapeEffect(:beh)
        zi = state.behaviors[:beh][1]
        want = (zi + 1 - dep.mean_val)^2 - (zi - dep.mean_val)^2
        @test compute_contribution(quad, state, data, 1, 1) ≈ want
    end

    @testset "Two-mode effects" begin
        Random.seed!(4)
        n_act, n_ev = 7, 5
        data = siena_data()
        add_nodeset!(data, NodeSet(n_act))
        add_nodeset!(data, NodeSet(n_ev; id=:events))
        nets = [[Int(rand() < 0.4) for i in 1:n_act, e in 1:n_ev] for _ in 1:2]
        add_dependent!(data, DependentNetwork(:aff, nets;
                                              type=:twomode, nodeset2=:events))
        add_covariate!(data, ConstantCovariate(:age, randn(n_act)))
        add_covariate!(data, ConstantDyadCovariate(:evcov, randn(n_act, n_ev);
                                                   nodeset2=:events))

        state = NetworkState()
        initialize!(state, data, 1)

        tm_effects = [
            TwoModeOutdegreeEffect(:aff), TwoModeIndegreeEffect(:aff),
            TwoModeIndegreeEffect(:aff; sqrt=true), FourCyclesEffect(:aff),
            SharedEventsEffect(:aff), SharedEventsEffect(:aff; sqrt=true),
            GWESPTwoModeEffect(:aff), TwoModeEgoEffect(:aff, :age),
            TwoModeEventEffect(:aff, :evcov), TwoModeSameEffect(:aff, :age),
            TwoModeSimilarityEffect(:aff, :age), TwoModeActivityEffect(:aff),
            TwoModePopularityAltEffect(:aff), TwoModeTransitiveClosureEffect(:aff),
            TwoModeActorAssortativityEffect(:aff),
        ]

        for eff in tm_effects
            ok = true
            for i in 1:n_act, e in 1:n_ev
                got = compute_contribution(eff, state, data, i, e)
                want = brute_network_contribution(eff, state, data, i, e)
                ok &= isapprox(got, want; atol=1e-10)
            end
            @test ok
            @test compute_statistic(eff, state, data) ≈
                  sum(evaluate_actor(eff, state, data, i) for i in 1:n_act)
        end

        # Two-mode choice probabilities range over the events, not the actors
        effects = SienaEffects()
        Siena.add_effect!(effects, EffectEntry(TwoModeOutdegreeEffect(:aff);
                                               shortname="outdegree2", include=true))
        probs, alters = compute_network_choice_probs(effects, [0.0], state, data, 1, :aff)
        @test alters == vcat(0, 1:n_ev)
        @test sum(probs) ≈ 1.0
    end

    @testset "Rate effects" begin
        Random.seed!(5)
        n = 6
        data = siena_data()
        add_nodeset!(data, NodeSet(n))
        net = [Int(rand() < 0.4 && i != j) for i in 1:n, j in 1:n]
        add_dependent!(data, DependentNetwork(:net, [net, net]))
        add_dependent!(data, DependentBehavior(:beh, [rand(1:5, n), rand(1:5, n)]))
        add_covariate!(data, ConstantCovariate(:age, randn(n)))
        state = NetworkState()
        initialize!(state, data, 1)

        @test rate_score(BasicRateEffect(:net, 1), state, data, 1) == 1.0
        @test rate_score(OutdegreeRateEffect(:net, :net, 1), state, data, 2) ==
              sum(net[2, :])
        @test rate_score(IndegreeRateEffect(:net, :net, 1), state, data, 2) ==
              sum(net[:, 2])
        cov = data.covariates[:age]
        @test rate_score(CovariateRateEffect(:net, :age, 1), state, data, 3) ==
              cov.values[3]

        # The rate function is multiplicative
        entry = EffectEntry(OutdegreeRateEffect(:net, :net, 1); include=true)
        λ = Siena.actor_rate(2.0, [entry], [0.3], state, data, 2)
        @test λ ≈ 2.0 * exp(0.3 * sum(net[2, :]))

        # Behavior rate score is mean-centered
        dep = data.dependents[:beh]
        @test rate_score(BehaviorRateEffect(:net, :beh, 1), state, data, 4) ≈
              state.behaviors[:beh][4] - dep.mean_val
    end

    @testset "Objective sign flip for deletions" begin
        n = 5
        net = zeros(Int, 5, 5)
        net[1, 2] = 1
        net[2, 1] = 1
        data = siena_data()
        add_nodeset!(data, NodeSet(n))
        add_dependent!(data, DependentNetwork(:net, [net, net]))

        effects = get_effects(data)
        include_effects!(effects, :net, [:outdegree, :recip])
        θ = [1.0, 2.0]  # outdegree, recip (objective part)

        state = NetworkState()
        initialize!(state, data, 1)

        # Adding tie 2->3: change +1 outdegree, no recip
        @test compute_objective(effects, θ, state, data, 2, 3, :net) ≈ 1.0
        # Adding tie 3->1 with 1->3 absent: outdegree only
        @test compute_objective(effects, θ, state, data, 3, 1, :net) ≈ 1.0
        # Deleting existing reciprocated tie 1->2: -(1*1 + 2*1)
        @test compute_objective(effects, θ, state, data, 1, 2, :net) ≈ -3.0

        # Fixed effects keep contributing with their initial value
        include_effects!(effects, :net, [:transTrip]; fix=true, initial_value=0.5)
        pm = build_param_map(effects)
        @test Siena.n_free_parameters(pm) == 3  # rate + outdegree + recip
        @test compute_objective(effects, θ, state, data, 2, 3, :net) ≈
              1.0 + 0.5 * compute_contribution(TransitiveTripletsEffect(:net),
                                               state, data, 2, 3)
    end

    @testset "Objective effect set (tuple hot path)" begin
        Random.seed!(11)
        n = 8
        data = siena_data()
        add_nodeset!(data, NodeSet(n))
        net1 = [Int(rand() < 0.25 && i != j) for i in 1:n, j in 1:n]
        add_dependent!(data, DependentNetwork(:net, [net1, copy(net1)]))
        add_dependent!(data, DependentBehavior(:beh, [rand(1:4, n) for _ in 1:2]))

        effects = get_effects(data)
        include_effects!(effects, :net, [:outdegree, :recip, :transTrip])
        include_effects!(effects, :beh, [:linear, :quad])
        include_effects!(effects, :net, [:cycle3]; fix=true, initial_value=0.3)

        oset = build_objective_set(effects)
        @test oset isa ObjectiveEffectSet
        @test length(oset) == length(get_objective_effects(effects))
        @test oset.specs isa Tuple  # tuple-backed: statically dispatched fold

        pm = build_param_map(effects)
        θ = [3.0, 3.0, -1.2, 0.9, 0.4, 0.2, -0.1]
        θ_obj = objective_theta(pm, θ)

        state = NetworkState()
        initialize!(state, data, 1)

        # The prebuilt set gives exactly the effects-table objective for every
        # candidate ministep (adds, deletions, behavior moves)
        for actor in 1:n, alter in 1:n
            actor == alter && continue
            @test compute_objective(oset, θ_obj, state, data, actor, alter, :net) ==
                  compute_objective(effects, θ_obj, state, data, actor, alter, :net)
        end
        for actor in 1:n, dir in (-1, 1)
            @test compute_objective(oset, θ_obj, state, data, actor, dir, :beh) ==
                  compute_objective(effects, θ_obj, state, data, actor, dir, :beh)
        end

        # Choice probabilities agree too
        p1, a1 = compute_network_choice_probs(oset, θ_obj, state, data, 2, :net)
        p2, a2 = compute_network_choice_probs(effects, θ_obj, state, data, 2, :net)
        @test p1 == p2 && a1 == a2
        b1, d1 = compute_behavior_choice_probs(oset, θ_obj, state, data, 3, :beh)
        b2, d2 = compute_behavior_choice_probs(effects, θ_obj, state, data, 3, :beh)
        @test b1 == b2 && d1 == d2

        # Whole-ministep equivalence for identical RNG streams
        stA = snapshot(state); stB = snapshot(state)
        chA = Siena.execute_network_ministep!(stA, oset, θ_obj, data, 1, :net,
                                              MersenneTwister(5))
        chB = Siena.execute_network_ministep!(stB, effects, θ_obj, data, 1, :net,
                                              MersenneTwister(5))
        @test chA == chB
        @test stA.networks[:net] == stB.networks[:net]
    end

    @testset "Parameter map" begin
        data = siena_data()
        add_nodeset!(data, NodeSet(10))
        nets = [rand(0:1, 10, 10) for _ in 1:3]
        for net in nets, i in 1:10
            net[i, i] = 0
        end
        add_dependent!(data, DependentNetwork(:net, nets))
        effects = get_effects(data)
        include_effects!(effects, :net, [:outdegree, :recip])

        pm = build_param_map(effects)
        @test Siena.n_free_parameters(pm) == 4      # 2 rates + 2 objective
        @test Siena.n_free_rate_parameters(pm) == 2
        θ = [2.0, 3.0, -1.0, 0.5]
        @test objective_theta(pm, θ) == [-1.0, 0.5]
        @test basic_rate(pm, θ, :net, 1) == 2.0
        @test basic_rate(pm, θ, :net, 2) == 3.0
        @test length(parameter_names(effects)) == 4
        @test_throws ArgumentError objective_theta(pm, [1.0, 2.0])
    end

    @testset "Simulation" begin
        Random.seed!(123)
        n = 10
        data = siena_data()
        add_nodeset!(data, NodeSet(n))
        net1 = [Int(rand() < 0.2 && i != j) for i in 1:n, j in 1:n]
        add_dependent!(data, DependentNetwork(:net, [net1, copy(net1), copy(net1)]))
        add_dependent!(data, DependentBehavior(:beh, [rand(1:4, n) for _ in 1:3]))

        effects = get_effects(data)
        include_effects!(effects, :net, [:outdegree, :recip])
        include_effects!(effects, :beh, [:linear, :quad])

        pm = build_param_map(effects)
        θ = zeros(Siena.n_free_parameters(pm))
        θ[1:Siena.n_free_rate_parameters(pm)] .= 3.0

        state, results = simulate_saom(data, effects, θ; seed=42)
        @test length(results) == 2  # two periods
        @test haskey(state.networks, :net)
        # per-period snapshots are independent states
        @test results[1].final_state !== results[2].final_state
        # behavior stays within its range
        for r in results
            @test all(1 .<= r.final_state.behaviors[:beh] .<= 4)
        end

        # Strongly negative density with high rate empties the network
        θ2 = copy(θ)
        θ2[findfirst(==("outdegree"), parameter_names(effects))] = -5.0
        θ2[1:Siena.n_free_rate_parameters(pm)] .= 10.0
        state2, _ = simulate_saom(data, effects, θ2; seed=7)
        @test sum(state2.networks[:net]) < sum(net1)

        # Wrong θ length errors
        @test_throws ArgumentError simulate_saom(data, effects, zeros(2); seed=1)
    end

    @testset "Structural zeros/ones (10/11 coding)" begin
        @testset "constructor decodes and validates codes" begin
            w1 = [0 1 11; 0 0 0; 10 0 0]
            w2 = [0 0 11; 1 0 0; 10 1 0]
            dep = DependentNetwork(:net, [w1, w2])
            @test has_structural(dep)
            @test dep.networks[1] == [0 1 1; 0 0 0; 0 0 0]   # 11 -> 1, 10 -> 0
            @test dep.networks[2] == [0 0 1; 1 0 0; 0 1 0]
            @test is_structural_dyad(dep, 1, 1, 3)
            @test is_structural_dyad(dep, 1, 3, 1)
            @test !is_structural_dyad(dep, 1, 1, 2)
            @test n_structural_dyads(dep, 1) == 2
            @test n_structural_dyads(dep, 2) == 2

            # No codes: no structural bookkeeping
            plain = DependentNetwork(:net, [[0 1; 0 0], [0 0; 1 0]])
            @test !has_structural(plain)
            @test n_structural_dyads(plain, 1) == 0
            @test !is_structural_dyad(plain, 1, 1, 2)

            # Invalid tie values are rejected
            @test_throws ArgumentError DependentNetwork(:net, [[0 5; 1 0]])
            @test_throws ArgumentError DependentNetwork(:net, [[0 -1; 1 0]])
            # Codes must be distinct and different from 0/1
            @test_throws ArgumentError DependentNetwork(:net, [w1];
                                                        structural_zero=11)
            @test_throws ArgumentError DependentNetwork(:net, [[0 1; 0 0]];
                                                        structural_one=1)

            # Configurable codes (and the defaults then reject 10/11)
            depc = DependentNetwork(:net, [[0 8; 9 0]];
                                    structural_zero=8, structural_one=9)
            @test depc.networks[1] == [0 0; 1 0]
            @test is_structural_dyad(depc, 1, 1, 2)
            @test is_structural_dyad(depc, 1, 2, 1)
            @test_throws ArgumentError DependentNetwork(:net, [[0 10; 0 0]];
                                                        structural_zero=8,
                                                        structural_one=9)
        end

        @testset "target statistics exclude structural dyads (hand-computed)" begin
            w1 = [0 1 11; 0 0 0; 10 0 0]
            w2 = [0 0 11; 1 0 0; 10 1 0]
            data = siena_data()
            add_nodeset!(data, NodeSet(3))
            add_dependent!(data, DependentNetwork(:net, [w1, w2]))
            effects = get_effects(data)
            include_effects!(effects, :net, [:outdegree, :recip])

            # Hand computation, excluding the structural dyads (1,3) and (3,1):
            # decoded wave 2 with structural entries zeroed is
            #   [0 0 0; 1 0 0; 0 1 0]
            # -> outdegree = 2, reciprocity = 0.
            # Rate target: Hamming distance wave1 -> wave2 over free dyads:
            #   (1,2): 1->0, (2,1): 0->1, (3,2): 0->1  => 3.
            targets = compute_target_statistics(data, effects)
            @test targets == [3.0, 2.0, 0.0]

            # Same data without codes for comparison: the structural dyads
            # would otherwise contribute
            data_plain = siena_data()
            add_nodeset!(data_plain, NodeSet(3))
            add_dependent!(data_plain,
                           DependentNetwork(:net, [[0 1 1; 0 0 0; 0 0 0],
                                                   [0 0 1; 1 0 0; 0 1 0]]))
            effects_plain = get_effects(data_plain)
            include_effects!(effects_plain, :net, [:outdegree, :recip])
            @test compute_target_statistics(data_plain, effects_plain) ==
                  [3.0, 3.0, 0.0]   # outdegree now counts the structural one
        end

        @testset "ministep candidate sets exclude structural dyads" begin
            w1 = [0 1 11; 0 0 0; 10 0 0]
            w2 = [0 0 11; 1 0 0; 10 1 0]
            data = siena_data()
            add_nodeset!(data, NodeSet(3))
            add_dependent!(data, DependentNetwork(:net, [w1, w2]))
            effects = get_effects(data)
            include_effects!(effects, :net, [:outdegree])

            state = NetworkState()
            initialize!(state, data, 1)
            oset = build_objective_set(effects)
            θ_obj = [0.0]

            _, alters1 = compute_network_choice_probs(oset, θ_obj, state, data, 1, :net)
            @test alters1 == [0, 2]        # 3 is structurally fixed for actor 1
            _, alters3 = compute_network_choice_probs(oset, θ_obj, state, data, 3, :net)
            @test alters3 == [0, 2]        # 1 is structurally fixed for actor 3
            _, alters2 = compute_network_choice_probs(oset, θ_obj, state, data, 2, :net)
            @test alters2 == [0, 1, 3]     # actor 2 has no structural dyads
        end

        @testset "simulation never toggles structural dyads" begin
            Random.seed!(99)
            n = 10
            base = [Int(rand() < 0.2 && i != j) for i in 1:n, j in 1:n]
            coded = copy(base)
            zeros_at = [(1, 4), (2, 7), (9, 3)]
            ones_at = [(5, 6), (8, 1)]
            for (i, j) in zeros_at
                coded[i, j] = 10
            end
            for (i, j) in ones_at
                coded[i, j] = 11
            end

            data = siena_data()
            add_nodeset!(data, NodeSet(n))
            add_dependent!(data, DependentNetwork(:net, [coded, copy(coded)]))
            effects = get_effects(data)
            include_effects!(effects, :net, [:outdegree, :recip])

            for seed in (1, 2, 3)
                # High rate, tie-friendly parameters: lots of ministeps
                state, results = simulate_saom(data, effects, [8.0, 0.5, 0.3];
                                               seed=seed)
                x = results[1].final_state.networks[:net]
                for (i, j) in zeros_at
                    @test x[i, j] == 0
                end
                for (i, j) in ones_at
                    @test x[i, j] == 1
                end
            end
        end

        @testset "siena07 runs end-to-end with structural dyads" begin
            Random.seed!(17)
            n = 20
            w1 = [Int(rand() < 0.12 && i != j) for i in 1:n, j in 1:n]
            zeros_at = [(1, 2), (3, 15), (7, 7 + 1)]
            ones_at = [(4, 9), (11, 5)]
            for (i, j) in zeros_at
                w1[i, j] = 10
            end
            for (i, j) in ones_at
                w1[i, j] = 11
            end

            # Generate wave 2 by simulating from the coded wave 1, so wave 2
            # inherits the structural face values, then re-code them
            gen = siena_data()
            add_nodeset!(gen, NodeSet(n))
            add_dependent!(gen, DependentNetwork(:net, [w1, w1]))
            geff = get_effects(gen)
            include_effects!(geff, :net, [:outdegree, :recip])
            gstate, _ = simulate_saom(gen, geff, [4.0, -1.5, 1.0]; seed=5)
            w2 = copy(gstate.networks[:net])
            for (i, j) in vcat(zeros_at, ones_at)
                @test w2[i, j] == (w1[i, j] == 11 ? 1 : 0)  # untouched by simulation
                w2[i, j] = w1[i, j]                          # restore the coding
            end

            data = siena_data()
            add_nodeset!(data, NodeSet(n))
            add_dependent!(data, DependentNetwork(:net, [w1, w2]))
            effects = get_effects(data)
            include_effects!(effects, :net, [:outdegree, :recip])

            alg = siena_algorithm(seed=21, verbose=false, phase1_iterations=20,
                                  n_subphases=2, phase3_iterations=150,
                                  derivative_sims=20)
            result = siena07(data, effects; algorithm=alg)
            @test result isa SienaResult
            @test all(isfinite, result.estimates)
            @test all(isfinite, result.standard_errors)
            @test all(isfinite, result.t_ratios)
            @test result.diverged == false
        end
    end

    @testset "Golden target statistics vs RSiena (s50)" begin
        s501 = read_int_matrix("s501.csv")
        s502 = read_int_matrix("s502.csv")
        s503 = read_int_matrix("s503.csv")
        s50a = read_int_matrix("s50a.csv")
        s50s = read_int_matrix("s50s.csv")

        data = siena_data()
        add_nodeset!(data, NodeSet(50))
        add_dependent!(data, DependentNetwork(:friendship, [s501, s502, s503]))
        add_dependent!(data, DependentBehavior(:alcohol, [s50a[:, w] for w in 1:3]))
        add_covariate!(data, ConstantCovariate(:smoke1, s50s[:, 1]))

        # Golden values from RSiena 4.x getTargets on the bundled s50 data
        # (unconditional MoM; statistics summed over the two periods)
        checks = [
            (BasicRateEffect(:friendship, 1), 115.0),
            (BasicRateEffect(:friendship, 2), 106.0),
            (BasicRateEffect(:alcohol, 1), 27.0),
            (BasicRateEffect(:alcohol, 2), 33.0),
            (OutdegreeEffect(:friendship), 238.0),
            (ReciprocityEffect(:friendship), 160.0),
            (TransitiveTripletsEffect(:friendship), 225.0),
            (TransitiveMediatedTripletsEffect(:friendship), 225.0),
            (TransitiveRecipTripletsEffect(:friendship), 175.0),
            (CyclicTripletsEffect(:friendship), 72.0),
            (TransitiveTiesEffect(:friendship), 154.0),
            (BetweennessEffect(:friendship), 291.0),
            (NbrDist2Effect(:friendship), 239.0),
            (DenseTriadsEffect(:friendship), 249.0),
            (IndegreePopularityEffect(:friendship), 798.0),
            (IndegreePopularityEffect(:friendship; sqrt=true), 426.025940),
            (OutdegreePopularityEffect(:friendship), 676.0),
            (OutdegreeActivityEffect(:friendship), 758.0),
            (OutdegreeActivityEffect(:friendship; sqrt=true), 416.092258),
            (IndegreeActivityEffect(:friendship), 676.0),
            (IsolateNetEffect(:friendship), 5.0),
            (AlterEffect(:friendship, :smoke1), -1.44),
            (EgoEffect(:friendship, :smoke1), 3.56),
            (SimilarityEffect(:friendship, :smoke1), 23.037143),
            (SameEffect(:friendship, :smoke1), 163.0),
            (EgoTimesAlterEffect(:friendship, :smoke1), 41.8272),
            (LinearShapeEffect(:alcohol), 11.666667),
            (QuadraticShapeEffect(:alcohol), 121.071111),
            (AverageSimilarityEffect(:alcohol, :friendship), 9.852772),
            (TotalSimilarityEffect(:alcohol, :friendship), 23.315204),
            (IndegreeEffect(:alcohol, :friendship), 51.046667),
            (BehaviorOutdegreeEffect(:alcohol, :friendship), 52.046667),
            (AverageAlterEffect(:alcohol, :friendship), 53.908644),
            (BehaviorCovariateEffect(:alcohol, :smoke1), 28.26),
        ]

        effects = SienaEffects()
        for (i, (eff, _)) in enumerate(checks)
            Siena.add_effect!(effects, EffectEntry(eff; shortname="e$i", include=true))
        end
        targets = compute_target_statistics(data, effects)
        @test length(targets) == length(checks)
        for (i, (_, want)) in enumerate(checks)
            @test targets[i] ≈ want atol = 2e-4
        end
    end

    @testset "Golden: RSiena siena07 fitted output (s50)" begin
        # The testset above checks TARGET STATISTICS against RSiena. Targets are
        # the easy half: a deterministic function of the observed waves, so they
        # prove the effect FORMULAS and nothing about the ESTIMATOR. This one
        # checks what siena07 actually returns — fitted coefficients, standard
        # errors, convergence diagnostics — against a real RSiena 1.6.6 run,
        # frozen with provenance in test/fixtures/s50_siena07.toml (generated by
        # test/fixtures/r/s50_siena07.R; read its header before touching the
        # tolerances, which are derived from a measured Monte-Carlo width, not
        # chosen to make this pass).
        #
        # HOW THIS COMPARISON IS MADE, AND WHY NOT THE OBVIOUS WAY.
        # Both sides are Method-of-Moments estimators driven by stochastic
        # approximation; a single run of either is a draw from a cloud. So a
        # single-run-vs-single-run comparison could only be given a tolerance so
        # wide it would test nothing (Siena.jl's seed-to-seed sd on the
        # similarity effect alone is 0.045 — a fifth of its standard error).
        # Instead we average FIVE Siena.jl fits at declared seeds, whose mean has
        # Monte-Carlo error sd/sqrt(5), and compare THAT to RSiena.
        #
        # The result, honestly: Siena.jl's coefficients agree with RSiena's — the
        # largest discrepancy on any objective parameter is 0.032 (reciprocity),
        # ~0.16 of its standard error, and every parameter is within 0.28 SE. But
        # Siena.jl's Robbins-Monro procedure is 3-19x NOISIER than RSiena's at
        # the same nominal budget, it does NOT reach RSiena's convergence
        # standard on this model (see the convergence block below), and roughly
        # 1 seed in 10 diverges outright. Right estimand, weaker algorithm.
        g = Networks.load_golden(joinpath(@__DIR__, "fixtures", "s50_siena07.toml"))
        @test g.provenance["rsiena_version"] == "1.6.6"
        report(key, actual) = begin
            ok = Networks.check_golden(g, key, actual)
            ok || println(stderr, Networks.golden_report(g, key, actual))
            ok
        end

        s501 = read_int_matrix("s501.csv")
        s502 = read_int_matrix("s502.csv")
        s503 = read_int_matrix("s503.csv")
        s50s = read_int_matrix("s50s.csv")

        data = siena_data()
        add_nodeset!(data, NodeSet(50))
        add_dependent!(data, DependentNetwork(:friendship, [s501, s502, s503]))
        add_covariate!(data, ConstantCovariate(:smoke1, s50s[:, 1]))
        effects = get_effects(data)
        include_effects!(effects, :friendship,
                         [:outdegree, :recip, :transTrip,
                          :altsmoke1, :egosmoke1, :simsmoke1])

        # Siena.jl orders ego BEFORE alter; RSiena orders alter before ego. Map
        # rather than assume: rate1 rate2 density recip transTrip [alter ego] sim
        to_rsiena_order = [1, 2, 3, 4, 5, 7, 6, 8]
        @test g.values["effect_names"] ==
              ["constant friendship rate (period 1)",
               "constant friendship rate (period 2)",
               "outdegree (density)", "reciprocity", "transitive triplets",
               "smoke1 alter", "smoke1 ego", "smoke1 similarity"]

        # --- targets: deterministic, so machine precision -------------------
        one = siena07(data, effects; algorithm=siena_algorithm(seed=1, verbose=false))
        @test report("targets", one.targets[to_rsiena_order])

        # --- fitted coefficients and SEs: the mean of five declared seeds ----
        # Deterministic given the seeds (Siena.jl's simulations are seeded per
        # run, so this testset is reproducible, not flaky).
        seeds = 1:5
        fits = [one; [siena07(data, effects;
                              algorithm=siena_algorithm(seed=s, verbose=false))
                      for s in 2:5]]
        θ = mean(f.estimates[to_rsiena_order] for f in fits)
        se = mean(f.standard_errors[to_rsiena_order] for f in fits)

        @test report("rate_coefficients", θ[1:2])
        @test report("coefficients", θ[3:8])
        @test report("rate_std_errors", se[1:2])
        @test report("std_errors", se[3:8])

        # --- convergence: the gap, stated rather than papered over -----------
        # RSiena's t-ratios and Siena.jl's are both mean-zero Monte-Carlo
        # quantities, so they are NOT expected to reproduce each other's values;
        # what is comparable is whether each meets the standard RSiena publishes
        # (|t| < 0.1 per parameter, overall convergence ratio < 0.25).
        #
        # RSiena meets it: the frozen tconv_max is 0.131.
        @test g.values["tconv_max"] < 0.25
        @test all(abs.(Float64.(g.values["t_ratios"])) .< 0.1)
        #
        # Siena.jl, at RSiena's budget, does NOT: on these five seeds tconv_max
        # is 0.255, 0.318, 0.677, 0.766, 0.255 — 2 to 6x RSiena's, and every one
        # of them above the 0.25 publication threshold. This is a real defect in
        # the Robbins-Monro procedure (Siena.jl reports it truthfully:
        # `converged` is false), NOT a tolerance to be widened, and it is
        # asserted here only loosely so a genuine blow-up (some seeds reach
        # tconv_max ~50) still fails the suite.
        for f in fits
            @test isfinite(f.tconv_max)
            @test f.tconv_max < 1.0          # characterization, not endorsement
            @test !f.diverged
        end
        @test any(f.tconv_max > Float64(g.values["tconv_max"]) for f in fits)
    end

    @testset "Multi-wave targets are per-period sums" begin
        s501 = read_int_matrix("s501.csv")
        s502 = read_int_matrix("s502.csv")
        s503 = read_int_matrix("s503.csv")

        build(nets) = begin
            d = siena_data()
            add_nodeset!(d, NodeSet(50))
            add_dependent!(d, DependentNetwork(:friendship, nets))
            e = get_effects(d)
            include_effects!(e, :friendship, [:outdegree, :recip, :transTrip])
            (d, e)
        end

        d3, e3 = build([s501, s502, s503])
        d12, e12 = build([s501, s502])
        d23, e23 = build([s502, s503])

        t3 = compute_target_statistics(d3, e3)
        t12 = compute_target_statistics(d12, e12)
        t23 = compute_target_statistics(d23, e23)
        # objective part (last 3 entries) is additive over periods
        @test t3[end-2:end] ≈ t12[end-2:end] .+ t23[end-2:end]
    end

    @testset "Algorithm Configuration" begin
        alg = SienaAlgorithm()
        @test alg.n_subphases == 4
        # RSiena publication standard: per-parameter |t| < 0.1, tconv.max < 0.25
        @test alg.convergence_threshold == 0.1
        @test alg.overall_convergence_threshold == 0.25
        @test alg.derivative_sims == 30
        @test alg.derivative_method == :score
        @test alg.conditional == false
        @test alg.condvar === nothing
        @test alg.n_simulations == 1
        @test alg.parallel                      # threaded by default...
        @test alg.max_iterations === nothing    # ...and no iteration budget

        alg2 = siena_algorithm(n_subphases=3, seed=42,
                               derivative_method=:finite_difference)
        @test alg2.n_subphases == 3
        @test alg2.seed == 42
        @test alg2.derivative_method == :finite_difference
        @test_throws ArgumentError siena_algorithm(derivative_method=:bogus)

        # `model_type` selects which dependent variables co-evolve (NOT RSiena's
        # `modelType`, which selects forcing/initiative network models and is not
        # implemented). It is a real control -- see "model_type restricts the
        # co-evolving variables" below.
        @test alg.model_type == :standard
        @test siena_algorithm(model_type=:networkonly).model_type == :networkonly
        @test siena_algorithm(model_type=:behavioronly).model_type == :behavioronly
        @test_throws ArgumentError siena_algorithm(model_type=:bogus)
        @test_throws ArgumentError siena_algorithm(model_type=:behavior)
        @test occursin("model_type=:standard", sprint(show, alg))

        # The remaining controls are validated instead of silently misbehaving
        @test_throws ArgumentError siena_algorithm(n_simulations=0)
        @test_throws ArgumentError siena_algorithm(max_iterations=0)
        @test_throws ArgumentError siena_algorithm(phase3_iterations=0)
        @test_throws ArgumentError siena_algorithm(derivative_sims=0)
        @test siena_algorithm(n_simulations=5).n_simulations == 5
        @test siena_algorithm(max_iterations=7).max_iterations == 7
        @test siena_algorithm(parallel=false).parallel == false
    end

    @testset "Algorithm controls change execution" begin
        # Every public control of SienaAlgorithm must change something observable
        # about the run (or be rejected -- see "Algorithm Configuration" above).
        # `SienaResult` reports the settings that were actually in effect.
        data = siena_data()
        add_nodeset!(data, NodeSet(8))
        Random.seed!(7)
        nets = [rand(0:1, 8, 8) for _ in 1:2]
        for net in nets, i in 1:8
            net[i, i] = 0
        end
        add_dependent!(data, DependentNetwork(:net, nets))

        # A deliberately tiny algorithm: 3 free parameters (1 rate + outdegree + recip)
        function fit(; kwargs...)
            effects = get_effects(data)
            include_effects!(effects, :net, [:outdegree, :recip])
            alg = siena_algorithm(; verbose=false, seed=3, n_subphases=1,
                                  phase1_iterations=2, phase3_iterations=5,
                                  derivative_sims=2, kwargs...)
            return siena07(data, effects; algorithm=alg)
        end

        @testset "parallel=false runs serially" begin
            # The dispatch helper itself: with parallel=false every loop body must
            # run on the calling thread, whatever JULIA_NUM_THREADS is.
            tids = zeros(Int, 8)
            Siena._run_simulations!(8, false) do i
                tids[i] = Threads.threadid()
            end
            @test all(==(Threads.threadid()), tids)

            fill!(tids, 0)
            Siena._run_simulations!(8, true) do i
                tids[i] = Threads.threadid()
            end
            @test all(>(0), tids)                              # every slot was visited
            @test length(unique(tids)) <= Threads.nthreads()

            # ...and the estimator reports the execution it actually performed.
            @test fit(parallel=false).n_threads_used == 1
            @test fit(parallel=true).n_threads_used == Threads.nthreads()

            # Threading is only about *where* the work runs: same seed, same answer.
            @test coef(fit(parallel=false)) ≈ coef(fit(parallel=true))
        end

        @testset "n_simulations changes the simulation count" begin
            r1 = fit(n_simulations=1)
            r3 = fit(n_simulations=3)
            @test r1.n_simulations_run > 0
            # Only the Robbins-Monro iterations of phases 1 and 2 draw the extra
            # simulations: (1 + n_subphases) * phase1_iterations per extra simulation.
            extra_per_sim = (1 + 1) * 2
            @test r3.n_simulations_run - r1.n_simulations_run == 2 * extra_per_sim
            @test r1.n_iterations == r3.n_iterations   # iterations unchanged, cost up
        end

        @testset "max_iterations caps the Robbins-Monro iterations" begin
            # No budget: phase 1 plus every subphase runs in full.
            full = fit()
            @test full.n_iterations == 2 * (1 + 1)   # phase1_iterations * (1 + n_subphases)

            # A binding budget stops the Robbins-Monro phases early (with a warning)
            # and leaves fewer iterations behind.
            capped = @test_logs (:warn, r"iteration budget") match_mode = :any fit(max_iterations=3)
            @test capped.n_iterations == 3
            @test capped.n_simulations_run < full.n_simulations_run

            # A budget larger than the schedule never binds and warns about nothing.
            @test fit(max_iterations=100).n_iterations == full.n_iterations
        end
    end

    @testset "model_type restricts the co-evolving variables" begin
        # `model_type` is a real control: it selects which dependent variables take
        # ministeps. The others stay in the state at their period-start values --
        # frozen, but still readable by the effects of the simulated variables -- and
        # their own (now unidentified) effects leave the parameter vector.
        rng = MersenneTwister(19)
        n = 10
        nets = [rand(rng, 0:1, n, n) for _ in 1:2]
        for net in nets, i in 1:n
            net[i, i] = 0
        end
        behs = [rand(rng, 1:4, n), rand(rng, 1:4, n)]

        codata = siena_data()
        add_nodeset!(codata, NodeSet(n))
        add_dependent!(codata, DependentNetwork(:net, nets))
        add_dependent!(codata, DependentBehavior(:beh, behs))

        netdata = siena_data()                     # networks only, no behavior
        add_nodeset!(netdata, NodeSet(n))
        add_dependent!(netdata, DependentNetwork(:net, nets))

        @testset "simulated_variables" begin
            @test Set(simulated_variables(codata, :standard)) == Set([:net, :beh])
            @test simulated_variables(codata, :networkonly) == [:net]
            @test simulated_variables(codata, :behavioronly) == [:beh]
            @test simulated_variables(codata, siena_algorithm(model_type=:networkonly)) ==
                  [:net]
            @test_throws ArgumentError simulated_variables(codata, :bogus)
            # A restriction that leaves nothing to simulate is an error, not an
            # empty (and silently degenerate) model.
            @test_throws ArgumentError simulated_variables(netdata, :behavioronly)
        end

        @testset "restrict_effects keeps the cross-variable effects" begin
            effects = get_effects(codata)
            include_effects!(effects, :net, [:outdegree, :recip])
            include_effects!(effects, :beh, [:linear, :quad, :avAltnet])
            # A network rate effect that *reads* the behavior: its target variable is
            # :net, so it survives a :networkonly restriction.
            Siena.add_effect!(effects, EffectEntry(BehaviorRateEffect(:net, :beh, 1);
                                             name="Rate net (behavior beh)",
                                             shortname="behRatebeh", include=true,
                                             initial_value=0.05))

            net_only = restrict_effects(effects, [:net])
            shortnames = Set(e.shortname for e in net_only.effects)
            @test "outdegree" in shortnames && "recip" in shortnames
            @test "behRatebeh" in shortnames          # reads :beh, but is a :net effect
            @test !("linear" in shortnames) && !("avAltnet" in shortnames)
            @test all(Siena.target_variable(e.effect) == :net for e in net_only.effects)
            # Entries are shared, not copied: flags stay in sync with the original
            entry = only(e for e in net_only.effects if e.shortname == "recip")
            @test entry === only(e for e in effects.effects if e.shortname == "recip")

            beh_only = restrict_effects(effects, [:beh])
            @test Set(e.shortname for e in beh_only.effects if e.include) ==
                  Set(["rate1", "linear", "quad", "avAltnet"])

            # Nothing to drop: the same object comes back (the :standard path is
            # exactly the pre-model_type code path).
            @test restrict_effects(effects, [:net, :beh]) === effects
        end

        # A frozen variable never changes during a simulated period, and the
        # simulated one does.
        @testset "frozen variables do not move" begin
            effects = get_effects(codata)
            include_effects!(effects, :net, [:outdegree, :recip])
            include_effects!(effects, :beh, [:linear, :quad])

            function final_state(vars)
                eff = restrict_effects(effects, vars)
                pm = build_param_map(eff)
                # generous rates, so the simulated variable certainly moves
                θ = [e.effect isa BasicRateEffect ? 6.0 : 0.3 for e in pm.free]
                state, _ = simulate_saom(codata, eff, θ; seed=99,
                                         variables=vars == [:net, :beh] ? nothing : vars)
                return state
            end

            st_net = final_state([:net])
            @test st_net.behaviors[:beh] == behs[1]          # behavior frozen...
            @test st_net.networks[:net] != nets[1]           # ...network moved

            st_beh = final_state([:beh])
            @test st_beh.networks[:net] == nets[1]           # network frozen...
            @test st_beh.behaviors[:beh] != behs[1]          # ...behavior moved

            st_std = final_state([:net, :beh])
            @test st_std.networks[:net] != nets[1]
            @test st_std.behaviors[:beh] != behs[1]
        end

        # Fitting: each model_type estimates its own variable's effects only, and
        # the three fits are genuinely different.
        function cofit(; effects=nothing, kwargs...)
            eff = effects === nothing ? get_effects(codata) : effects
            if effects === nothing
                include_effects!(eff, :net, [:outdegree, :recip])
                include_effects!(eff, :beh, [:linear, :quad])
            end
            alg = siena_algorithm(; verbose=false, seed=13, n_subphases=1,
                                  phase1_iterations=2, phase3_iterations=8,
                                  derivative_sims=2, kwargs...)
            return siena07(codata, eff; algorithm=alg)
        end

        @testset "each model_type estimates only its own variable's effects" begin
            r_std = cofit()
            r_net = cofit(model_type=:networkonly)
            r_beh = cofit(model_type=:behavioronly)

            @test r_std.model_type == :standard
            @test Set(r_std.parameter_names) ==
                  Set(["Rate net (period 1)", "Rate beh (period 1)",
                       "outdegree", "recip", "linear", "quad"])

            @test r_net.model_type == :networkonly
            @test Set(r_net.parameter_names) ==
                  Set(["Rate net (period 1)", "outdegree", "recip"])
            @test keys(r_net.rate_estimates) == Set([:net])
            @test all(Siena.target_variable(e.effect) == :net
                      for e in r_net.effects.effects)

            @test r_beh.model_type == :behavioronly
            @test Set(r_beh.parameter_names) ==
                  Set(["Rate beh (period 1)", "linear", "quad"])
            @test keys(r_beh.rate_estimates) == Set([:beh])

            # ...and the restriction changes the estimates it does share with the
            # standard model: freezing the behavior changes the network process.
            std_of(r, name) = r.estimates[findfirst(==(name), r.parameter_names)]
            @test std_of(r_net, "outdegree") != std_of(r_std, "outdegree")
            @test std_of(r_net, "recip") != std_of(r_std, "recip")
            @test std_of(r_beh, "linear") != std_of(r_std, "linear")
            @test coef(r_net) != coef(r_beh)         # trivially: different models
        end

        @testset "frozen variables stay readable by the simulated ones" begin
            # :networkonly with a network rate effect that reads the frozen behavior
            eff = get_effects(codata)
            include_effects!(eff, :net, [:outdegree, :recip])
            include_effects!(eff, :beh, [:linear, :quad])
            Siena.add_effect!(eff, EffectEntry(BehaviorRateEffect(:net, :beh, 1);
                                         name="Rate net (behavior beh)",
                                         shortname="behRatebeh", include=true,
                                         initial_value=0.05))
            r = cofit(effects=eff, model_type=:networkonly)
            @test "Rate net (behavior beh)" in r.parameter_names   # estimated...
            state = initialize!(NetworkState(), codata, 1)
            # ...and it really reads the (frozen) behavior
            @test any(Siena.rate_score(BehaviorRateEffect(:net, :beh, 1), state,
                                       codata, i) != 0 for i in 1:n)

            # :behavioronly with a behavior effect that reads the frozen network
            eff2 = get_effects(codata)
            include_effects!(eff2, :net, [:outdegree, :recip])
            include_effects!(eff2, :beh, [:linear, :quad, :avAltnet])
            r2 = cofit(effects=eff2, model_type=:behavioronly)
            @test "avAltnet" in r2.parameter_names
            @test any(Siena.evaluate_actor(AverageAlterEffect(:beh, :net), state,
                                           codata, i) != 0 for i in 1:n)
        end

        @testset "restrictions that leave nothing to estimate throw" begin
            # No dependent variable of the requested kind
            neteff = get_effects(netdata)
            include_effects!(neteff, :net, [:outdegree, :recip])
            @test_throws ArgumentError siena07(netdata, neteff;
                algorithm=siena_algorithm(verbose=false, model_type=:behavioronly))

            # Variables of the right kind, but no free parameter left for them:
            # every :net entry is excluded, so :networkonly has nothing to estimate.
            eff = get_effects(codata)
            include_effects!(eff, :beh, [:linear, :quad])
            for e in eff.effects
                Siena.target_variable(e.effect) == :net && (e.include = false)
            end
            @test_throws ArgumentError siena07(codata, eff;
                algorithm=siena_algorithm(verbose=false, model_type=:networkonly))
        end

        @testset "conditioning on a frozen variable throws" begin
            eff = get_effects(codata)
            include_effects!(eff, :net, [:outdegree, :recip])
            include_effects!(eff, :beh, [:linear, :quad])
            @test_throws ArgumentError siena07(codata, eff;
                algorithm=siena_algorithm(verbose=false, model_type=:networkonly,
                                          conditional=true, condvar=:beh))
            # simulate_saom rejects it at its own level too
            @test_throws ArgumentError simulate_saom(codata, restrict_effects(eff, [:net]),
                                                     zeros(3); condvar=:beh,
                                                     cond_targets=[1],
                                                     variables=[:net])
            # Conditioning on the simulated variable is fine -- and with only one
            # variable left, condvar even defaults to it.
            r = cofit(model_type=:networkonly, conditional=true)
            @test haskey(r.rate_estimates, :net)
        end

        @testset ":standard is the unrestricted model, unchanged" begin
            # The :standard path must be bit-for-bit the pre-model_type behaviour:
            # explicit :standard, the default, and an explicit full variable list all
            # give the same seeded answer.
            @test coef(cofit(model_type=:standard)) == coef(cofit())

            effects = get_effects(codata)
            include_effects!(effects, :net, [:outdegree, :recip])
            include_effects!(effects, :beh, [:linear, :quad])
            pm = build_param_map(effects)
            θ = [e.effect isa BasicRateEffect ? 4.0 : 0.2 for e in pm.free]
            s1, _ = simulate_saom(codata, effects, θ; seed=7)
            s2, _ = simulate_saom(codata, effects, θ; seed=7,
                                  variables=collect(keys(codata.dependents)))
            @test s1.networks[:net] == s2.networks[:net]
            @test s1.behaviors[:beh] == s2.behaviors[:beh]
        end
    end

    @testset "Convergence standards" begin
        cs = ConvergenceStats(2)
        Siena.update_convergence!(cs, [0.05, -0.2], [1.0, 1.0])
        @test cs.max_t_ratio ≈ 0.2
        cs.tconv_max = 0.2
        # per-parameter t-ratio above 0.1 fails even though tconv.max is fine
        @test !Siena.is_converged(cs, 0.1, 0.25)
        Siena.update_convergence!(cs, [0.05, -0.05], [1.0, 1.0])
        cs.tconv_max = 0.3
        # tconv.max above 0.25 fails even though all t-ratios pass
        @test !Siena.is_converged(cs, 0.1, 0.25)
        cs.tconv_max = 0.2
        @test Siena.is_converged(cs, 0.1, 0.25)
        # two-argument form only checks the per-parameter t-ratios
        @test Siena.is_converged(cs, 0.1)
    end

    @testset "Divergence clamp" begin
        data = siena_data()
        add_nodeset!(data, NodeSet(6))
        nets = [rand(0:1, 6, 6) for _ in 1:2]
        for net in nets, i in 1:6
            net[i, i] = 0
        end
        add_dependent!(data, DependentNetwork(:net, nets))
        effects = get_effects(data)
        include_effects!(effects, :net, [:outdegree, :recip])
        pm = build_param_map(effects)

        θ = [2000.0, 12.0, 0.5]        # rate above cap, objective above cap
        @test Siena._clamp_parameters!(θ, pm)
        @test θ == [1e3, 10.0, 0.5]
        @test !Siena._clamp_parameters!(θ, pm)   # already inside the box
        θ2 = [0.01, -12.0, 0.0]
        @test Siena._clamp_parameters!(θ2, pm)
        @test θ2 == [0.05, -10.0, 0.0]           # rate floor is not divergence...
        θ3 = [0.01, 0.0, 0.0]
        @test !Siena._clamp_parameters!(θ3, pm)  # ...on its own
        @test θ3[1] == 0.05
    end

    @testset "Gain Sequence" begin
        gs = GainSequence(0.2, 0.001)
        @test gs.current == 0.2
        @test next_gain!(gs) == 0.2
        @test next_gain!(gs) == 0.1
        reset_gain!(gs)
        @test gs.iteration == 0
        @test gs.current == 0.2
    end

    @testset "include_effects!" begin
        data = siena_data()
        add_nodeset!(data, NodeSet(10))
        nets = [rand(0:1, 10, 10) for _ in 1:2]
        for net in nets, i in 1:10
            net[i, i] = 0
        end
        add_dependent!(data, DependentNetwork(:net, nets))
        effects = get_effects(data)

        include_effects!(effects, :net, [:outdegree]; initial_value=0.0)
        entry = only(e for e in effects.effects if e.shortname == "outdegree")
        @test entry.include
        @test entry.initial_value == 0.0  # explicit zero is honored

        # A requested effect that does not exist cannot silently disappear: strict
        # matching is the default, and the error names what is available.
        err = try
            include_effects!(effects, :net, [:nosuch])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("nosuch", err.msg)
        @test occursin("outdegree", err.msg)   # the available effects are listed
        @test occursin("strict=false", err.msg)

        # A typo does not quietly estimate the wrong model...
        @test_throws ArgumentError include_effects!(effects, :net, [:transTrpt])
        @test !any(e -> e.shortname == "transTrip" && e.include, effects.effects)

        # ...nor does a right name on the wrong variable.
        @test_throws ArgumentError include_effects!(effects, :nosuchvar, [:outdegree])

        # Even one bad name in a batch is rejected, and nothing about the requested
        # batch is left half-applied under the default.
        @test_throws ArgumentError include_effects!(effects, :net, [:recip, :nosuch])

        # Explicit opt-out: warn and skip the missing ones, include the rest.
        @test_logs (:warn, r"NOT included") include_effects!(effects, :net,
                                                             [:recip, :nosuch];
                                                             strict=false)
        @test only(e for e in effects.effects if e.shortname == "recip").include
        @test !any(e -> e.shortname == "nosuch", effects.effects)
    end

    @testset "include_interaction! throws" begin
        data = siena_data()
        add_nodeset!(data, NodeSet(6))
        nets = [zeros(Int, 6, 6) for _ in 1:2]
        add_dependent!(data, DependentNetwork(:net, nets))
        effects = get_effects(data)
        include_effects!(effects, :net, [:outdegree, :recip])
        n_included = length(get_included_effects(effects))

        # Interactions are not implemented; the old no-op returned a model silently
        # missing the requested interaction.
        @test_throws ArgumentError include_interaction!(effects, :net, :outdegree, :recip)
        @test length(get_included_effects(effects)) == n_included   # model unchanged
    end

    @testset "Score function and derivative estimators" begin
        Random.seed!(21)
        n = 16
        net1 = [Int(rand() < 0.15 && i != j) for i in 1:n, j in 1:n]

        gen = siena_data()
        add_nodeset!(gen, NodeSet(n))
        add_dependent!(gen, DependentNetwork(:net, [net1, net1]))
        geff = get_effects(gen)
        include_effects!(geff, :net, [:outdegree, :recip])
        θ = [3.0, -1.3, 0.8]  # rate, density, recip
        st, _ = simulate_saom(gen, geff, θ; seed=5)
        net2 = copy(st.networks[:net])

        data = siena_data()
        add_nodeset!(data, NodeSet(n))
        add_dependent!(data, DependentNetwork(:net, [net1, net2]))
        effects = get_effects(data)
        include_effects!(effects, :net, [:outdegree, :recip])
        pm = build_param_map(effects)

        # Score accumulation does not consume extra randomness: statistics are
        # unchanged for the same seed
        sacc = ScoreAccumulator(pm)
        s_plain = compute_simulated_statistics(data, effects,
                      simulate_saom(data, effects, θ; seed=99)[2])
        s_scored = compute_simulated_statistics(data, effects,
                      simulate_saom(data, effects, θ; seed=99, scores=sacc)[2])
        @test s_plain == s_scored
        @test any(!=(0.0), sacc.scores)
        reset_scores!(sacc)
        @test all(==(0.0), sacc.scores)

        # E[score] = 0 at any θ: mean accumulated score is ~0 within MC error
        nsim = 500
        S = zeros(nsim, 3)
        for s in 1:nsim
            reset_scores!(sacc)
            simulate_saom(data, effects, θ; seed=1000 + s, scores=sacc)
            S[s, :] = sacc.scores
        end
        for j in 1:3
            @test abs(mean(S[:, j])) < 4 * std(S[:, j]) / sqrt(nsim)
        end

        # Score-function and finite-difference derivative estimators agree
        D_fd = estimate_derivative_matrix(data, effects, θ, 200, MersenneTwister(1))
        D_sc = estimate_derivative_matrix_score(data, effects, θ, 800,
                                                MersenneTwister(2))
        @test size(D_sc) == (3, 3)
        # statistics increase in their own parameter
        @test all(diag(D_fd) .> 0)
        @test all(diag(D_sc) .> 0)
        # agreement to Monte-Carlo/finite-difference-bias tolerance
        @test norm(D_sc - D_fd) < 0.25 * norm(D_fd)

        # E[score] = 0 also holds with a co-evolving behavior and a non-basic rate
        # effect (exercises the behavior-choice and rate-effect score terms)
        codata = siena_data()
        add_nodeset!(codata, NodeSet(n))
        add_dependent!(codata, DependentNetwork(:net, [net1, net2]))
        add_dependent!(codata, DependentBehavior(:beh, [rand(1:4, n), rand(1:4, n)]))
        coeff = get_effects(codata)
        include_effects!(coeff, :net, [:outdegree, :recip])
        include_effects!(coeff, :beh, [:linear, :quad])
        Siena.add_effect!(coeff, EffectEntry(OutdegreeRateEffect(:net, :net, 1);
                                             shortname="outRate", include=true,
                                             initial_value=0.1))
        copm = build_param_map(coeff)
        θvals = Dict("outdegree" => -1.3, "recip" => 0.8,
                     "linear" => 0.2, "quad" => -0.1)
        θco = [e.effect isa BasicRateEffect ? 3.0 :
               e.effect isa RateEffect ? 0.1 : θvals[e.shortname]
               for e in copm.free]
        @test Siena.n_free_parameters(copm) == 7
        cosacc = ScoreAccumulator(copm)
        Sco = zeros(nsim, length(θco))
        for s in 1:nsim
            reset_scores!(cosacc)
            simulate_saom(codata, coeff, θco; seed=5000 + s, scores=cosacc)
            Sco[s, :] = cosacc.scores
        end
        for j in 1:length(θco)
            @test abs(mean(Sco[:, j])) < 4 * std(Sco[:, j]) / sqrt(nsim)
        end
    end

    @testset "StatsAPI integration" begin
        # the accessors are StatsAPI methods, so they are the same bindings that
        # StatsBase re-exports: `using Siena, StatsBase` must not shadow them
        @test Siena.coef === StatsBase.coef
        @test Siena.stderror === StatsBase.stderror
        @test Siena.vcov === StatsBase.vcov
        @test Siena.confint === StatsBase.confint
        @test coef === StatsBase.coef  # unqualified use is unambiguous
    end

    @testset "Estimation end-to-end (siena07)" begin
        Random.seed!(7)
        n = 30
        net1 = [Int(rand() < 0.10 && i != j) for i in 1:n, j in 1:n]

        # Generate wave 2 from known parameters, then recover them
        gen = siena_data()
        add_nodeset!(gen, NodeSet(n))
        add_dependent!(gen, DependentNetwork(:friendship, [net1, net1]))
        geff = get_effects(gen)
        include_effects!(geff, :friendship, [:outdegree, :recip])
        θtrue = [4.0, -1.5, 1.0]  # rate, density, recip
        state, _ = simulate_saom(gen, geff, θtrue; seed=11)
        net2 = copy(state.networks[:friendship])

        data = siena_data()
        add_nodeset!(data, NodeSet(n))
        add_dependent!(data, DependentNetwork(:friendship, [net1, net2]))
        effects = get_effects(data)
        include_effects!(effects, :friendship, [:outdegree, :recip])

        alg = siena_algorithm(seed=42, verbose=false, phase1_iterations=30,
                              n_subphases=3, phase3_iterations=400, derivative_sims=20)
        result = siena07(data, effects; algorithm=alg)

        @test result.parameter_names ==
              ["Rate friendship (period 1)", "outdegree", "recip"]
        @test all(isfinite, result.estimates)
        @test all(result.standard_errors .> 0)
        @test all(isfinite, result.t_ratios)
        @test maximum(abs.(result.t_ratios)) < 1.0  # deviations small relative to noise

        # Parameter recovery (generous stochastic tolerances)
        @test abs(result.estimates[1] - θtrue[1]) < 2.0    # rate
        @test abs(result.estimates[2] - θtrue[2]) < 0.5    # density
        @test abs(result.estimates[3] - θtrue[3]) < 1.0    # recip
        @test result.rate_estimates[:friendship][1] == result.estimates[1]

        # Convergence report and divergence flag
        @test isfinite(result.tconv_max) && result.tconv_max >= 0
        @test result.diverged == false

        # Accessors (StatsAPI methods, exercised with StatsBase co-loaded)
        @test coef(result) === result.estimates
        @test stderror(result) === result.standard_errors
        @test size(vcov(result)) == (3, 3)
        ci = confint(result)
        @test size(ci) == (3, 2)
        @test all(ci[:, 1] .< result.estimates .< ci[:, 2])

        # show() works, rendering the shared ecosystem coefficient table
        # (Networks.jl print_coeftable: z / Pr(>|z|) columns + signif codes)
        out = sprint(show, result)
        @test occursin("outdegree", out)
        @test occursin("overall max convergence ratio", out)
        @test occursin("Pr(>|z|)", out)
        @test occursin("Signif. codes", out)

        # fit_siena is the standardized entry point; siena07 is the
        # RSiena-name alias and gives bitwise-identical results
        result_fs = fit_siena(data, effects; algorithm=alg)
        @test result_fs.estimates == result.estimates
        @test result_fs.standard_errors == result.standard_errors

        # The finite-difference cross-check path still runs end-to-end and, given
        # enough derivative simulations, gives comparable standard errors (with few
        # simulations its D is much noisier — the reason :score is the default)
        alg_fd = siena_algorithm(seed=43, verbose=false, phase1_iterations=30,
                                 n_subphases=3, phase3_iterations=400,
                                 derivative_sims=200,
                                 derivative_method=:finite_difference)
        result_fd = siena07(data, effects; algorithm=alg_fd)
        @test all(result_fd.standard_errors .> 0)
        @test all(0.5 .< result_fd.standard_errors ./ result.standard_errors .< 2.0)

        # GOF end-to-end
        g = siena_gof_indegree(result, data, :friendship; n_sims=40, seed=3)
        @test 0.0 <= g.p_overall <= 1.0
        @test sum(g.observed) == n
        @test size(g.simulated, 1) == 40
        # per-level Monte-Carlo p-values use (1 + k)/(N + 1): never exactly 0,
        # bounded below by 1/(N + 1)
        @test all(p -> 1 / 41 <= p <= 1.0, g.p_values)
        g2 = siena_gof_triad(result, data, :friendship; n_sims=40, seed=4)
        @test length(g2.observed) == 16
        @test all(p -> 1 / 41 <= p <= 1.0, g2.p_values)
        @test sum(g2.observed) == binomial(n, 3)
        @test all(vec(sum(g2.simulated, dims=2)) .== binomial(n, 3))
        @test occursin("p-value", sprint(show, g2))

        # RSiena-style detail converts to the ecosystem-wide GOFResult
        gc = GOFResult(g2)
        @test gc isa GOFResult
        @test gc.p_overall == g2.p_overall
        @test occursin("Goodness-of-fit", sprint(show, gc))

        # gof() is a method of the ONE shared Networks.jl generic and
        # returns the ecosystem-wide GOFResult directly
        @test Siena.gof === Networks.gof
        gr = gof(result, data, IndegreeDistribution(:friendship);
                 n_sims=20, seed=5)
        @test gr isa GOFResult
        @test Networks.n_simulations(gr) == 20
        @test 0.0 < gr.p_overall <= 1.0
        gv = gof(result, data,
                 [IndegreeDistribution(:friendship),
                  OutdegreeDistribution(:friendship)]; n_sims=10, seed=6)
        @test length(gv.statistics) == 2
        @test occursin("outdegree distribution", sprint(show, gv))
    end

    @testset "Conditional estimation (siena07)" begin
        Random.seed!(19)
        n = 30
        net1 = [Int(rand() < 0.10 && i != j) for i in 1:n, j in 1:n]

        # Generate wave 2 from known parameters
        gen = siena_data()
        add_nodeset!(gen, NodeSet(n))
        add_dependent!(gen, DependentNetwork(:friendship, [net1, net1]))
        geff = get_effects(gen)
        include_effects!(geff, :friendship, [:outdegree, :recip])
        θtrue = [4.0, -1.5, 1.0]  # rate, density, recip
        gstate, _ = simulate_saom(gen, geff, θtrue; seed=13)
        net2 = copy(gstate.networks[:friendship])

        build() = begin
            d = siena_data()
            add_nodeset!(d, NodeSet(n))
            add_dependent!(d, DependentNetwork(:friendship, [net1, net2]))
            e = get_effects(d)
            include_effects!(e, :friendship, [:outdegree, :recip])
            (d, e)
        end

        # Conditional simulation stops exactly at the observed distance
        data, effects = build()
        target = Siena._observed_distance(data, :friendship, 1)
        @test target == sum(abs.(net2 .- net1))
        _, results = simulate_saom(data, effects, θtrue; seed=3,
                                   condvar=:friendship, cond_targets=[target])
        x = copy(results[1].final_state.networks[:friendship])
        @test sum(abs.(x .- net1)) == target
        @test results[1].final_state.time != 1.0
        # guards
        @test_throws ArgumentError simulate_saom(data, effects, θtrue; seed=3,
                                                 condvar=:nosuch,
                                                 cond_targets=[target])
        @test_throws ArgumentError simulate_saom(data, effects, θtrue; seed=3,
                                                 condvar=:friendship)
        pm = build_param_map(effects)
        @test_throws ArgumentError simulate_saom(data, effects, θtrue; seed=3,
                                                 condvar=:friendship,
                                                 cond_targets=[target],
                                                 scores=ScoreAccumulator(pm))

        # Conditional and unconditional estimation agree in expectation
        data_u, effects_u = build()
        alg_u = siena_algorithm(seed=42, verbose=false, phase1_iterations=30,
                                n_subphases=3, phase3_iterations=300,
                                derivative_sims=20)
        result_u = siena07(data_u, effects_u; algorithm=alg_u)

        data_c, effects_c = build()
        alg_c = siena_algorithm(seed=44, verbose=false, phase1_iterations=30,
                                n_subphases=3, phase3_iterations=300,
                                derivative_sims=20, conditional=true)
        result_c = siena07(data_c, effects_c; algorithm=alg_c)

        # The conditioned basic rate leaves the parameter vector...
        @test result_c.parameter_names == ["outdegree", "recip"]
        @test length(result_c.estimates) == 2
        @test all(isfinite, result_c.estimates)
        @test all(result_c.standard_errors .> 0)
        @test result_c.diverged == false
        @test maximum(abs.(result_c.t_ratios)) < 1.0
        # ...its entry is marked fixed on the effects object...
        rate_entry = only(e for e in effects_c.effects
                          if e.effect isa BasicRateEffect)
        @test rate_entry.fix
        # ...and its conditional estimate (simulation rate x mean stopping time)
        # is reported in rate_estimates
        ρc = result_c.rate_estimates[:friendship][1]
        @test isfinite(ρc) && ρc > 0
        @test rate_entry.initial_value == ρc

        # Agreement in expectation (generous stochastic tolerances): objective
        # parameters against the unconditional fit, and the conditional rate
        # (fixed by the observed change count) against the estimated one
        @test abs(result_c.estimates[1] - result_u.estimates[2]) < 0.5  # density
        @test abs(result_c.estimates[2] - result_u.estimates[3]) < 1.0  # recip
        @test abs(ρc - result_u.estimates[1]) < 2.0                     # rate

        # condvar is required when several dependent variables are present
        data_m, effects_m = build()
        add_dependent!(data_m, DependentBehavior(:beh, [rand(1:4, n), rand(1:4, n)]))
        @test_throws ArgumentError siena07(data_m, effects_m;
                                           algorithm=siena_algorithm(verbose=false,
                                                                     conditional=true))
    end

    @testset "Composition change (joiners/leavers)" begin
        @testset "is_present semantics" begin
            cc = CompositionChange()
            add_change!(cc, 4, 2, :join)
            add_change!(cc, 2, 3, :leave)
            add_change!(cc, 5, 2, :join)
            add_change!(cc, 5, 3, :leave)
            @test !is_present(cc, 4, 1) && is_present(cc, 4, 2) && is_present(cc, 4, 3)
            @test is_present(cc, 2, 1) && is_present(cc, 2, 2) && !is_present(cc, 2, 3)
            @test !is_present(cc, 5, 1) && is_present(cc, 5, 2) && !is_present(cc, 5, 3)
            @test is_present(cc, 1, 1) && is_present(cc, 1, 3)  # no events
            @test_throws ArgumentError add_change!(cc, 1, 2, :vanish)
        end

        @testset "sequence validation" begin
            # Actor/wave bounds that do not depend on the data
            @test_throws ArgumentError add_change!(CompositionChange(), 0, 1, :join)
            @test_throws ArgumentError add_change!(CompositionChange(), -1, 1, :leave)
            @test_throws ArgumentError add_change!(CompositionChange(), 1, 0, :join)
            @test_throws ArgumentError CompositionChange([(2, 0, :join)])
            @test_throws ArgumentError CompositionChange([(0, 2, :leave)])

            # Duplicate events: one actor cannot have two events at the same wave
            @test_throws ArgumentError CompositionChange([(1, 2, :join), (1, 2, :leave)])
            @test_throws ArgumentError CompositionChange([(1, 2, :join), (1, 2, :join)])
            cc = CompositionChange()
            add_change!(cc, 1, 2, :join)
            @test_throws ArgumentError add_change!(cc, 1, 2, :leave)

            # Contradictory transitions: an actor cannot join while already present
            # or leave while already absent
            @test_throws ArgumentError CompositionChange([(1, 2, :join), (1, 4, :join)])
            @test_throws ArgumentError CompositionChange([(1, 2, :leave), (1, 3, :leave)])
            # ...also when the events are given out of wave order
            @test_throws ArgumentError CompositionChange([(1, 4, :join), (1, 2, :join)])
            @test_throws ArgumentError add_change!(cc, 1, 5, :join)   # 1 already joined

            # A rejected event leaves the object untouched
            @test cc.changes == [(1, 2, :join)]

            # Legal alternating histories are accepted, in any order, for any actor
            @test_throws ArgumentError add_change!(cc, 1, 3, :join)
            add_change!(cc, 1, 3, :leave)
            add_change!(cc, 1, 5, :join)
            add_change!(cc, 2, 4, :leave)     # a different actor is independent
            @test length(cc.changes) == 4
            @test !is_present(cc, 1, 1) && is_present(cc, 1, 2)
            @test !is_present(cc, 1, 3) && is_present(cc, 1, 5)
            cc2 = CompositionChange([(3, 4, :join), (3, 2, :leave), (3, 6, :leave)])
            @test length(cc2.changes) == 3
        end

        @testset "actor/wave range validation against the data" begin
            data = siena_data()
            add_nodeset!(data, NodeSet(5))
            add_dependent!(data, DependentNetwork(:net, [zeros(Int, 5, 5) for _ in 1:3]))
            @test data.n_waves == 3

            # In range: fine
            ok = CompositionChange([(5, 3, :join), (1, 2, :leave)])
            @test add_composition_change!(data, ok) === data
            @test data.composition_change === ok

            # Actor beyond the node set
            @test_throws ArgumentError add_composition_change!(
                data, CompositionChange([(6, 2, :join)]))
            # Wave beyond the observed waves
            @test_throws ArgumentError add_composition_change!(
                data, CompositionChange([(2, 4, :join)]))
            # A rejected sequence is not attached
            @test data.composition_change === ok

            # Composition change cannot be attached before there are any waves
            empty_data = siena_data()
            add_nodeset!(empty_data, NodeSet(5))
            @test_throws ArgumentError add_composition_change!(
                empty_data, CompositionChange([(1, 1, :join)]))
        end

        @testset "targets exclude absent actors (hand-computed)" begin
            # Actor 4 joins at wave 2: inactive in period 1, active in period 2
            w1 = [0 1 0 0; 0 0 1 0; 1 0 0 0; 0 0 0 0]
            w2 = [0 1 1 0; 1 0 1 0; 0 0 0 1; 1 0 0 0]
            w3 = [0 1 1 1; 1 0 0 0; 0 1 0 1; 1 1 0 0]
            data = siena_data()
            add_nodeset!(data, NodeSet(4))
            add_dependent!(data, DependentNetwork(:net, [w1, w2, w3]))
            cc = CompositionChange()
            add_change!(cc, 4, 2, :join)
            add_composition_change!(data, cc)
            effects = get_effects(data)
            include_effects!(effects, :net, [:outdegree, :recip])

            # Period 1 (actors 1-3): distance 3, outdegree 4, recip 2.
            # Period 2 (all actors): distance 4, outdegree 8, recip 4.
            targets = compute_target_statistics(data, effects)
            @test targets == [3.0, 4.0, 12.0, 6.0]

            # Without the composition change, the joiner's dyads count in
            # period 1: distance 5, outdegree 6 (total 14)
            data_plain = siena_data()
            add_nodeset!(data_plain, NodeSet(4))
            add_dependent!(data_plain, DependentNetwork(:net, [w1, w2, w3]))
            effects_plain = get_effects(data_plain)
            include_effects!(effects_plain, :net, [:outdegree, :recip])
            @test compute_target_statistics(data_plain, effects_plain) ==
                  [5.0, 4.0, 14.0, 6.0]
        end

        @testset "absent actors take no part in simulation" begin
            Random.seed!(23)
            n = 10
            w1 = [Int(rand() < 0.2 && i != j) for i in 1:n, j in 1:n]
            w1[n, :] .= 0
            w1[:, n] .= 0        # joiner starts with no ties
            data = siena_data()
            add_nodeset!(data, NodeSet(n))
            add_dependent!(data, DependentNetwork(:net, [w1, w1, w1]))
            cc = CompositionChange()
            add_change!(cc, n, 2, :join)
            add_composition_change!(data, cc)
            effects = get_effects(data)
            include_effects!(effects, :net, [:outdegree, :recip])

            # Candidate sets: in period 1 the joiner is neither ego nor alter
            state = NetworkState()
            initialize!(state, data, 1)
            @test state.active == vcat(trues(n - 1), falses(1))
            oset = build_objective_set(effects)
            _, alters1 = compute_network_choice_probs(oset, [0.0, 0.0], state,
                                                      data, 1, :net)
            @test !(n in alters1)
            probs_n, alters_n = compute_network_choice_probs(oset, [0.0, 0.0],
                                                             state, data, n, :net)
            @test alters_n == [0] && probs_n == [1.0]
            # in period 2 the joiner is a regular actor
            initialize!(state, data, 2)
            @test state.active === nothing || all(state.active)
            _, alters2 = compute_network_choice_probs(oset, [0.0, 0.0], state,
                                                      data, 1, :net)
            @test n in alters2

            # High-rate simulation: the joiner's dyads never change in period 1
            # but do change in period 2
            for seed in (1, 2, 3)
                _, results = simulate_saom(data, effects, [8.0, 8.0, 0.5, 0.3];
                                           seed=seed)
                x1 = results[1].final_state.networks[:net]
                @test all(x1[n, j] == 0 for j in 1:n)
                @test all(x1[i, n] == 0 for i in 1:n)
            end
            changed = false
            for seed in (1, 2, 3)
                _, results = simulate_saom(data, effects, [8.0, 8.0, 0.5, 0.3];
                                           seed=seed)
                x2 = results[2].final_state.networks[:net]
                changed |= any(x2[n, j] == 1 for j in 1:n) ||
                           any(x2[i, n] == 1 for i in 1:n)
            end
            @test changed
        end

        @testset "siena07 converges with a wave-2 joiner" begin
            Random.seed!(29)
            n = 20
            w1 = [Int(rand() < 0.12 && i != j) for i in 1:n, j in 1:n]
            w1[n, :] .= 0
            w1[:, n] .= 0

            cc = CompositionChange()
            add_change!(cc, n, 2, :join)

            # Generate wave 2 with the joiner frozen, wave 3 with everyone active
            gen1 = siena_data()
            add_nodeset!(gen1, NodeSet(n))
            add_dependent!(gen1, DependentNetwork(:net, [w1, w1]))
            add_composition_change!(gen1, cc)
            geff1 = get_effects(gen1)
            include_effects!(geff1, :net, [:outdegree, :recip])
            g1, _ = simulate_saom(gen1, geff1, [3.0, -1.5, 1.0]; seed=7)
            w2 = copy(g1.networks[:net])
            @test all(w2[n, :] .== 0) && all(w2[:, n] .== 0)

            gen2 = siena_data()
            add_nodeset!(gen2, NodeSet(n))
            add_dependent!(gen2, DependentNetwork(:net, [w2, w2]))
            geff2 = get_effects(gen2)
            include_effects!(geff2, :net, [:outdegree, :recip])
            g2, _ = simulate_saom(gen2, geff2, [3.0, -1.5, 1.0]; seed=9)
            w3 = copy(g2.networks[:net])

            data = siena_data()
            add_nodeset!(data, NodeSet(n))
            add_dependent!(data, DependentNetwork(:net, [w1, w2, w3]))
            add_composition_change!(data, cc)
            effects = get_effects(data)
            include_effects!(effects, :net, [:outdegree, :recip])

            # Sensible targets: the period-1 rate target excludes the joiner
            targets = compute_target_statistics(data, effects)
            @test targets[1] == sum(abs.(w2[1:n-1, 1:n-1] .- w1[1:n-1, 1:n-1]))
            @test all(isfinite, targets)

            alg = siena_algorithm(seed=31, verbose=false, phase1_iterations=30,
                                  n_subphases=3, phase3_iterations=300,
                                  derivative_sims=30)
            result = siena07(data, effects; algorithm=alg)
            @test result isa SienaResult
            @test all(isfinite, result.estimates)
            @test all(isfinite, result.standard_errors)
            @test all(isfinite, result.t_ratios)
            @test result.diverged == false
            @test maximum(abs.(result.t_ratios)) < 1.0
        end
    end

    @testset "Estimation guards" begin
        data = siena_data()
        add_nodeset!(data, NodeSet(5))
        add_dependent!(data, DependentNetwork(:net, [zeros(Int, 5, 5), zeros(Int, 5, 5)]))
        effects = get_effects(data)

        # endowment effects are rejected
        Siena.add_effect!(effects,
            EffectEntry(EndowmentEffect(ReciprocityEffect(:net));
                        shortname="recipEndow", include=true))
        include_effects!(effects, :net, [:outdegree])
        @test_throws ArgumentError siena07(data, effects;
                                           algorithm=siena_algorithm(verbose=false))
    end

    @testset "Triad census classification" begin
        # 1 -> 2, 2 -> 1, 3 -> 1 on 5 actors
        net = zeros(Int, 5, 5)
        net[1, 2] = net[2, 1] = net[3, 1] = 1
        data = siena_data()
        add_nodeset!(data, NodeSet(5))
        add_dependent!(data, DependentNetwork(:net, [net, net]))
        state = NetworkState()
        initialize!(state, data, 1)

        labels, counts = compute_gof_statistic(TriadCensus(:net), state, data)
        @test labels == Siena.TRIAD_LABELS
        census = Dict(zip(labels, counts))
        @test census["003"] == 5
        @test census["012"] == 2
        @test census["102"] == 2
        @test census["111D"] == 1
        @test sum(counts) == binomial(5, 3)

        # 030C vs 030T
        cyc = zeros(Int, 3, 3)
        cyc[1, 2] = cyc[2, 3] = cyc[3, 1] = 1
        data2 = siena_data()
        add_nodeset!(data2, NodeSet(3))
        add_dependent!(data2, DependentNetwork(:net, [cyc, cyc]))
        state2 = NetworkState()
        initialize!(state2, data2, 1)
        _, c2 = compute_gof_statistic(TriadCensus(:net), state2, data2)
        @test c2[findfirst(==("030C"), Siena.TRIAD_LABELS)] == 1

        trans = zeros(Int, 3, 3)
        trans[1, 2] = trans[2, 3] = trans[1, 3] = 1
        state2.networks[:net] = trans
        _, c3 = compute_gof_statistic(TriadCensus(:net), state2, data2)
        @test c3[findfirst(==("030T"), Siena.TRIAD_LABELS)] == 1
    end

    @testset "GOF statistics" begin
        data = siena_data()
        add_nodeset!(data, NodeSet(10))
        net = [Int(rand() < 0.3 && i != j) for i in 1:10, j in 1:10]
        add_dependent!(data, DependentNetwork(:net, [net, net]))

        state = NetworkState()
        initialize!(state, data, 1)

        levls, counts = compute_gof_statistic(IndegreeDistribution(:net), state, data)
        @test sum(counts) == 10
        levls, counts = compute_gof_statistic(OutdegreeDistribution(:net), state, data)
        @test sum(counts) == 10

        # explicit levels are respected (alignment for simulations)
        stat = IndegreeDistribution(:net; levls=collect(0:20))
        levls, counts = compute_gof_statistic(stat, state, data)
        @test length(counts) == 21
        @test sum(counts) == 10

        labels, counts = compute_gof_statistic(GeodesicDistribution(:net), state, data)
        @test length(labels) == 6  # 1..5 + unreachable
    end

    @testset "Networks.jl bridge (SienaNetworkExt)" begin
        @test Base.get_extension(Siena, :SienaNetworkExt) !== nothing

        # Directed panel: DependentNetwork from Vector{<:Network}
        nets = [network(4) for _ in 1:3]
        add_edge!(nets[1], 1, 2)
        add_edge!(nets[2], 1, 2); add_edge!(nets[2], 2, 1)
        add_edge!(nets[3], 2, 3)
        dep = DependentNetwork(:friendship, nets)
        @test dep isa DependentNetwork
        @test n_waves(dep) == 3
        @test n_actors(dep) == 4
        @test dep.directed == true
        @test dep.type == :onemode
        @test dep.allow_self_loops == false
        for w in 1:3
            @test dep.networks[w] == Int.(as_matrix(nets[w]))
        end
        @test dep.networks[2][1, 2] == 1 && dep.networks[2][2, 1] == 1

        # siena_dependent dispatches to the same conversion
        dep2 = siena_dependent(:f2, nets)
        @test dep2 isa DependentNetwork
        @test dep2.networks == dep.networks

        # Undirected panel: directedness preserved, matrices symmetric
        unets = [network(4; directed=false) for _ in 1:2]
        add_edge!(unets[1], 1, 2)
        add_edge!(unets[2], 1, 2); add_edge!(unets[2], 3, 4)
        depu = DependentNetwork(:acquaintance, unets)
        @test depu.directed == false
        @test all(M == M' for M in depu.networks)
        @test depu.networks[2][3, 4] == 1 == depu.networks[2][4, 3]

        # Panel validation: node-set size, directedness, mode structure
        @test_throws ArgumentError DependentNetwork(:x, [network(4), network(5)])
        @test_throws ArgumentError DependentNetwork(
            :x, [network(4), network(4; directed=false)])
        @test_throws ArgumentError DependentNetwork(
            :x, Network{Int}[])
        @test_throws ArgumentError DependentNetwork(
            :x, [network(5; bipartite=2), network(5)])
        @test_throws ArgumentError DependentNetwork(
            :x, [network(5; bipartite=2), network(5; bipartite=3)])

        # Vertex names must agree across waves when present
        na, nb, nc = network(3), network(3), network(3)
        set_vertex_attribute!(na, :vertex_names, Dict(1 => "u", 2 => "v", 3 => "w"))
        set_vertex_attribute!(nb, :vertex_names, Dict(1 => "u", 2 => "x", 3 => "w"))
        set_vertex_attribute!(nc, :vertex_names, Dict(1 => "u", 2 => "v", 3 => "w"))
        @test_throws ArgumentError DependentNetwork(:x, [na, nb])
        @test DependentNetwork(:x, [na, nc]) isa DependentNetwork
        @test DependentNetwork(:x, [na, network(3)]) isa DependentNetwork  # unnamed OK

        # Two-mode metadata carries over (incidence matrix, :twomode type)
        bns = [network(5; bipartite=2) for _ in 1:2]
        add_edge!(bns[1], 1, 3); add_edge!(bns[2], 2, 5)
        depb = DependentNetwork(:affiliation, bns)
        @test depb.type == :twomode
        @test depb.nodeset2 == :mode2
        @test size(depb.networks[1]) == (2, 3)
        @test depb.networks[2][2, 3] == 1  # vertex 5 is mode-2 column 3

        # Dyadic covariates from networks (binary and edge-attribute values)
        w = network(3)
        add_edge!(w, 1, 2); add_edge!(w, 2, 3)
        set_edge_attribute!(w, :weight, Dict((1, 2) => 2.5, (2, 3) => 0.5))
        dcb = ConstantDyadCovariate(:tie, w; center=false)
        @test dcb.values == Float64.(as_matrix(w))
        dcw = constant_dyad_covariate(:prox, w; attr=:weight, center=false)
        @test dcw isa ConstantDyadCovariate
        @test dcw.values[1, 2] == 2.5 && dcw.values[2, 3] == 0.5
        @test dcw.values[1, 3] == 0.0
        vdc = varying_dyad_covariate(:vprox, [w, w]; attr=:weight, center=false)
        @test vdc isa VaryingDyadCovariate
        @test vdc.values[2][1, 2] == 2.5
        @test_throws ArgumentError varying_dyad_covariate(:x, [w, network(4)])

        # Round trip: describe-style Network objects -> Siena fit, no matrices
        Random.seed!(31)
        n = 30
        wave1 = network(n)
        for i in 1:n, j in 1:n
            i != j && rand() < 0.10 && add_edge!(wave1, i, j)
        end
        wave2 = copy(wave1)
        for _ in 1:40   # a modest amount of change between observations
            i, j = rand(1:n), rand(1:n)
            i == j && continue
            has_edge(wave2, i, j) ? rem_edge!(wave2, i, j) : add_edge!(wave2, i, j)
        end
        data = siena_data()
        add_nodeset!(data, NodeSet(n))
        add_dependent!(data, DependentNetwork(:friendship, [wave1, wave2]))
        @test data.dependents[:friendship].networks[1] == Int.(as_matrix(wave1))
        effects = get_effects(data)
        include_effects!(effects, :friendship, [:outdegree, :recip])
        alg = siena_algorithm(seed=8, verbose=false, phase1_iterations=20,
                              n_subphases=2, phase3_iterations=100,
                              derivative_sims=10)
        result = siena07(data, effects; algorithm=alg)
        @test result.parameter_names ==
              ["Rate friendship (period 1)", "outdegree", "recip"]
        @test all(isfinite, coef(result))
        @test all(stderror(result) .> 0)
        @test size(vcov(result)) == (3, 3)
    end

    # Conversion invariants (see the ecosystem table in
    # Networks.jl/docs/src/guide/conversion_invariants.md).
    #
    # Siena's own per-wave mask records STRUCTURAL zeros/ones — ties that are
    # determined — which is a different claim from Networks.jl's missing-dyad
    # mask, which records ties that are UNOBSERVED. Coding an unobserved dyad
    # as a structural zero would tell the estimator the tie is known to be
    # impossible. There is no faithful encoding, so a masked network is
    # REJECTED rather than silently written into the matrix as a 0.
    @testset "Networks.jl bridge: conversion invariants" begin
        for directed in (true, false)
            waves = [network(4; directed=directed) for _ in 1:2]
            add_edge!(waves[1], 1, 2)
            add_edge!(waves[2], 2, 3)

            # Preserved: directedness, loops flag, node-set size, tie values
            dep, rep = DependentNetwork(:friendship, waves; report=true)
            @test dep.directed == directed
            @test n_actors(dep) == 4
            @test n_waves(dep) == 2
            @test dep.networks[1] == Int.(as_matrix(waves[1]))
            # Dropped by nature, and named
            @test :attributes in dropped_fields(rep)
            @test !(:missing_dyads in dropped_fields(rep))

            # Mask a dyad with a PRESENT face value and one with an ABSENT one
            set_missing_dyad!(waves[1], 1, 2)
            set_missing_dyad!(waves[2], 1, 4)

            @test_throws ArgumentError DependentNetwork(:friendship, waves)
            @test_throws ArgumentError siena_dependent(:friendship, waves)
            @test_throws ArgumentError DependentNetwork(:friendship, waves;
                                                        missing=:error)
            @test_throws ArgumentError ConstantDyadCovariate(:prox, waves[1])
            @test_throws ArgumentError VaryingDyadCovariate(:prox, waves)
            @test_throws ArgumentError constant_dyad_covariate(:prox, waves[1])

            # Explicit opt-in writes face values and reports the cost
            dep_face, rep_face = DependentNetwork(:friendship, waves;
                                                  missing=:face, report=true)
            @test dep_face.networks[1] == Int.(as_matrix(waves[1]))
            @test :missing_dyads in dropped_fields(rep_face)
            cov_face, cov_rep = ConstantDyadCovariate(:prox, waves[1];
                                                      missing=:face, report=true)
            @test cov_face isa ConstantDyadCovariate
            @test :missing_dyads in dropped_fields(cov_rep)

            # Declaring the dyads observed makes the conversion legal again
            clear_missing_dyads!(waves[1])
            clear_missing_dyads!(waves[2])
            @test DependentNetwork(:friendship, waves) isa DependentNetwork
            @test ConstantDyadCovariate(:prox, waves[1]) isa ConstantDyadCovariate
            @test VaryingDyadCovariate(:prox, waves) isa VaryingDyadCovariate
        end
    end

    @testset "Effects table" begin
        data = siena_data()
        add_nodeset!(data, NodeSet(10))
        nets = [rand(0:1, 10, 10) for _ in 1:2]
        add_dependent!(data, DependentNetwork(:net, nets))
        add_covariate!(data, ConstantCovariate(:age, randn(10)))
        effects = get_effects(data)
        tbl = effects_table(effects)
        @test tbl isa DataFrame
        @test nrow(tbl) == length(effects)
        @test "rate1" in tbl.shortname
    end

    @testset "Result metadata protocol" begin
        rng = Random.Xoshiro(303)
        n = 20
        w1 = zeros(Int, n, n)
        for i in 1:n, j in 1:n
            i != j && rand(rng) < 0.15 && (w1[i, j] = 1)
        end
        gen = siena_data()
        add_nodeset!(gen, NodeSet(n))
        add_dependent!(gen, DependentNetwork(:net, [w1, w1]))
        geff = get_effects(gen)
        include_effects!(geff, :net, [:outdegree, :recip])
        gstate, _ = simulate_saom(gen, geff, [4.0, -1.5, 1.0]; seed=9)
        w2 = copy(gstate.networks[:net])

        data = siena_data()
        add_nodeset!(data, NodeSet(n))
        add_dependent!(data, DependentNetwork(:net, [w1, w2]))
        effects = get_effects(data)
        include_effects!(effects, :net, [:outdegree, :recip])
        alg = siena_algorithm(seed=31, verbose=false, phase1_iterations=20,
                              n_subphases=2, phase3_iterations=100,
                              derivative_sims=20)
        result = siena07(data, effects; algorithm=alg)

        md = fit_metadata(result)
        # Method of Moments — the ONLY estimator implemented. No likelihood is
        # ever evaluated, so the fit is never exact, whatever the effect set.
        @test md.estimand == :saom
        @test md.objective == :moment
        @test !md.is_exact
        # D⁻¹ Σ D⁻ᵀ from the phase-3 simulations: a moment-estimator sandwich,
        # not an inverse Hessian
        @test md.se_method == :sandwich
        # Missing (NA) tie values are not handled at all
        @test md.missing_method == :rejected
        @test md.tie_method == :not_applicable

        @test any(occursin("Monte-Carlo error", a) for a in md.approximations)
        @test any(occursin("ridge-regularized", a) for a in md.approximations)
        # A standard-model fit reports no model_type restriction
        @test result.model_type == :standard
        @test !any(occursin("FROZEN", a) for a in md.approximations)
    end

end
