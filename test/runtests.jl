using Siena
using Test
using Random
using Statistics
using DataFrames

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
            CyclicTripletsEffect(:net), BalanceEffect(:net), BetweennessEffect(:net),
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
            AverageAttHigherEffect(:beh, :net), AverageAttLowerEffect(:beh, :net),
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
        @test alg.convergence_threshold == 0.25
        @test alg.derivative_sims == 30

        alg2 = siena_algorithm(n_subphases=3, seed=42)
        @test alg2.n_subphases == 3
        @test alg2.seed == 42
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

        @test_logs (:warn, r"not found") include_effects!(effects, :net, [:nosuch])
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

        # Accessors
        @test coef(result) === result.estimates
        @test stderror(result) === result.standard_errors
        @test size(vcov(result)) == (3, 3)
        ci = confint(result)
        @test size(ci) == (3, 2)
        @test all(ci[:, 1] .< result.estimates .< ci[:, 2])

        # show() works
        @test occursin("outdegree", sprint(show, result))

        # GOF end-to-end
        g = siena_gof_indegree(result, data, :friendship; n_sims=40, seed=3)
        @test 0.0 <= g.p_overall <= 1.0
        @test sum(g.observed) == n
        @test size(g.simulated, 1) == 40
        g2 = siena_gof_triad(result, data, :friendship; n_sims=40, seed=4)
        @test length(g2.observed) == 16
        @test sum(g2.observed) == binomial(n, 3)
        @test all(vec(sum(g2.simulated, dims=2)) .== binomial(n, 3))
        @test occursin("p-value", sprint(show, g2))
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

end
