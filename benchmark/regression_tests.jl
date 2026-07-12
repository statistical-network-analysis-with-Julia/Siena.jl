#!/usr/bin/env julia
# benchmark/regression_tests.jl — allocation-regression assertions for the
# Siena.jl ministep hot loop. Standalone; run with
#     julia --project=benchmark benchmark/regression_tests.jl
#
# The per-candidate objective evaluation (`compute_objective` over the
# tuple-backed `ObjectiveEffectSet`) runs once per candidate alter per
# ministep and is allocation-free as of the P0–P2 optimization sprints
# (measured 0 bytes). A full network ministep allocates only its choice
# probability/alternative vectors (~2.9 KB measured at n = 60); the bound
# below is 2x that measurement, so it catches regressions, not noise.

using Random
using Siena
using Siena: execute_network_ministep!
using Test

function make_fixture(n::Int)
    rng = Random.Xoshiro(20260712)
    data = siena_data()
    add_nodeset!(data, NodeSet(n))
    net1 = [Int(rand(rng) < 0.08 && i != j) for i in 1:n, j in 1:n]
    net2 = copy(net1)
    for _ in 1:(5 * n)
        i, j = rand(rng, 1:n), rand(rng, 1:n)
        i != j && (net2[i, j] = 1 - net2[i, j])
    end
    add_dependent!(data, DependentNetwork(:net, [net1, net2]))
    effects = get_effects(data)
    include_effects!(effects, :net, [:outdegree, :recip, :transTrip])
    oset = build_objective_set(effects)
    pm = build_param_map(effects)
    θ = [3.0, -1.5, 1.0, 0.4]
    θ_obj = objective_theta(pm, θ)
    state = NetworkState()
    initialize!(state, data, 1)
    return data, oset, θ_obj, state
end

@testset "Siena allocation regressions" begin
    n = 60
    data, oset, θ_obj, state = make_fixture(n)

    @testset "compute_objective is allocation-free" begin
        # warm up (compile) every dispatch path first
        compute_objective(oset, θ_obj, state, data, 1, 2, :net)
        worst = 0
        for actor in 1:20, alter in 1:20
            actor == alter && continue
            worst = max(worst, @allocated compute_objective(oset, θ_obj, state,
                                                            data, actor, alter,
                                                            :net))
        end
        @test worst == 0
    end

    @testset "network ministep stays within its vector budget" begin
        rng = Random.Xoshiro(42)
        execute_network_ministep!(state, oset, θ_obj, data, 3, :net, rng)
        worst = 0
        for actor in 4:14
            worst = max(worst, @allocated execute_network_ministep!(state, oset,
                                                                    θ_obj, data,
                                                                    actor, :net,
                                                                    rng))
        end
        # 2x the measured ~2880 bytes (choice probs/alters vectors at n = 60)
        @test worst <= 5760
    end
end
