#!/usr/bin/env julia
# benchmark/benchmarks.jl — BenchmarkTools suite for Siena.jl's hot loops.
#
# Covers the ministep machinery that the estimation loop hammers: the
# per-candidate objective evaluation (tuple-backed `ObjectiveEffectSet`
# fold, allocation-free), single network ministeps, and a full
# `simulate_saom` trajectory (two periods, rate-cached ministep loop).
#
# Defines the standard `SUITE::BenchmarkGroup`. Run standalone with
#     julia --project=benchmark benchmark/benchmarks.jl
# which tunes + runs the suite and prints one tab-separated `BENCHJL` line
# per benchmark (consumed by the site repo's tools/run_benchmarks.jl).

using BenchmarkTools
using Random
using Siena
using Siena: execute_network_ministep!

# ---------------------------------------------------------------------------
# Fixture: n = 60 actors, two waves, outdegree + recip + transTrip objective
# ---------------------------------------------------------------------------

const N_ACTORS = 60

function make_data(n::Int)
    rng = Random.Xoshiro(20260712)
    data = siena_data()
    add_nodeset!(data, NodeSet(n))
    net1 = [Int(rand(rng) < 0.08 && i != j) for i in 1:n, j in 1:n]
    net2 = copy(net1)
    for _ in 1:(5 * n)                       # ~5 toggles per actor between waves
        i, j = rand(rng, 1:n), rand(rng, 1:n)
        i != j && (net2[i, j] = 1 - net2[i, j])
    end
    add_dependent!(data, DependentNetwork(:net, [net1, net2]))
    return data
end

const DATA = make_data(N_ACTORS)
const EFFECTS = let e = get_effects(DATA)
    include_effects!(e, :net, [:outdegree, :recip, :transTrip])
    e
end
const OSET = build_objective_set(EFFECTS)
const PM = build_param_map(EFFECTS)
const THETA = [3.0, -1.5, 1.0, 0.4]          # rate, outdegree, recip, transTrip
const THETA_OBJ = objective_theta(PM, THETA)

const STATE = let s = NetworkState()
    initialize!(s, DATA, 1)
    s
end

"Objective evaluation for every candidate alter of one actor (one ministep's work)."
function objective_sweep(oset, θ_obj, state, data, actor, n)
    s = 0.0
    for alter in 1:n
        alter == actor && continue
        s += compute_objective(oset, θ_obj, state, data, actor, alter, :net)
    end
    return s
end

# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

const SUITE = BenchmarkGroup()

let g = addgroup!(SUITE, "ministep")
    g["objective_sweep_n$(N_ACTORS)"] =
        @benchmarkable objective_sweep($OSET, $THETA_OBJ, $STATE, $DATA, 7, $N_ACTORS)
    # A full network ministep: choice probabilities + sampling + toggle.
    # Uses its own state/rng copies so repeated evaluations stay comparable.
    g["network_ministep_n$(N_ACTORS)"] =
        @benchmarkable execute_network_ministep!(state, $OSET, $THETA_OBJ, $DATA,
                                                 7, :net, rng) setup =
            (state = let s = NetworkState(); initialize!(s, DATA, 1); s end;
             rng = Random.Xoshiro(42))
end

let g = addgroup!(SUITE, "simulation")
    g["simulate_saom_n$(N_ACTORS)"] =
        @benchmarkable simulate_saom($DATA, $EFFECTS, $THETA; rng = rng) setup =
            (rng = Random.Xoshiro(42))
end

# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

function print_benchjl(results::BenchmarkGroup)
    for (path, trial) in BenchmarkTools.leaves(results)
        est = median(trial)
        println("BENCHJL\t", join(path, "/"), "\t",
                BenchmarkTools.time(est), "\t",
                BenchmarkTools.allocs(est), "\t",
                BenchmarkTools.memory(est))
    end
end

function main()
    tune!(SUITE)
    results = run(SUITE; verbose=false, seconds=1)
    print_benchjl(results)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
