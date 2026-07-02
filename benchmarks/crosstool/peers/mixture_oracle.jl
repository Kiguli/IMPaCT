# IntervalMDP.jl oracle for mix_demo.odimdp: the SAME 2x2 mixture-of-orthogonal IMDP,
# solved as MixtureIntervalMarkovDecisionProcess (infinite-horizon reachability to (2,2),
# Pessimistic/Optimistic x Maximize). All states carry K=2 components in IntervalMDP.jl;
# for states 1..3 both components are identical (weights then don't matter).
# Run: JULIA_DEPOT_PATH=/opt/jldepot:/root/.julia julia --startup-file=no <this>
using IntervalMDP

# component 1 marginals (cols = source states s1..s4)
l1c1 = [0.3 0.0 1.0 0.0;  0.3 0.8 0.0 1.0]
u1c1 = [0.7 0.2 1.0 0.0;  0.7 1.0 0.0 1.0]
l2c1 = [0.4 0.5 0.0 0.0;  0.4 0.5 1.0 1.0]
u2c1 = [0.6 0.5 0.0 0.0;  0.6 0.5 1.0 1.0]
# component 2: differs only in column 1 (state (1,1))
l1c2 = [0.0 0.0 1.0 0.0;  0.8 0.8 0.0 1.0]
u1c2 = [0.2 0.2 1.0 0.0;  1.0 1.0 0.0 1.0]
l2c2 = [0.1 0.5 0.0 0.0;  0.7 0.5 1.0 1.0]
u2c2 = [0.3 0.5 0.0 0.0;  0.9 0.5 1.0 1.0]
# interval mixture weights (rows = components, cols = source states)
wl = [0.3 0.5 0.5 0.5;  0.2 0.5 0.5 0.5]
wu = [0.8 0.5 0.5 0.5;  0.7 0.5 0.5 0.5]

prob1 = OrthogonalIntervalProbabilities(
    (IntervalProbabilities(; lower = l1c1, upper = u1c1),
     IntervalProbabilities(; lower = l2c1, upper = u2c1)), (Int32(2), Int32(2)))
prob2 = OrthogonalIntervalProbabilities(
    (IntervalProbabilities(; lower = l1c2, upper = u1c2),
     IntervalProbabilities(; lower = l2c2, upper = u2c2)), (Int32(2), Int32(2)))
wp = IntervalProbabilities(; lower = wl, upper = wu)
mix = MixtureIntervalProbabilities((prob1, prob2), wp)
stateptr = Int32[1, 2, 3, 4, 5]
mdp = MixtureIntervalMarkovDecisionProcess(mix, stateptr, [CartesianIndex(1, 1)])

for (sat, name) in ((Pessimistic, "pess"), (Optimistic, "opt"))
    prop = InfiniteTimeReachability([CartesianIndex(2, 2)], 1e-9)
    spec = Specification(prop, sat, Maximize)
    sol  = solve(VerificationProblem(mdp, spec))
    V = value_function(sol)
    println("oracle\tbound=", name, "\tV(1,1)=", V[1, 1], "\tV(2,1)=", V[2, 1],
            "\titers=", num_iterations(sol))
end
