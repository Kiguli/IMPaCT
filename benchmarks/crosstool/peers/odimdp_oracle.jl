# IntervalMDP.jl oracle for the orthogonal IMDP od_demo.odimdp: build the SAME 2x2
# factored model with OrthogonalIntervalProbabilities and solve infinite-horizon
# reachability to (2,2), Pessimistic and Optimistic (Maximize). Columns are the
# (state,action) pairs in linearised order s = i + 2(j-1) (dim 1 fastest), one action each.
# Run: JULIA_DEPOT_PATH=/opt/jldepot:/root/.julia julia --startup-file=no <this>
using IntervalMDP

# dim-1 marginals: rows = destination i' in {1,2}; cols = source states s1..s4
# (s3 = (1,2) is DEAD absorbing: dim1 stays at 1, dim2 stays at 2)
l1 = [0.3 0.0 1.0 0.0;
      0.3 0.8 0.0 1.0]
u1 = [0.7 0.2 1.0 0.0;
      0.7 1.0 0.0 1.0]
# dim-2 marginals
l2 = [0.4 0.5 0.0 0.0;
      0.4 0.5 1.0 1.0]
u2 = [0.6 0.5 0.0 0.0;
      0.6 0.5 1.0 1.0]

prob = OrthogonalIntervalProbabilities(
    (IntervalProbabilities(; lower = l1, upper = u1),
     IntervalProbabilities(; lower = l2, upper = u2)),
    (Int32(2), Int32(2)),
)
stateptr = Int32[1, 2, 3, 4, 5]          # one action per state
mdp = OrthogonalIntervalMarkovDecisionProcess(prob, stateptr, [CartesianIndex(1, 1)])

for (sat, name) in ((Pessimistic, "pess"), (Optimistic, "opt"))
    prop = InfiniteTimeReachability([CartesianIndex(2, 2)], 1e-9)
    spec = Specification(prop, sat, Maximize)
    sol  = solve(VerificationProblem(mdp, spec))
    V = value_function(sol)
    println("oracle\tbound=", name, "\tV(1,1)=", V[1, 1], "\tV(2,1)=", V[2, 1],
            "\tV(1,2)=", V[1, 2], "\titers=", num_iterations(sol))
end
