# Same model, but solve at successively tighter convergence thresholds to check
# whether the ~1.2e-6 gap to IMPaCT is just IntervalMDP.jl's default 1e-6 stopping.
using IntervalMDP, IntervalMDP.Data
base = "/opt/jldepot/packages/IntervalMDP/15smo/test/data/multiObj_robotIMDP"
problem = read_prism_file(base)
sys = system(problem)
reachset = reach(system_property(specification(problem)))
for thr in (1e-6, 1e-9, 1e-12)
    prop = InfiniteTimeReachability(reachset, thr)
    spec = Specification(prop, Pessimistic, Maximize)
    sol = solve(VerificationProblem(sys, spec))
    V = value_function(sol)
    println("thr=", thr, "  INIT_VALUE=", V[1], "  iters=", num_iterations(sol))
end
