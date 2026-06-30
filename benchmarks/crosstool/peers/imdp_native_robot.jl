# Native IntervalMDP.jl solve of the shipped multiObj_robotIMDP (PRISM-explicit),
# spec Pmaxmin=?[F reach] (Pessimistic, Maximize). Prints init-state value + timing
# for cross-tool comparison against IMPaCT (tools/imdp_solve on the converted .imdp).
# Run: JULIA_DEPOT_PATH=/opt/jldepot:/root/.julia julia --startup-file=no <this>
using IntervalMDP, IntervalMDP.Data
base = "/opt/jldepot/packages/IntervalMDP/15smo/test/data/multiObj_robotIMDP"
problem = read_prism_file(base)
spec = specification(problem)
println("satisfaction=", satisfaction_mode(spec), " strategy=", strategy_mode(spec))
sol = solve(problem)                                 # warm up (JIT compile)
t = @elapsed (sol2 = solve(problem))
V = value_function(sol)
println("INIT_VALUE=", V[1], " iters=", num_iterations(sol), " residual=", residual(sol)[1])
println("TIME_S=", t)
