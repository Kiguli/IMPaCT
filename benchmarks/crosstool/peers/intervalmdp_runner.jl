# ============================================================================
# IntervalMDP.jl runner for the neutral .imdp exchange format (src/imdp_io.h).
#
# Parses an .imdp model, builds an IntervalMarkovDecisionProcess, and solves a
# robust reachability/safety property with RobustValueIteration in BOTH senses:
#   pess (Pessimistic + Maximize) = nature adversarial  = IMPaCT lower bound
#   opt  (Optimistic  + Maximize) = nature cooperative   = IMPaCT upper bound
#
# Prints one machine-parseable line per sense:
#   result tool=intervalmdp prop=reach bound=pess state=0 value=0.4 iters=2 seconds=0.001
#
# Usage:
#   julia intervalmdp_runner.jl MODEL.imdp PROP [--label NAME] [--horizon H]
#                                              [--eps E] [--state S]
#   PROP : reach | safety        (reach uses label "target", safety label "avoid")
# ============================================================================
using IntervalMDP, SparseArrays

function parse_imdp(path::String)
    nstates = 0
    init = 0
    labels = Dict{String, Vector{Int}}()
    rewards = Dict{Int, Float64}()
    # actions[s] = vector of action distributions; each dist = Vector{Tuple{to,lo,hi}}
    actions = Dict{Int, Vector{Vector{Tuple{Int,Float64,Float64}}}}()
    for raw in eachline(path)
        line = strip(replace(raw, '\r' => ""))
        (isempty(line) || startswith(line, "#")) && continue
        tok = split(line)
        kw = tok[1]
        if kw == "states"
            nstates = parse(Int, tok[2])
        elseif kw == "init"
            init = parse(Int, tok[2])
        elseif kw == "label"
            name = tok[2]
            sset = [parse(Int, t) for t in tok[3:end]]
            labels[name] = get(labels, name, Int[])
            append!(labels[name], sset)
        elseif kw == "reward"
            rewards[parse(Int, tok[2])] = parse(Float64, tok[3])
        elseif kw == "tran"
            s = parse(Int, tok[2])
            dist = Tuple{Int,Float64,Float64}[]
            for t in tok[4:end]
                parts = split(t, ':')
                to = parse(Int, parts[1]); lo = parse(Float64, parts[2]); hi = parse(Float64, parts[3])
                push!(dist, (to, lo, hi))
            end
            actions[s] = get(actions, s, Vector{Vector{Tuple{Int,Float64,Float64}}}())
            push!(actions[s], dist)
        end
    end
    rvec = zeros(Float64, nstates)
    for (s, r) in rewards; rvec[s+1] = r; end
    return nstates, init, labels, actions, rvec
end

function build_mdp(nstates, init, actions)
    # Columns = action distributions grouped by source state in increasing order.
    Ilow=Int[]; Jlow=Int[]; Vlow=Float64[]
    Iup=Int[];  Jup=Int[];  Vup=Float64[]
    stateptr = Int32[1]
    col = 0
    for s in 0:(nstates-1)
        acts = get(actions, s, nothing)
        if acts === nothing || isempty(acts)
            # state with no outgoing transitions -> absorbing self loop
            col += 1
            push!(Ilow, s+1); push!(Jlow, col); push!(Vlow, 1.0)
            push!(Iup,  s+1); push!(Jup,  col); push!(Vup,  1.0)
        else
            for dist in acts
                col += 1
                for (to, lo, hi) in dist
                    push!(Ilow, to+1); push!(Jlow, col); push!(Vlow, lo)
                    push!(Iup,  to+1); push!(Jup,  col); push!(Vup,  hi)
                end
            end
        end
        push!(stateptr, col+1)
    end
    lower = sparse(Ilow, Jlow, Vlow, nstates, col)
    upper = sparse(Iup,  Jup,  Vup,  nstates, col)
    tp = IntervalProbabilities(; lower=lower, upper=upper)
    return IntervalMarkovDecisionProcess(tp, stateptr, [Int32(init+1)])
end

function main()
    path = ARGS[1]
    prop = ARGS[2]
    label = prop == "reach" ? "target" : "avoid"
    horizon = 0
    eps = 1e-6
    state = -1
    dumpdir = ""
    discount = 0.9
    i = 3
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--label"; label = ARGS[i+1]; i += 2
        elseif a == "--horizon"; horizon = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--eps"; eps = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--discount"; discount = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--state"; state = parse(Int, ARGS[i+1]); i += 2
        elseif a == "--dumpdir"; dumpdir = ARGS[i+1]; i += 2
        else; i += 1; end
    end

    nstates, init, labels, actions, rvec = parse_imdp(path)
    s = state >= 0 ? state : init
    mdp = build_mdp(nstates, init, actions)
    tgt = [x+1 for x in get(labels, label, Int[])]

    for (sat, name) in ((Pessimistic, "pess"), (Optimistic, "opt"))
        if prop == "reward"
            pr = InfiniteTimeReward(rvec, discount, eps)   # discounted expected reward
        elseif prop == "reach"
            pr = horizon > 0 ? FiniteTimeReachability(tgt, horizon) : InfiniteTimeReachability(tgt, eps)
        else
            pr = horizon > 0 ? FiniteTimeSafety(tgt, horizon) : InfiniteTimeSafety(tgt, eps)
        end
        spec = Specification(pr, sat, Maximize)
        problem = VerificationProblem(mdp, spec)
        t0 = time()
        sol = solve(problem)
        dt = time() - t0
        V = sol.value_function
        println("result\ttool=intervalmdp\tprop=$(prop)\tbound=$(name)\tstate=$(s)\t",
                "value=$(V[s+1])\titers=$(sol.num_iterations)\tseconds=$(round(dt, digits=4))")
        if !isempty(dumpdir)
            open(joinpath(dumpdir, "intervalmdp_$(name).txt"), "w") do io
                for v in V; println(io, v); end
            end
        end
    end
end

main()
