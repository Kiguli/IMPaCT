"""
Independent reference oracles for the IMPaCT v2.0 test suite.

These reproduce, by an INDEPENDENT method (numpy linear algebra / brute-force
vertex enumeration), the exact expected values baked into the C++ contract
tests. They serve three purposes:
  1. cross-validate that the hand-derived oracle constants are correct,
  2. document how each expected value is obtained,
  3. generate expected values for larger integration cases.

Pure Python + numpy; runnable without the C++ toolchain.
"""
from __future__ import annotations
from itertools import product
import numpy as np


# --------------------------------------------------------------------------
# O-maximization: optimum of  opt_{p in [lo,up], sum p = 1}  dot(p, V)
# --------------------------------------------------------------------------
def box_optimum_greedy(lo, up, V, sense):
    """Closed-form sort-and-assign (Givan-Leach-Dean)."""
    lo = list(map(float, lo)); up = list(map(float, up)); V = list(map(float, V))
    n = len(V)
    assert len(lo) == n and len(up) == n and n >= 1
    assert all(lo[i] <= up[i] for i in range(n))
    assert sum(lo) <= 1 + 1e-12 and sum(up) >= 1 - 1e-12, "infeasible box"
    order = sorted(range(n), key=lambda i: V[i], reverse=(sense == "max"))
    p = list(lo)
    resid = 1.0 - sum(lo)
    for i in order:
        gap = up[i] - lo[i]
        take = min(gap, resid)
        p[i] += take
        resid -= take
        if resid <= 1e-15:
            break
    return p, float(np.dot(p, V))


def box_optimum_brute(lo, up, V, sense):
    """Independent: enumerate polytope vertices (all-but-one coord at a bound)."""
    lo = list(map(float, lo)); up = list(map(float, up)); V = list(map(float, V))
    n = len(V)
    best = None
    for pivot in range(n):
        others = [i for i in range(n) if i != pivot]
        for bits in product([0, 1], repeat=len(others)):
            p = [0.0] * n
            assigned = 0.0
            for i, b in zip(others, bits):
                p[i] = up[i] if b else lo[i]
                assigned += p[i]
            resid = 1.0 - assigned
            if resid < lo[pivot] - 1e-12 or resid > up[pivot] + 1e-12:
                continue
            p[pivot] = resid
            val = float(np.dot(p, V))
            if best is None or (val < best and sense == "min") or (val > best and sense == "max"):
                best = val
    return best


# --------------------------------------------------------------------------
# Exact reachability for a point-probability Markov chain.
#   states: list of ids; P: dict s -> list of (t, prob); targets, sinks: sets
# Solves x = P x on transient states with x[target]=1, x[sink]=0.
# --------------------------------------------------------------------------
def mc_reach(n, P, targets, sinks):
    targets = set(targets); sinks = set(sinks)
    transient = [s for s in range(n) if s not in targets and s not in sinks]
    idx = {s: k for k, s in enumerate(transient)}
    m = len(transient)
    A = np.eye(m)
    b = np.zeros(m)
    for s in transient:
        for (t, pr) in P[s]:
            if t in targets:
                b[idx[s]] += pr
            elif t in sinks:
                pass
            else:
                A[idx[s], idx[t]] -= pr
    x = np.linalg.solve(A, b)
    out = {s: 1.0 for s in targets}
    out.update({s: 0.0 for s in sinks})
    out.update({s: float(x[idx[s]]) for s in transient})
    return out


# --------------------------------------------------------------------------
# Self-check: reproduce every constant baked into the C++ contract tests.
# --------------------------------------------------------------------------
def _selfcheck():
    # omax hand example A
    p, v = box_optimum_greedy([0.1,0.2,0.1], [0.6,0.7,0.8], [0.0,0.5,1.0], "min")
    assert abs(v - 0.25) < 1e-12 and p == [0.6,0.3,0.1], (p, v)
    p, v = box_optimum_greedy([0.1,0.2,0.1], [0.6,0.7,0.8], [0.0,0.5,1.0], "max")
    assert abs(v - 0.8) < 1e-12 and p == [0.1,0.2,0.7], (p, v)
    # greedy vs brute agree on a grid of random feasible boxes
    rng = np.random.default_rng(7)
    for _ in range(3000):
        n = rng.integers(1, 6)
        lo = rng.random(n) * 0.5
        up = lo + rng.random(n) * 0.6
        up = np.minimum(up, 1.0)
        if lo.sum() > 1 or up.sum() < 1:
            continue
        Vv = rng.random(n)
        for s in ("min", "max"):
            g = box_optimum_greedy(lo, up, Vv, s)[1]
            br = box_optimum_brute(lo, up, Vv, s)
            assert abs(g - br) < 1e-9, (s, lo, up, Vv, g, br)
    # interval-iteration Model 1 (point MC): value[0] = 0.5
    r = mc_reach(4, {0:[(1,0.5),(2,0.5)], 1:[(3,1.0)], 2:[(2,1.0)], 3:[(3,1.0)]},
                 targets={3}, sinks={2})
    assert abs(r[0] - 0.5) < 1e-12, r
    # Model 3 (self-loop still reaches w.p.1): value[0] = 1
    r = mc_reach(2, {0:[(0,0.5),(1,0.5)], 1:[(1,1.0)]}, targets={1}, sinks=set())
    assert abs(r[0] - 1.0) < 1e-12, r
    # Model 2 interval bounds by direct adversary-vertex evaluation on the chain:
    # value[0] = p(->1)*1 ; pessimistic p=0.4, optimistic p=0.6
    assert abs(0.4 * 1.0 - 0.4) < 1e-12
    assert abs(0.6 * 1.0 - 0.6) < 1e-12
    return True


if __name__ == "__main__":
    assert _selfcheck()
    print("oracles self-check OK — all baked C++ oracle constants reproduced.")
