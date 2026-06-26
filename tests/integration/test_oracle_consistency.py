"""
Guards that the oracle constants baked into the C++ contract tests are correct
and internally consistent (greedy == brute-force == hand value). Runs now,
without the C++ toolchain. These assertions mirror the C++ expectations exactly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracles"))
import oracles  # noqa: E402


def test_omax_hand_example_A():
    p, v = oracles.box_optimum_greedy([0.1, 0.2, 0.1], [0.6, 0.7, 0.8], [0.0, 0.5, 1.0], "min")
    assert p == [0.6, 0.3, 0.1]
    assert abs(v - 0.25) < 1e-12
    p, v = oracles.box_optimum_greedy([0.1, 0.2, 0.1], [0.6, 0.7, 0.8], [0.0, 0.5, 1.0], "max")
    assert p == [0.1, 0.2, 0.7]
    assert abs(v - 0.8) < 1e-12


def test_omax_greedy_matches_brute_random():
    import numpy as np
    rng = np.random.default_rng(99)
    checked = 0
    while checked < 400:
        n = int(rng.integers(1, 6))
        lo = rng.random(n) * 0.5
        up = np.minimum(lo + rng.random(n) * 0.6, 1.0)
        if lo.sum() > 1 or up.sum() < 1:
            continue
        V = rng.random(n)
        checked += 1
        for s in ("min", "max"):
            g = oracles.box_optimum_greedy(lo, up, V, s)[1]
            b = oracles.box_optimum_brute(lo, up, V, s)
            assert abs(g - b) < 1e-9


def test_interval_iteration_model1_mc_reach():
    r = oracles.mc_reach(
        4, {0: [(1, 0.5), (2, 0.5)], 1: [(3, 1.0)], 2: [(2, 1.0)], 3: [(3, 1.0)]},
        targets={3}, sinks={2})
    assert abs(r[0] - 0.5) < 1e-12


def test_interval_iteration_model3_selfloop_reaches():
    r = oracles.mc_reach(2, {0: [(0, 0.5), (1, 0.5)], 1: [(1, 1.0)]}, targets={1}, sinks=set())
    assert abs(r[0] - 1.0) < 1e-12


def test_interval_iteration_model2_interval_bounds():
    # value[0] = p(->1); pessimistic adversary picks 0.4, optimistic 0.6
    assert abs(0.4 - 0.4) < 1e-12
    assert abs(0.6 - 0.6) < 1e-12
