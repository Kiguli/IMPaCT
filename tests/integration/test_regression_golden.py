"""
Numerical regression: the synthesized controller / probability bounds for the
existing reach-avoid & safety examples must NOT change (within ABS_TOL) as we
refactor the solver in Phase 1 (GLPK LP -> O-maximization, sound interval
iteration). This is the safety net for "same answers, faster + sound".

Workflow:
  * Generate goldens once on the baseline (main branch):
        IMPACT_GEN_GOLDEN=1 pytest tests/integration/test_regression_golden.py
  * Thereafter every run compares against tests/golden/<example>/*.h5.

Skips cleanly if the toolchain/h5py are unavailable or no golden exists yet.
"""
import os
from pathlib import Path

import pytest

from conftest import REPO, GOLDEN, ABS_TOL, build_and_run, load_arma_hdf5

# (example dir relative to repo, executable name, output files to compare)
CASES = [
    ("examples/ex_2Drobot-RA-U", "robot2D", ["controller.h5"]),
    ("examples/ex_4DBAS-S", None, ["controller.h5"]),
    ("examples/ARCH-COMP/2025/VP", None, ["controller.h5"]),
]


@pytest.mark.parametrize("rel,exe,outputs", CASES, ids=[c[0] for c in CASES])
def test_controller_matches_golden(toolchain, rel, exe, outputs):
    import numpy as np

    example_dir = REPO / rel
    if not example_dir.exists():
        pytest.skip(f"missing example {rel}")

    run_dir = build_and_run(example_dir, exe)
    gold_dir = GOLDEN / Path(rel).name

    gen = os.environ.get("IMPACT_GEN_GOLDEN") == "1"
    if gen:
        gold_dir.mkdir(parents=True, exist_ok=True)

    for out in outputs:
        produced = run_dir / out
        assert produced.exists(), f"{out} not produced by {rel}"
        golden = gold_dir / out
        if gen:
            import shutil
            shutil.copy(produced, golden)
            continue
        if not golden.exists():
            pytest.skip(f"no golden for {rel}/{out}; run with IMPACT_GEN_GOLDEN=1 on baseline")
        a = load_arma_hdf5(produced)
        b = load_arma_hdf5(golden)
        assert a.shape == b.shape, f"shape changed for {rel}/{out}: {a.shape} vs {b.shape}"
        max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
        assert max_abs <= ABS_TOL, f"{rel}/{out} drifted by {max_abs} > {ABS_TOL}"
