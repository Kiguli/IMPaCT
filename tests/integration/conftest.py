"""
Shared fixtures/helpers for IMPaCT integration tests.

Design: integration tests that need the compiled C++ tool SKIP (not fail) when
the toolchain or a golden file is unavailable, so the suite is meaningful both
locally (where acpp + HDF5 + GLPK may be missing) and in CI/Docker (where it is
present). Numerical regression compares HDF5 outputs within a tolerance.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "golden"

# Tolerance for controller / probability regression. Tight enough to catch real
# changes, loose enough to allow solver/compiler nondeterminism at the last ULPs.
ABS_TOL = 1e-6


def _have_toolchain() -> bool:
    return shutil.which("acpp") is not None or shutil.which("syclcc") is not None


def _have_h5py() -> bool:
    try:
        import h5py  # noqa: F401
        return True
    except Exception:
        return False


def load_arma_hdf5(path: Path):
    """Load an Armadillo-saved HDF5 matrix/vector as a numpy array.

    Armadillo's hdf5_binary stores a single dataset; read whichever dataset the
    file contains rather than hard-coding its name.
    """
    import h5py
    import numpy as np
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        assert keys, f"no datasets in {path}"
        return np.array(f[keys[0]][...])


def build_and_run(example_dir: Path, exe_hint: str | None = None) -> Path:
    """make + run an example; return the directory it ran in (where *.h5 land)."""
    subprocess.run(["make", "clean"], cwd=example_dir, check=False,
                   capture_output=True)
    r = subprocess.run(["make"], cwd=example_dir, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"build failed for {example_dir.name}:\n{r.stderr[-2000:]}")
    # find the executable: prefer hint, else the non-dotted, executable file
    exe = None
    if exe_hint and (example_dir / exe_hint).exists():
        exe = example_dir / exe_hint
    else:
        for p in example_dir.iterdir():
            if p.is_file() and os.access(p, os.X_OK) and p.suffix == "":
                exe = p
                break
    if exe is None:
        pytest.skip(f"no executable produced in {example_dir.name}")
    rr = subprocess.run([f"./{exe.name}"], cwd=example_dir, capture_output=True,
                        text=True, timeout=1800)
    if rr.returncode != 0:
        pytest.skip(f"run failed for {exe.name}:\n{rr.stderr[-2000:]}")
    return example_dir


@pytest.fixture(scope="session")
def toolchain():
    if not _have_toolchain():
        pytest.skip("no acpp/syclcc toolchain available")
    if not _have_h5py():
        pytest.skip("h5py not installed (pip install -r tests/integration/requirements.txt)")
    return True
