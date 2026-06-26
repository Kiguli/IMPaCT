#!/usr/bin/env python3
"""
ARCH-COMP Stochastic-Models benchmark runner for IMPaCT v2.0.

Builds and runs the examples under examples/ARCH-COMP/2025/, timing each, and
records a row per case. Skips cleanly when the acpp toolchain is unavailable
(so it is safe to run anywhere; real numbers come from Docker/CI or a configured
machine). Results are meant to be pasted into RESULTS.md / paper §8.

Usage:
  python3 benchmarks/run_archcomp.py [--only NAME] [--tag baseline]
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCH = REPO / "examples" / "ARCH-COMP" / "2025"


def have_toolchain() -> bool:
    return shutil.which("acpp") is not None or shutil.which("syclcc") is not None


def run_case(case_dir: Path) -> dict:
    name = case_dir.name
    rec = {"case": name, "status": "skip", "seconds": None, "outputs": 0, "note": ""}
    subprocess.run(["make", "clean"], cwd=case_dir, capture_output=True)
    b = subprocess.run(["make"], cwd=case_dir, capture_output=True, text=True)
    if b.returncode != 0:
        rec["status"] = "build-fail"
        rec["note"] = b.stderr.strip().splitlines()[-1] if b.stderr.strip() else "build error"
        return rec
    exes = [p for p in case_dir.iterdir()
            if p.is_file() and p.suffix == "" and p.stat().st_mode & 0o111]
    if not exes:
        rec["status"] = "no-exe"
        return rec
    t0 = time.perf_counter()
    ok = True
    for exe in exes:
        r = subprocess.run([f"./{exe.name}"], cwd=case_dir, capture_output=True,
                           text=True, timeout=3600)
        ok = ok and (r.returncode == 0)
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    rec["outputs"] = len(list(case_dir.glob("*.h5")))
    rec["status"] = "ok" if ok and rec["outputs"] > 0 else "run-fail"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single case by dir name")
    ap.add_argument("--tag", default="baseline", help="label for this run")
    args = ap.parse_args()

    if not have_toolchain():
        print("SKIP: no acpp/syclcc toolchain found — run in Docker or a configured host.")
        return
    if not ARCH.exists():
        print(f"SKIP: {ARCH} not found.")
        return

    cases = sorted(d for d in ARCH.iterdir() if d.is_dir())
    if args.only:
        cases = [d for d in cases if d.name == args.only]

    print(f"# ARCH-COMP run ({args.tag})")
    print(f"{'case':16} {'status':12} {'secs':>8} {'h5':>4}  note")
    for d in cases:
        rec = run_case(d)
        print(f"{rec['case']:16} {rec['status']:12} "
              f"{str(rec['seconds']):>8} {rec['outputs']:>4}  {rec['note']}")


if __name__ == "__main__":
    main()
