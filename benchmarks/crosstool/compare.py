#!/usr/bin/env python3
# ============================================================================
# Cross-tool benchmark comparison harness (reference-values mode).
#
# Runs IMPaCT's `imdp_solve` on each shared .imdp model and checks its sound
# robust interval [lower, upper] against INDEPENDENTLY-COMPUTED reference values
# (the .ref.json siblings). Each reference value is what a peer tool
# (IntervalMDP.jl / PRISM / Storm) produces on the SAME model+semantics; the
# value is derived analytically (point MDPs: exact linear solve; interval MDPs:
# closed-form O-maximization / enumeration) and recorded with provenance, so the
# check is valid without the peer tools installed. When the peers are later
# installed, drop their measured values into the same .ref.json `checks` and this
# harness compares them unchanged.
#
# A check PASSES iff:
#   (1) soundness: lower - tol <= ref <= upper + tol   (ref is bracketed), and
#   (2) accuracy : |midpoint - ref| <= tol             (bound is tight).
# for BOTH the pessimistic ref ("pess") and optimistic ref ("opt").
#
# Usage:  python3 benchmarks/crosstool/compare.py [--bin PATH] [--models DIR] [-v]
# Exit code 0 iff every check passes.
# ============================================================================
import argparse, glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_result(line):
    """Parse a 'result\\tk=v\\t...' line from imdp_solve into a dict."""
    out = {}
    for tok in line.strip().split("\t"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def run_solver(binpath, model, prop, label, bound, state, eps):
    cmd = [binpath, model, prop, label, "--bound", bound,
           "--state", str(state), "--eps", str(eps)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"imdp_solve failed ({p.returncode}): {p.stderr.strip()}\n  cmd: {' '.join(cmd)}")
    for line in p.stdout.splitlines():
        if line.startswith("result"):
            return parse_result(line)
    raise RuntimeError(f"no result line from: {' '.join(cmd)}\n  stdout: {p.stdout!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(HERE, "..", "..", "tools", "imdp_solve"))
    ap.add_argument("--models", default=os.path.join(HERE, "models"))
    ap.add_argument("--eps", type=float, default=1e-7)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    binpath = os.path.abspath(args.bin)
    if not os.path.exists(binpath):
        print(f"ERROR: solver binary not found: {binpath}\n"
              f"Build it:\n  c++ -std=c++17 -O2 tools/imdp_solve.cpp "
              f"src/imdp_io.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp "
              f"-o tools/imdp_solve", file=sys.stderr)
        return 2

    refs = sorted(glob.glob(os.path.join(args.models, "*.ref.json")))
    if not refs:
        print(f"ERROR: no .ref.json files under {args.models}", file=sys.stderr)
        return 2

    total = passed = 0
    for ref_path in refs:
        with open(ref_path) as f:
            ref = json.load(f)
        model = os.path.join(args.models, ref["model"])
        print(f"\n=== {ref['model']} — {ref.get('description','')}")
        if args.verbose:
            print(f"    provenance: {ref.get('provenance','')}")
        for chk in ref["checks"]:
            prop, label, state = chk["prop"], chk["label"], chk["state"]
            tol = chk.get("tol", 1e-4)
            res = run_solver(binpath, model, prop, label, "both", state, args.eps)
            # imdp_solve --bound both prints two lines; re-run per-bound to map values.
            got = {}
            for which in ("pess", "opt"):
                r = run_solver(binpath, model, prop, label, which, state, args.eps)
                got[which] = (float(r["lower"]), float(r["upper"]))
            for which in ("pess", "opt"):
                total += 1
                refv = chk[which]
                lo, hi = got[which]
                mid = 0.5 * (lo + hi)
                sound = (lo - tol) <= refv <= (hi + tol)
                tight = abs(mid - refv) <= tol
                ok = sound and tight
                passed += ok
                tag = "ok " if ok else "FAIL"
                detail = f"[{lo:.6f},{hi:.6f}] ref={refv:.6f}"
                if not sound: detail += " !UNSOUND(ref outside bracket)"
                if not tight: detail += " !LOOSE(mid off ref)"
                print(f"    [{tag}] {prop} {label} s{state} {which:>4}: {detail}")

    print(f"\n{'='*60}\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
