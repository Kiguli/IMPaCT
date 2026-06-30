#!/usr/bin/env python3
# Cross-tool SOLVE-phase sweep over shared .imdp models. Runs IMPaCT (imdp_solve),
# IntervalMDP.jl (intervalmdp_runner.jl), and PRISM (explicit interval import) on the
# SAME interval-MDP and prints a TSV. Storm is run separately on the host (.drn).
#
# Semantics (robust = pessimistic = nature adversarial; opt = cooperative):
#   reach  pess/opt = Pmaxmin / Pmaxmax [ F  label ]
#   safety pess/opt = Pmaxmin / Pmaxmax [ !(F label) ]   (= 1 - reach-to-avoid)
#   omega (buchi/persist/patrol): IMPaCT only; peers N/A.
#
# Usage: sweep_compare.py ref|arch|all   > results.tsv
import subprocess, sys, time, re, os

ROOT = "/work/benchmarks/crosstool"
IMPACT = "/tmp/imdp_solve"
RUNNER = ROOT + "/peers/intervalmdp_runner.jl"
PCONV  = ROOT + "/peers/imdp_to_prism_interval.py"
JLENV  = dict(os.environ, JULIA_DEPOT_PATH="/opt/jldepot:/root/.julia")

# (name, imdp, group, prop, label, init)
REF = [
    ("chain_point",       "models/chain_point.imdp",       "ref", "reach",  "target", 0),
    ("safety_point",      "models/safety_point.imdp",      "ref", "safety", "avoid",  0),
    ("choice_interval",   "models/choice_interval.imdp",   "ref", "reach",  "target", 0),
    ("fork_interval",     "models/fork_interval.imdp",     "ref", "reach",  "target", 0),
    ("multiObj_robotIMDP","models/multiObj_robotIMDP.imdp","ref", "reach",  "reach",  0),
    ("buchi_reach",       "models/buchi_reach.imdp",       "ref", "buchi",  "acc",    2),
    ("buchi_routearound", "models/buchi_routearound.imdp", "ref", "buchi",  "acc",    0),
    ("persist_leak",      "models/persist_leak.imdp",      "ref", "persist","safe",   0),
    ("patrol_cycle",      "models/patrol_cycle.imdp",      "ref", "patrol", "r0,r2",  0),
]
ARCH = [
    ("AS",       "arch/models/AS.imdp",       "arch", "reach",  "target", 0),
    ("BA",       "arch/models/BA.imdp",       "arch", "safety", "avoid",  0),
    ("IC_reach", "arch/models/IC_reach.imdp", "arch", "reach",  "target", 0),
    ("IC_safe",  "arch/models/IC_safe.imdp",  "arch", "safety", "avoid",  0),
    ("PD_p1",    "arch/models/PD_p1.imdp",    "arch", "reach",  "target", 0),
    ("PD_p3",    "arch/models/PD_p3.imdp",    "arch", "reach",  "target", 0),
    ("VP",       "arch/models/VP.imdp",       "arch", "reach",  "target", 0),
]

def run(cmd, env=None, timeout=180):
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        return p.stdout + p.stderr, time.monotonic() - t0
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__", time.monotonic() - t0

def emit(tool, name, group, prop, sense, value, t, status):
    v = "%.10g" % value if isinstance(value, float) else value
    print("\t".join([tool, name, group, prop, sense, str(v), "%.3f" % t if t is not None else "-", status]), flush=True)

def impact(name, imdp, group, prop, label, init):
    eps = ["--eps", "1e-4"] if group == "arch" else []   # large abstracted IMDPs: 1e-4 (ISSUE-0010)
    if prop in ("reach", "safety"):
        out, t = run([IMPACT, ROOT+"/"+imdp, prop, label, "--bound", "both", "--state", str(init)] + eps)
        for sense in ("pess", "opt"):
            m = re.search(r"bound=%s\s+state=\d+\s+lower=([\d.eE+-]+)\s+upper=([\d.eE+-]+)" % sense, out)
            if m:
                emit("IMPaCT", name, group, prop, sense, float(m.group(1)), t/2, "ok")
            else:
                to = (out == "__TIMEOUT__")
                emit("IMPaCT", name, group, prop, sense, "TIMEOUT" if to else "ERR", t, "timeout" if to else "err")
    else:  # omega: buchi / persist / patrol
        out, t = run([IMPACT, ROOT+"/"+imdp, prop, label, "--bound", "both", "--state", str(init)])
        for sense in ("pess", "opt"):
            m = re.search(r"bound=%s\s+state=\d+\s+lower=([\d.eE+-]+)\s+upper=([\d.eE+-]+)" % sense, out)
            if m:
                emit("IMPaCT", name, group, prop, sense, float(m.group(1)), t/2, "ok")
            else:
                to = (out == "__TIMEOUT__")
                emit("IMPaCT", name, group, prop, sense, "TIMEOUT" if to else "ERR", t, "timeout" if to else "err")

def intervalmdp(name, imdp, group, prop, label, init):
    if prop not in ("reach", "safety"):
        for sense in ("pess", "opt"):
            emit("IntervalMDP.jl", name, group, prop, sense, "N/A", None, "unsupported")
        return
    eps = ["--eps", "1e-4"] if group == "arch" else []
    out, t = run(["julia", "--startup-file=no", RUNNER, ROOT+"/"+imdp, prop, "--label", label, "--state", str(init)] + eps, env=JLENV)
    for sense in ("pess", "opt"):
        m = re.search(r"bound=%s\s+state=\d+\s+value=([\d.eE+-]+)\s+iters=\d+\s+seconds=([\d.eE+-]+)" % sense, out)
        if m:
            emit("IntervalMDP.jl", name, group, prop, sense, float(m.group(1)), float(m.group(2)), "ok")
        else:
            to = (out == "__TIMEOUT__")
            emit("IntervalMDP.jl", name, group, prop, sense, "TIMEOUT" if to else "ERR", t, "timeout" if to else "err")

def prism(name, imdp, group, prop, label, init):
    if prop not in ("reach", "safety"):
        for sense in ("pess", "opt"):
            emit("PRISM", name, group, prop, sense, "N/A", None, "unsupported")
        return
    base = "/tmp/pex_" + name
    # PRISM rejects [0,hi] interval edges -> floor exact-zero lowers to 1e-9 (perturbation,
    # mark the resulting ARCH values with an asterisk).
    subprocess.run(["python3", PCONV, ROOT+"/"+imdp, base, "--floor", "1e-9"], check=True)
    inner = 'F "%s"' % label if prop == "reach" else '!(F "%s")' % label
    for sense, op in (("pess", "Pmaxmin"), ("opt", "Pmaxmax")):
        pf = "%s=?[ %s ]" % (op, inner)
        out, t = run(["prism", "-importmodel", base+".tra,sta,lab", "-pf", pf, "-fixdl"])
        m = re.search(r"Result:\s*([\d.eE+-]+)", out)
        tc = re.search(r"Time for model checking:\s*([\d.eE+-]+)", out)
        if m:
            emit("PRISM", name, group, prop, sense, float(m.group(1)), float(tc.group(1)) if tc else t, "ok")
        else:
            bad = "unsupported" if re.search(r"not implemented", out) else "err"
            emit("PRISM", name, group, prop, sense, "N/A" if bad=="unsupported" else "ERR", None if bad=="unsupported" else t, bad)

def storm(name, imdp, group, prop, label, init):
    # native storm (built in-container); reads the .drn sibling of the .imdp.
    # models/*.drn are gitignored (regenerable) -> build it from the .imdp if missing.
    drn = ROOT + "/" + imdp[:-5] + ".drn"
    if not os.path.exists(drn):
        subprocess.run(["python3", ROOT + "/peers/imdp_to_drn.py", ROOT + "/" + imdp, drn], check=True)
    if prop not in ("reach", "safety"):
        for sense in ("pess", "opt"):
            emit("Storm", name, group, prop, sense, "N/A", None, "unsupported")
        return
    op = "Pmax" if prop == "reach" else "Pmin"      # safety = 1 - min-reach-to-avoid
    pf = '%s=? [F "%s"]' % (op, label)
    for sense, mode in (("pess", "robust"), ("opt", "cooperative")):
        out, t = run(["storm", "--explicit-drn", drn, "--uncertainty-resolution", mode, "--prop", pf])
        m = re.search(r"Result \(for initial states\):\s*([\d.eE+-]+)", out)
        tc = re.search(r"Time for model checking:\s*([\d.eE+-]+)", out)
        if m:
            val = float(m.group(1))
            if prop == "safety":
                val = 1.0 - val
            emit("Storm", name, group, prop, sense, val, float(tc.group(1)) if tc else t, "ok")
        else:
            bad = "unsupported" if re.search(r"not.*implement", out, re.I) else "err"
            emit("Storm", name, group, prop, sense, "N/A" if bad=="unsupported" else "ERR", None if bad=="unsupported" else t, bad)

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    models = (REF if which in ("ref","all") else []) + (ARCH if which in ("arch","all") else [])
    print("tool\tmodel\tgroup\tprop\tsense\tvalue\ttime_s\tstatus")
    for (name, imdp, group, prop, label, init) in models:
        impact(name, imdp, group, prop, label, init)
        prism(name, imdp, group, prop, label, init)
        storm(name, imdp, group, prop, label, init)
        intervalmdp(name, imdp, group, prop, label, init)

if __name__ == "__main__":
    main()
