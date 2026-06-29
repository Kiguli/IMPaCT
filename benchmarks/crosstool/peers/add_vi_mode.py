#!/usr/bin/env python3
# Add a selectable pure-VALUE-ITERATION mode to the infinite-horizon synthesis loops,
# alongside the existing (default) sound interval iteration. Peer tools (PRISM / Storm
# / IntervalMDP.jl) use plain value iteration from 0 with RESIDUAL stopping.
#
# Trick that needs NO per-branch controller rewrites: in ValueIteration mode we
# initialise the "upper" iterate (first1/second1, normally fill::ones) to ZEROS, so
# BOTH the from-0 and the (former) from-1 iterates run value iteration from 0 and
# converge to the least fixed point (= the robust value the peers compute). Every
# existing controller assignment (which reports first0 / second1 / ones-first1 / ...)
# then yields the correct pure-VI value automatically. The gap |upper-lower| is then
# identically 0, so the stopping criterion switches to the residual ||V_{k+1}-V_k||.
#
# Three edits per loop, all in the two infinite-horizon functions only:
#   (1) after `vec (first1|second1)(..., fill::ones);` -> `if (VI) X1.zeros();`
#   (2) before `X0 = <new>;` reassignment -> compute `viResid` (VI mode only)
#   (3) `max_diff = max(abs(X1-X0));` -> `(VI) ? viResid : max(abs(X1-X0))`
# Interval iteration remains the default and pays no extra per-sweep cost (the
# residual is only computed when VI is selected).
import re, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "src/GPU_synthesis.cpp"
with open(PATH, newline="") as f:
    text = f.read()
nl = "\r\n" if "\r\n" in text else "\n"
lines = [l.rstrip("\r") for l in text.split("\n")]
n = len(lines)

TARGET = ("infiniteHorizonReachControllerSorted",
          "infiniteHorizonSafeControllerSorted")

def in_target_function(idx):
    for j in range(idx, -1, -1):
        s = lines[j].lstrip()
        if s.startswith("void IMDP::") and lines[j].rstrip().endswith("{"):
            return any(t in s for t in TARGET)
    return False

ONES_RE     = re.compile(r"^(\s*)vec (first1|second1)\(state_space_size, 1, fill::ones\);\s*$")
REASSIGN_RE = re.compile(r"^(\s*)(first|second)0 = (check0|firstnew0|secondnew0);\s*$")
MAXDIFF_RE  = re.compile(r"^(\s*)max_diff = max\(abs\((first|second)1-(first|second)0\)\);\s*$")
VI = "iterMethod == IterationMethod::ValueIteration"

n_zero = n_resid = n_stop = 0
out = []
for i in range(n):
    line = lines[i]
    if not in_target_function(i):
        out.append(line); continue
    mo, mr, md = ONES_RE.match(line), REASSIGN_RE.match(line), MAXDIFF_RE.match(line)
    if mo:
        ind, name = mo.group(1), mo.group(2)
        out.append(line)
        out.append("%sif (%s) %s.zeros();" % (ind, VI, name))
        n_zero += 1
    elif mr:
        ind, pre, src0 = mr.group(1), mr.group(2), mr.group(3)
        out.append("%sdouble viResid = (%s) ? (double)(max(abs(%s - %s0))) : 0.0;" % (ind, VI, src0, pre))
        out.append(line)
        n_resid += 1
    elif md:
        ind, p1, p0 = md.group(1), md.group(2), md.group(3)
        out.append("%smax_diff = (%s) ? viResid : max(abs(%s1-%s0));" % (ind, VI, p1, p0))
        n_stop += 1
    else:
        out.append(line)

with open(PATH, "w", newline="") as f:
    f.write(nl.join(out))
print("zeroed upper-iterates:", n_zero, " residual inserts:", n_resid, " stop edits:", n_stop)
