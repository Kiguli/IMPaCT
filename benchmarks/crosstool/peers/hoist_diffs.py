#!/usr/bin/env python3
# ISSUE-0017 fix: hoist the loop-invariant `diff` matrix/vector computations out of
# the value-iteration convergence loops in the two INFINITE-horizon synthesis
# functions. `maxTransitionM-minTransitionM` (and the target/avoid analogues, and the
# fixed-input `tempTmax-tempTmin` variants) never change inside the loop, yet were
# reallocated + recomputed on every sweep (a full (state*input) x state dense matrix),
# dominating the CPU/OMP per-sweep cost. Each loop is wrapped in a fresh block scope
#   {  mat diffT = ...; vec diffR = ...; vec diffA = ...;  while(...) { ... }  }
# so the (now once-computed) diffs do not clash across sibling loops. Loop-invariant
# code motion -> identical result, far fewer allocations.
#
# Only the two infinite-horizon functions are touched (finite-horizon loops run a
# handful of sweeps, so the per-sweep cost is irrelevant there).
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

WHILE_RE = re.compile(r"^(\s*)while \(max_diff > epsilon\)\s*\{\s*$")
DIFF_START_RE = re.compile(r"^\s*mat diffT = (maxTransitionM-minTransitionM|tempTmax-tempTmin);\s*$")
DIFF_CONT_RE = re.compile(r"^\s*vec diff[RA] = .*;\s*$")

def loop_end(i):
    """Index of the line closing the while loop opened on line i (brace match)."""
    depth = 0
    started = False
    for k in range(i, n):
        depth += lines[k].count("{") - lines[k].count("}")
        if lines[k].count("{") > 0:
            started = True
        if started and depth == 0:
            return k
    raise RuntimeError("unbalanced braces from line %d" % i)

close_brace = {}   # line index -> indent of the extra '}' to add AFTER it
moved = 0
out = []
skip = set()
i = 0
while i < n:
    if i in skip:
        i += 1
        continue
    m = WHILE_RE.match(lines[i])
    if m and in_target_function(i):
        indent = m.group(1)
        # diff block inside this loop body
        diff_block = None
        j = i + 1
        while j < n and not WHILE_RE.match(lines[j]):
            if DIFF_START_RE.match(lines[j]):
                blk = [j]
                k = j + 1
                while k < n and DIFF_CONT_RE.match(lines[k]):
                    blk.append(k); k += 1
                diff_block = blk
                break
            j += 1
        if diff_block:
            e = loop_end(i)
            out.append(indent + "{")                 # open block scope
            for bidx in diff_block:
                out.append(lines[bidx]); skip.add(bidx)
            close_brace[e] = indent
            moved += 1
    out.append(lines[i])
    if i in close_brace:
        out.append(close_brace[i] + "}")             # close block scope
    i += 1

with open(PATH, "w", newline="") as f:
    f.write(nl.join(out))
print("hoisted+wrapped loops:", moved)
