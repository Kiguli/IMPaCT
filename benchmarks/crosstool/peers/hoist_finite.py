#!/usr/bin/env python3
# Apply the loop-invariant hoist (diff matrices, sycl::queue, and constant read-only
# SYCL buffers) to the FINITE-horizon synthesis loops, exactly as was done for the
# infinite-horizon functions. Finite loops use `while (k < timeHorizon)` (only the
# three finite functions: finiteHorizonReachControllerSorted,
# finiteHorizonSafeControllerSorted, finiteHorizonReachControllerSortedStoreMDP), so
# matching that condition leaves the already-hoisted infinite functions untouched.
#
# Per loop, in one pass: wrap in a { } block; hoist the diff block
# (mat diffT = max-min | tempTmax-tempTmin; + vec diffR/diffA) before the while; hoist
# the queue; and hoist the constant buffers (minTransitionM/minTargetM/minAvoidM or the
# fixed-input reduced tempTmin/tempTTmin/tempATmin, plus diffT/diffR/diffA) after the
# queue. Loop-invariant code motion -> identical results, far fewer per-sweep allocs.
import re, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "src/GPU_synthesis.cpp"
with open(PATH, newline="") as f:
    text = f.read()
nl = "\r\n" if "\r\n" in text else "\n"
lines = [l.rstrip("\r") for l in text.split("\n")]
n = len(lines)

WHILE_RE     = re.compile(r"^(\s*)while \(k\s*<\s*timeHorizon\)\s*\{\s*$")
DIFF_START   = re.compile(r"^\s*mat diffT = (maxTransitionM-minTransitionM|tempTmax-tempTmin);\s*$")
DIFF_CONT    = re.compile(r"^\s*vec diff[RA] = .*;\s*$")
QUEUE_RE     = re.compile(r"^\s*sycl::queue \w+;\s*$")
CONST_BUF_RE = re.compile(
    r"^\s*sycl::buffer<\w+> \w+\((minTransitionM|minTargetM|minAvoidM|tempTmin|tempTTmin|tempATmin|diffT|diffR|diffA)\.memptr\(\)")

def code_only(s):
    # strip // line comments so braces inside comments (e.g. `//}`, `//if(..){`)
    # are not counted by the brace matcher
    idx = s.find("//")
    return s[:idx] if idx >= 0 else s

def loop_end(i):
    depth = 0; started = False
    for k in range(i, n):
        s = code_only(lines[k])
        depth += s.count("{") - s.count("}")
        if s.count("{") > 0: started = True
        if started and depth == 0: return k
    raise RuntimeError("unbalanced braces from %d" % i)

out = []; skip = set(); close = {}
nloops = 0
i = 0
while i < n:
    if i in skip: i += 1; continue
    m = WHILE_RE.match(lines[i])
    if m:
        ind = m.group(1); e = loop_end(i)
        D = []; q = None; B = []
        j = i + 1
        while j < e:
            if DIFF_START.match(lines[j]):
                D.append(j); k = j + 1
                while k < e and DIFF_CONT.match(lines[k]): D.append(k); k += 1
            elif QUEUE_RE.match(lines[j]) and q is None:
                q = j
            elif CONST_BUF_RE.match(lines[j]):
                B.append(j)
            j += 1
        out.append(ind + "{")
        for d in D: out.append(lines[d]); skip.add(d)
        if q is not None: out.append(lines[q]); skip.add(q)
        for b in B: out.append(lines[b]); skip.add(b)
        close[e] = ind
        out.append(lines[i])
        nloops += 1
    else:
        out.append(lines[i])
    if i in close: out.append(close[i] + "}")
    i += 1

with open(PATH, "w", newline="") as f:
    f.write(nl.join(out))
print("finite loops hoisted:", nloops)
