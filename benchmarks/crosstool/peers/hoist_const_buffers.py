#!/usr/bin/env python3
# Improve per-sweep SYCL buffer construction: hoist the CONSTANT (read-only) buffers
# out of the per-sweep inner buffer scope into the per-loop block scope, so they are
# constructed ONCE per loop instead of once per sweep. The constant buffers wrap data
# that never changes during value iteration:
#   minTransitionM, minTargetM, minAvoidM  (abstraction matrices/vectors), and
#   diffT, diffR, diffA                     (the once-hoisted max-min differences).
# The CHANGING buffers (sorted_indices, firstnew*/secondnew*, first*/second*, U_pos)
# stay inside the per-sweep scope. Accessors created inside queue.submit() simply
# reference the now-outer buffers. Runs AFTER hoist_diffs.py + hoist_queue.py, which
# already created the per-loop block scope and moved `sycl::queue` before each while.
# Only the two infinite-horizon functions are touched.
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

QUEUE_RE = re.compile(r"^\s*sycl::queue \w+;\s*$")
CONST_BUF_RE = re.compile(
    r"^\s*sycl::buffer<\w+> \w+\((minTransitionM|minTargetM|minAvoidM|diffT|diffR|diffA)\.memptr\(\)")

moved = 0
skip = set()
out = []
i = 0
while i < n:
    if i in skip:
        i += 1
        continue
    line = lines[i]
    if QUEUE_RE.match(line) and in_target_function(i):
        # collect this loop's constant buffers (up to the next queue / function end)
        bufs = []
        j = i + 1
        while j < n and not QUEUE_RE.match(lines[j]):
            s = lines[j].lstrip()
            if s.startswith("void IMDP::") and lines[j].rstrip().endswith("{"):
                break
            if CONST_BUF_RE.match(lines[j]):
                bufs.append(j)
            j += 1
        out.append(line)                       # the queue line
        for b in bufs:
            out.append(lines[b]); skip.add(b)  # hoist constant buffer right after queue
        moved += len(bufs)
        i += 1
        continue
    out.append(line)
    i += 1

with open(PATH, "w", newline="") as f:
    f.write(nl.join(out))
print("hoisted constant buffers:", moved)
