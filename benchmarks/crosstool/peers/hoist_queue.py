#!/usr/bin/env python3
# ISSUE-0017 fix (part 2): hoist the per-sweep `sycl::queue` construction out of the
# infinite-horizon convergence loops. A fresh queue was created every sweep; one queue
# per loop is sufficient and avoids repeated runtime/queue setup. Runs AFTER
# hoist_diffs.py (which already wrapped each loop in a { } block); the queue is moved
# into that block, just before the `while`.
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
QUEUE_RE = re.compile(r"^\s*sycl::queue \w+;\s*$")

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
        # find the queue line in this loop body (before the next while)
        j = i + 1
        qidx = None
        while j < n and not WHILE_RE.match(lines[j]):
            if QUEUE_RE.match(lines[j]):
                qidx = j
                break
            j += 1
        if qidx is not None:
            qname = lines[qidx].strip()
            out.append(indent + qname)   # emit queue before the while (in the block)
            skip.add(qidx)
            moved += 1
    out.append(lines[i])
    i += 1

with open(PATH, "w", newline="") as f:
    f.write(nl.join(out))
print("hoisted queues:", moved)
