#!/usr/bin/env python3
# Convert a PRISM-explicit interval-MDP (.tra + .lab, as shipped by IntervalMDP.jl,
# e.g. test/data/multiObj_robotIMDP) into IMPaCT's neutral .imdp exchange format so
# tools/imdp_solve can solve the SAME model IntervalMDP.jl solves.
#
#   .tra line:  src act dest [lo,hi] actlabel     (header: nstates npairs ntrans)
#   .lab:       0="init" 1="deadlock" 2="reach"   then  state: labelidx [labelidx...]
#   .imdp out:  states N / init I / label NAME s... / tran src act to:lo:hi ...
#
# Usage: tra_to_imdp.py BASE  (reads BASE.tra, BASE.lab) > out.imdp
import sys, re

def main():
    base = sys.argv[1]
    # --- labels: index->name, then per-state label indices ---
    idx2name, state_labels = {}, {}
    with open(base + ".lab") as f:
        first = f.readline()
        for m in re.finditer(r'(\d+)="([^"]+)"', first):
            idx2name[int(m.group(1))] = m.group(2)
        for line in f:
            line = line.strip()
            if not line:
                continue
            s, rest = line.split(":", 1)
            state_labels[int(s)] = [int(x) for x in rest.split()]
    name2states = {n: [] for n in idx2name.values()}
    for s, idxs in state_labels.items():
        for i in idxs:
            name2states[idx2name[i]].append(s)

    # --- transitions: group by (src, act) preserving first-seen order ---
    nstates = None
    rows, order = {}, []
    with open(base + ".tra") as f:
        header = f.readline().split()
        nstates = int(header[0])
        for line in f:
            t = line.split()
            if len(t) < 4:
                continue
            src, act, dest = int(t[0]), int(t[1]), int(t[2])
            lo, hi = re.sub(r"[\[\]]", "", t[3]).split(",")
            key = (src, act)
            if key not in rows:
                rows[key] = []
                order.append(key)
            rows[key].append((dest, float(lo), float(hi)))

    init = name2states.get("init", [0])[0]
    out = [f"# converted from {base} (PRISM-explicit interval MDP, IntervalMDP.jl)",
           f"states {nstates}", f"init {init}"]
    for name, states in name2states.items():
        if name in ("init", "deadlock") or not states:
            continue
        out.append("label " + name + " " + " ".join(str(s) for s in sorted(states)))
    for (src, act) in order:
        edges = " ".join(f"{d}:{lo:.9g}:{hi:.9g}" for (d, lo, hi) in rows[(src, act)])
        out.append(f"tran {src} {act} {edges}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
