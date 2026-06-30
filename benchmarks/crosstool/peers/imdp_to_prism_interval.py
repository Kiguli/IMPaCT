#!/usr/bin/env python3
# Convert a neutral .imdp (INTERVAL or point) model to PRISM's explicit interval format
# (BASE.tra/.sta/.lab). PRISM >= 4.8 auto-detects an IMDP from the [lo,hi] interval
# entries in the .tra and model-checks robust properties (Pmaxmin/Pmaxmax=?[F label]).
# Unlike imdp_to_prism.py (point-only), this preserves intervals.
#
# PRISM's IMDP checker REQUIRES every interval transition to have a strictly positive
# lower bound (fixed support graph): it rejects [0,hi] edges with "Transition probability
# has lower bound of 0". Abstracted IMDPs (IMPaCT/Storm/IntervalMDP.jl all accept [0,hi])
# routinely contain such edges. Pass --floor EPS to replace exact-zero lower bounds with
# EPS so PRISM will run; the resulting PRISM value is then only an EPS-perturbation of the
# true robust value (mark it with an asterisk in any comparison).
#
# Usage: imdp_to_prism_interval.py MODEL.imdp OUTBASE [--floor EPS]
import sys

def main():
    src, base = sys.argv[1], sys.argv[2]
    floor = 0.0
    if "--floor" in sys.argv:
        floor = float(sys.argv[sys.argv.index("--floor") + 1])
    nstates = 0; init = 0; labels = {}; actions = {}
    with open(src) as f:
        for raw in f:
            line = raw.replace('\r', '').strip()
            if not line or line.startswith('#'):
                continue
            tok = line.split()
            if tok[0] == 'states':
                nstates = int(tok[1])
            elif tok[0] == 'init':
                init = int(tok[1])
            elif tok[0] == 'label':
                labels.setdefault(tok[1], set()).update(int(x) for x in tok[2:])
            elif tok[0] == 'tran':
                s = int(tok[1]); dist = []
                for t in tok[3:]:
                    to, lo, hi = t.split(':')
                    dist.append((int(to), float(lo), float(hi)))
                actions.setdefault(s, []).append(dist)

    # .tra (PRISM explicit, interval transitions: "src act dest [lo,hi]"; header "n c m")
    rows = []; nchoices = 0; ntrans = 0
    for s in range(nstates):
        acts = actions.get(s) or [[(s, 1.0, 1.0)]]      # deadlock -> self-loop
        for k, dist in enumerate(acts):
            nchoices += 1
            for to, lo, hi in dist:
                if floor > 0.0 and lo <= 0.0:
                    lo = floor
                rows.append("%d %d %d [%.12g,%.12g]" % (s, k, to, lo, hi)); ntrans += 1
    with open(base + ".tra", "w") as g:
        g.write("%d %d %d\n" % (nstates, nchoices, ntrans))
        g.write("\n".join(rows) + "\n")

    # .sta (single variable s = state index)
    with open(base + ".sta", "w") as g:
        g.write("(s)\n")
        for s in range(nstates):
            g.write("%d:(%d)\n" % (s, s))

    # .lab (index 0 reserved for "init"; then one index per model label)
    names = ["init"] + list(labels.keys())
    name_idx = {n: i for i, n in enumerate(names)}
    with open(base + ".lab", "w") as g:
        g.write(" ".join('%d="%s"' % (i, n) for i, n in enumerate(names)) + "\n")
        for s in range(nstates):
            idxs = ([0] if s == init else []) + [name_idx[n] for n, ss in labels.items() if s in ss]
            if idxs:
                g.write("%d: %s\n" % (s, " ".join(map(str, sorted(idxs)))))

if __name__ == '__main__':
    main()
