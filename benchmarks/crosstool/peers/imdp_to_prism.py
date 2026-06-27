#!/usr/bin/env python3
# Convert a POINT .imdp model (degenerate intervals lo==hi) to a PRISM MDP model.
# PRISM has no interval-MDP support, so this is only valid when every interval is a
# point; we assert that. Used to add PRISM as a third oracle on the point crosstool
# models (chain_point, safety_point).
#
# Usage: imdp_to_prism.py MODEL.imdp OUT.pm
import sys

def main():
    src, out = sys.argv[1], sys.argv[2]
    nstates = 0; init = 0; labels = {}; actions = {}
    with open(src) as f:
        for raw in f:
            line = raw.replace('\r', '').strip()
            if not line or line.startswith('#'):
                continue
            tok = line.split()
            if tok[0] == 'states': nstates = int(tok[1])
            elif tok[0] == 'init': init = int(tok[1])
            elif tok[0] == 'label': labels.setdefault(tok[1], set()).update(int(x) for x in tok[2:])
            elif tok[0] == 'tran':
                s = int(tok[1]); dist = []
                for t in tok[3:]:
                    to, lo, hi = t.split(':')
                    lo = float(lo); hi = float(hi)
                    if abs(hi - lo) > 1e-9:
                        sys.exit("imdp_to_prism: model is not a point MDP (interval %s:%s); PRISM has no IMDP support" % (lo, hi))
                    dist.append((int(to), lo))
                actions.setdefault(s, []).append(dist)
    with open(out, 'w') as g:
        g.write('mdp\n\nmodule M\n')
        g.write('  s : [0..%d] init %d;\n\n' % (nstates - 1, init))
        for s in range(nstates):
            for a, dist in enumerate(actions.get(s, [[(s, 1.0)]])):
                terms = ' + '.join('%.12g:(s\'=%d)' % (p, to) for to, p in dist)
                g.write('  [a%d_%d] s=%d -> %s;\n' % (a, s, s, terms))
        g.write('endmodule\n\n')
        for name, sset in labels.items():
            cond = ' | '.join('s=%d' % s for s in sorted(sset))
            g.write('label "%s" = %s;\n' % (name, cond))

if __name__ == '__main__':
    main()
