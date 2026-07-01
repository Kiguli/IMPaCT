#!/usr/bin/env python3
# Parameter Lifting: a parametric Markov model + a parameter region -> an INTERVAL MDP whose
# robust [pess,opt] reachability bounds the parametric value over the region. This is the
# reduction of parametric REGION verification to robust interval-MDP model checking, which
# IMPaCT already solves (imdp_solve reach --bound both). Each parameter OCCURRENCE is relaxed
# independently, so the bound is EXACT when a parameter appears once per row and a SOUND
# over-approximation otherwise (Quatmann, Dehnert, Jansen, Junges, Katoen, "Parameter
# Synthesis for Markov Models: Faster Than Ever", ATVA 2016).
#
# Parametric .pimdp input: like .imdp but a `param <name> <lo> <hi>` region directive and
# `tran <s> <a> <to>:<expr> ...` where <expr> is affine in ONE parameter: c, p, 1-p, a*p+b.
# Usage: param_lift.py MODEL.pimdp > MODEL.imdp
import sys, re

def affine(expr, pname):
    # return (a, b) with expr = a*p + b (affine in the single parameter pname)
    e = expr.replace(' ', '')
    if e == pname: return (1.0, 0.0)
    if e == '1-' + pname: return (-1.0, 1.0)
    m = re.fullmatch(r'([-\d.]*)\*?' + re.escape(pname) + r'([+-][\d.]+)?', e)
    if m:
        a = float(m.group(1)) if m.group(1) not in ('', '+', '-') else (1.0 if m.group(1) != '-' else -1.0)
        b = float(m.group(2)) if m.group(2) else 0.0
        return (a, b)
    return (0.0, float(e))     # constant

def main():
    lo = hi = None; pname = None; out = []
    for raw in open(sys.argv[1]):
        line = raw.replace('\r', '').rstrip('\n')
        t = line.split()
        if not t or line.lstrip().startswith('#'): out.append(line); continue
        if t[0] == 'param':
            pname, lo, hi = t[1], float(t[2]), float(t[3]); continue
        if t[0] == 'tran':
            edges = []
            for e in t[3:]:
                to, expr = e.split(':', 1)
                a, b = affine(expr, pname or 'p')
                v1, v2 = a * lo + b, a * hi + b
                edges.append('%s:%.12g:%.12g' % (to, min(v1, v2), max(v1, v2)))
            out.append(' '.join(t[:3] + edges)); continue
        out.append(line)
    sys.stdout.write('\n'.join(out) + '\n')

if __name__ == '__main__':
    main()
