#!/usr/bin/env python3
# Convert the neutral .imdp exchange format (src/imdp_io.h) to Storm's DRN format
# for interval MDPs (@type: MDP, @value_type: double-interval). Preserves state
# order (so result vectors align across tools) and emits the model's labels.
#
# Usage: imdp_to_drn.py MODEL.imdp OUT.drn
import sys

def main():
    src, out = sys.argv[1], sys.argv[2]
    nstates = 0
    init = 0
    labels = {}                      # name -> set(states)
    actions = {}                     # state -> list of list[(to,lo,hi)]
    rewards = {}                     # state -> reward (optional)
    with open(src) as f:
        for raw in f:
            line = raw.replace('\r', '').strip()
            if not line or line.startswith('#'):
                continue
            tok = line.split()
            kw = tok[0]
            if kw == 'states':
                nstates = int(tok[1])
            elif kw == 'init':
                init = int(tok[1])
            elif kw == 'label':
                labels.setdefault(tok[1], set()).update(int(x) for x in tok[2:])
            elif kw == 'reward':
                rewards[int(tok[1])] = float(tok[2])
            elif kw == 'tran':
                s = int(tok[1])
                dist = []
                for t in tok[3:]:
                    to, lo, hi = t.split(':')
                    dist.append((int(to), float(lo), float(hi)))
                actions.setdefault(s, []).append(dist)

    # per-state labels (Storm DRN puts them on the `state` line; "init" is special)
    statelabels = {s: [] for s in range(nstates)}
    statelabels[init].append('init')
    for name, sset in labels.items():
        for s in sset:
            if 0 <= s < nstates:
                statelabels[s].append(name)

    nchoices = 0
    for s in range(nstates):
        acts = actions.get(s)
        nchoices += len(acts) if acts else 1   # actionless -> 1 self-loop choice

    has_rew = len(rewards) > 0
    def srew(s):    # state-reward tag "[r] " for the state line, action-reward "[0]"
        return ('[%.12g] ' % rewards.get(s, 0.0)) if has_rew else ''
    arew = ' [0]' if has_rew else ''
    with open(out, 'w') as g:
        g.write('@type: MDP\n@value_type: double-interval\n@parameters\n\n@reward_models\n%s\n'
                % ('r ' if has_rew else ''))
        g.write('@nr_states\n%d\n@nr_choices\n%d\n@model\n' % (nstates, nchoices))
        for s in range(nstates):
            lab = (' ' + ' '.join(statelabels[s])) if statelabels[s] else ''
            g.write('state %d %s%s\n' % (s, srew(s), lab.strip()) if has_rew
                    else 'state %d%s\n' % (s, lab))
            acts = actions.get(s)
            if not acts:
                g.write('\taction 0%s\n\t\t%d : [1, 1]\n' % (arew, s))
                continue
            for a, dist in enumerate(acts):
                g.write('\taction %d%s\n' % (a, arew))
                for to, lo, hi in dist:
                    g.write('\t\t%d : [%.12g, %.12g]\n' % (to, lo, hi))

if __name__ == '__main__':
    main()
