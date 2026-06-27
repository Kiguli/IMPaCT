#!/usr/bin/env python3
# Compare IMPaCT's per-cell value vector (controller.h5 extract, cols pess opt) to a
# peer's full value vector dump (one value per line, all states incl. sinks). Only
# the first <cells> entries (the abstraction cells) are compared, since IMPaCT only
# synthesizes over cells. Prints max/mean abs difference and where the max occurs.
#
# Usage: compare_vectors.py IMPACT_VALUE.txt PEER_PESS.txt PEER_OPT.txt
import sys
import numpy as np

impact = np.loadtxt(sys.argv[1])           # columns: pess opt (one row per cell)
if impact.ndim == 1:
    impact = impact.reshape(1, -1)
i_pess = impact[:, 0]; i_opt = impact[:, 1]
cells = len(i_pess)

p_pess = np.loadtxt(sys.argv[2])[:cells]
p_opt  = np.loadtxt(sys.argv[3])[:cells]

dp = np.abs(i_pess - p_pess)
do = np.abs(i_opt - p_opt)
print("cells=%d" % cells)
print("pess: maxabs=%.3e  meanabs=%.3e  at cell %d (impact=%.6f peer=%.6f)"
      % (dp.max(), dp.mean(), int(dp.argmax()), i_pess[dp.argmax()], p_pess[dp.argmax()]))
print("opt : maxabs=%.3e  meanabs=%.3e  at cell %d (impact=%.6f peer=%.6f)"
      % (do.max(), do.mean(), int(do.argmax()), i_opt[do.argmax()], p_opt[do.argmax()]))
