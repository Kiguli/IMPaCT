#!/usr/bin/env python3
# Per-benchmark correctness comparison: IMPaCT (controller.h5 extract) vs
# IntervalMDP.jl (full value-vector dump), over the abstraction cells only.
# Emits one TSV row: name cells i_pmin i_pmax i_pmean m_pmin m_pmax m_pmean
#                    pess_maxabs opt_maxabs
#
# Usage: arch_compare.py NAME IMPACT_VALUE.txt IMDP_PESS.txt IMDP_OPT.txt
import sys
import numpy as np

name = sys.argv[1]
imp = np.loadtxt(sys.argv[2])
if imp.ndim == 1:
    imp = imp.reshape(1, -1)
i_pess, i_opt = imp[:, 0], imp[:, 1]
cells = len(i_pess)
m_pess = np.loadtxt(sys.argv[3])[:cells]
m_opt = np.loadtxt(sys.argv[4])[:cells]
pd = np.abs(i_pess - m_pess)
od = np.abs(i_opt - m_opt)
print("\t".join(str(x) for x in [
    name, cells,
    "%.6f" % i_pess.min(), "%.6f" % i_pess.max(), "%.6f" % i_pess.mean(),
    "%.6f" % m_pess.min(), "%.6f" % m_pess.max(), "%.6f" % m_pess.mean(),
    "%.3e" % pd.max(), "%.3e" % od.max(),
]))
