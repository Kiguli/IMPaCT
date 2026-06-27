#!/usr/bin/env python3
# Extract IMPaCT's per-cell synthesized value vectors from a controller.h5 written
# by the SYCL synthesis. The controller matrix columns are:
#   [ state coords (dim_x) | optimal input (dim_u) | value_lower | value_upper ]
# value_lower is the pessimistic (robust / nature-adversarial) bound and
# value_upper the optimistic (cooperative) bound. We print/dump the two value
# columns (one entry per abstraction cell, in cell order = .imdp state order).
#
# Usage: extract_impact_value.py controller.h5 DIM_X DIM_U [--dump OUT.txt]
import sys, h5py, numpy as np

def main():
    path = sys.argv[1]; dim_x = int(sys.argv[2]); dim_u = int(sys.argv[3])
    dump = None
    if '--dump' in sys.argv:
        dump = sys.argv[sys.argv.index('--dump') + 1]
    with h5py.File(path, 'r') as f:
        key = list(f.keys())[0]              # armadillo hdf5_binary default dataset
        A = np.array(f[key])
    # Armadillo stores column-major; h5py reads it as shape (ncols, nrows) for a
    # 2-D dataset written by Armadillo -> transpose to (nrows=cells, ncols).
    if A.ndim == 2 and A.shape[0] == dim_x + dim_u + 2:
        A = A.T
    ncols = A.shape[1]
    lo = A[:, dim_x + dim_u]
    hi = A[:, dim_x + dim_u + 1]
    print('cells=%d cols=%d pess[min,max]=[%.6f,%.6f] opt[min,max]=[%.6f,%.6f]'
          % (A.shape[0], ncols, lo.min(), lo.max(), hi.min(), hi.max()))
    print('pess_mean=%.6f opt_mean=%.6f pess[0]=%.6f opt[0]=%.6f'
          % (lo.mean(), hi.mean(), lo[0], hi[0]))
    if dump:
        np.savetxt(dump, np.column_stack([lo, hi]), fmt='%.10f', header='pess opt')
        print('dumped', dump)

if __name__ == '__main__':
    main()
