#!/usr/bin/env python3
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], 'r') as f:
    key = list(f.keys())[0]
    A = np.array(f[key])
print("dataset key:", key, "raw shape:", A.shape)
# show both orientations of the first few rows
print("first 3 rows as-is:")
print(A[:3])
if A.ndim == 2:
    print("first 3 rows transposed:")
    print(A.T[:3])
    print("col ranges (as-is, per row-index across):",
          [(round(float(A[i].min()),3), round(float(A[i].max()),3)) for i in range(min(A.shape[0], 8))])
