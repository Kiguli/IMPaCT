#!/usr/bin/env python3
# Monte-Carlo soundness check for the BA finite-safety abstraction (ISSUE-0020).
# Simulates the BA closed loop under the DENSE-synthesised policy (controller.h5) with
# NOMINAL Gaussian noise. The robust lower bound IMPaCT reports must be <= this nominal
# empirical safety (nominal nature is inside the abstraction intervals). If the dense
# lower bound exceeds the empirical safety, the dense nlopt abstraction OVER-claims
# (unsound), and the smaller sparse value is the valid bound.
import sys, h5py, numpy as np
rng = np.random.default_rng(0)

A = np.array([[0.6682,0,0.02632,0],[0,0.6830,0,0.02096],[1.0005,0,-0.000499,0],[0,0.8004,0,0.1996]])
B = np.array([0.1320,0.1402,0,0]); Q = np.array([3.4378,2.9272,13.0207,10.4166])
sigma = np.sqrt([0.0774,0.0774,0.3872,0.3098])
lb = np.array([19.,19,30,30]); ub = np.array([21.,21,36,36]); eta = np.array([0.5,0.5,1,1])
Npts = np.round((ub-lb)/eta).astype(int) + 1     # dense grid points per dim
H = 6; nP = 2000

with h5py.File(sys.argv[1] if len(sys.argv)>1 else "run/BA/controller.h5") as f:
    C = np.array(f[list(f.keys())[0]])
if C.shape[0] == 7: C = C.T                       # (cells, 7): x0..x3, u, safe_lo, safe_hi
coords, inputs, safelo = C[:, :4], C[:, 4], C[:, 5]

idx = np.round((coords - lb) / eta).astype(int)
flat = np.ravel_multi_index(np.clip(idx, 0, Npts-1).T, Npts)
inpArr = np.full(int(np.prod(Npts)), np.nan); inpArr[flat] = inputs
umid = 18.5                                        # fallback input (mid of [17,20])

# start from RANDOM states WITHIN each cell (a sound cell abstraction must bound the
# safety for EVERY state in the cell, not just the centre)
incell = (sys.argv[2] == "incell") if len(sys.argv) > 2 else False
x = np.repeat(coords, nP, axis=0)
if incell:
    x = x + (rng.random((len(x), 4)) - 0.5) * eta[None, :]
safe = np.ones(len(x), bool)
for t in range(H):
    qi = np.clip(np.round((x - lb) / eta).astype(int), 0, Npts-1)
    u = inpArr[np.ravel_multi_index(qi.T, Npts)]
    u = np.where(np.isnan(u), umid, u)
    x = x @ A.T + B[None,:]*u[:,None] + Q[None,:] + rng.standard_normal((len(x),4))*sigma
    safe &= np.all((x >= lb) & (x <= ub), axis=1)

emp = safe.reshape(len(coords), nP).mean(axis=1)   # empirical nominal safety per cell
viol = emp < safelo - 0.02                          # dense lower bound exceeds empirical?
print("cells=%d  dense_lo[mean]=%.3f  empirical_nominal[mean]=%.3f  sparse[mean]=0.107"
      % (len(coords), safelo.mean(), emp.mean()))
print("cells where dense_lower > empirical (unsound, >0.02 margin): %d / %d (%.1f%%)"
      % (viol.sum(), len(coords), 100.0*viol.mean()))
if viol.sum():
    i = np.argmax(safelo - emp)
    print("worst: cell %s  dense_lo=%.3f  empirical=%.3f  (dense over-claims by %.3f)"
          % (coords[i].round(2).tolist(), safelo[i], emp[i], safelo[i]-emp[i]))
