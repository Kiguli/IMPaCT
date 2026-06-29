#!/usr/bin/env python3
# One-step abstraction check for a BA source cell (ISSUE-0020): is the sparse
# product-of-per-dimension interval the EXACT joint worst case, or a conservative
# over-approximation for COUPLED (non-diagonal A) dynamics?
#   true (joint)         = 1 - min over source x in the cell of  prod_d massIn_d(mean(x))
#   sparse (per-dim)     = 1 - prod_d ( min over mean_d range of massIn_d )
# For coupled A the per-dim minima need not be attained at the same x, so
# sparse_massout >= true_massout (sparse is MORE conservative). Diagonal A => equal.
import math, numpy as np
rng = np.random.default_rng(0)
S2 = math.sqrt(2.0)
def ncdf(z): return 0.5*math.erfc(-z/S2)
def massIn(mu, sig, a, b):   # vectorised P(N(mu,sig) in [a,b])
    return 0.5*(np.vectorize(math.erf)((b-mu)/(sig*S2)) - np.vectorize(math.erf)((a-mu)/(sig*S2)))

A = np.array([[0.6682,0,0.02632,0],[0,0.6830,0,0.02096],[1.0005,0,-0.000499,0],[0,0.8004,0,0.1996]])
B = np.array([0.1320,0.1402,0,0]); Q = np.array([3.4378,2.9272,13.0207,10.4166])
sig = np.sqrt([0.0774,0.0774,0.3872,0.3098])
eta = np.array([0.5,0.5,1,1]); u = 18.5
dlo = np.array([19,19,30,30]) - eta/2     # aligned domain box (dense convention)
dhi = np.array([21,21,36,36]) + eta/2

for center in [np.array([20.,20,33,33]), np.array([21.,21,36,36]), np.array([19.,21,30,36])]:
    cl, ch = center - eta/2, center + eta/2
    # brute force: joint min mass-in over source cell
    N = 400000
    X = cl + (ch-cl)*rng.random((N,4))
    mean = X @ A.T + B*u + Q
    massin = np.ones(N)
    for d in range(4): massin *= massIn(mean[:,d], sig[d], dlo[d], dhi[d])
    true_out = 1.0 - massin.min()
    # sparse: per-dim mean range (affine interval) then per-dim min mass-in, product
    prod = 1.0
    for d in range(4):
        lo = hi = Q[d] + B[d]*u
        for j in range(4):
            a = A[d][j]
            if a >= 0: lo += a*cl[j]; hi += a*ch[j]
            else: lo += a*ch[j]; hi += a*cl[j]
        # min over mean in [lo,hi] of massIn (unimodal, min at farther endpoint)
        m = min(massIn(np.array([lo]), sig[d], dlo[d], dhi[d])[0],
                massIn(np.array([hi]), sig[d], dlo[d], dhi[d])[0])
        prod *= m
    sparse_out = 1.0 - prod
    print("cell center %-16s  true_massout=%.4f  sparse_massout=%.4f  (sparse - true = %+.4f)"
          % (center.tolist(), true_out, sparse_out, sparse_out - true_out))
