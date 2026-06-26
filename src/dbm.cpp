#include "dbm.h"

namespace impact {
namespace dbm {

Bound addB(const Bound& a, const Bound& b) {
    if (a.isInf() || b.isInf()) return Bound::inf();
    return { a.c + b.c, a.strict || b.strict };
}

// a strictly tighter (smaller) than b: smaller constant, or equal constant but
// strict (<) beats non-strict (<=).
bool tighterB(const Bound& a, const Bound& b) {
    if (a.c != b.c) return a.c < b.c;
    return a.strict && !b.strict;
}

const Bound& minB(const Bound& a, const Bound& b) { return tighterB(a, b) ? a : b; }

Zone::Zone(int clocks) : n(clocks), m(clocks + 1, std::vector<Bound>(clocks + 1, Bound::inf())) {
    // universe: all clocks >= 0, no upper bounds. x_i - x_i <= 0; x_0 - x_i <= 0 (x_i>=0).
    for (int i = 0; i <= n; ++i) m[i][i] = Bound::leq(0);
    for (int i = 1; i <= n; ++i) m[0][i] = Bound::leq(0);   // 0 - x_i <= 0  i.e. x_i >= 0
}

void canonicalize(Zone& z) {
    const int N = z.n + 1;
    for (int k = 0; k < N; ++k)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                Bound through = addB(z.m[i][k], z.m[k][j]);
                if (tighterB(through, z.m[i][j])) z.m[i][j] = through;
            }
}

bool isEmpty(const Zone& z) {
    // negative diagonal after tightening => infeasible. Check x_i - x_i path < 0.
    const int N = z.n + 1;
    for (int i = 0; i < N; ++i)
        if (z.m[i][i].c < 0 || (z.m[i][i].c == 0 && z.m[i][i].strict)) return true;
    // also any pair whose round-trip is negative
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            Bound rt = addB(z.m[i][j], z.m[j][i]);
            if (rt.c < 0 || (rt.c == 0 && rt.strict)) return true;
        }
    return false;
}

void constrain(Zone& z, int i, int j, Bound b) {
    z.m[i][j] = minB(z.m[i][j], b);
    canonicalize(z);
}

Zone intersect(const Zone& a, const Zone& b) {
    Zone r = a;
    for (int i = 0; i <= a.n; ++i)
        for (int j = 0; j <= a.n; ++j)
            r.m[i][j] = minB(a.m[i][j], b.m[i][j]);
    canonicalize(r);
    return r;
}

void up(Zone& z) {
    // let time pass: each clock can grow unboundedly -> drop upper bounds x_i - 0.
    for (int i = 1; i <= z.n; ++i) z.m[i][0] = Bound::inf();
    canonicalize(z);
}

void reset(Zone& z, int r) {
    // x_r := 0  ==  identify clock r with the zero clock 0.
    for (int j = 0; j <= z.n; ++j) {
        z.m[r][j] = z.m[0][j];
        z.m[j][r] = z.m[j][0];
    }
    z.m[r][0] = Bound::leq(0);
    z.m[0][r] = Bound::leq(0);
    canonicalize(z);
}

bool includes(const Zone& outer, const Zone& inner) {
    // inner subset of outer iff every inner bound is tighter-or-equal to outer's.
    for (int i = 0; i <= outer.n; ++i)
        for (int j = 0; j <= outer.n; ++j)
            if (tighterB(outer.m[i][j], inner.m[i][j])) return false;   // outer strictly tighter -> inner not contained
    return true;
}

bool contains(const Zone& z, const std::vector<double>& val) {
    auto x = [&](int idx) -> double { return idx == 0 ? 0.0 : val[idx - 1]; };
    for (int i = 0; i <= z.n; ++i)
        for (int j = 0; j <= z.n; ++j) {
            if (z.m[i][j].isInf()) continue;
            double diff = x(i) - x(j);
            if (z.m[i][j].strict) { if (!(diff <  z.m[i][j].c - 1e-9)) return false; }
            else                  { if (!(diff <= z.m[i][j].c + 1e-9)) return false; }
        }
    return true;
}

} // namespace dbm
} // namespace impact
