// ============================================================================
// Sparse-abstraction ARCH benchmarks (ISSUE-0006 scalability).
//
// Re-runs the ARCH-COMP cases through IMPaCT's SPARSE abstraction
// (src/abstraction.cpp, O(nnz) memory like IntervalMDP.jl) + the sparse robust
// solver (src/solve.cpp, OVI — sound and convergent on nature-confinable end
// components, ISSUE-0003), instead of the dense SYCL IMDP class (which stores a
// (state*input) x state dense matrix and OOMs on the large cases).
//
// Covers the AFFINE-in-state systems, where the per-cell mean RANGE is EXACT, so
// the sparse abstraction has NO accuracy loss vs the dense (nlopt) abstraction:
//   AS, BA, IC_reach, IC_safe, PD_p1, PD_p3, PR_minimal.
// (Nonlinear VP/AV/LM need interval-arithmetic mean enclosures — separate pass.)
//
// Build (pure std C++, no SYCL/Armadillo):
//   c++ -std=c++17 -O2 benchmarks/sparse_arch.cpp \
//       src/abstraction.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp \
//       -o /tmp/sparse_arch
// ============================================================================
#include "../src/abstraction.h"
#include "../src/solve.h"
#include "../src/omaximization.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <functional>
#include <set>
#include <string>
#include <vector>

using namespace impact;

// ---- robust Bellman backup at one state (for the finite-horizon cases) --------
static double backup(const solve::StateActions& acts, const std::vector<double>& V,
                     omax::Sense sense, bool ctrlMax) {
    double best = 0.0; bool any = false;
    std::vector<double> lo, hi, vv;
    for (const solve::ActionDist& a : acts) {
        lo.clear(); hi.clear(); vv.clear();
        for (const solve::Interval& iv : a) { lo.push_back(iv.lo); hi.push_back(iv.hi); vv.push_back(V[iv.to]); }
        const double q = lo.empty() ? 0.0 : omax::optimize(lo, hi, vv, sense).value;
        if (!any) { best = q; any = true; } else best = ctrlMax ? std::max(best, q) : std::min(best, q);
    }
    return any ? best : 0.0;
}

// finite-horizon robust VI: H backups, controller max/min, nature `sense`.
static std::vector<double> finiteVI(const solve::IMDPModel& m, const std::set<int>& tgt,
                                    int H, bool ctrlMax, omax::Sense sense) {
    const int n = (int)m.size();
    std::vector<double> V(n, 0.0);
    for (int t : tgt) V[t] = 1.0;
    for (int k = 0; k < H; ++k) {
        std::vector<double> Vn(n, 0.0);
        for (int s = 0; s < n; ++s) Vn[s] = tgt.count(s) ? 1.0 : backup(m[s], V, sense, ctrlMax);
        V.swap(Vn);
    }
    return V;
}

// Is cell `lin` entirely inside the box [lo,hi]? (target / avoid membership)
static bool inBox(long long lin, const abstraction::GridSpec& g,
                  const std::vector<double>& lo, const std::vector<double>& hi) {
    const int dx = g.dim_x; long long s = 1;
    for (int i = 0; i < dx; ++i) {
        int Ni = std::max(1, (int)std::llround((g.xub[i]-g.xlb[i])/g.eta[i]));
        int mi = (int)((lin / s) % Ni); s *= Ni;
        double cl = g.xlb[i] + mi * g.eta[i], ch = cl + g.eta[i];
        if (!(cl >= lo[i] - 1e-12 && ch <= hi[i] + 1e-12)) return false;
    }
    return true;
}

struct Stat { double mn, mx, mean; int n; };

// Mark cells whose box lies inside the avoid box as dead absorbing (value 0):
// replace their actions with a single self-loop (so they are never the target and
// reaching them yields 0 = reach-avoid semantics).
static void markAvoid(abstraction::SparseReach& R, const abstraction::GridSpec& g,
                      const std::vector<double>& alo, const std::vector<double>& ahi) {
    const int dx = g.dim_x;
    std::vector<int> Nd(dx); std::vector<long long> stride(dx); long long N = 1;
    for (int i = 0; i < dx; ++i) { Nd[i] = std::max(1, (int)std::llround((g.xub[i]-g.xlb[i])/g.eta[i])); stride[i] = N; N *= Nd[i]; }
    for (long long lin = 0; lin < N; ++lin) {
        bool inAvoid = true;
        for (int i = 0; i < dx && inAvoid; ++i) {
            int mi = (int)((lin / stride[i]) % Nd[i]);
            double lo = g.xlb[i] + mi * g.eta[i], hi = lo + g.eta[i];
            if (!(lo >= alo[i] - 1e-12 && hi <= ahi[i] + 1e-12)) inAvoid = false;
        }
        if (inAvoid) R.model[lin] = { { {(int)lin, 1.0, 1.0} } };
    }
}

enum Mode { INF_REACH, FIN_REACH, FIN_SAFETY };

// Point dynamics: mu = f(x, u). For affine, f_i(x,u) = A_i . x + offset_i(u).
using PointMeanFn = std::function<void(const std::vector<double>& x, const std::vector<double>& u,
                                       std::vector<double>& mu)>;

struct Bench {
    const char* name;
    abstraction::GridSpec g;
    PointMeanFn mean;
    Mode mode; int H;
    bool hasAvoid; std::vector<double> alo, ahi;
};

// affine-in-state mean: mu_i range = [A_i . cell] + offset_i(u), exact.
static abstraction::MeanBoundFn affine(std::vector<std::vector<double>> A,
                                       std::function<void(const std::vector<double>&, std::vector<double>&)> offset) {
    int dx = (int)A.size();
    return [A, offset, dx](const std::vector<double>& cl, const std::vector<double>& ch,
                           const std::vector<double>& u, std::vector<double>& muLo, std::vector<double>& muHi) {
        std::vector<double> off(dx); offset(u, off);
        muLo.assign(dx, 0.0); muHi.assign(dx, 0.0);
        for (int i = 0; i < dx; ++i) {
            double lo = off[i], hi = off[i];
            for (int j = 0; j < dx; ++j) { double a = A[i][j];
                if (a >= 0) { lo += a*cl[j]; hi += a*ch[j]; } else { lo += a*ch[j]; hi += a*cl[j]; } }
            muLo[i] = lo; muHi[i] = hi;
        }
    };
}

// ---- JOINT enclosure (matches the dense joint optimisation; no coupled-A over-approx) ----
static PointMeanFn affinePoint(std::vector<std::vector<double>> A,
                               std::function<void(const std::vector<double>&, std::vector<double>&)> offset) {
    int dx = (int)A.size();
    return [A, offset, dx](const std::vector<double>& x, const std::vector<double>& u, std::vector<double>& mu) {
        std::vector<double> off(dx); offset(u, off);
        mu.assign(dx, 0.0);
        for (int i = 0; i < dx; ++i) { double v = off[i];
            for (int j = 0; j < dx; ++j) v += A[i][j] * x[j];
            mu[i] = v; }
    };
}

// Build the sparse IMDP with a JOINT box-mass enclosure over each source cell:
//   lo (min mass) = min over the 2^dim source-cell CORNERS  (the box mass is log-concave
//                   in x, so its minimum over the cell is attained at a vertex -> EXACT),
//   hi (max mass) = product of per-dimension maxima  (a sound upper bound).
// This removes the per-dimension MIN over-approximation that made the sparse abstraction
// more conservative than the dense joint nlopt on coupled (non-diagonal A) systems.
static abstraction::SparseReach buildSparseReachJoint(const abstraction::GridSpec& g,
                                                      const PointMeanFn& fmean, double prune) {
    const int dx = g.dim_x, du = g.dim_u;
    std::vector<int> Nd(dx); std::vector<long long> stride(dx); long long N = 1;
    for (int i = 0; i < dx; ++i) { Nd[i] = std::max(1, (int)std::llround((g.xub[i]-g.xlb[i])/g.eta[i])); stride[i]=N; N*=Nd[i]; }
    const int TARGET = (int)N, SINK = (int)N + 1;

    std::vector<std::vector<double>> upts(du);
    for (int k = 0; k < du; ++k) { int Mk = std::max(0,(int)std::llround((g.uub[k]-g.ulb[k])/g.ueta[k]));
        for (int t = 0; t <= Mk; ++t) upts[k].push_back(g.ulb[k] + t*g.ueta[k]); }
    std::vector<std::vector<double>> actions;
    if (du == 0) actions.push_back({});
    else { std::vector<int> id(du,0); while (true) { std::vector<double> u(du);
        for (int k=0;k<du;++k) u[k]=upts[k][id[k]]; actions.push_back(std::move(u));
        int k=0; for(;k<du;++k){ if(++id[k]<(int)upts[k].size()) break; id[k]=0; } if(k==du) break; } }

    abstraction::SparseReach out; out.nCells=(int)N; out.nnz=0;
    out.model.assign((size_t)N+2, {}); out.targets.insert(TARGET); out.actions = actions;

    auto cellLoDim = [&](int i, int j){ return g.xlb[i] + j*g.eta[i]; };
    std::vector<int> mi(dx), jt(dx);
    std::vector<double> cl(dx), ch(dx), muLo(dx), muHi(dx);
    const int nCorner = 1 << dx;
    std::vector<std::vector<double>> cornerMu(nCorner, std::vector<double>(dx));
    std::vector<double> xc(dx), mu(dx);

    auto isTargetMi = [&](const std::vector<int>& m){ for(int i=0;i<dx;++i){ double lo=cellLoDim(i,m[i]),hi=lo+g.eta[i];
        if(!(lo>=g.tlo[i]-1e-12 && hi<=g.thi[i]+1e-12)) return false; } return true; };

    for (long long lin = 0; lin < N; ++lin) {
        for (int i=0;i<dx;++i) mi[i]=(int)((lin/stride[i])%Nd[i]);
        if (isTargetMi(mi)) { out.model[lin].push_back({ {TARGET,1.0,1.0} }); out.nnz+=1; continue; }
        for (int i=0;i<dx;++i){ cl[i]=cellLoDim(i,mi[i]); ch[i]=cl[i]+g.eta[i]; }
        for (const auto& u : actions) {
            // evaluate the dynamics at the 2^dim source-cell corners; cache the means
            for (int c=0;c<nCorner;++c){ for(int i=0;i<dx;++i) xc[i]=((c>>i)&1)?ch[i]:cl[i];
                fmean(xc,u,mu); cornerMu[c]=mu; }
            for (int i=0;i<dx;++i){ muLo[i]=1e300; muHi[i]=-1e300;
                for(int c=0;c<nCorner;++c){ muLo[i]=std::min(muLo[i],cornerMu[c][i]); muHi[i]=std::max(muHi[i],cornerMu[c][i]); } }

            // window of candidate dest cells (per-dim mean range +/- 6 sigma)
            std::vector<int> wlo(dx), whi(dx); bool any=true;
            for (int i=0;i<dx;++i){ double W=6.0*g.sigma[i];
                wlo[i]=std::max(0,(int)std::floor((muLo[i]-W-g.xlb[i])/g.eta[i]));
                whi[i]=std::min(Nd[i]-1,(int)std::floor((muHi[i]+W-g.xlb[i])/g.eta[i]));
                if(wlo[i]>whi[i]) any=false; jt[i]=wlo[i]; }
            solve::ActionDist row;
            if (any) while (true) {
                // hi = product of per-dim maxima (sound); lo = min over corners (exact joint min)
                double ph=1.0;
                for(int i=0;i<dx;++i){ double a=cellLoDim(i,jt[i]), b=a+g.eta[i];
                    abstraction::Bound bd=abstraction::transitionInterval1D(muLo[i],muHi[i],g.sigma[i],a,b); ph*=bd.hi; }
                double pl=1e300;
                for(int c=0;c<nCorner;++c){ double p=1.0;
                    for(int i=0;i<dx;++i){ double a=cellLoDim(i,jt[i]), b=a+g.eta[i];
                        p*=abstraction::massInInterval(cornerMu[c][i],g.sigma[i],a,b); }
                    pl=std::min(pl,p); }
                if (ph>prune){ if(pl>ph) pl=ph;
                    if(isTargetMi(jt)) row.push_back({TARGET,pl,ph});
                    else { long long lj=0; for(int i=0;i<dx;++i) lj+=(long long)jt[i]*stride[i]; row.push_back({(int)lj,pl,ph}); } }
                int i=0; for(;i<dx;++i){ if(++jt[i]<=whi[i]) break; jt[i]=wlo[i]; } if(i==dx) break;
            }
            // outside-grid (SINK): mass-in-grid hi via per-dim product (sound), lo via corner min (exact)
            double gh=1.0;
            for(int i=0;i<dx;++i){ abstraction::Bound gg=abstraction::transitionInterval1D(muLo[i],muHi[i],g.sigma[i],g.xlb[i],g.xub[i]); gh*=gg.hi; }
            double gl=1e300;
            for(int c=0;c<nCorner;++c){ double p=1.0;
                for(int i=0;i<dx;++i) p*=abstraction::massInInterval(cornerMu[c][i],g.sigma[i],g.xlb[i],g.xub[i]); gl=std::min(gl,p); }
            row.push_back({SINK, std::max(0.0,1.0-gh), std::min(1.0,1.0-gl)});
            out.nnz += (long long)row.size();
            out.model[lin].push_back(std::move(row));
        }
    }
    out.model[TARGET].push_back({ {TARGET,1.0,1.0} });
    out.model[SINK].push_back({ {SINK,1.0,1.0} });
    return out;
}

int main(int argc, char** argv) {
    const double eps = 1e-6, prune = 1e-7;
    const bool fast = (argc > 1 && std::string(argv[1]) == "fast");  // skip the slow PR_minimal
    std::vector<Bench> benches;

    // ---- AS: 3D anaesthesia, affine, finite reach H=10 ----
    { abstraction::GridSpec g; g.dim_x=3; g.dim_u=2;
      g.xlb={1,0,0}; g.xub={6,10,10}; g.eta={0.25,1,1};
      g.ulb={0,0}; g.uub={7,30}; g.ueta={1,30};
      g.sigma={std::sqrt(0.001),std::sqrt(0.001),std::sqrt(0.001)};
      g.tlo={4,8,8}; g.thi={6,10,10};
      std::vector<std::vector<double>> A={{0.8192,0.03412,0.01265},{0.01646,0.9822,0.0001},{0.0009,0.00002,0.9989}};
      auto off=[](const std::vector<double>& u, std::vector<double>& o){ double s=u[0]+u[1]; o={0.01883*s,0.0002*s,0.00001*s}; };
      benches.push_back({"AS", g, affinePoint(A,off), FIN_REACH, 10, false, {}, {}}); }

    // ---- BA: 4D building automation, affine, finite safety H=6 (avoid = leave grid) ----
    { abstraction::GridSpec g; g.dim_x=4; g.dim_u=1;
      g.xlb={19,19,30,30}; g.xub={21,21,36,36}; g.eta={0.5,0.5,1,1};
      g.ulb={17}; g.uub={20}; g.ueta={1};
      g.sigma={std::sqrt(0.0774),std::sqrt(0.0774),std::sqrt(0.3872),std::sqrt(0.3098)};
      g.tlo={100,100,100,100}; g.thi={101,101,101,101};   // no target
      std::vector<std::vector<double>> A={{0.6682,0,0.02632,0},{0,0.6830,0,0.02096},{1.0005,0,-0.000499,0},{0,0.8004,0,0.1996}};
      auto off=[](const std::vector<double>& u, std::vector<double>& o){ double v=u[0]; o={0.1320*v+3.4378,0.1402*v+2.9272,13.0207,10.4166}; };
      benches.push_back({"BA", g, affinePoint(A,off), FIN_SAFETY, 6, false, {}, {}}); }

    // ---- IC_reach: 2D integrator, affine, finite reach H=5 ----
    { abstraction::GridSpec g; g.dim_x=2; g.dim_u=1;
      g.xlb={-10,-10}; g.xub={10,10}; g.eta={0.5,0.5};
      g.ulb={-1}; g.uub={1}; g.ueta={0.5};
      g.sigma={std::sqrt(0.01),std::sqrt(0.01)};
      g.tlo={-8,-8}; g.thi={8,8};
      std::vector<std::vector<double>> A={{1,0.1},{0,1}};
      auto off=[](const std::vector<double>& u, std::vector<double>& o){ o={0.005*u[0],0.1*u[0]}; };
      benches.push_back({"IC_reach", g, affinePoint(A,off), FIN_REACH, 5, false, {}, {}}); }

    // ---- IC_safe: same dynamics, finite safety H=5 ----
    { abstraction::GridSpec g; g.dim_x=2; g.dim_u=1;
      g.xlb={-10,-10}; g.xub={10,10}; g.eta={0.5,0.5};
      g.ulb={-1}; g.uub={1}; g.ueta={0.5};
      g.sigma={std::sqrt(0.01),std::sqrt(0.01)};
      g.tlo={100,100}; g.thi={101,101};   // no target
      std::vector<std::vector<double>> A={{1,0.1},{0,1}};
      auto off=[](const std::vector<double>& u, std::vector<double>& o){ o={0.005*u[0],0.1*u[0]}; };
      benches.push_back({"IC_safe", g, affinePoint(A,off), FIN_SAFETY, 5, false, {}, {}}); }

    // ---- PD_p1: 2D package delivery, affine, infinite reach-avoid ----
    { abstraction::GridSpec g; g.dim_x=2; g.dim_u=2;
      g.xlb={-6,-6}; g.xub={6,6}; g.eta={0.5,0.5};
      g.ulb={-1,-1}; g.uub={1,1}; g.ueta={0.1,0.1};
      g.sigma={std::sqrt(0.02),std::sqrt(0.02)};
      g.tlo={5,-1}; g.thi={6,1};
      std::vector<std::vector<double>> A={{0.9,0},{0,0.8}};
      auto off=[](const std::vector<double>& u, std::vector<double>& o){ o={1.4*u[0],1.4*u[1]}; };
      benches.push_back({"PD_p1", g, affinePoint(A,off), INF_REACH, 0, true, {0,-5}, {1,1}}); }

    // ---- PD_p3: different target ----
    { abstraction::GridSpec g; g.dim_x=2; g.dim_u=2;
      g.xlb={-6,-6}; g.xub={6,6}; g.eta={0.5,0.5};
      g.ulb={-1,-1}; g.uub={1,1}; g.ueta={0.1,0.1};
      g.sigma={std::sqrt(0.02),std::sqrt(0.02)};
      g.tlo={-4,-4}; g.thi={-2,-3};
      std::vector<std::vector<double>> A={{0.9,0},{0,0.8}};
      auto off=[](const std::vector<double>& u, std::vector<double>& o){ o={1.4*u[0],1.4*u[1]}; };
      benches.push_back({"PD_p3", g, affinePoint(A,off), INF_REACH, 0, true, {0,-5}, {1,1}}); }

    // ---- PR_minimal: 2D robot, affine-in-state (cos/sin of u), infinite reach-avoid.
    //      This is the ISSUE-0006 dense-OOM case (698103 x 1583 ~ 8.84 GB dense). ----
    { abstraction::GridSpec g; g.dim_x=2; g.dim_u=2;
      g.xlb={-10,-10}; g.xub={10,10}; g.eta={0.5,0.5};
      g.ulb={-1,-1}; g.uub={1,1}; g.ueta={0.1,0.1};
      g.sigma={std::sqrt(1.0/1.3333),std::sqrt(1.0/1.3333)};
      g.tlo={5,5}; g.thi={8,8};
      std::vector<std::vector<double>> A={{1,0},{0,1}};
      auto off=[](const std::vector<double>& u, std::vector<double>& o){ o={2*u[0]*std::cos(u[1]),2*u[0]*std::sin(u[1])}; };
      benches.push_back({"PR_minimal", g, affinePoint(A,off), INF_REACH, 0, true, {-1.5,-1.5}, {1.5,1.5}}); }

    std::printf("%-11s %7s %7s %12s %11s %9s %8s %9s  interior[min,max,mean]\n",
                "bench","cells","actions","nnz","sparse(MB)","dense(GB)","build_s","solve_s");
    for (auto& b : benches) {
        if (fast && std::string(b.name) == "PR_minimal") continue;
        // Align to the DENSE convention so sparse == dense (then sparse is purely a
        // memory win): the dense IMDP class grids the state space as (ub-lb)/eta + 1
        // POINTS at lb+k*eta with cells CENTRED on each point ([point +/- eta/2]).
        // abstraction.cpp instead grids into (ub-lb)/eta corner-anchored cells
        // [lb+j*eta, lb+(j+1)*eta]. Shifting the grid by -eta/2 (and +eta/2 at the top)
        // makes the sparse cells centre on the dense points (N+1 cells); expanding the
        // target/avoid boxes by eta/2 makes the box-subset-region test equal the dense
        // centre-in-region test.
        for (int i = 0; i < b.g.dim_x; ++i) {
            const double h = b.g.eta[i] / 2.0;
            b.g.xlb[i] -= h; b.g.xub[i] += h;
            b.g.tlo[i] -= h; b.g.thi[i] += h;
            if (b.hasAvoid) { b.alo[i] -= h; b.ahi[i] += h; }
        }
        auto t0 = std::chrono::steady_clock::now();
        abstraction::SparseReach R = buildSparseReachJoint(b.g, b.mean, prune);
        if (b.hasAvoid) markAvoid(R, b.g, b.alo, b.ahi);
        auto t1 = std::chrono::steady_clock::now();

        std::vector<double> V;
        const int TARGET = R.nCells, SINK = R.nCells + 1;
        if (b.mode == INF_REACH) {
            solve::IntervalResult r = solve::maxReachPessimistic(R.model, {TARGET}, eps);
            V = r.lower;
        } else if (b.mode == FIN_REACH) {
            V = finiteVI(R.model, {TARGET}, b.H, /*ctrlMax=*/true, omax::Sense::Min);
        } else { // FIN_SAFETY: safety = 1 - min_ctrl max_nature reach-to-SINK in H steps
            std::vector<double> reachSink = finiteVI(R.model, {SINK}, b.H, /*ctrlMax=*/false, omax::Sense::Max);
            V.assign(reachSink.size(), 0.0);
            for (size_t s = 0; s < reachSink.size(); ++s) V[s] = 1.0 - reachSink[s];
        }
        auto t2 = std::chrono::steady_clock::now();

        // interior stats: exclude absorbing target/avoid cells (comparable to dense)
        Stat st{1e18, -1e18, 0, 0};
        for (int s = 0; s < R.nCells; ++s) {
            if (inBox(s, b.g, b.g.tlo, b.g.thi)) continue;
            if (b.hasAvoid && inBox(s, b.g, b.alo, b.ahi)) continue;
            st.mn = std::min(st.mn, V[s]); st.mx = std::max(st.mx, V[s]); st.mean += V[s]; ++st.n;
        }
        st.mean /= std::max(1, st.n);

        const int nA = (int)R.actions.size();
        const double sparseMB = R.nnz * (double)sizeof(solve::Interval) / 1e6;
        const double denseGB = (double)R.nCells * nA * R.nCells * 8.0 * 2.0 / 1e9; // min+max dense
        std::printf("%-11s %7d %7d %12lld %11.1f %9.2f %8.2f %9.2f  [%.4f, %.4f, %.4f] (n=%d)\n",
                    b.name, R.nCells, nA, R.nnz, sparseMB, denseGB,
                    std::chrono::duration<double>(t1-t0).count(),
                    std::chrono::duration<double>(t2-t1).count(),
                    st.mn, st.mx, st.mean, st.n);
    }
    return 0;
}
