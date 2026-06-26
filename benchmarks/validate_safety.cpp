// End-to-end safety synthesis over a CONTINUOUS system. Build the sparse full-
// dynamics IMDP, synthesize a robust safety controller (maximize P(never enter the
// avoid region)), simulate the continuous closed loop, and check the empirical
// safety probability >= the synthesized robust lower bound.
//
// 2-D affine: x0'=0.95 x0 + 1.0 u0 ; x1'=0.95 x1 + 1.0 u1 + N(0,sigma^2).
// Avoid = centre box [-0.6,0.6]^2 (dynamics drift toward the origin, so staying out
// requires active control -> safety in (0,1): a discriminating check).
//
// Build: c++ -std=c++17 -O2 benchmarks/validate_safety.cpp \
//   src/abstraction.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp -o /tmp/safe
#include "../src/abstraction.h"
#include "../src/solve.h"
#include "../src/omaximization.h"
#include <cstdio>
#include <vector>
#include <set>
#include <random>
#include <cmath>

using namespace impact;

int main() {
    abstraction::SystemND s;
    s.dim_x = 2; s.dim_u = 2;
    s.xlb = {-3, -3}; s.xub = {3, 3}; s.eta = {0.3, 0.3};
    s.ulb = {-1, -1}; s.uub = {1, 1}; s.ueta = {0.5, 0.5};
    s.A = {{0.9, 0.0}, {0.0, 0.9}};
    s.B = {{0.3, 0.0}, {0.0, 0.3}};
    s.c = {0.0, 0.0};
    const double sd = 0.3;
    s.sigma = {sd, sd};
    s.tlo = {1e18, 1e18}; s.thi = {-1e18, -1e18};       // empty -> full IMDP

    auto ab = abstraction::buildSparseReachND(s, 1e-9);
    const int N = ab.nCells;
    std::vector<int> Nd(2); std::vector<long long> stride(2); long long NN=1;
    for (int i=0;i<2;++i){ Nd[i]=(int)std::llround((s.xub[i]-s.xlb[i])/s.eta[i]); stride[i]=NN; NN*=Nd[i]; }
    auto locate=[&](double x0,double x1,int& lin)->bool{
        int j0=(int)std::floor((x0-s.xlb[0])/s.eta[0]), j1=(int)std::floor((x1-s.xlb[1])/s.eta[1]);
        if(j0<0||j0>=Nd[0]||j1<0||j1>=Nd[1]) return false; lin=(int)(j0*stride[0]+j1*stride[1]); return true; };
    auto inAvoidCell=[&](double x0,double x1){           // cell fully inside [-0.6,0.6]^2
        int j0=(int)std::floor((x0-s.xlb[0])/s.eta[0]), j1=(int)std::floor((x1-s.xlb[1])/s.eta[1]);
        if(j0<0||j0>=Nd[0]||j1<0||j1>=Nd[1]) return false;
        double a0=s.xlb[0]+j0*s.eta[0], a1=s.xlb[1]+j1*s.eta[1];
        return a0>=-0.6&&a0+s.eta[0]<=0.6 && a1>=-0.6&&a1+s.eta[1]<=0.6; };

    std::set<int> avoid;
    for (int c=0;c<N;++c){ double x0=s.xlb[0]+((c%Nd[0])+0.5)*s.eta[0], x1=s.xlb[1]+((c/Nd[0])+0.5)*s.eta[1];
        if (inAvoidCell(x0,x1)) avoid.insert(c); }

    auto sf = solve::maxSafetyPessimistic(ab.model, avoid, 1e-6);

    // safety policy: argmax_a min_nature E[safety]  (omax Sense::Min on safety lower)
    std::vector<int> policy(ab.model.size(), 0);
    for (int c=0;c<N;++c){
        double best=-1; int bi=0;
        for (size_t a=0;a<ab.model[c].size();++a){
            std::vector<double> lo,hi,V;
            for (const auto& iv: ab.model[c][a]){ lo.push_back(iv.lo); hi.push_back(iv.hi); V.push_back(sf.lower[iv.to]); }
            double q=omax::optimize(lo,hi,V,omax::Sense::Min).value;
            if (q>best){best=q;bi=(int)a;}
        }
        policy[c]=bi;
    }

    std::mt19937 rng(3); std::normal_distribution<double> g(0.0,sd);
    const int TRIALS=3000, HORIZON=200;
    printf("Safety: max P(never enter [-0.6,0.6]^2) over a continuous robot\n");
    printf("%-12s %8s %10s %6s\n","start","lower","empirical","ok?");
    double tests[][2]={{1.0,1.0},{1.5,0.0},{-1.2,1.2},{2.0,2.0}};
    int ok=0,ntot=0;
    for (auto& st: tests){
        int c0; if(!locate(st[0],st[1],c0)) continue;
        double lower=sf.lower[c0];
        int safe=0;
        for (int t=0;t<TRIALS;++t){
            double x0=st[0],x1=st[1]; bool unsafe=false;
            for (int k=0;k<HORIZON;++k){
                if (inAvoidCell(x0,x1)){ unsafe=true; break; }
                int c; if(!locate(x0,x1,c)){ break; }       // leaving grid: treat as safe (out of avoid)
                const auto& u=ab.actions[policy[c]];
                x0=0.9*x0+0.3*u[0]+g(rng); x1=0.9*x1+0.3*u[1]+g(rng);
            }
            if(!unsafe) ++safe;
        }
        double emp=(double)safe/TRIALS;
        double ci=1.96*std::sqrt(std::max(emp*(1-emp),1e-9)/TRIALS);
        bool good = emp >= lower - ci - 1e-2;
        ok+=good; ++ntot;
        printf("(%4.1f,%4.1f) %8.3f %8.3f   %4s\n", st[0],st[1],lower,emp, good?"yes":"NO");
    }
    printf("\n%d/%d starts: empirical safety >= robust lower bound.\n", ok, ntot);
    printf("cells=%d nnz=%lld\n", N, ab.nnz);
    return ok==ntot?0:1;
}
