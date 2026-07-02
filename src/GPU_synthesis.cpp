#include "IMDP.h"
#include "IO_utils.h"
#include "solve.h"          // shared CPU OVI solver (OptimisticVI dispatch)
#include "omaximization.h"  // robust inner Bellman (controller extraction)
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <nlopt.hpp>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <AdaptiveCpp/sycl/sycl.hpp>
#include <chrono>
#include <armadillo>
#include <hdf5/serial/hdf5.h>

using namespace std;
using namespace arma;

/// Destructor
IMDP::~IMDP(){
}

/// Save/Load functions using IO_utils.h
void IMDP::saveMinTargetTransitionVector(){
    IMPaCT_IO::saveData(minTargetM, "minttm.h5", "Min Target Transition Vector");
}
void IMDP::loadMinTargetTransitionVector(string filename){
    IMPaCT_IO::loadData(minTargetM, filename, "minimum target transition Vector");
}
void IMDP::saveMinAvoidTransitionVector(){
    IMPaCT_IO::saveData(minAvoidM, "minatm.h5", "Min Avoid Transition Vector");
}
void IMDP::loadMinAvoidTransitionVector(string filename){
    IMPaCT_IO::loadData(minAvoidM, filename, "minimum avoid transition Vector");
}
void IMDP::saveMinTransitionMatrix(){
    IMPaCT_IO::saveData(minTransitionM, "mintm.h5", "Min Transition Matrix");
}
void IMDP::loadMinTransitionMatrix(string filename){
    IMPaCT_IO::loadData(minTransitionM, filename, "minimum transition matrix");
}
void IMDP::saveMaxTargetTransitionVector(){
    IMPaCT_IO::saveData(maxTargetM, "maxttm.h5", "Max Target Transition Vector");
}
void IMDP::loadMaxTargetTransitionVector(string filename){
    IMPaCT_IO::loadData(maxTargetM, filename, "maximum target transition Vector");
}
void IMDP::saveMaxAvoidTransitionVector(){
    IMPaCT_IO::saveData(maxAvoidM, "maxatm.h5", "Max Avoid Transition Vector");
}
void IMDP::loadMaxAvoidTransitionVector(string filename){
    IMPaCT_IO::loadData(maxAvoidM, filename, "maximum avoid transition Vector");
}
void IMDP::saveMaxTransitionMatrix(){
    IMPaCT_IO::saveData(maxTransitionM, "maxtm.h5", "Max Transition Matrix");
}
void IMDP::loadMaxTransitionMatrix(string filename){
    IMPaCT_IO::loadData(maxTransitionM, filename, "maximum transition matrix");
}
void IMDP::saveController(){
    IMPaCT_IO::saveData(controller, "controller.h5", "Controller");
}
void IMDP::loadController(string filename){
    IMPaCT_IO::loadData(controller, filename, "controller");
}

/// Export the abstracted Interval MDP to the neutral .imdp text format so peer
/// tools (IntervalMDP.jl / Storm / PRISM) can solve the SAME model. Cells are
/// states 0..state_space_size-1; an accepting target sink and a dead avoid sink
/// (explicit avoid region + mass leaving the bounded domain) are appended. Because
/// cells + target + avoid partition the one-step probability mass, every
/// (state,action) interval row is a sound interval distribution. The disturbance
/// dimension is assumed 0 (true for the ARCH-COMP cross-tool benchmarks).
void IMDP::exportIMDP(const string& filename){
    auto start = chrono::steady_clock::now();
    cout << "Exporting abstracted IMDP to neutral .imdp format: " << filename << endl;

    const bool has_target = !minTargetM.is_empty();
    const bool has_avoid  = !minAvoidM.is_empty();
    const size_t ss = state_space_size;
    const size_t n_actions = (input_space_size == 0 ? 1 : input_space_size);

    if (disturb_space_size != 0)
        cout << "  warning: exportIMDP ignores the disturbance dimension (dim_w>0)." << endl;
    if (minTransitionM.is_empty()) {
        cout << "  error: transition matrices not computed; call transitionMatrixBounds() first." << endl;
        return;
    }

    size_t N = ss;
    long long T_idx = -1, A_idx = -1;
    if (has_target) { T_idx = (long long)N; N++; }
    if (has_avoid)  { A_idx = (long long)N; N++; }

    ofstream f(filename);
    if(!f){ cout << "  error: cannot open " << filename << " for writing." << endl; return; }
    f << setprecision(12);
    f << "# Exported by IMPaCT abstraction. cells=" << ss
      << " actions/state=" << n_actions
      << (has_target ? " target-sink" : "")
      << (has_avoid  ? " avoid-sink"  : "") << "\n";
    f << "states " << N << "\n";
    f << "init 0\n";
    if (has_target) f << "label target " << T_idx << "\n";
    if (has_avoid)  f << "label avoid " << A_idx << "\n";

    auto clamp01 = [](double v){ return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); };
    const double EPS = 1e-12;

    for (size_t i = 0; i < ss; ++i) {
        for (size_t k = 0; k < n_actions; ++k) {
            const size_t row = k * ss + i;          // (state i, action k), dim_w == 0
            ostringstream line;
            line << "tran " << i << " " << k;
            bool any = false;
            for (size_t col = 0; col < ss; ++col) {
                double hi = clamp01(maxTransitionM(row, col));
                if (hi <= EPS) continue;
                double lo = clamp01(minTransitionM(row, col));
                if (lo > hi) lo = hi;
                line << " " << col << ":" << lo << ":" << hi;
                any = true;
            }
            if (has_target) {
                double hi = clamp01(maxTargetM(row));
                if (hi > EPS) {
                    double lo = clamp01(minTargetM(row));
                    if (lo > hi) lo = hi;
                    line << " " << T_idx << ":" << lo << ":" << hi;
                    any = true;
                }
            }
            if (has_avoid) {
                double hi = clamp01(maxAvoidM(row));
                if (hi > EPS) {
                    double lo = clamp01(minAvoidM(row));
                    if (lo > hi) lo = hi;
                    line << " " << A_idx << ":" << lo << ":" << hi;
                    any = true;
                }
            }
            if (!any) line << " " << i << ":1:1";   // isolated cell -> self loop
            f << line.str() << "\n";
        }
    }
    if (has_target) f << "tran " << T_idx << " 0 " << T_idx << ":1:1\n";
    if (has_avoid)  f << "tran " << A_idx << " 0 " << A_idx << ":1:1\n";
    f.close();

    auto end = chrono::steady_clock::now();
    cout << "  wrote " << N << " states ("
         << ss << " cells x " << n_actions << " actions) in "
         << chrono::duration<double>(end - start).count() << " s." << endl;
}

/// OptimisticVI dispatch (IterationMethod::OptimisticVI). Build the abstracted interval
/// MDP from the in-memory min/max matrices (the same construction exportIMDP writes to
/// text) and solve it through the shared CPU OVI solver (src/solve.cpp), which gives a
/// SOUND two-sided [lower,upper] certificate and converges on nature-confinable end
/// components where the SYCL interval iteration cannot (ISSUE-0003). The greedy controller
/// is extracted per cell from the converged values via the same O-maximization the SYCL
/// kernels use. Returns false (fall back to SYCL) when dim_w>0 (export is dim_w==0 only).
bool IMDP::infiniteHorizonOVIDispatch(bool IMDP_lower, bool is_reach){
    using namespace impact;
    if (disturb_space_size != 0) {
        cout << "  OptimisticVI dispatch supports dim_w==0 only; falling back to SYCL." << endl;
        return false;
    }
    if (minTransitionM.is_empty()) {
        cout << "  error: transition matrices not computed; call transitionMatrixBounds() first." << endl;
        return false;
    }
    const bool has_target = !minTargetM.is_empty();
    const bool has_avoid  = !minAvoidM.is_empty();
    const size_t ss = state_space_size;
    const size_t n_actions = (input_space_size == 0 ? 1 : input_space_size);

    // --- build the interval model: cells 0..ss-1, then target / avoid / rest sinks ---
    size_t N = ss; long long T_idx = -1, A_idx = -1;
    if (has_target) { T_idx = (long long)N; N++; }
    if (has_avoid)  { A_idx = (long long)N; N++; }
    const long long R_idx = (long long)N; N++;        // value-0 rest sink (residual mass)

    auto clamp01 = [](double v){ return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); };
    const double EPS = 1e-12;
    solve::IMDPModel model(N);
    for (size_t i = 0; i < ss; ++i) {
        for (size_t k = 0; k < n_actions; ++k) {
            const size_t row = k * ss + i;            // (state i, action k), dim_w == 0
            solve::ActionDist a; double sumLo = 0.0, sumHi = 0.0;
            for (size_t col = 0; col < ss; ++col) {
                double hi = clamp01(maxTransitionM(row, col));
                if (hi <= EPS) continue;
                double lo = clamp01(minTransitionM(row, col)); if (lo > hi) lo = hi;
                a.push_back({(int)col, lo, hi}); sumLo += lo; sumHi += hi;
            }
            if (has_target) { double hi = clamp01(maxTargetM(row));
                if (hi > EPS) { double lo = clamp01(minTargetM(row)); if (lo > hi) lo = hi;
                    a.push_back({(int)T_idx, lo, hi}); sumLo += lo; sumHi += hi; } }
            if (has_avoid)  { double hi = clamp01(maxAvoidM(row));
                if (hi > EPS) { double lo = clamp01(minAvoidM(row)); if (lo > hi) lo = hi;
                    a.push_back({(int)A_idx, lo, hi}); sumLo += lo; sumHi += hi; } }
            // Make the row a FEASIBLE interval distribution for O-maximization
            // (sum_lo <= 1 <= sum_hi) without granting nature spurious freedom:
            //  - sum_hi < 1: genuinely missing (leaving-domain) mass -> value-0 rest sink
            //    with the EXACT residual interval [1-sum_hi, 1-sum_lo]. Sound: unreachable
            //    mass cannot reach the target (pessimistic) and is value 0.
            //  - sum_hi >= 1: the cells/target/avoid already carry all the mass (matches
            //    exportIMDP, which the peer tools accept) -> add NOTHING, so nature can only
            //    redistribute within the real successors.
            if (sumHi < 1.0 - EPS) {
                double rLo = 1.0 - sumHi, rHi = 1.0 - sumLo;
                if (rHi > 1.0) rHi = 1.0; if (rLo < 0.0) rLo = 0.0;
                a.push_back({(int)R_idx, rLo, rHi});
            } else if (sumLo > 1.0 + EPS) {
                for (auto& iv : a) iv.lo /= sumLo;        // over-constrained row -> renormalise lowers
            }
            if (a.empty()) a.push_back({(int)i, 1.0, 1.0});
            model[i].push_back(std::move(a));
        }
    }
    if (has_target) model[T_idx].push_back({ {(int)T_idx, 1.0, 1.0} });
    if (has_avoid)  model[A_idx].push_back({ {(int)A_idx, 1.0, 1.0} });
    model[R_idx].push_back({ {(int)R_idx, 1.0, 1.0} });

    // --- solve via the shared OVI; build the per-cell value vector the controller optimizes ---
    const double eps = epsilon;
    std::vector<double> Vlo(N), Vup(N), Qval(N);   // Qval = value the controller acts on
    omax::Sense natureSense;
    bool ctrlMax;
    if (is_reach) {
        std::set<int> tgt; if (T_idx >= 0) tgt.insert((int)T_idx);
        solve::IntervalResult r = IMDP_lower ? solve::maxReachPessimistic(model, tgt, eps)
                                             : solve::maxReachOptimistic(model, tgt, eps);
        Vlo = r.lower; Vup = r.upper; Qval = r.lower;            // reach value
        natureSense = IMDP_lower ? omax::Sense::Min : omax::Sense::Max;
        ctrlMax = true;                                          // controller maximizes reach
    } else {
        std::set<int> avo; if (A_idx >= 0) avo.insert((int)A_idx);
        solve::IntervalResult r = IMDP_lower ? solve::maxSafetyPessimistic(model, avo, eps)
                                             : solve::maxSafetyOptimistic(model, avo, eps);
        Vlo = r.lower; Vup = r.upper;                            // safety bounds
        for (size_t s = 0; s < N; ++s) Qval[s] = 1.0 - r.upper[s]; // reach-to-avoid value
        natureSense = IMDP_lower ? omax::Sense::Max : omax::Sense::Min;
        ctrlMax = false;                                        // controller minimizes reach-to-avoid
    }

    // --- greedy controller: argopt action per cell on the converged value (same O-max) ---
    std::vector<arma::uword> U_pos(ss, 0);
    std::vector<double> lo, hi, vv;
    for (size_t i = 0; i < ss; ++i) {
        double best = 0.0; bool any = false; arma::uword bestK = 0;
        const solve::StateActions& acts = model[i];
        for (size_t k = 0; k < acts.size(); ++k) {
            const solve::ActionDist& d = acts[k];
            lo.clear(); hi.clear(); vv.clear();
            for (const solve::Interval& iv : d) { lo.push_back(iv.lo); hi.push_back(iv.hi); vv.push_back(Qval[iv.to]); }
            double q = lo.empty() ? 0.0 : omax::optimize(lo, hi, vv, natureSense).value;
            if (!any) { best = q; bestK = (arma::uword)k; any = true; }
            else if (ctrlMax ? (q > best) : (q < best)) { best = q; bestK = (arma::uword)k; }
        }
        U_pos[i] = bestK;
    }

    // --- write the controller in the dense format: [state | (input) | lower | upper] ---
    arma::vec lowerCol(ss), upperCol(ss);
    for (size_t i = 0; i < ss; ++i) { lowerCol(i) = Vlo[i]; upperCol(i) = Vup[i]; }
    if (input_space_size == 0) {
        controller.set_size(ss, dim_x + 2);
        controller.cols(0, dim_x - 1) = state_space;
        controller.col(dim_x) = lowerCol;
        controller.col(dim_x + 1) = upperCol;
    } else {
        controller.set_size(ss, dim_x + dim_u + 2);
        controller.cols(0, dim_x - 1) = state_space;
        for (size_t i = 0; i < ss; ++i)
            controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos[i]);
        controller.col(dim_x + dim_u) = lowerCol;
        controller.col(dim_x + dim_u + 1) = upperCol;
    }
    double loMin = 1e18, loMax = -1e18, gapMax = 0.0;
    for (size_t i = 0; i < ss; ++i) {
        loMin = std::min(loMin, Vlo[i]); loMax = std::max(loMax, Vlo[i]);
        gapMax = std::max(gapMax, Vup[i] - Vlo[i]);
    }
    cout << "  OptimisticVI (shared CPU solver) done: " << N << " states, " << n_actions
         << " actions/cell; cell value lower in [" << loMin << ", " << loMax
         << "], max certified gap " << gapMax << "." << endl;
    return true;
}

/// Sorted Implementation of infinite horizon reachability
void IMDP::infiniteHorizonReachControllerSorted(bool IMDP_lower){
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for infinite horizon reach controller using sorted approach... " << endl;
    if (iterMethod == IterationMethod::OptimisticVI && infiniteHorizonOVIDispatch(IMDP_lower, /*is_reach=*/true)) {
        cout << "Infinite horizon reach (OptimisticVI) completed in "
             << chrono::duration<double>(chrono::steady_clock::now() - start).count() << " s." << endl;
        return;
    }
    
    if (input_space_size == 0 && disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size, 1, fill::zeros);

            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                //Get difference between max and min for incrementing values

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;

                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }

                            // maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }

                            //return final values
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;

                        });
                    });
                }
                queue.wait_and_throw();

                vec check0 = firstnew0;
                vec check1 = firstnew1;
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;

                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);

            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;

            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;


                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                //Get difference between max and min for incrementing values


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;

                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s+= accdTT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            //rest is avoid state transitions we don't need to calculate

                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(secondnew0 - second0))) : 0.0;
                second0 = secondnew0;
                second1 = secondnew1;

                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first0;
            controller.col(dim_x + 1) = second1;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                //Get difference between max and min for incrementing values

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                vec check0 = firstnew0;
                vec check1 = firstnew1;
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;
                
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                //Get difference between max and min for incrementing values
                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(secondnew0 - second0))) : 0.0;
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second0;
            controller.col(dim_x + 1) = first1;
        }
    }else if (disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                //Get difference between max and min for incrementing values

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                //Get difference between max and min for incrementing values
                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s+= accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(secondnew0 - second0))) : 0.0;
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(secondnew0 - second0))) : 0.0;
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = second0;
            controller.col(dim_x+dim_u + 1) = first1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }else if (input_space_size==0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                //Get difference between max and min for incrementing values

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                //Get difference between max and min for incrementing values
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - second0))) : 0.0;
                second0 = check0;
                second1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first0;
            controller.col(dim_x + 1) = second1;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - second0))) : 0.0;
                second0 = check0;
                second1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second0;
            controller.col(dim_x + 1) = first1;
        }
    }
    else{
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                //Get difference between max and min for incrementing values

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);
                input_and_state1 = min(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(max(input_and_state1,1));
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                //Get difference between max and min for incrementing values
                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s+= accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - second0))) : 0.0;
                second0 = check0;
                second1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);
                input_and_state1 = min(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(max(input_and_state1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            temp1 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                temp1 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - second0))) : 0.0;
                second0 = check0;
                second1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = second0;
            controller.col(dim_x+dim_u + 1) = first1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

// Sorted Method for finite horizon reachability
void IMDP::finiteHorizonReachControllerSorted(bool IMDP_lower, size_t timeHorizon){
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for finite horizon reach controller using sorted approach... " << endl;
    
    if (input_space_size == 0 && disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                queue.wait_and_throw();
                k++;
                first0 = firstnew0;
                
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            k = 0;
            
            cout << "Create reduced matrix where input is fixed." << endl;
            
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;

            
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, false);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            
                            double temp0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                                
                            }
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;
                
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
            cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, false);


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), 0);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), 0);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), 0);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                k++;
                first0 = firstnew0;
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            k = 0;
            
            cout << "Create reduced matrix where input is fixed." << endl;
            
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;

            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, true);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;
                
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
        }
    }else if (input_space_size==0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, true);


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);

                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                first0 = conv_to< colvec >::from(min(firstnew0,1)); 
                k++;
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            k = 0;
            cout << "second loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, false);


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                
                k++;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first0;
            controller.col(dim_x + 1) = second0;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, false);


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                first0 = conv_to< colvec >::from(min(firstnew0,1));
                k++;
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                k++;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second0;
            controller.col(dim_x + 1) = first0;
        }
    }
    
    else if (disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
            cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, true);


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                //TODO: throw an error here.
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
               
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(firstnew0,1));
                k++;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            k=0;
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
           
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, false);

                //Get difference between max and min for incrementing values


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, false);

                //Get difference between max and min for incrementing values

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(firstnew0,1));
                k++;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, true);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[col];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;
                
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }else{
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            
            size_t k=0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, true);



                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(input_and_state0,1));
                
                k++;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
                
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
           
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, false);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                
                k++;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            
            size_t k=0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
            cout << "." << endl; 
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, false);

                
                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                                
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                                s = s+accdTT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            accAvoidM[i] = (1.0-s);
                            cdfAccessor0[i] =  temp0;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(input_and_state0,1));
                k++;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, true);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminTT[i];
                            s = s + accminTT[i];
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                k++;
                
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}


// Sorted Method for infinite horizon safety
void IMDP::infiniteHorizonSafeControllerSorted(bool IMDP_lower){
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for infinite horizon safe controller using sorted approach... " << endl;
    if (iterMethod == IterationMethod::OptimisticVI && infiniteHorizonOVIDispatch(IMDP_lower, /*is_reach=*/false)) {
        cout << "Infinite horizon safety (OptimisticVI) completed in "
             << chrono::duration<double>(chrono::steady_clock::now() - start).count() << " s." << endl;
        return;
    }

    if (input_space_size == 0 && disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                                
                            }
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                vec check0 = firstnew0;
                vec check1 = firstnew1;
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(secondnew0 - second0))) : 0.0;
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = ones(state_space_size)-first1;
            controller.col(dim_x + 1) = ones(state_space_size)-second1;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                vec check0 = firstnew0;
                vec check1 = firstnew1;
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(secondnew0 - second0))) : 0.0;
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = ones(state_space_size)-second1;
            controller.col(dim_x + 1) = ones(state_space_size)-first1;
        }
    }else if (disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).min(U_pos[i]);
                }
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(secondnew0 - second0))) : 0.0;
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = ones(state_space_size)-first1;
            controller.col(dim_x+dim_u + 1) = ones(state_space_size)-second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).min(U_pos[i]);
                }
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                         
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(secondnew0 - second0))) : 0.0;
                second0 = secondnew0;
                second1 = secondnew1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = ones(state_space_size)-second1;
            controller.col(dim_x+dim_u + 1) = ones(state_space_size)-first1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }else if (input_space_size==0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - second0))) : 0.0;
                second0 = check0;
                second1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = ones(state_space_size)-first1;
            controller.col(dim_x + 1) = ones(state_space_size)-second1;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - second0))) : 0.0;
                second0 = check0;
                second1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = ones(state_space_size)-second1;
            controller.col(dim_x + 1) = ones(state_space_size)-first1;
        }
    }
    else{
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = max(firstnew0,1);
                input_and_state1 = max(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(min(input_and_state1,1));
                
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).min(U_pos[i]);
                }
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
           
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - second0))) : 0.0;
                second0 = check0;
                second1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = ones(state_space_size)-first1;
            controller.col(dim_x+dim_u + 1) = ones(state_space_size)-second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            vec first1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) first1.zeros();
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            double max_diff = 1.0;
            double min_diff = 1.0;
            size_t converge = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
            sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = max(firstnew0,1);
                input_and_state1 = max(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(min(input_and_state1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - first0))) : 0.0;
                first0 = check0;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).min(U_pos[i]);
                }
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(first1-first0));
                min_diff = min(abs(first1-first0));
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            vec second1(state_space_size, 1, fill::ones);
            if (iterMethod == IterationMethod::ValueIteration) second1.zeros();
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            max_diff = 1.0;
            min_diff = 1.0;
            converge = 0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
            sycl::queue Q;
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (max_diff > epsilon) {
                converge++;
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            double temp0;
                            
                            temp1 = 0;
                            temp0 = 0;
                            s = 0.0;
                            
                            temp0 += accminAT[i];
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            //maximize transitions to target
                            if ((1.0-s) <= accdAT[i]){
                                temp0 += (1.0-s);
                                temp1 += (1.0-s);
                            }else{
                                temp0 += accdAT[i];
                                temp1 += accdAT[i];
                            }
                            
                            cdfAccessor0[i] =  temp0;
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                double viResid = (iterMethod == IterationMethod::ValueIteration) ? (double)(max(abs(check0 - second0))) : 0.0;
                second0 = check0;
                second1 = check1;
                
                max_diff = (iterMethod == IterationMethod::ValueIteration) ? viResid : max(abs(second1-second0));
                min_diff = min(abs(second1-second0));
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = ones(state_space_size)-second1;
            controller.col(dim_x+dim_u + 1) = ones(state_space_size)-first1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

//Sorted Method for finite horizon safety
void IMDP::finiteHorizonSafeControllerSorted(bool IMDP_lower, size_t timeHorizon){
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for finite horizon safe controller using sorted approach... " << endl;
    
    if (input_space_size == 0 && disturb_space_size == 0){
        if (!IMDP_lower){
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size, 1, fill::zeros);

            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                vec check1 = firstnew1;
                k++;
                first1 = check1;
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;

                            // ISSUE-0013: avoid mass is value 0 for finite (direct stay-safe) VI,
                            // so it must NOT be added to temp1 (the value) — only to s
                            // (normalization), as in the matching lower-bound loop. Removed an
                            // erroneous `temp1 += accminAT[i];` here that over-counted the bound.
                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second1 = secondnew1;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first1;
            controller.col(dim_x + 1) = second1;
        }
        else{
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                vec check1 = firstnew1;
                k++;
                first1 = check1;
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;

                            temp1 = 0;          // ISSUE-0013: temp1 was used uninitialized (UB)
                            s = 0.0;

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];   // ISSUE-0013: base transition*value term was missing
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            //maximize transitions between states: nature MAXIMIZES the stay-safe
                            //value, so the residual mass goes to the highest-value states FIRST
                            //(descending sort); the avoid set (value 0) receives only the leftover.
                            //ISSUE-0013 (verification round 2): the avoid-residual block belongs
                            //AFTER this loop for the maximizing sense — placing it before (as in
                            //the minimizing sibling) hands nature's maximizer the zero-value avoid
                            //slack first and systematically depresses the upper bound (measured:
                            //0.6738 vs oracle 0.9400 on the no-input verification model).
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];

                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            // leftover (if any) is absorbed by the avoid set at value 0 — no
                            // temp1 contribution; s bookkeeping ends with the loop.

                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second1 = secondnew1;
                
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second1;
            controller.col(dim_x + 1) = first1;
        }
    }else if (disturb_space_size == 0){
        if (!IMDP_lower){
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            //transitions to avoid
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew1.reshape(state_space_size, input_space_size);
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                k++;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew1.row(i).max(U_pos[i]);
                }
                
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k <timeHorizon) {
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second1 = secondnew1;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first1;
            controller.col(dim_x+dim_u + 1) = second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*input_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }
                            
                            
                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*input_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew1.reshape(state_space_size, input_space_size);
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                k++;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew1.row(i).max(U_pos[i]);
                }
                
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }
                         
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second1 = secondnew1;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first1;
            controller.col(dim_x+dim_u + 1) = second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }else if (input_space_size==0){
        if (!IMDP_lower){
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        //auto accs = bufS.get_access<sycl::access::mode::read_write>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            temp1 += accminAT[i];
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                
                k++;
                first1 = check1;
                
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[val];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                k++;
                second1 = check1;
                
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first1;
            controller.col(dim_x + 1) = second1;
        }
        else{
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                k++;
                first1 = check1;
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> buff1(second1.memptr(),second1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                k++;
                second1 = check1;
                
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second1;
            controller.col(dim_x + 1) = first1;
        }
    }
    else{
        if (!IMDP_lower){
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, false);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }

                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state1 = min(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check1 = conv_to< colvec >::from(max(input_and_state1,1));
                k++;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew1.row(i).max(U_pos[i]);
                }
                
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
           
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, true);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }
                            
                            
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                k++;
                second1 = check1;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first1;
            controller.col(dim_x+dim_u + 1) = second1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first1(state_space_size, 1, fill::ones);
            mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
            
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first1, true);

                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    sycl::buffer<double> buff1(first1.memptr(),first1.n_rows);
                    //sycl::buffer<double> bufS(s.memptr(),s.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf1 = buff1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf1[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }
                            
                            
                            // maximize transitions to target set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }
                            
                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accf1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf1[val];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            
                            cdfAccessor1[i] =  temp1;
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state1 = min(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check1 = conv_to< colvec >::from(max(input_and_state1,1));
                k++;
                first1 = check1;
                
                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew1.row(i).max(U_pos[i]);
                }
                
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;
            
            vec second1(state_space_size, 1, fill::ones);
            mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }
            
            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second1, false);

                
                
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    sycl::buffer<double> bufs1(second1.memptr(),second1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs1 = bufs1.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        
                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp1;
                            
                            temp1 = 0;
                            s = 0.0;
                            
                            s = s + accminAT[i];
                            
                            for (size_t col = 0; col < state_space_size; col++) {
                                temp1 += accminT[(col*state_space_size*disturb_space_size) +i]*accs1[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }
                            
                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp1 += (1.0-s)*accs1[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp1 += accdT[(val*state_space_size*disturb_space_size) +i]*accs1[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }
                            
                            cdfAccessor1[i] =  temp1;
                        });
                    });
                }
                Q.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                k++;
                second1 = check1;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;
            
            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = second1;
            controller.col(dim_x+dim_u + 1) = first1;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/*Collect MDP that is synthesized for counter-factual analysis*/

// Sorted Method for finite horizon safety
void IMDP::finiteHorizonReachControllerSortedStoreMDP(bool IMDP_lower, size_t timeHorizon){
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for finite horizon reach controller using sorted approach... " << endl;

    if (input_space_size == 0 && disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);
            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, true);



                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), 0);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), 0);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), 0);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //only used if buffer is bigger than zero
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                //if (storeMDP==true) {
                                accAvoidM[i] += (1.0-s);
                                //}
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                                //if (storeMDP==true) {
                                accAvoidM[i] += accdAT[i];
                                //}
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size) +i] += (1.0-s);
                                    //}
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size) +i] += accdT[(val*state_space_size) +i];
                                    //}
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                //if (storeMDP==true) {
                                accTargetM[i] += (1.0-s);
                                //}
                            }else{
                                temp0 += accdTT[i];
                                //if (storeMDP==true) {
                                accTargetM[i] += accdTT[i];
                                //}
                            }
                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                queue.wait_and_throw();
                k++;
                first0 = firstnew0;

            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);

            k = 0;

            cout << "Create reduced matrix where input is fixed." << endl;

                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;



            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, false);



                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;

                            double temp0;
                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];

                            }

                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;

            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size, 1, fill::zeros);


            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
            cout << "." << flush;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, false);


                //if (storeMDP==true) {
                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), 0);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), 0);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), 0);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        //only used if buffer is bigger than zero
                       auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                       auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                       auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];


                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }


                            // maximize transitions to target set

                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                //if (storeMDP==true) {
                                accTargetM[i] += (1.0-s);
                                //}
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                //if (storeMDP==true) {
                                accTargetM[i] += accdTT[i];
                                //}
                                s = s+accdTT[i];
                            }

                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size) +i] += (1.0-s);
                                    //}
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accf0[val];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size) +i] += accdT[(val*state_space_size) +i];
                                    //}
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            //if (storeMDP==true) {
                            accAvoidM[i] += (1.0-s);
                                //}
                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();
                k++;
                first0 = firstnew0;
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;


            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);

            k = 0;

            cout << "Create reduced matrix where input is fixed." << endl;

                tempTmin = minTransitionM;
                tempTmax = maxTransitionM;
                tempTTmin= minTargetM;
                tempTTmax= maxTargetM;
                tempATmin = minAvoidM;
                tempATmax = maxAvoidM;

            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, true);



                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;
                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }

                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;

            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
        }
    }else if (input_space_size==0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);

            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, true);


                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;
                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];


                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                                accAvoidM[i] += (1.0-s);
                            }else{
                                s = s+accdAT[i];
                                accAvoidM[i] += accdAT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*disturb_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*disturb_space_size) +i] += accdT[(val*state_space_size*disturb_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);

                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();

                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                first0 = conv_to< colvec >::from(min(firstnew0,1));
                k++;
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            k = 0;
            cout << "second loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, false);


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];


                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();

                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));

                k++;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = first0;
            controller.col(dim_x + 1) = second0;
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);

            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, false);


                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*disturb_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*disturb_space_size) +i] += accdT[(val*state_space_size*disturb_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            accAvoidM[i] += (1.0-s);
                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();


                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size,disturb_space_size);
                first0 = conv_to< colvec >::from(min(firstnew0,1));
                k++;
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "second loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, true);


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> buff0(second0.memptr(),second0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s = s+accdAT[i];
                            }

                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accf0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();

                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                k++;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x) = second0;
            controller.col(dim_x + 1) = first0;
        }
    }

    else if (disturb_space_size == 0){
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);

            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
            cout << "." << flush;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, true);


                //if (storeMDP==true) {
                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);
                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];


                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                                //if (storeMDP==true) {
                                accAvoidM[i] += (1.0-s);
                                //}
                            }else{
                                s = s+accdAT[i];
                                //if (storeMDP==true) {
                                accAvoidM[i] += accdAT[i];
                                //}
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size*input_space_size) +i] += (1.0-s);
                                    //}
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    //if (storeMDP==true) {
                                    accTransitionM[(val*state_space_size*input_space_size) +i] += accdT[(val*state_space_size*input_space_size) +i];
                                    //}
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                //if (storeMDP==true) {
                                accTargetM[i] += (1.0-s);
                                //}
                            }else{
                                //TODO: throw an error here.
                                temp0 += accdTT[i];
                                //if (storeMDP==true) {
                                accTargetM[i] += accdTT[i];
                                //}
                            }

                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();


                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(firstnew0,1));
                k++;

                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;


            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);

            k=0;

            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }


            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, false);

                //Get difference between max and min for incrementing values


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);

            size_t k = 0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, false);
                //Get difference between max and min for incrementing values

                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size) +i];
                            }


                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                                s = s+accdTT[i];
                            }

                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size) +i] += accdT[(val*state_space_size*input_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*input_space_size) +i];
                                }
                            }

                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            accAvoidM[i] += (1.0-s);
                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();

                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(firstnew0,1));
                k++;

                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }

            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size, 1, fill::zeros);

            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
                tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
                tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
                tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
                tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
                tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
            }



            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, true);


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size) +i];
                            }

                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }

                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[col];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size) +i];
                                }
                            }

                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                k++;
                second0 = secondnew0;

            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }else{
        if (IMDP_lower){
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);

            size_t k=0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, true);


                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];


                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];
                            }

                            // maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                accAvoidM[i] += (1.0-s);
                                s = 1.0;
                            }else{
                                accAvoidM[i] += accdAT[i];
                                s = s+accdAT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }

                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();


                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);

                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(input_and_state0,1));

                k++;

                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }

            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }


            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, false);



                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                s+= accdTT[i];
                            }


                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();

                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));

                k++;
            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
        else{
            vec first0(state_space_size, 1, fill::zeros);
            mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
            uvec U_pos(state_space_size, 1, fill::zeros);
            mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);

            size_t k=0;
            cout << "first loop iterations: " << endl;
            {
                mat diffT = maxTransitionM-minTransitionM;
                vec diffR = maxTargetM - minTargetM;
                vec diffA = maxAvoidM - minAvoidM;
                sycl::queue queue;
                    sycl::buffer<double> bufminT(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(minTargetM.memptr(),minTargetM.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(minAvoidM.memptr(),minAvoidM.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k < timeHorizon) {
            cout << "." << endl;

                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(first0, false);


                TargetM = minTargetM;
                AvoidM = minAvoidM;
                TransitionM = minTransitionM;
                //}

                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> buff0(first0.memptr(),first0.n_rows);
                    sycl::buffer<double> bufTargetM(TargetM.memptr(), TargetM.n_rows);
                    sycl::buffer<double> bufAvoidM(AvoidM.memptr(), AvoidM.n_rows);
                    sycl::buffer<double> bufTransitionM(TransitionM.memptr(), TransitionM.n_rows * TransitionM.n_cols);


                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::read_write>(cgh);
                        auto accf0 = buff0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);
                        auto accTargetM = bufTargetM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accAvoidM = bufAvoidM.get_access<sycl::access::mode::read_write>(cgh);
                        auto accTransitionM = bufTransitionM.get_access<sycl::access::mode::read_write>(cgh);

                        //ASSUMING MINIMAL LP SOLVING
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*input_space_size*disturb_space_size) +i]*accf0[col];
                                s = s+ accminT[(col*state_space_size*input_space_size*disturb_space_size) +i];

                            }


                            // maximize transitions to target set
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                                accTargetM[i] += (1.0-s);
                                s = 1.0;
                            }else{
                                temp0 += accdTT[i];
                                accTargetM[i] += accdTT[i];
                                s = s+accdTT[i];
                            }

                            //maximize state to state transitions
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += (1.0-s);
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i]*accf0[val];
                                    accTransitionM[(val*state_space_size*input_space_size*disturb_space_size) +i] += accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                    s = s+ accdT[(val*state_space_size*input_space_size*disturb_space_size) +i];
                                }
                            }

                            // maximize transitions to avoid set
                            // no need to add code here since its the rest of the probabilities and doesnt add to the output
                            accAvoidM[i] = (1.0-s);
                            cdfAccessor0[i] =  temp0;

                        });
                    });
                }
                queue.wait_and_throw();

                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);

                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                first0 = conv_to< colvec >::from(max(input_and_state0,1));
                k++;

                for (size_t i = 0; i < state_space_size; ++i){
                    firstnew0.row(i).max(U_pos[i]);
                }
            }
            }
            cout << endl;
            cout << "control policy for lower bound found, finding upper bound." << endl;

            vec second0(state_space_size, 1, fill::zeros);
            mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
            cout << "second loop iterations: " << endl;
            mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
            vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
            vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);

            k=0;
            cout << "Create reduced matrix where input is fixed." << endl;
            for (size_t j = 0; j < disturb_space_size; j++){
                for (size_t i = 0; i < state_space_size; i++){
                    tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                    tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                }
            }

            cout << "Matrix Fixed" << endl;
            {
                mat diffT = tempTmax-tempTmin;
                vec diffR = tempTTmax - tempTTmin;
                vec diffA = tempATmax - tempATmin;
                sycl::queue Q;
                    sycl::buffer<double> bufminT(tempTmin.memptr(),tempTmin.n_rows*tempTmin.n_cols);
                    sycl::buffer<double> bufdT(diffT.memptr(),diffT.n_rows*diffT.n_cols);
                    sycl::buffer<double> bufminTT(tempTTmin.memptr(),tempTTmin.n_rows);
                    sycl::buffer<double> bufdTT(diffR.memptr(),diffR.n_rows);
                    sycl::buffer<double> bufminAT(tempATmin.memptr(),tempATmin.n_rows);
                    sycl::buffer<double> bufdAT(diffA.memptr(),diffA.n_rows);
            while (k<timeHorizon) {
                cout << "." << flush;
                std::vector<int> sorted_indices = IMPaCT_IO::getSortedIndices(second0, true);


                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<int> bufsort(sorted_indices.data(), sorted_indices.size());
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> bufs0(second0.memptr(),second0.n_rows);

                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto accsort = bufsort.get_access<sycl::access::mode::read>(cgh);
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto accs0 = bufs0.get_access<sycl::access::mode::read>(cgh);
                        auto accminT = bufminT.get_access<sycl::access::mode::read>(cgh);
                        auto accdT = bufdT.get_access<sycl::access::mode::read>(cgh);
                        auto accminTT = bufminTT.get_access<sycl::access::mode::read>(cgh);
                        auto accdTT = bufdTT.get_access<sycl::access::mode::read>(cgh);
                        auto accminAT = bufminAT.get_access<sycl::access::mode::read>(cgh);
                        auto accdAT = bufdAT.get_access<sycl::access::mode::read>(cgh);

                        //ASSUMING MAXIMAL LP SOLVING
                        cgh.parallel_for<class maxTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::id<1> i) {
                            // set base values to be equal to the minimal transition probabilities
                            double s;
                            double temp0;

                            temp0 = 0;
                            s = 0.0;

                            temp0 += accminTT[i];
                            s = s + accminTT[i];

                            s = s + accminAT[i];

                            for (size_t col = 0; col < state_space_size; col++) {
                                temp0 += accminT[(col*state_space_size*disturb_space_size) +i]*accs0[col];
                                s = s+ accminT[(col*state_space_size*disturb_space_size) +i];
                            }

                            //maximize transitions to avoid set
                            if ((1.0-s) <= accdAT[i]){
                                s = 1.0;
                            }else{
                                s+= accdAT[i];
                            }

                            //maximize transitions between states
                            for(size_t col = 0; col < state_space_size; col++){
                                size_t val = accsort[col];
                                if ((1.0-s) <= accdT[(val*state_space_size*disturb_space_size) +i]){
                                    temp0 += (1.0-s)*accs0[val];
                                    s = 1.0;
                                    break;
                                }else {
                                    temp0 += accdT[(val*state_space_size*disturb_space_size) +i]*accs0[val];
                                    s = s+ accdT[(val*state_space_size*disturb_space_size) +i];
                                }
                            }

                            //maximize transitions to target
                            if ((1.0-s) <= accdTT[i]){
                                temp0 += (1.0-s);
                            }else{
                                temp0 += accdTT[i];
                            }

                            cdfAccessor0[i] =  temp0;
                        });
                    });
                }
                Q.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                second0 = conv_to< colvec >::from(min(secondnew0,1));
                k++;

            }
            }
            cout << endl;
            cout << "Upper bound found." << endl;

            controller.set_size(state_space_size, dim_x + dim_u + 2);
            controller.cols(0,dim_x-1) = state_space;
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second0;
            for (size_t i = 0; i < state_space_size; ++i) {
                controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}