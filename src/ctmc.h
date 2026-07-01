#ifndef IMPACT_CTMC_H
#define IMPACT_CTMC_H

// ============================================================================
// Continuous-time Markov chain (CTMC) analysis by UNIFORMISATION.
//
// A CTMC is given in the neutral .imdp format with a `ctmc` header: each state has a
// single "action" whose entries `to:r:r` are the transition RATES s->to (point, lo==hi==r;
// no self rate). Uniformisation (Jensen) builds the embedded DTMC P = I + Q/Λ with
// Λ = max exit rate; this DTMC has the SAME stationary distribution as the CTMC, so the
// steady-state (S) operator reuses IMPaCT's long-run-average solver on P. Time-bounded
// CSL reachability P(F<=t goal) is the Poisson-weighted (Fox-Glynn) transient of P with
// `goal` made absorbing.
//
// Refs: C. Baier, B. Haverkort, H. Hermanns, J.-P. Katoen, "Model-Checking Algorithms for
// Continuous-Time Markov Chains", IEEE Trans. Software Eng. 29(6):524-541, 2003
// (DOI 10.1109/TSE.2003.1205180); B. L. Fox, P. W. Glynn, "Computing Poisson probabilities",
// CACM 31(4):440-445, 1988 (the numerically stable Poisson truncation).
//
// The ROBUST/interval extension (uncertain rates -> interval CTMC) is IMPaCT's niche and a
// natural follow-up: uniformise the rate intervals and resolve them by O-maximisation.
// ============================================================================

#include <vector>
#include <set>
#include "solve.h"

namespace impact {
namespace ctmc {

    struct Uniformized { solve::IMDPModel dtmc; double lambda; };

    // Treat each state's single action as outgoing RATES and uniformise to a DTMC.
    Uniformized uniformize(const solve::IMDPModel& rateModel);

    // CSL time-bounded reachability P(F<=t goal), per state (goal absorbing).
    std::vector<double> timeBoundedReach(const Uniformized& u, const std::set<int>& goal,
                                         double t, double eps);

} // namespace ctmc
} // namespace impact

#endif // IMPACT_CTMC_H
