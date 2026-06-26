#include "product.h"

namespace impact {
namespace product {

Product build(const solve::IMDPModel& m,
              const std::vector<Label>& labels,
              const ltl::DFA& dfa,
              int s0) {
    const int n = (int)m.size();
    const int nQ = dfa.nStates;
    Product P;
    P.nProd = n * nQ;
    P.model.assign((size_t)n * nQ, {});

    auto pidx = [&](int s, int q) { return s * nQ + q; };
    auto advance = [&](int q, int s) {                 // DFA step on entering state s
        return dfa.trans[q][ltl::letterIndex(dfa, labels[s])];
    };

    for (int s = 0; s < n; ++s) {
        for (int q = 0; q < nQ; ++q) {
            const int p = pidx(s, q);
            for (const solve::ActionDist& act : m[s]) {
                solve::ActionDist pa;
                pa.reserve(act.size());
                for (const solve::Interval& iv : act) {
                    int q2 = advance(q, iv.to);         // DFA reads the label of the entered state
                    pa.push_back({pidx(iv.to, q2), iv.lo, iv.hi});
                }
                P.model[p].push_back(std::move(pa));
            }
            if (dfa.accepting[q]) P.targets.insert(p);  // good-prefix: accepting => satisfied
        }
    }

    P.start = pidx(s0, advance(dfa.start, s0));          // read L(s0) first
    return P;
}

} // namespace product
} // namespace impact
