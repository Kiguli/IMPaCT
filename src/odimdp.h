#ifndef IMPACT_ODIMDP_H
#define IMPACT_ODIMDP_H

// ============================================================================
// Orthogonally-decoupled interval MDPs (odIMDPs): the transition ambiguity set of
// each (state, action) FACTORS as a product of per-dimension marginal interval
// distributions over a product state space S = S_1 x ... x S_n. Storing n marginals
// of size |S_d| instead of one joint distribution of size prod |S_d| is the
// memory/compute scalability feature of IntervalMDP.jl, and matches the structure
// IMPaCT's own sparse abstraction produces for decoupled dynamics (per-dimension
// transitionInterval1D bounds, src/abstraction.cpp).
//
// Robust Bellman backup: the joint opt over the product ambiguity decomposes into a
// RECURSIVE per-dimension O-maximization — reduce the value hyper-rectangle one
// dimension at a time, optimizing the dim-d marginal per prefix of the outer
// destination coordinates (nature's factored choice may depend on the already-
// realised coordinates; this per-prefix rectangular semantics is the odIMDP
// ambiguity set of Mathiesen-Haesaert-Laurenti).
//
// Refs: F. B. Mathiesen, S. Haesaert, L. Laurenti, "Scalable control synthesis for
// stochastic systems via structural IMDP abstractions" (arXiv:2411.11803); the
// per-marginal inner solve is O-maximization (Givan-Leach-Dean AIJ 2000). Oracle:
// IntervalMDP.jl's OrthogonalIntervalMarkovDecisionProcess on the identical model
// (benchmarks/crosstool/peers/odimdp_oracle.jl).
// ============================================================================

#include <vector>
#include <set>
#include <string>
#include "solve.h"

namespace impact {
namespace odimdp {

    // marginal[d] = interval distribution over dim-d destination coordinates 0..dims[d]-1
    using Marginals = std::vector<solve::ActionDist>;

    // One action = a MIXTURE of orthogonal components (Mathiesen-Haesaert-Laurenti,
    // arXiv:2411.11803): nature picks a weight vector within the `weights` intervals AND
    // each component's marginals within theirs; the backup is the per-component factored
    // O-max followed by an O-max over the mixture weights. A single component with empty
    // weights is the plain odIMDP case.
    struct Action {
        std::vector<Marginals> comps;   // >= 1 orthogonal components
        solve::ActionDist weights;      // interval distribution over component index (empty if 1 comp)
    };

    struct Model {
        std::vector<int> dims;                       // per-dimension sizes; nStates = prod
        std::vector<std::vector<Action>> actions;    // indexed by linearised state (dim 0 fastest)
        std::set<int> targets;                       // linearised target states
        int init = 0;
        long long nStates() const { long long n = 1; for (int d : dims) n *= d; return n; }
    };

    // Parse the .odimdp text format:
    //   odimdp / dims n1 n2 ... / init s / label NAME s... /
    //   otran  <s> <a> <d> to:lo:hi ...      (dim-d marginal of action a, single component)
    //   mtran  <s> <a> <k> <d> to:lo:hi ...  (dim-d marginal of mixture component k)
    //   mweight <s> <a> k:lo:hi ...          (interval mixture weights over components)
    Model parseFile(const std::string& path, const std::string& targetLabel);

    // Robust factored Bellman backup of value V (indexed linearised, dim 0 fastest) at
    // state s: max over actions of the recursive per-dimension opt (pess: nature Min).
    double backup(const Model& m, int s, const std::vector<double>& V, bool pessimistic);

    // Robust reachability VI to m.targets; returns per-state values.
    std::vector<double> reach(const Model& m, double eps, bool pessimistic, int* iters = nullptr);

} // namespace odimdp
} // namespace impact

#endif // IMPACT_ODIMDP_H
