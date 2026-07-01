#ifndef IMPACT_LTL_SPOT_H
#define IMPACT_LTL_SPOT_H

// ============================================================================
// Arbitrary-LTL robust synthesis on interval MDPs via a deterministic automaton.
//
// The LTL formula is translated to a DETERMINISTIC transition-based (generalized)
// Büchi automaton by Spot's `ltl2tgba -D` (Duret-Lutz et al., "Spot 2.0"), parsed
// from HOA, and synchronously producted with the IMDP; the product's acceptance is
// (generalized) Büchi, solved robustly by the EXISTING robust ω-regular engine
// (omega::maxGenBuchi{Pessimistic,Optimistic}). This lifts IMPaCT from the hard-coded
// F/G/U/X/GF/FG/patrol fragment to any LTL formula whose deterministic automaton has
// (generalized) Büchi acceptance — the "deterministic Büchi expressible" class.
//
// Refs: Sickert, Esparza, Jaax, Křetínský, "Limit-Deterministic Büchi Automata for
// LTL", CAV 2016 (the LDBA/product route); Hahn, Perez, Schewe, Somenzi, Trivedi,
// Wojtczak, "Good-for-MDPs Automata", TACAS 2020 (a deterministic automaton is always
// good-for-MDPs, so the product value is the LTL value); Baier & Katoen, Principles of
// Model Checking, Ch. 10 (automata-theoretic model checking).
//
// A formula whose deterministic automaton is NOT (generalized) Büchi (e.g. needs
// Rabin/parity or co-Büchi `Fin`) throws std::runtime_error — that needs an LDBA
// (Owl) / robust parity solver, tracked in ISSUE-0016.
// ============================================================================

#include <string>
#include <map>
#include <set>
#include "solve.h"

namespace impact {
namespace ltlspot {

    // Solve robust LTL. `labels` maps atomic-proposition name -> state set (the .imdp
    // labels). `ltl2tgba` is the command to run Spot's translator (may embed
    // LD_LIBRARY_PATH etc.); default "ltl2tgba". Returns the [lower,upper] value with the
    // reported value at the initial product state. pessimistic = robust nature.
    solve::IntervalResult synthesizeLTL(const solve::IMDPModel& m,
                                        const std::map<std::string, std::set<int>>& labels,
                                        int init, const std::string& formula,
                                        bool pessimistic, double eps,
                                        const std::string& ltl2tgba = "ltl2tgba");

} // namespace ltlspot
} // namespace impact

#endif // IMPACT_LTL_SPOT_H
