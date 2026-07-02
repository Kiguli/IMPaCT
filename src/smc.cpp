#include "smc.h"

#include <random>
#include <cmath>
#include <stdexcept>

namespace impact {
namespace smc {

namespace {
// one sampled path: does it hit `target` within `horizon` steps from `init`?
bool samplePath(const solve::IMDPModel& m, const std::set<int>& target, int init,
                int horizon, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    int s = init;
    for (int k = 0; k <= horizon; ++k) {
        if (target.count(s)) return true;
        if (m[s].empty()) return false;
        if (m[s].size() != 1) throw std::runtime_error("smc: model must be a point DTMC (one action per state)");
        const solve::ActionDist& d = m[s][0];
        double u = unif(rng), acc = 0.0;
        int next = d.empty() ? s : d.back().to;
        for (const solve::Interval& iv : d) {
            if (iv.lo != iv.hi) throw std::runtime_error("smc: model must be a point chain (lo==hi); "
                                                         "simulate interval models via their solved bounds");
            acc += iv.lo;
            if (u <= acc) { next = iv.to; break; }
        }
        if (next == s && d.size() == 1 && d[0].to == s) return target.count(s) > 0;  // absorbing
        s = next;
    }
    return false;
}
} // namespace

Estimate estimateReach(const solve::IMDPModel& m, const std::set<int>& target,
                       int init, int horizon, long long samples, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    long long succ = 0;
    for (long long i = 0; i < samples; ++i)
        if (samplePath(m, target, init, horizon, rng)) ++succ;
    const double n = (double)samples, ph = succ / n;
    // Wilson 95% CI
    const double z = 1.959963985, z2 = z * z;
    const double denom = 1.0 + z2 / n;
    const double centre = (ph + z2 / (2 * n)) / denom;
    const double half = (z / denom) * std::sqrt(ph * (1 - ph) / n + z2 / (4 * n * n));
    // APMC / Chernoff-Hoeffding half-width for confidence 1-delta, delta=0.05:
    // P(|ph - p| >= eps) <= 2 exp(-2 n eps^2)  =>  eps = sqrt(ln(2/delta)/(2n)).
    const double eps = std::sqrt(std::log(2.0 / 0.05) / (2.0 * n));
    return { ph, std::max(0.0, centre - half), std::min(1.0, centre + half), eps, succ, samples };
}

int sprt(const solve::IMDPModel& m, const std::set<int>& target, int init, int horizon,
         double theta, double delta, long long maxSamples, std::uint64_t seed,
         long long* samplesUsed) {
    // Wald SPRT: H0 p = p0 = theta+delta vs H1 p = p1 = theta-delta; alpha = beta = 0.01.
    const double alpha = 0.01, beta = 0.01;
    const double p0 = std::min(1.0 - 1e-12, theta + delta), p1 = std::max(1e-12, theta - delta);
    const double logA = std::log((1 - beta) / alpha);      // accept H1 (reject) above
    const double logB = std::log(beta / (1 - alpha));      // accept H0 below
    std::mt19937_64 rng(seed);
    double llr = 0.0;                                       // log likelihood ratio  log(P1/P0)
    for (long long i = 1; i <= maxSamples; ++i) {
        const bool hit = samplePath(m, target, init, horizon, rng);
        llr += hit ? std::log(p1 / p0) : std::log((1 - p1) / (1 - p0));
        if (llr >= logA) { if (samplesUsed) *samplesUsed = i; return -1; }   // p <= theta-delta
        if (llr <= logB) { if (samplesUsed) *samplesUsed = i; return +1; }   // p >= theta+delta
    }
    if (samplesUsed) *samplesUsed = maxSamples;
    return 0;
}

} // namespace smc
} // namespace impact
