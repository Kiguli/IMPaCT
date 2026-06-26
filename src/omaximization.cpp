#include "omaximization.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace impact {
namespace omax {

namespace {
    constexpr double kFeasTol = 1e-12;
}

Result optimize(const std::vector<double>& lower,
                const std::vector<double>& upper,
                const std::vector<double>& V,
                Sense sense) {
    const std::size_t n = V.size();
    if (n == 0 || lower.size() != n || upper.size() != n) {
        throw std::invalid_argument("omax::optimize: empty or mismatched vector sizes");
    }

    double sum_lo = 0.0, sum_up = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (lower[i] > upper[i]) {
            throw std::invalid_argument("omax::optimize: lower[i] > upper[i]");
        }
        sum_lo += lower[i];
        sum_up += upper[i];
    }
    if (sum_lo > 1.0 + kFeasTol || sum_up < 1.0 - kFeasTol) {
        throw std::invalid_argument("omax::optimize: infeasible interval box (no distribution sums to 1)");
    }

    // Start every successor at its lower bound; distribute the residual mass.
    std::vector<double> p(lower);
    double residual = 1.0 - sum_lo;
    if (residual < 0.0) residual = 0.0;  // clamp fp noise when sum_lo ~ 1

    // Allocation order: lowest value first to minimize, highest first to maximize.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return (sense == Sense::Min) ? (V[a] < V[b]) : (V[a] > V[b]);
    });

    for (std::size_t idx : order) {
        if (residual <= 0.0) break;
        const double headroom = upper[idx] - lower[idx];
        const double take = std::min(headroom, residual);
        p[idx] += take;
        residual -= take;
    }

    double value = 0.0;
    for (std::size_t i = 0; i < n; ++i) value += p[i] * V[i];
    return Result{std::move(p), value};
}

} // namespace omax
} // namespace impact
