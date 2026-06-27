#include "ltlspec.h"
#include "pctl.h"
#include "omega.h"

#include <memory>
#include <vector>
#include <stdexcept>
#include <cctype>

namespace impact {
namespace ltlspec {

namespace {

struct Node {
    enum K { Atom, True, False, Not, And, Or, Impl, Next, Fin, Glob, Until } k;
    std::string name;
    std::shared_ptr<Node> a, b;
};
using NP = std::shared_ptr<Node>;
NP mk(Node::K k, NP a = nullptr, NP b = nullptr) { auto n = std::make_shared<Node>(); n->k = k; n->a = a; n->b = b; return n; }
NP atom(const std::string& s) { auto n = std::make_shared<Node>(); n->k = Node::Atom; n->name = s; return n; }

// ---- tokenizer --------------------------------------------------------------
std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> t;
    size_t i = 0;
    while (i < s.size()) {
        char c = s[i];
        if (std::isspace((unsigned char)c) || c == ',') { ++i; continue; }   // commas treated as separators
        if (c == '-' && i + 1 < s.size() && s[i + 1] == '>') { t.push_back("->"); i += 2; continue; }
        if (c == '&' && i + 1 < s.size() && s[i + 1] == '&') { t.push_back("&"); i += 2; continue; }
        if (c == '|' && i + 1 < s.size() && s[i + 1] == '|') { t.push_back("|"); i += 2; continue; }
        if (c == '(' || c == ')' || c == '!' || c == '&' || c == '|') { t.push_back(std::string(1, c)); ++i; continue; }
        if (std::isalnum((unsigned char)c) || c == '_') {
            size_t j = i; while (j < s.size() && (std::isalnum((unsigned char)s[j]) || s[j] == '_')) ++j;
            t.push_back(s.substr(i, j - i)); i = j; continue;
        }
        throw std::runtime_error(std::string("ltlspec: bad character '") + c + "'");
    }
    return t;
}

bool isOp(const std::string& s) { return s == "X" || s == "F" || s == "G" || s == "U"; }

struct Parser {
    std::vector<std::string> t;
    size_t p = 0;
    const std::string& peek() { static std::string e; return p < t.size() ? t[p] : e; }
    std::string next() { return p < t.size() ? t[p++] : std::string(); }
    bool eat(const std::string& s) { if (peek() == s) { ++p; return true; } return false; }

    NP parse() { NP n = pImpl(); if (p != t.size()) throw std::runtime_error("ltlspec: trailing tokens"); return n; }
    NP pImpl() { NP a = pOr(); while (eat("->")) a = mk(Node::Impl, a, pOr()); return a; }
    NP pOr()   { NP a = pAnd(); while (eat("|")) a = mk(Node::Or, a, pAnd()); return a; }
    NP pAnd()  { NP a = pUntil(); while (eat("&")) a = mk(Node::And, a, pUntil()); return a; }
    NP pUntil(){ NP a = pUnary(); while (peek() == "U") { ++p; a = mk(Node::Until, a, pUnary()); } return a; }
    NP pUnary() {
        std::string x = peek();
        if (x == "!") { ++p; return mk(Node::Not, pUnary()); }
        if (x == "X") { ++p; return mk(Node::Next, pUnary()); }
        if (x == "F") { ++p; return mk(Node::Fin, pUnary()); }
        if (x == "G") { ++p; return mk(Node::Glob, pUnary()); }
        return pAtom();
    }
    NP pAtom() {
        std::string x = next();
        if (x == "(") { NP n = pImpl(); if (!eat(")")) throw std::runtime_error("ltlspec: missing ')'"); return n; }
        if (x.empty()) throw std::runtime_error("ltlspec: unexpected end of formula");
        if (x == "true")  return mk(Node::True);
        if (x == "false") return mk(Node::False);
        if (isOp(x))      throw std::runtime_error("ltlspec: operator '" + x + "' used as atom");
        return atom(x);
    }
};

// ---- state-formula evaluation (boolean over atoms) --------------------------
std::set<int> stateSet(const NP& n, const Labels& L, int N) {
    auto all = [&]{ std::set<int> s; for (int i = 0; i < N; ++i) s.insert(i); return s; };
    switch (n->k) {
        case Node::True:  return all();
        case Node::False: return {};
        case Node::Atom: {
            auto it = L.find(n->name);
            if (it == L.end()) throw std::runtime_error("ltlspec: unknown atom '" + n->name + "'");
            return it->second;
        }
        case Node::Not: { std::set<int> s = stateSet(n->a, L, N), r; for (int i = 0; i < N; ++i) if (!s.count(i)) r.insert(i); return r; }
        case Node::And: { std::set<int> x = stateSet(n->a, L, N), y = stateSet(n->b, L, N), r; for (int v : x) if (y.count(v)) r.insert(v); return r; }
        case Node::Or:  { std::set<int> r = stateSet(n->a, L, N), y = stateSet(n->b, L, N); r.insert(y.begin(), y.end()); return r; }
        case Node::Impl:{ NP e = mk(Node::Or, mk(Node::Not, n->a), n->b); return stateSet(e, L, N); }
        default: throw std::runtime_error("ltlspec: out of supported fragment — a temporal operator appears where a state formula is required (arbitrary LTL needs the LDBA route, ISSUE-0016)");
    }
}

bool isGF(const NP& n, NP& inner) { if (n->k == Node::Glob && n->a->k == Node::Fin) { inner = n->a->a; return true; } return false; }

solve::IntervalResult indicator(const std::set<int>& s, int N) {
    solve::IntervalResult r; r.lower.assign(N, 0.0); r.upper.assign(N, 0.0); r.iterations = 0;
    for (int v : s) if (v >= 0 && v < N) { r.lower[v] = 1.0; r.upper[v] = 1.0; }
    return r;
}

void flattenAnd(const NP& n, std::vector<NP>& out) {
    if (n->k == Node::And) { flattenAnd(n->a, out); flattenAnd(n->b, out); } else out.push_back(n);
}

} // namespace

solve::IntervalResult synthesize(const solve::IMDPModel& m, const Labels& labels,
                                 const std::string& formula, bool pess, double eps) {
    const int N = (int)m.size();
    Parser ps; ps.t = tokenize(formula);
    NP root = ps.parse();

    auto outOfFragment = [] {
        throw std::runtime_error("ltlspec: out of supported fragment — needs an LDBA translation (Spot/Owl); see ISSUE-0016");
        return solve::IntervalResult{};
    };

    switch (root->k) {
        case Node::Fin: {
            if (root->a->k == Node::Glob)   // F G phi -> persistence
                return pess ? omega::maxPersistencePessimistic(m, stateSet(root->a->a, labels, N), eps)
                            : omega::maxPersistenceOptimistic (m, stateSet(root->a->a, labels, N), eps);
            return pess ? solve::maxReachPessimistic(m, stateSet(root->a, labels, N), eps)   // F phi -> reach
                        : solve::maxReachOptimistic (m, stateSet(root->a, labels, N), eps);
        }
        case Node::Glob: {
            if (root->a->k == Node::Fin)    // G F phi -> Büchi (recurrence)
                return pess ? omega::maxBuchiPessimistic(m, stateSet(root->a->a, labels, N), eps)
                            : omega::maxBuchiOptimistic (m, stateSet(root->a->a, labels, N), eps);
            // G phi -> stay in phi (safety): avoid the complement of phi
            std::set<int> phi = stateSet(root->a, labels, N), avoid;
            for (int i = 0; i < N; ++i) if (!phi.count(i)) avoid.insert(i);
            return pess ? solve::maxSafetyPessimistic(m, avoid, eps)
                        : solve::maxSafetyOptimistic (m, avoid, eps);
        }
        case Node::Until:
            return pess ? pctl::untilPessimistic(m, stateSet(root->a, labels, N), stateSet(root->b, labels, N), eps)
                        : pctl::untilOptimistic (m, stateSet(root->a, labels, N), stateSet(root->b, labels, N), eps);
        case Node::Next:
            return pess ? pctl::nextPessimistic(m, stateSet(root->a, labels, N), eps)
                        : pctl::nextOptimistic (m, stateSet(root->a, labels, N), eps);
        case Node::And: {
            // patrol iff every conjunct is G F (state formula)
            std::vector<NP> cs; flattenAnd(root, cs);
            std::vector<std::set<int>> sets; NP inner;
            for (const NP& c : cs) { if (!isGF(c, inner)) return outOfFragment(); sets.push_back(stateSet(inner, labels, N)); }
            return pess ? omega::maxGenBuchiPessimistic(m, sets, eps)
                        : omega::maxGenBuchiOptimistic (m, sets, eps);
        }
        case Node::Atom: case Node::True: case Node::False: case Node::Not: case Node::Or: case Node::Impl:
            return indicator(stateSet(root, labels, N), N);   // pure state formula -> 0/1 indicator
    }
    return outOfFragment();
}

} // namespace ltlspec
} // namespace impact
