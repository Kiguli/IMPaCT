#include "ltl.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

namespace impact {
namespace ltl {

// --- AST -------------------------------------------------------------------
// Tail is an internal marker used only during DFA progression: it holds iff the
// remaining trace is non-empty (nu(Tail)=false; der(Tail,a)=true).
enum class Op { Atom, True, False, Not, And, Or, Implies, Iff, Next, Event, Glob, Until, Release, Tail };

struct Node {
    Op op;
    std::string ap;   // for Atom
    int a = -1;       // child / left
    int b = -1;       // right (binary)
};

struct Automaton {
    std::vector<Node> nodes;   // arena
    int root = -1;
    std::vector<std::string> aps;
};

namespace {

// --- Tokenizer -------------------------------------------------------------
enum class Tok { LParen, RParen, Not, And, Or, Implies, Iff,
                 Next, Event, Glob, Until, Release, Atom, True, False, End };

struct Token { Tok kind; std::string text; };

std::vector<Token> tokenize(const std::string& s) {
    std::vector<Token> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        char c = s[i];
        if (std::isspace((unsigned char)c)) { ++i; continue; }
        switch (c) {
            case '(': out.push_back({Tok::LParen, "("}); ++i; continue;
            case ')': out.push_back({Tok::RParen, ")"}); ++i; continue;
            case '!': case '~': out.push_back({Tok::Not, "!"}); ++i; continue;
            case '&': out.push_back({Tok::And, "&"}); i += (i+1<n && s[i+1]=='&') ? 2 : 1; continue;
            case '|': out.push_back({Tok::Or, "|"}); i += (i+1<n && s[i+1]=='|') ? 2 : 1; continue;
            case '-':
                if (i+1 < n && s[i+1] == '>') { out.push_back({Tok::Implies, "->"}); i += 2; continue; }
                throw std::invalid_argument("ltl: unexpected '-'");
            case '<':
                if (i+2 < n && s[i+1] == '-' && s[i+2] == '>') { out.push_back({Tok::Iff, "<->"}); i += 3; continue; }
                throw std::invalid_argument("ltl: unexpected '<'");
            default: break;
        }
        if (std::isalpha((unsigned char)c) || c == '_') {
            size_t j = i;
            while (j < n && (std::isalnum((unsigned char)s[j]) || s[j] == '_')) ++j;
            std::string id = s.substr(i, j - i);
            i = j;
            if (id == "true" || id == "True")       out.push_back({Tok::True, id});
            else if (id == "false" || id == "False") out.push_back({Tok::False, id});
            else if (id == "X") out.push_back({Tok::Next, id});
            else if (id == "F") out.push_back({Tok::Event, id});
            else if (id == "G") out.push_back({Tok::Glob, id});
            else if (id == "U") out.push_back({Tok::Until, id});
            else if (id == "R") out.push_back({Tok::Release, id});
            else out.push_back({Tok::Atom, id});
            continue;
        }
        throw std::invalid_argument(std::string("ltl: unexpected character '") + c + "'");
    }
    out.push_back({Tok::End, ""});
    return out;
}

// --- Recursive-descent parser ----------------------------------------------
// Precedence (loose -> tight): <-> , -> , | , & , (U,R) , unary(! X F G), atom.
struct Parser {
    const std::vector<Token>& t;
    size_t pos = 0;
    Automaton* aut;
    const std::unordered_set<std::string>& apset;

    Parser(const std::vector<Token>& toks, Automaton* a, const std::unordered_set<std::string>& aps)
        : t(toks), aut(a), apset(aps) {}

    const Token& peek() const { return t[pos]; }
    const Token& next() { return t[pos++]; }
    bool accept(Tok k) { if (t[pos].kind == k) { ++pos; return true; } return false; }

    int add(Node node) { aut->nodes.push_back(node); return (int)aut->nodes.size() - 1; }

    int parse() {
        int r = parseIff();
        if (peek().kind != Tok::End) throw std::invalid_argument("ltl: trailing tokens after formula");
        return r;
    }
    int parseIff() {
        int l = parseImplies();
        while (accept(Tok::Iff)) { int r = parseImplies(); l = add({Op::Iff, "", l, r}); }
        return l;
    }
    int parseImplies() {
        int l = parseOr();
        if (accept(Tok::Implies)) { int r = parseImplies(); return add({Op::Implies, "", l, r}); } // right assoc
        return l;
    }
    int parseOr() {
        int l = parseAnd();
        while (accept(Tok::Or)) { int r = parseAnd(); l = add({Op::Or, "", l, r}); }
        return l;
    }
    int parseAnd() {
        int l = parseTemporalBin();
        while (accept(Tok::And)) { int r = parseTemporalBin(); l = add({Op::And, "", l, r}); }
        return l;
    }
    int parseTemporalBin() {
        int l = parseUnary();
        while (peek().kind == Tok::Until || peek().kind == Tok::Release) {
            Tok k = next().kind;
            int r = parseUnary();
            l = add({k == Tok::Until ? Op::Until : Op::Release, "", l, r});
        }
        return l;
    }
    int parseUnary() {
        switch (peek().kind) {
            case Tok::Not:   next(); return add({Op::Not,   "", parseUnary(), -1});
            case Tok::Next:  next(); return add({Op::Next,  "", parseUnary(), -1});
            case Tok::Event: next(); return add({Op::Event, "", parseUnary(), -1});
            case Tok::Glob:  next(); return add({Op::Glob,  "", parseUnary(), -1});
            default: return parseAtom();
        }
    }
    int parseAtom() {
        const Token& tk = peek();
        if (tk.kind == Tok::LParen) {
            next();
            int r = parseIff();
            if (!accept(Tok::RParen)) throw std::invalid_argument("ltl: missing ')'");
            return r;
        }
        if (tk.kind == Tok::True)  { next(); return add({Op::True,  "", -1, -1}); }
        if (tk.kind == Tok::False) { next(); return add({Op::False, "", -1, -1}); }
        if (tk.kind == Tok::Atom) {
            if (apset.find(tk.text) == apset.end())
                throw std::invalid_argument("ltl: unknown atom '" + tk.text + "'");
            std::string name = tk.text; next();
            return add({Op::Atom, name, -1, -1});
        }
        throw std::invalid_argument("ltl: expected atom or '(' at token '" + tk.text + "'");
    }
};

// --- LTLf finite-trace semantics: does node hold at position i (0 <= i < n)? --
bool sat(const Automaton* A, const FiniteTrace& w, int i, int idx) {
    const int n = (int)w.size();
    const Node& nd = A->nodes[idx];
    switch (nd.op) {
        case Op::True:  return true;
        case Op::False: return false;
        case Op::Atom:  return w[i].count(nd.ap) > 0;
        case Op::Not:   return !sat(A, w, i, nd.a);
        case Op::And:   return sat(A, w, i, nd.a) && sat(A, w, i, nd.b);
        case Op::Or:    return sat(A, w, i, nd.a) || sat(A, w, i, nd.b);
        case Op::Implies: return !sat(A, w, i, nd.a) || sat(A, w, i, nd.b);
        case Op::Iff:   return sat(A, w, i, nd.a) == sat(A, w, i, nd.b);
        case Op::Next:  return (i + 1 < n) && sat(A, w, i + 1, nd.a);   // strong next
        case Op::Event:                                                 // F a
            for (int j = i; j < n; ++j) if (sat(A, w, j, nd.a)) return true;
            return false;
        case Op::Glob:                                                  // G a
            for (int j = i; j < n; ++j) if (!sat(A, w, j, nd.a)) return false;
            return true;
        case Op::Until:                                                 // a U b
            for (int j = i; j < n; ++j) {
                if (sat(A, w, j, nd.b)) return true;
                if (!sat(A, w, j, nd.a)) return false;
            }
            return false;
        case Op::Release: {                                             // a R b
            for (int j = i; j < n; ++j) {
                if (!sat(A, w, j, nd.b)) return false;
                if (sat(A, w, j, nd.a)) return true;
            }
            return true;   // b held throughout (release never triggered) => holds on finite trace
        }
        case Op::Tail: return false;   // internal DFA marker; never in a parsed formula
    }
    return false;
}

} // namespace

Automaton* compileFinite(const std::string& formula, const std::vector<std::string>& aps) {
    Automaton* a = new Automaton();
    a->aps = aps;
    std::unordered_set<std::string> apset(aps.begin(), aps.end());
    try {
        std::vector<Token> toks = tokenize(formula);
        Parser p(toks, a, apset);
        a->root = p.parse();
    } catch (...) {
        delete a;
        throw;
    }
    return a;
}

bool acceptsFinite(const Automaton* a, const FiniteTrace& trace) {
    if (a == nullptr || a->root < 0) return false;
    if (trace.empty()) return false;            // LTLf over the empty trace: tests use n>=1
    return sat(a, trace, 0, a->root);
}

void destroy(Automaton* a) { delete a; }

// ===========================================================================
// LTLf -> DFA via formula progression (Brzozowski-style derivatives).
//
// der(phi, a)  : residual formula the remaining suffix must satisfy after letter a
// nu(phi)      : does phi hold on the EMPTY suffix (accepting predicate)
// The X operator is "strong next"; the marker Tail (non-empty remaining trace)
// encodes that: X phi -> phi & Tail. Correctness is validated differentially
// against acceptsFinite. States are interned canonical formulas, so the residual
// set is finite (boolean combinations of subformula anchors + Tail).
// ===========================================================================
namespace {

struct Builder {
    std::vector<Node> nodes;
    std::vector<std::string> key;                 // canonical key per node
    std::unordered_map<std::string, int> intern;
    int TRUE_, FALSE_, TAIL_;

    Builder() {
        TRUE_  = mk({Op::True,  "", -1, -1}, "T");
        FALSE_ = mk({Op::False, "", -1, -1}, "F");
        TAIL_  = mk({Op::Tail,  "", -1, -1}, "@");
    }
    int mk(Node n, const std::string& k) {
        auto it = intern.find(k);
        if (it != intern.end()) return it->second;
        int id = (int)nodes.size();
        nodes.push_back(n); key.push_back(k); intern[k] = id;
        return id;
    }
    int mkAtom(const std::string& p) { return mk({Op::Atom, p, -1, -1}, "p:" + p); }

    int mkNot(int a) {
        if (a == TRUE_) return FALSE_;
        if (a == FALSE_) return TRUE_;
        if (nodes[a].op == Op::Not) return nodes[a].a;       // double negation
        return mk({Op::Not, "", a, -1}, "~(" + key[a] + ")");
    }
    void collect(Op op, int idx, std::vector<int>& out) {
        if (nodes[idx].op == op) { collect(op, nodes[idx].a, out); collect(op, nodes[idx].b, out); }
        else out.push_back(idx);
    }
    int assoc(Op op, std::vector<int> ops) {            // op in {And, Or}
        const int ANNI = (op == Op::And) ? FALSE_ : TRUE_;   // annihilator
        const int IDEN = (op == Op::And) ? TRUE_ : FALSE_;   // identity
        std::vector<int> flat;
        for (int o : ops) collect(op, o, flat);
        std::vector<int> kept;
        std::unordered_map<std::string, int> seen;
        for (int o : flat) {
            if (o == ANNI) return ANNI;
            if (o == IDEN) continue;
            if (seen.insert({key[o], o}).second) kept.push_back(o);
        }
        if (kept.empty()) return IDEN;
        std::sort(kept.begin(), kept.end(), [&](int x, int y) { return key[x] < key[y]; });
        if (kept.size() == 1) return kept[0];
        const char* sym = (op == Op::And) ? "&" : "|";
        int acc = kept.back();
        for (int i = (int)kept.size() - 2; i >= 0; --i) {
            std::string k = std::string(sym) + "(" + key[kept[i]] + "," + key[acc] + ")";
            acc = mk({op, "", kept[i], acc}, k);
        }
        return acc;
    }
    int mkAnd(int a, int b) { return assoc(Op::And, {a, b}); }
    int mkOr (int a, int b) { return assoc(Op::Or,  {a, b}); }
    int mkUn(Op op, const char* sym, int a) { return mk({op, "", a, -1}, std::string(sym) + "(" + key[a] + ")"); }
    int mkBin(Op op, const char* sym, int a, int b) {
        return mk({op, "", a, b}, std::string(sym) + "(" + key[a] + "," + key[b] + ")");
    }

    // Import an Automaton AST node into this builder (expanding ->, <->).
    int import(const Automaton* A, int idx) {
        const Node& n = A->nodes[idx];
        switch (n.op) {
            case Op::Atom:    return mkAtom(n.ap);
            case Op::True:    return TRUE_;
            case Op::False:   return FALSE_;
            case Op::Not:     return mkNot(import(A, n.a));
            case Op::And:     return mkAnd(import(A, n.a), import(A, n.b));
            case Op::Or:      return mkOr(import(A, n.a), import(A, n.b));
            case Op::Implies: return mkOr(mkNot(import(A, n.a)), import(A, n.b));
            case Op::Iff: { int a = import(A, n.a), b = import(A, n.b);
                            return mkOr(mkAnd(a, b), mkAnd(mkNot(a), mkNot(b))); }
            case Op::Next:    return mkUn(Op::Next,  "X", import(A, n.a));
            case Op::Event:   return mkUn(Op::Event, "F", import(A, n.a));
            case Op::Glob:    return mkUn(Op::Glob,  "G", import(A, n.a));
            case Op::Until:   return mkBin(Op::Until,   "U", import(A, n.a), import(A, n.b));
            case Op::Release: return mkBin(Op::Release, "R", import(A, n.a), import(A, n.b));
            case Op::Tail:    return TAIL_;
        }
        return FALSE_;
    }

    bool nu(int idx) {
        const Node& n = nodes[idx];
        switch (n.op) {
            case Op::True: return true;   case Op::False: return false;
            case Op::Tail: return false;  case Op::Atom: return false;
            case Op::Not: return !nu(n.a);
            case Op::And: return nu(n.a) && nu(n.b);
            case Op::Or:  return nu(n.a) || nu(n.b);
            case Op::Next: return false;  case Op::Event: return false;
            case Op::Glob: return true;   case Op::Until: return false;
            case Op::Release: return true;
            case Op::Implies: case Op::Iff: return false;  // expanded away on import
        }
        return false;
    }

    int der(int idx, unsigned letter, const std::unordered_map<std::string,int>& apbit) {
        const Op op = nodes[idx].op;       // read fields before the arena can grow
        const int a = nodes[idx].a, b = nodes[idx].b;
        const std::string ap = nodes[idx].ap;
        switch (op) {
            case Op::True: case Op::Tail: return TRUE_;   // Tail: remaining (after a) seen non-empty
            case Op::False: return FALSE_;
            case Op::Atom: { auto it = apbit.find(ap);
                             bool on = (it != apbit.end()) && ((letter >> it->second) & 1u);
                             return on ? TRUE_ : FALSE_; }
            case Op::Not: return mkNot(der(a, letter, apbit));
            case Op::And: return mkAnd(der(a, letter, apbit), der(b, letter, apbit));
            case Op::Or:  return mkOr(der(a, letter, apbit), der(b, letter, apbit));
            case Op::Next:  return mkAnd(a, TAIL_);                                // X phi -> phi & Tail
            case Op::Event: return mkOr(der(a, letter, apbit), mkAnd(idx, TAIL_)); // F = phi | (F & Tail)
            case Op::Glob:  return mkAnd(der(a, letter, apbit), mkOr(idx, mkNot(TAIL_)));
            case Op::Until: return mkOr(der(b, letter, apbit),
                                        mkAnd(der(a, letter, apbit), mkAnd(idx, TAIL_)));
            case Op::Release: return mkAnd(der(b, letter, apbit),
                                           mkOr(der(a, letter, apbit), mkOr(idx, mkNot(TAIL_))));
            case Op::Implies: case Op::Iff: return FALSE_;  // expanded away on import
        }
        return FALSE_;
    }

    // --- Semantic canonicalization over anchor subformulas --------------------
    // A residual is a Boolean function of "anchors" (atoms, temporal subformulas,
    // and Tail). der is a Boolean homomorphism over anchors, so two residuals
    // with the same truth table over anchors are interchangeable => keying DFA
    // states by truth table gives a finite, minimal state set (no syntactic blow-up).
    std::unordered_map<int,int> abit;     // anchor node id -> bit index
    int K = 0;
    static bool isAnchorOp(Op o) {
        return o == Op::Atom || o == Op::Next || o == Op::Event ||
               o == Op::Glob || o == Op::Until || o == Op::Release || o == Op::Tail;
    }
    void collectAnchors() {
        for (int i = 0; i < (int)nodes.size(); ++i)
            if (isAnchorOp(nodes[i].op) && !abit.count(i)) abit[i] = K++;
    }
    bool evalStruct(int idx, unsigned m) {
        const Node& n = nodes[idx];
        if (isAnchorOp(n.op)) return (m >> abit.at(idx)) & 1u;
        switch (n.op) {
            case Op::True: return true;   case Op::False: return false;
            case Op::Not: return !evalStruct(n.a, m);
            case Op::And: return evalStruct(n.a, m) && evalStruct(n.b, m);
            case Op::Or:  return evalStruct(n.a, m) || evalStruct(n.b, m);
            case Op::Implies: return !evalStruct(n.a, m) || evalStruct(n.b, m);
            case Op::Iff: return evalStruct(n.a, m) == evalStruct(n.b, m);
            default: return false;
        }
    }
    std::string truthKey(int idx) {                 // canonical key = packed truth table
        const unsigned N = 1u << K;
        std::string s((N + 7) / 8, '\0');
        for (unsigned m = 0; m < N; ++m)
            if (evalStruct(idx, m)) s[m >> 3] |= (char)(1u << (m & 7u));
        return s;
    }
    unsigned nuAssignment() {                        // empty-trace values of anchors
        unsigned m = 0;                              // G,R hold on empty; atoms,X,F,U,Tail don't
        for (auto& kv : abit) {
            Op o = nodes[kv.first].op;
            if (o == Op::Glob || o == Op::Release) m |= (1u << kv.second);
        }
        return m;
    }
};

} // namespace

DFA toDFA(const Automaton* A, int maxStates) {
    DFA dfa;
    if (A == nullptr || A->root < 0) { dfa.nStates = 1; dfa.start = 0; dfa.trans = {{}}; dfa.accepting = {0}; return dfa; }
    dfa.aps = A->aps;
    const int m = (int)dfa.aps.size();
    const unsigned nLetters = (m >= 31) ? 0u : (1u << m);  // guard
    std::unordered_map<std::string,int> apbit;
    for (int i = 0; i < m; ++i) apbit[dfa.aps[i]] = i;

    Builder B;
    int rootB = B.import(A, A->root);
    B.collectAnchors();
    if (B.K > 24)  // truth table is 2^K bits; 24 anchors is already enormous
        throw std::runtime_error("ltl::toDFA: too many anchor subformulas (" + std::to_string(B.K) + ")");
    const unsigned nuM = B.nuAssignment();

    // States are canonical truth tables (semantic dedup). Keep a representative
    // builder node per state to apply der to.
    std::unordered_map<std::string,int> stateId;          // truth-table key -> DFA state
    std::vector<int> stateNode;                           // DFA state -> representative builder idx
    auto getState = [&](int bidx) {
        std::string k = B.truthKey(bidx);
        auto it = stateId.find(k);
        if (it != stateId.end()) return it->second;
        int id = (int)stateNode.size();
        stateId[std::move(k)] = id; stateNode.push_back(bidx);
        return id;
    };

    int s0 = getState(rootB);
    for (int s = 0; s < (int)stateNode.size(); ++s) {
        if ((int)stateNode.size() > maxStates)
            throw std::runtime_error("ltl::toDFA: state explosion (formula too large)");
        std::vector<int> row(nLetters, 0);
        for (unsigned L = 0; L < nLetters; ++L)
            row[L] = getState(B.der(stateNode[s], L, apbit));
        dfa.trans.push_back(std::move(row));
    }
    dfa.nStates = (int)stateNode.size();
    dfa.start = s0;
    dfa.accepting.resize(dfa.nStates);
    for (int s = 0; s < dfa.nStates; ++s)
        dfa.accepting[s] = B.evalStruct(stateNode[s], nuM) ? 1 : 0;  // accept iff holds on empty suffix
    return dfa;
}

int letterIndex(const DFA& dfa, const Letter& letter) {
    int idx = 0;
    for (int i = 0; i < (int)dfa.aps.size(); ++i)
        if (letter.count(dfa.aps[i])) idx |= (1 << i);
    return idx;
}

bool dfaAccepts(const DFA& dfa, const FiniteTrace& trace) {
    int s = dfa.start;
    for (const Letter& a : trace) s = dfa.trans[s][letterIndex(dfa, a)];
    return dfa.accepting[s] != 0;
}

} // namespace ltl
} // namespace impact
