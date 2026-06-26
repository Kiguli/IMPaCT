#include "ltl.h"

#include <cctype>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <string>

namespace impact {
namespace ltl {

// --- AST -------------------------------------------------------------------
enum class Op { Atom, True, False, Not, And, Or, Implies, Iff, Next, Event, Glob, Until, Release };

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

} // namespace ltl
} // namespace impact
