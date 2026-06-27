#include "expr.h"

#include <stdexcept>
#include <cmath>
#include <cctype>
#include <algorithm>

namespace impact {
namespace expr {

using abstraction::Ival;
using abstraction::isquare;

// AST node (Expr = shared_ptr<Node>, forward-declared in the header).
struct Node {
    enum K { Num, Var, Add, Sub, Mul, Div, Neg, Pow, Fn } k;
    double num = 0; int var = -1; int powN = 0; std::string fn; Expr a, b;
};

namespace {

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("expr: " + m); }

// ---- sound interval elementary operations ----------------------------------
Ival idiv(const Ival& a, const Ival& b) {
    if (b.lo <= 0.0 && b.hi >= 0.0) fail("interval division by an interval containing 0");
    double c1=a.lo/b.lo, c2=a.lo/b.hi, c3=a.hi/b.lo, c4=a.hi/b.hi;
    return Ival(std::min(std::min(c1,c2),std::min(c3,c4)), std::max(std::max(c1,c2),std::max(c3,c4)));
}
Ival ineg(const Ival& a) { return Ival(-a.hi, -a.lo); }
Ival ipow(const Ival& a, int n) {                 // n >= 0; tight even powers via isquare
    if (n == 0) return Ival(1.0);
    if (n == 1) return a;
    if (n % 2 == 0) { Ival h = ipow(a, n/2); return isquare(h); }
    return a * ipow(a, n-1);
}
Ival iexp(const Ival& a) { return Ival(std::exp(a.lo), std::exp(a.hi)); }
Ival isqrtI(const Ival& a) { double lo=a.lo<0?0:a.lo, hi=a.hi<0?0:a.hi; return Ival(std::sqrt(lo), std::sqrt(hi)); }
Ival iabs(const Ival& a) {
    if (a.lo >= 0) return a;
    if (a.hi <= 0) return Ival(-a.hi, -a.lo);
    return Ival(0.0, std::max(-a.lo, a.hi));
}
Ival itrig(const Ival& a, bool isSin) {           // sin/cos over an interval (scan endpoints + extrema)
    if (a.hi - a.lo >= 2*M_PI) return Ival(-1.0, 1.0);
    auto f = [&](double x){ return isSin ? std::sin(x) : std::cos(x); };
    double lo = std::min(f(a.lo), f(a.hi)), hi = std::max(f(a.lo), f(a.hi));
    double base = isSin ? M_PI/2 : 0.0;            // extrema of sin at pi/2+k*pi, cos at k*pi
    long k0 = (long)std::floor((a.lo - base) / M_PI) - 1;
    for (long k = k0; k <= k0 + 4; ++k) {
        double x = base + k*M_PI;
        if (x >= a.lo && x <= a.hi) { lo = std::min(lo, f(x)); hi = std::max(hi, f(x)); }
    }
    return Ival(std::max(-1.0, lo), std::min(1.0, hi));
}

// ---- tokenizer -------------------------------------------------------------
struct Tok { char kind; std::string s; double num; };   // kind: 'n' num, 'i' ident, 'o' operator/paren
std::vector<Tok> lex(const std::string& s) {
    std::vector<Tok> t; size_t i = 0;
    while (i < s.size()) {
        char c = s[i];
        if (std::isspace((unsigned char)c)) { ++i; continue; }
        if (std::isdigit((unsigned char)c) || c == '.') {
            size_t j = i;
            while (j < s.size() && (std::isdigit((unsigned char)s[j]) || s[j]=='.' || s[j]=='e' || s[j]=='E'
                   || ((s[j]=='+'||s[j]=='-') && j>0 && (s[j-1]=='e'||s[j-1]=='E')))) ++j;
            t.push_back({'n', "", std::stod(s.substr(i, j-i))}); i = j; continue;
        }
        if (std::isalpha((unsigned char)c) || c == '_') {
            size_t j = i; while (j < s.size() && (std::isalnum((unsigned char)s[j]) || s[j]=='_')) ++j;
            t.push_back({'i', s.substr(i, j-i), 0}); i = j; continue;
        }
        if (std::string("+-*/^()").find(c) != std::string::npos) { t.push_back({'o', std::string(1,c), 0}); ++i; continue; }
        fail(std::string("bad character '") + c + "'");
    }
    return t;
}

Expr mk(Node::K k, Expr a, Expr b) { auto n = std::make_shared<Node>(); n->k = k; n->a = a; n->b = b; return n; }
Expr number(double v) { auto n = std::make_shared<Node>(); n->k = Node::Num; n->num = v; return n; }

struct Parser {
    std::vector<Tok> t; size_t p = 0; const std::vector<std::string>& vars;
    Parser(std::vector<Tok> tk, const std::vector<std::string>& v) : t(std::move(tk)), vars(v) {}
    bool has() const { return p < t.size(); }
    const Tok& peek() const { static Tok e{'e',"",0}; return p<t.size()?t[p]:e; }
    bool isOp(char c) const { return has() && peek().kind=='o' && peek().s[0]==c; }
    int varIndex(const std::string& nm) const { for (size_t i=0;i<vars.size();++i) if (vars[i]==nm) return (int)i; return -1; }

    Expr expr()  { Expr a=term();  while(isOp('+')||isOp('-')){ char o=peek().s[0]; ++p; a=mk(o=='+'?Node::Add:Node::Sub,a,term()); } return a; }
    Expr term()  { Expr a=unary(); while(isOp('*')||isOp('/')){ char o=peek().s[0]; ++p; a=mk(o=='*'?Node::Mul:Node::Div,a,unary()); } return a; }
    Expr unary() { if(isOp('-')){ ++p; return mk(Node::Neg, unary(), nullptr); } return powf(); }
    Expr powf()  { Expr a=atom(); if(isOp('^')){ ++p; if(!has()||peek().kind!='n') fail("exponent must be an integer constant"); double e=peek().num; ++p; if(e!=std::floor(e)||e<0) fail("exponent must be a non-negative integer"); Expr n=mk(Node::Pow,a,number(e)); n->b->powN=(int)e; return n; } return a; }
    Expr atom() {
        if (isOp('(')) { ++p; Expr e=expr(); if(!isOp(')')) fail("missing ')'"); ++p; return e; }
        const Tok& tk = peek();
        if (tk.kind=='n') { ++p; return number(tk.num); }
        if (tk.kind=='i') {
            std::string nm = tk.s; ++p;
            if (isOp('(')) {
                ++p; Expr arg=expr(); if(!isOp(')')) fail("missing ')' after "+nm); ++p;
                static const std::vector<std::string> fns={"sin","cos","exp","sqrt","abs"};
                if (std::find(fns.begin(),fns.end(),nm)==fns.end()) fail("unknown function '"+nm+"'");
                auto n=std::make_shared<Node>(); n->k=Node::Fn; n->fn=nm; n->a=arg; return n;
            }
            int vi = varIndex(nm);
            if (vi<0) fail("unknown variable '"+nm+"' (known: x0,x1,... u0,u1,...)");
            auto n=std::make_shared<Node>(); n->k=Node::Var; n->var=vi; return n;
        }
        fail("unexpected token");
    }
};

} // namespace

Expr parse(const std::string& s, const std::vector<std::string>& vars) {
    Parser ps(lex(s), vars);
    if (!ps.has()) fail("empty expression");
    Expr e = ps.expr();
    if (ps.has()) fail("trailing tokens in expression");
    return e;
}

Ival evalInterval(const Expr& e, const std::vector<Ival>& v) {
    switch (e->k) {
        case Node::Num: return Ival(e->num);
        case Node::Var: return v[e->var];
        case Node::Add: return evalInterval(e->a,v) + evalInterval(e->b,v);
        case Node::Sub: return evalInterval(e->a,v) - evalInterval(e->b,v);
        case Node::Mul: return evalInterval(e->a,v) * evalInterval(e->b,v);
        case Node::Div: return idiv(evalInterval(e->a,v), evalInterval(e->b,v));
        case Node::Neg: return ineg(evalInterval(e->a,v));
        case Node::Pow: return ipow(evalInterval(e->a,v), e->b->powN);
        case Node::Fn: {
            Ival x = evalInterval(e->a,v);
            if (e->fn=="sin") return itrig(x,true);
            if (e->fn=="cos") return itrig(x,false);
            if (e->fn=="exp") return iexp(x);
            if (e->fn=="sqrt") return isqrtI(x);
            return iabs(x);
        }
    }
    return Ival(0.0);
}

double evalPoint(const Expr& e, const std::vector<double>& v) {
    switch (e->k) {
        case Node::Num: return e->num;
        case Node::Var: return v[e->var];
        case Node::Add: return evalPoint(e->a,v) + evalPoint(e->b,v);
        case Node::Sub: return evalPoint(e->a,v) - evalPoint(e->b,v);
        case Node::Mul: return evalPoint(e->a,v) * evalPoint(e->b,v);
        case Node::Div: return evalPoint(e->a,v) / evalPoint(e->b,v);
        case Node::Neg: return -evalPoint(e->a,v);
        case Node::Pow: return std::pow(evalPoint(e->a,v), e->b->powN);
        case Node::Fn: {
            double x = evalPoint(e->a,v);
            if (e->fn=="sin") return std::sin(x);
            if (e->fn=="cos") return std::cos(x);
            if (e->fn=="exp") return std::exp(x);
            if (e->fn=="sqrt") return std::sqrt(x);
            return std::fabs(x);
        }
    }
    return 0.0;
}

} // namespace expr
} // namespace impact
