#!/usr/bin/env python3
# Assemble the comparison tables for TOOL_COMPARISONS.md from the collected TSVs:
#   arch/run/summary.tsv      (IMPaCT timings: abstraction, synthesis, RSS)
#   arch/run/arch_compare.tsv (IMPaCT vs IntervalMDP.jl per-cell correctness)
#   arch/run/arch_storm.tsv   (Storm robust/cooperative aggregates + timing)
#   run_peers/crosstool_peers.tsv
import os, csv

ROOT = "/work/benchmarks/crosstool"

def read_tsv(p):
    if not os.path.exists(p): return []
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))

summary = {r["name"]: r for r in read_tsv(f"{ROOT}/arch/run/summary.tsv")}
comp    = {r["name"]: r for r in read_tsv(f"{ROOT}/arch/run/arch_compare.tsv")}
storm   = read_tsv(f"{ROOT}/arch/run/arch_storm.tsv")
storm_by = {}
for r in storm:
    storm_by.setdefault(r["name"], {})[r["mode"]] = r

print("### ARCH-COMP — correctness (robust / pessimistic, per-cell over the abstracted IMDP)\n")
print("| Benchmark | cells | IMPaCT pess [min,max] | IntervalMDP pess [min,max] | max|IMPaCT−IntervalMDP| (pess) | max diff (opt) | verdict |")
print("|---|---|---|---|---|---|---|")
for n, r in comp.items():
    v = "MATCH" if float(r["pess_maxabs"]) < 1e-4 and float(r["opt_maxabs"]) < 1e-4 else "CHECK"
    print(f"| {n} | {r['cells']} | [{r['i_pess_min']}, {r['i_pess_max']}] | "
          f"[{r['m_pess_min']}, {r['m_pess_max']}] | {r['pess_maxabs']} | {r['opt_maxabs']} | {v} |")

print("\n### ARCH-COMP — solve-time comparison (seconds)\n")
print("| Benchmark | IMPaCT abstraction | IMPaCT VI (synth) | IntervalMDP VI | Storm VI | Storm construct |")
print("|---|---|---|---|---|---|")
for n, r in comp.items():
    s = summary.get(n, {})
    st = storm_by.get(n, {}).get("robust", {})
    print(f"| {n} | {s.get('impact_abs_s','?')} | {s.get('impact_syn_s','?')} | "
          f"{r.get('imdp_s','?')} | {st.get('mc_s','?')} | {st.get('con_s','?')} |")

print("\n### ARCH-COMP — Storm cross-check (all-state aggregates)\n")
print("| Benchmark | mode | Storm min | Storm max | Storm avg | Storm init |")
print("|---|---|---|---|---|---|")
for n in comp:
    for mode in ("robust","coop"):
        st = storm_by.get(n, {}).get(mode)
        if st:
            print(f"| {n} | {mode} | {st['min']} | {st['max']} | {st['avg']} | {st['init']} |")

print("\n### Crosstool small models\n")
print("| Model | prop | state | bound | IMPaCT | IntervalMDP.jl | PRISM | ref |")
print("|---|---|---|---|---|---|---|---|")
for r in read_tsv(f"{ROOT}/run_peers/crosstool_peers.tsv"):
    print(f"| {r['model']} | {r['prop']} | {r['state']} | {r['bound']} | {r['impact']} | "
          f"{r['intervalmdp']} | {r['prism']} | {r['ref']} |")
