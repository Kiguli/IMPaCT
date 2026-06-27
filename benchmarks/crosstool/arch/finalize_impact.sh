#!/bin/bash
# Re-extract IMPaCT value vectors (correct dims), re-solve with IntervalMDP.jl
# dumping the full vector, and compare per-cell. Reads the same benchmark table.
# Produces run/arch_compare.tsv (correctness) using already-built controller.h5 +
# exported .imdp (does NOT rebuild IMPaCT).
export JULIA_DEPOT_PATH=/opt/jldepot
ROOT=/work/benchmarks/crosstool/arch
PEERS=/work/benchmarks/crosstool/peers
OUTSUM="$ROOT/run/arch_compare.tsv"
echo -e "name\tcells\ti_pess_min\ti_pess_max\ti_pess_mean\tm_pess_min\tm_pess_max\tm_pess_mean\tpess_maxabs\topt_maxabs\timdp_pess0\timdp_iters\timdp_s" > "$OUTSUM"

cmp_bench () {
  NAME=$1; PROP=$2; H=$3; DX=$4; DU=$5
  OUT="$ROOT/run/$NAME"
  [ -f "$OUT/controller.h5" ] || { echo "$NAME: no controller.h5 (skipped/OOM)"; return; }
  python3 "$PEERS/extract_impact_value.py" "$OUT/controller.h5" "$DX" "$DU" --dump "$OUT/impact_value.txt" >/dev/null 2>&1
  HARG=""; [ "$H" != "0" ] && HARG="--horizon $H"
  julia "$PEERS/intervalmdp_runner.jl" "$ROOT/models/$NAME.imdp" "$PROP" $HARG --state 0 --dumpdir "$OUT" > "$OUT/intervalmdp.txt" 2>&1
  P0=$(grep 'bound=pess' "$OUT/intervalmdp.txt" | grep -oE 'value=[0-9.eE+-]+' | head -1 | cut -d= -f2)
  IT=$(grep 'bound=pess' "$OUT/intervalmdp.txt" | grep -oE 'iters=[0-9]+' | head -1 | cut -d= -f2)
  SEC=$(grep 'bound=pess' "$OUT/intervalmdp.txt" | grep -oE 'seconds=[0-9.]+' | head -1 | cut -d= -f2)
  ROW=$(python3 "$PEERS/arch_compare.py" "$NAME" "$OUT/impact_value.txt" "$OUT/intervalmdp_pess.txt" "$OUT/intervalmdp_opt.txt" 2>"$OUT/cmp.err")
  if [ -z "$ROW" ]; then echo "$NAME: compare failed: $(cat $OUT/cmp.err)"; return; fi
  echo -e "$ROW\t$P0\t$IT\t$SEC" >> "$OUTSUM"
  echo "$NAME done"
}

cmp_bench AS       reach  10 3 2
cmp_bench BA       safety 6  4 1
cmp_bench VP       reach  0  2 0
cmp_bench IC_reach reach  5  2 1
cmp_bench IC_safe  safety 5  2 1
cmp_bench PD_p1    reach  0  2 2
cmp_bench PD_p3    reach  0  2 2

echo "=== arch_compare.tsv ==="
column -t -s $'\t' "$OUTSUM"
