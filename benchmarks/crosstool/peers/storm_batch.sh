#!/bin/bash
# Run Storm on every exported ARCH DRN (robust + cooperative), collecting
# all-state aggregates + init value + timing. Run inside movesrwth/storm with
# /work mounted. Writes run/arch_storm.tsv.
ROOT=/work/benchmarks/crosstool/arch
PEERS=/work/benchmarks/crosstool/peers
OUT="$ROOT/run/arch_storm.tsv"
echo -e "tool\tname\tprop\tmode\tcomp\tmin\tmax\tavg\tinit\tmc_s\tcon_s" > "$OUT"

storm_bench () {
  NAME=$1; PROP=$2; H=$3
  [ -f "$ROOT/models/$NAME.drn" ] || { echo "$NAME: no DRN"; return; }
  tr -d '\r' < "$PEERS/storm_arch.sh" | bash -s -- "$NAME" "$PROP" "$H" | tee -a "$OUT"
}

storm_bench AS       reach  10
storm_bench BA       safety 6
storm_bench VP       reach  0
storm_bench IC_reach reach  5
storm_bench IC_safe  safety 5
storm_bench PD_p1    reach  0
storm_bench PD_p3    reach  0
echo "=== arch_storm.tsv written ==="
