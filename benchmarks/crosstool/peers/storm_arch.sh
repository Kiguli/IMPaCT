#!/bin/bash
# ============================================================================
# Storm runner for an exported ARCH .imdp model (consumes the DRN sibling).
# Storm cannot export per-state result vectors for interval models, so we compare
# tool-independent AGGREGATES over ALL states via filter(min|max|avg, prop, true),
# plus the init-state value. Runs robust (= IMPaCT pessimistic) and cooperative
# (= IMPaCT optimistic).
#
#   reach , H=0 : Pmax=?[F "target"]          safety,H=0 : Pmin=?[F "avoid"]  (value=1-x)
#   reach , H>0 : Pmax=?[F<=H "target"]       safety,H>0 : Pmin=?[F<=H "avoid"]
#
# Usage: storm_arch.sh NAME PROP HORIZON   (run inside movesrwth/storm, /work mounted)
# Emits TSV lines: storm NAME PROP mode comp init min max avg mc_seconds total_seconds
# ============================================================================
NAME=$1; PROP=$2; H=$3
ROOT=/work/benchmarks/crosstool/arch
DRN=$ROOT/models/$NAME.drn
OUT=$ROOT/run/$NAME
mkdir -p "$OUT"

if [ "$PROP" = "reach" ]; then
  LBL=target; OP=Pmax; COMP=0
else
  LBL=avoid;  OP=Pmin; COMP=1
fi
if [ "$H" = "0" ]; then F="F \"$LBL\""; else F="F<=$H \"$LBL\""; fi
BASE="$OP=?[$F]"

run_one () {
  MODE=$1; TAG=$2
  LOG=$OUT/storm_$TAG.log
  # three aggregate properties + plain (init) in one invocation
  storm --explicit-drn "$DRN" --uncertainty-resolution "$MODE" \
        --prop "filter(min, $BASE, true); filter(max, $BASE, true); filter(avg, $BASE, true); $BASE" \
        > "$LOG" 2>&1
  mapfile -t R < <(grep -oE 'Result \(for [^)]*\): [0-9.eE+-]+' "$LOG" | grep -oE '[0-9.eE+-]+$')
  MC=$(grep -oE 'Time for model checking: [0-9.]+' "$LOG" | grep -oE '[0-9.]+' | awk '{s+=$1} END{printf "%.4f", s}')
  CON=$(grep -oE 'Time for model construction: [0-9.]+' "$LOG" | grep -oE '[0-9.]+' | head -1)
  echo "storm	$NAME	$PROP	$TAG	comp=$COMP	min=${R[0]}	max=${R[1]}	avg=${R[2]}	init=${R[3]}	mc_s=$MC	con_s=$CON"
}

run_one robust robust
run_one cooperative coop
