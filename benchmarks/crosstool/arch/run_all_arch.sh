#!/bin/bash
# ============================================================================
# Master ARCH cross-tool runner (IMPaCT side + IntervalMDP.jl side).
# For each benchmark: build+run the IMPaCT SYCL example (abstraction + robust VI),
# export the abstracted .imdp, convert to Storm DRN, extract IMPaCT's value vector
# from controller.h5, and solve the same .imdp with IntervalMDP.jl. Storm is run
# separately (different container). Results land under run/<NAME>/.
#
# Benchmark table columns: NAME DIR SRC PROP HORIZON(0=infinite) DIM_X DIM_U
# ============================================================================
export JULIA_DEPOT_PATH=/opt/jldepot
ROOT=/work/benchmarks/crosstool/arch
PEERS=/work/benchmarks/crosstool/peers
SUM="$ROOT/run/summary.tsv"
mkdir -p "$ROOT/run"
echo -e "name\tprop\thorizon\tcells\timpact_abs_s\timpact_syn_s\timpact_rss_kb\timpact_pess0\timpact_opt0\timdp_pess0\timdp_opt0\timdp_iters\timdp_s" > "$SUM"

run_bench () {
  NAME=$1; DIR=$2; SRC=$3; PROP=$4; H=$5; DX=$6; DU=$7
  OUT="$ROOT/run/$NAME"
  echo "############## $NAME ($PROP H=$H) ##############"
  set +e
  # 1) IMPaCT build + run
  tr -d '\r' < "$ROOT/build_run.sh" | bash -s -- "$NAME" "$DIR" "$SRC" > "$ROOT/run/$NAME.buildrun.log" 2>&1
  if [ ! -f "$ROOT/models/$NAME.imdp" ]; then
     echo "  $NAME: NO .imdp produced (build/run failed or OOM); see $NAME.buildrun.log"
     echo -e "$NAME\t$PROP\t$H\tFAIL\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA" >> "$SUM"
     tail -5 "$ROOT/run/$NAME.buildrun.log"
     return
  fi
  CELLS=$(grep -m1 '^states' "$ROOT/models/$NAME.imdp" | awk '{print $2-2}')
  SYN=$(grep '^synthesis_seconds=' "$OUT/time.txt" | cut -d= -f2)
  RSS=$(grep '^max_rss_kb=' "$OUT/time.txt" | cut -d= -f2)
  # abstraction time = wall - synthesis (rough)
  WALL=$(grep '^wall_clock=' "$OUT/time.txt" | cut -d= -f2)

  # 2) extract IMPaCT value vector
  python3 "$PEERS/extract_impact_value.py" "$OUT/controller.h5" "$DX" "$DU" --dump "$OUT/impact_value.txt" > "$OUT/impact_summary.txt" 2>&1
  IPESS0=$(awk '/pess\[0\]/{for(i=1;i<=NF;i++){if($i ~ /pess\[0\]=/){split($i,a,"=");print a[2]}}}' "$OUT/impact_summary.txt")
  IOPT0=$(awk '/opt\[0\]/{for(i=1;i<=NF;i++){if($i ~ /opt\[0\]=/){split($i,a,"=");print a[2]}}}' "$OUT/impact_summary.txt")

  # 3) Storm DRN
  python3 "$PEERS/imdp_to_drn.py" "$ROOT/models/$NAME.imdp" "$ROOT/models/$NAME.drn" 2>/dev/null

  # 4) IntervalMDP.jl
  HARG=""; [ "$H" != "0" ] && HARG="--horizon $H"
  julia "$PEERS/intervalmdp_runner.jl" "$ROOT/models/$NAME.imdp" "$PROP" $HARG --state 0 > "$OUT/intervalmdp.txt" 2>&1
  MPESS=$(grep 'bound=pess' "$OUT/intervalmdp.txt" | grep -oE 'value=[0-9.eE+-]+' | head -1 | cut -d= -f2)
  MOPT=$(grep 'bound=opt'  "$OUT/intervalmdp.txt" | grep -oE 'value=[0-9.eE+-]+' | head -1 | cut -d= -f2)
  MIT=$(grep 'bound=pess'  "$OUT/intervalmdp.txt" | grep -oE 'iters=[0-9]+' | head -1 | cut -d= -f2)
  MSEC=$(grep 'bound=pess' "$OUT/intervalmdp.txt" | grep -oE 'seconds=[0-9.]+' | head -1 | cut -d= -f2)

  ABS=$(python3 -c "w='$WALL'.split(':'); s=float(w[-1])+ (float(w[-2])*60 if len(w)>1 else 0); print(round(s-float('$SYN' or 0),2))" 2>/dev/null)
  echo -e "$NAME\t$PROP\t$H\t$CELLS\t$ABS\t$SYN\t$RSS\t$IPESS0\t$IOPT0\t$MPESS\t$MOPT\t$MIT\t$MSEC" >> "$SUM"
  echo "  $NAME: cells=$CELLS abs=${ABS}s syn=${SYN}s | IMPaCT pess0=$IPESS0 | IntervalMDP pess0=$MPESS (it=$MIT, ${MSEC}s)"
}

# ---- benchmark table (feasible dense set) ----
run_bench AS       examples/ARCH-COMP/2025/AS        AS.cpp           reach  10 3 2
run_bench BA       examples/ARCH-COMP/2025/BA        BA_4d.cpp        safety 6  4 1
run_bench VP       examples/ARCH-COMP/2025/VP        VP.cpp           reach  0  2 0
run_bench IC_reach examples/ARCH-COMP/2025/IC        IC_reach_2d.cpp  reach  5  2 1
run_bench IC_safe  examples/ARCH-COMP/2025/IC        IC_safe_2d.cpp   safety 5  2 1
run_bench PD_p1    examples/ARCH-COMP/2025/PD        PD_target_p1.cpp reach  0  2 2
run_bench PD_p3    examples/ARCH-COMP/2025/PD        PD_target_p3.cpp reach  0  2 2

echo "=== SUMMARY ==="
column -t -s $'\t' "$SUM"
