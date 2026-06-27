#!/bin/bash
# Cross-tool comparison on the small shared crosstool models. Runs IMPaCT
# (imdp_solve), IntervalMDP.jl, and PRISM (point models only) on reach/safety
# models and prints a comparison table vs the analytic .ref.json values.
# Storm is run separately (storm container). The omega-regular models
# (buchi/persist/patrol) are handled by IMPaCT + analytic ref only here.
export JULIA_DEPOT_PATH=/opt/jldepot
M=/work/benchmarks/crosstool/models
PEERS=/work/benchmarks/crosstool/peers
BIN=/work/tools/imdp_solve
OUT=/work/benchmarks/crosstool/run_peers
mkdir -p "$OUT"
TSV="$OUT/crosstool_peers.tsv"
echo -e "model\tprop\tlabel\tstate\tbound\timpact\tintervalmdp\tprism\tref" > "$TSV"

# args: model prop label state bound ref pointmodel(1/0)
row () {
  MODEL=$1; PROP=$2; LBL=$3; ST=$4; BOUND=$5; REF=$6; POINT=$7
  # IMPaCT (imdp_solve): use midpoint of [lower,upper]
  IL=$("$BIN" "$M/$MODEL" "$PROP" "$LBL" --bound "$BOUND" --state "$ST" --eps 1e-9 2>/dev/null | grep -oE 'lower=[0-9.eE+-]+' | cut -d= -f2)
  IU=$("$BIN" "$M/$MODEL" "$PROP" "$LBL" --bound "$BOUND" --state "$ST" --eps 1e-9 2>/dev/null | grep -oE 'upper=[0-9.eE+-]+' | cut -d= -f2)
  IMP=$(python3 -c "print(round((float('$IL')+float('$IU'))/2,6))" 2>/dev/null)
  # IntervalMDP.jl
  MDP=$(julia "$PEERS/intervalmdp_runner.jl" "$M/$MODEL" "$PROP" --label "$LBL" --state "$ST" --eps 1e-9 2>/dev/null | grep "bound=$BOUND" | grep -oE 'value=[0-9.eE+-]+' | cut -d= -f2)
  MDP=$(python3 -c "print(round(float('$MDP'),6))" 2>/dev/null)
  # PRISM (point models only; PRISM has no interval-MDP support)
  PR="n/a"
  if [ "$POINT" = "1" ]; then
    python3 "$PEERS/imdp_to_prism.py" "$M/$MODEL" "$OUT/${MODEL%.imdp}.pm" 2>/dev/null
    if [ "$PROP" = "reach" ]; then PF="Pmax=?[F \"$LBL\"]"; else PF="Pmax=?[G !\"$LBL\"]"; fi
    PR=$(prism "$OUT/${MODEL%.imdp}.pm" -pf "$PF" 2>/dev/null | awk '/^Result:/{print $2; exit}')
    [ -z "$PR" ] && PR="ERR"
  fi
  echo -e "$MODEL\t$PROP\t$LBL\t$ST\t$BOUND\t$IMP\t$MDP\t$PR\t$REF" >> "$TSV"
}

# point models (PRISM applies)
row chain_point.imdp  reach  target 0 pess 0.25 1
row chain_point.imdp  reach  target 1 pess 0.50 1
row safety_point.imdp safety avoid  0 pess 0.72 1
# interval models (no PRISM)
row choice_interval.imdp reach target 0 pess 0.4 0
row choice_interval.imdp reach target 0 opt  0.5 0
row fork_interval.imdp   reach target 0 pess 0.4 0
row fork_interval.imdp   reach target 0 opt  0.6 0

echo "=== crosstool_peers.tsv ==="
column -t -s $'\t' "$TSV"
