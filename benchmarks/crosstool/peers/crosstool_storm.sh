#!/bin/bash
# Storm on the small crosstool models (reach/safety/interval) + an omega-regular
# LTL probe (Buchi/persistence/patrol) to see whether Storm supports robust LTL on
# interval MDPs. Run inside movesrwth/storm with /work mounted.
M=/work/benchmarks/crosstool/models
OUT=/work/benchmarks/crosstool/run_peers
mkdir -p "$OUT"
TSV="$OUT/crosstool_storm.tsv"
echo -e "model\tquery\trobust\tcooperative" > "$TSV"

q () {  # model  propstr  label-for-print
  MODEL=$1; PROP=$2; NAME=$3
  R=$(storm --explicit-drn "$M/$MODEL" --prop "$PROP" --uncertainty-resolution robust 2>&1 | grep -oE 'Result \(for initial states\): [0-9.eE+-]+' | grep -oE '[0-9.eE+-]+$' | head -1)
  C=$(storm --explicit-drn "$M/$MODEL" --prop "$PROP" --uncertainty-resolution cooperative 2>&1 | grep -oE 'Result \(for initial states\): [0-9.eE+-]+' | grep -oE '[0-9.eE+-]+$' | head -1)
  [ -z "$R" ] && R="unsupported"; [ -z "$C" ] && C="unsupported"
  echo -e "$MODEL\t$NAME\t$R\t$C" | tee -a "$TSV"
}

q chain_point.drn      'Pmax=?[F "target"]'                 'reach target'
q safety_point.drn     'Pmin=?[F "avoid"]'                  'reach avoid (safety=1-x)'
q choice_interval.drn  'Pmax=?[F "target"]'                 'reach target'
q fork_interval.drn    'Pmax=?[F "target"]'                 'reach target'
# omega-regular probes (robust LTL on interval MDP)
q buchi_reach.drn      'Pmax=?[G F "acc"]'                  'GF acc (Buchi)'
q persist_leak.drn     'Pmax=?[F G "safe"]'                 'FG safe (persistence)'
q patrol_cycle.drn     'Pmax=?[(G F "r0") & (G F "r2")]'    'GF r0 & GF r2 (patrol)'
q buchi_routearound.drn 'Pmax=?[G F "acc"]'                 'GF acc (route-around)'
