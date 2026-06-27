#!/bin/bash
# ============================================================================
# Build and run one ARCH-COMP IMPaCT example (CPU/OMP), capturing the abstraction
# wall-clock, the synthesis ("Execution time") wall-clock, and the exported
# neutral .imdp model (written by mdp.exportIMDP). Reuses prebuilt /tmp/IMDP.o
# and /tmp/MDP.o so only the small example TU is compiled per benchmark.
#
# Usage: build_run.sh <NAME> <example_dir> <source.cpp>
#   e.g. build_run.sh BA examples/ARCH-COMP/2025/BA BA_4d.cpp
# Outputs land in benchmarks/crosstool/arch/models/<NAME>.imdp and
#          benchmarks/crosstool/arch/run/<NAME>/{controller.h5,run.log,time.txt}
# ============================================================================
set -e
NAME="$1"; EXDIR="/work/$2"; SRC="$3"
ROOT=/work/benchmarks/crosstool/arch
OUT="$ROOT/run/$NAME"
mkdir -p "$OUT" "$ROOT/models"

FLAGS='--acpp-targets=omp -O3 -lnlopt -lm -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -lglpk -lgsl -lgslcblas -DH5_USE_110_API -larmadillo'

echo "[build] $NAME"
cd "$EXDIR"
acpp "$SRC" /tmp/IMDP.o /tmp/MDP.o $FLAGS -o "$OUT/$NAME.bin"

echo "[run] $NAME"
cd "$OUT"
rm -f "$NAME.imdp" controller.h5
/usr/bin/time -v "./$NAME.bin" > run.log 2>&1 || { echo "RUN FAILED"; tail -20 run.log; exit 1; }

# Collect the exported model (the example writes <NAME>.imdp to cwd)
if [ -f "$NAME.imdp" ]; then cp "$NAME.imdp" "$ROOT/models/$NAME.imdp"; fi

# Extract timings: abstraction = sum of "Calculating ... s." lines is not printed
# uniformly, so we report wall-clock peak + the synthesis "Execution time".
SYN=$(grep -oE 'Execution time: [0-9.]+' run.log | tail -1 | grep -oE '[0-9.]+' || echo "NA")
WALL=$(grep -oE 'Elapsed \(wall clock\) time.*' run.log | sed 's/.*: //' || echo "NA")
MEM=$(grep -oE 'Maximum resident set size \(kbytes\): [0-9]+' run.log | grep -oE '[0-9]+$' || echo "NA")
echo "synthesis_seconds=$SYN" | tee time.txt
echo "wall_clock=$WALL"       | tee -a time.txt
echo "max_rss_kb=$MEM"        | tee -a time.txt
echo "[done] $NAME -> models/$NAME.imdp"
