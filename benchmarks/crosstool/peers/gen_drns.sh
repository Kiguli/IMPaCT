#!/bin/bash
# Generate Storm DRN siblings for every exported ARCH .imdp model.
ROOT=/work/benchmarks/crosstool/arch
PEERS=/work/benchmarks/crosstool/peers
for f in "$ROOT"/models/*.imdp; do
  b=$(basename "$f" .imdp)
  python3 "$PEERS/imdp_to_drn.py" "$f" "$ROOT/models/$b.drn" && echo "drn: $b"
done
