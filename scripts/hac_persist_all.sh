#!/bin/bash
# Run scripts/hac_persistence.py on each given field, one memory-capped process per field (laptop rule:
# 512^3 persistence can need 11-21 GB; an uncapped run crashed the laptop once). Skips existing diagrams.
#   bash scripts/hac_persist_all.sh results/fields/hac/scaffold_*_r256_off0.0.npy ...
cd "$(dirname "$0")/.."
for f in "$@"; do
  systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=4G .venv/bin/python scripts/hac_persistence.py "$f" \
    || echo "FAILED $f (exit $?)"
done
echo PERSIST_DONE
