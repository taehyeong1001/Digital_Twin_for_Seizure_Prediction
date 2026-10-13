#!/usr/bin/env bash
# Run the 4-step multi-node SCDT pipeline (final_code README order) inside an
# isolated copy of final_code/CODE, so experiments/ and figures/ land next to
# the copy instead of inside final_code/.
# usage: survival_dist/run_multinode_pipeline.sh <run_dir>   (run_dir contains CODE/)
set -euo pipefail
RUN_DIR="$(cd "$1" && pwd)"
PY="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python"
cd "$RUN_DIR/CODE"
for step in experiment_three_node_scdt_empirical_boundary_bo \
            experiment_three_node_scdt_block_bo100 \
            experiment_three_node_scdt_block_peak_bo100 \
            experiment_three_node_scdt_block_peak_confirm; do
  echo "=== START $step $(date '+%F %T')"
  start=$(date +%s)
  "$PY" "$step.py" > "$RUN_DIR/$step.log" 2>&1
  echo "=== DONE  $step $(( $(date +%s) - start )) s"
done
echo "=== PIPELINE FINISHED $(date '+%F %T')"
