#!/usr/bin/env bash
set -euo pipefail

python run_pipeline.py \
  --config config/config_alt.yaml \
  --results-out data/results/analysis_results_alt.json \
  --alt-model
