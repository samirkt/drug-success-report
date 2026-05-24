#!/bin/bash

# Full training
uv run python -m model train --time-split-year 2019 --output ./model_runs/t2019

# Full training with isotonic calibration on the held-out 2018 slice
# (three-way split: train <= 2017, calibrate == 2018, test > 2018)
#uv run python -m model train --calibration-year 2018 --output ./model_runs/t2018cal

# Group-level RFE
uv run python -m model rfe --time-split-year 2019 --output ./model_runs/rfe_t2019

# Ablation
uv run python -m model ablate --mode loo --time-split-year 2019 --output ./model_runs/loo_t2019

# Baselines (1:1 comparison vs full model)
uv run python -m model baselines --time-split-year 2019 --output ./model_runs/baselines

# HINT trial-level inference. Writes per-phase metrics next to the input dataset:
#   model_runs/t2019/trial/hint_test.csv  →  model_runs/t2019/trial/hint_results.csv
./run_hint.sh ./model_runs/t2019/trial/hint_test.csv ./model_runs/t2019/trial/hint_results.csv

# Consolidated PDF report.
uv run python scripts/build_consolidated_report.py \
  --run-root ./model_runs \
  --train-name t2019 \
  --rfe-name rfe_t2019 \
  --ablate-name loo_t2019 \
  --baselines-name baselines \
  --hint-metrics ./model_runs/t2019/trial/hint_results.csv \
  --output ./outputs/consolidated_report.pdf
