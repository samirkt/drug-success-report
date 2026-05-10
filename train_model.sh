#!/bin/bash

# Full training
uv run python -m model train --time-split-year 2019 --output ./model_runs/t2019

# Ablation
uv run python -m model ablate --mode loo --time-split-year 2019 --output ./model_runs/loo_t2019

# Baselines (1:1 comparison vs full model)
uv run python -m model baselines --time-split-year 2019 --output ./model_runs/baselines
