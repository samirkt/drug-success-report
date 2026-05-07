#!/bin/bash

# Full training
uv run python -m model train --time-split-year 2005 --output ./model_runs/t2005

# Ablation
uv run python -m model ablate --mode loo --time-split-year 2005 --output ./model_runs/loo_t2005
