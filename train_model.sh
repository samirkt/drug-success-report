#!/bin/bash

# Full training
uv run python -m model train --time-split-year 2020 --output ./model_runs/t2020

# Ablation
uv run python -m model ablate --mode loo --time-split-year 2020 --output ./model_runs/loo_t2020
