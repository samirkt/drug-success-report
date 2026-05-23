#!/bin/bash
HINT_ROOT="/Users/samirtownsley/Documents/projects/hint_standalone/repo"
cd "$HINT_ROOT" && uv run python run_hint_on_dataset.py --input "$@"
