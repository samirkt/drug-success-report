#!/bin/bash

uv run python run_pipeline.py --output ../docs/ --max-trials 76000 --drugbank-csv ../drugbank_approvals.csv --all-modalities "$@"
