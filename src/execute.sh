#!/bin/bash

#uv run python run_pipeline.py --output ../docs/ --max-trials 76000 --drugbank-csv ../drugbank_approvals.csv --all-modalities "$@"
uv run python run_pipeline.py --output ../outputs/ --max-trials -1 --max-candidates 999999 --sample-seed 42 --drugbank-csv ../drugbank_approvals_smiles.csv --all-modalities --chembl-snapshot ../chembl_targets.sqlite  --opentargets-snapshot ../opentargets_snapshot.sqlite --year-range 2009-2026 --drop-uncached-candidates --require-smiles --use-ct-cache "$@"

