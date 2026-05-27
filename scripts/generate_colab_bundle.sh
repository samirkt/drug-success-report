#!/bin/bash

cd src && ../.venv/bin/python -m scripts.export_ndc_work \
      --source aact \
      --max-trials 76000 \
      --year-range 2000-2006 \
      --max-candidates 600 --sample-seed 42 \
      --use-ct-cache \
      --ct-cache-path aact_cache.pkl \
      --drugbank-csv ../drugbank_approvals_smiles.csv \
      --chembl-snapshot ../chembl_targets.sqlite \
      --opentargets-snapshot ../opentargets_snapshot.sqlite \
      --output ../tmp/colab_smoke/work_units.jsonl \
      --candidates-pickle ../tmp/colab_smoke/clustered.pkl \
      --skip-cached

wc -l ./tmp/colab_smoke/work_units.jsonl
head -1 ./tmp/colab_smoke/work_units.jsonl | uv run python -m json.tool

zip -r ./tmp/colab_smoke/colab_bundle.zip colab_inference \
      -x 'colab_inference/__pycache__/*' 'colab_inference/*.pyc'
ls -la ./tmp/colab_smoke/colab_bundle.zip ./tmp/colab_smoke/work_units.jsonl
