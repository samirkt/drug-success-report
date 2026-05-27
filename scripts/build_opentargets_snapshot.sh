#!/bin/bash

uv run python scripts/download_opentargets.py --release 25.03 --dest data/opentargets
uv run python scripts/build_opentargets_snapshot.py \
      --opentargets-dir data/opentargets/25.03 \
      --chembl-snapshot chembl_targets.sqlite \
      --out opentargets_snapshot.sqlite
