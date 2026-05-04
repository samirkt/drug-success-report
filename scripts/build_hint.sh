#!/bin/bash

uv run python build_hint_dataset.py --candidates ../docs/candidate_detail.parquet --trials ../docs/trial_detail.parquet --output ../docs/features/hint_dataset.csv
