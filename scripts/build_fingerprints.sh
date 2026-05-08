#!/bin/bash

uv run build_fingerprints.py --candidates ../outputs/candidate_detail.parquet --output ../outputs/features/fingerprints.parquet
