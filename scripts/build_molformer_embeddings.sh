#!/bin/bash

uv run build_molformer_embeddings.py --candidates ../outputs/candidate_detail.parquet --output ../outputs/features/molformer_embeddings.parquet
