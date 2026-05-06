"""Modeling pipeline for drug-indication candidate approval prediction.

Trains and evaluates classifiers on six toggleable feature groups
(fingerprints, embeddings, drug targets, ADMET, pathway, disease) sourced
from the published candidate parquets in `docs/`. See `model/cli.py` for
the `train` / `ablate` entrypoints.
"""
