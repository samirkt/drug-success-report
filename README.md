# drug-success-report

Automated pipeline for reconstructing drug-development trajectories from public clinical trial data and computing phase-transition success rates stratified by drug modality and disease area.

Minimum setup:
src/execute.sh → scripts/build_fingerprints.sh + scripts/build_molformer_embeddings.sh → train_model.sh
