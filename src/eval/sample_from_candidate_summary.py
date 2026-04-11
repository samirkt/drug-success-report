from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "reports" / "candidate_summary.csv"
DEFAULT_CANDIDATE_PAIRS = Path(__file__).resolve().parent / "candidate_pairs.csv"
RANDOM_SEED = 42
SAMPLES_PER_LABEL = {
	"Clinical": 0,
	"Post-Clinical": 20,
	"Other": 0,
}


def main():
	df = pd.read_csv(DEFAULT_INPUT)

	df["weak_label"] = df.outcome.apply(
		lambda x: "Clinical" if "Failed" in x else 
		"Post-Clinical" if x in ["Approved", "Commercialized"] 
		else "Other"
	)
	df_filtered = df[df.outcome != "Ongoing"].copy()

	existing_df = pd.DataFrame()
	if DEFAULT_CANDIDATE_PAIRS.exists():
		existing_df = pd.read_csv(DEFAULT_CANDIDATE_PAIRS)
		if "drug_name" in existing_df.columns and "drug_name" in df_filtered.columns:
			existing_drugs = set(existing_df["drug_name"].dropna().astype(str).str.strip().str.lower())
			df_filtered = df_filtered[
				~df_filtered["drug_name"].astype(str).str.strip().str.lower().isin(existing_drugs)
			]

	sampled_groups = []
	for label, group in df_filtered.groupby("weak_label"):
		target_n = SAMPLES_PER_LABEL.get(label, 0)
		n = min(len(group), target_n)
		if n > 0:
			sampled_groups.append(group.sample(n=n, random_state=RANDOM_SEED))

	df_stratified = pd.concat(sampled_groups, ignore_index=True) if sampled_groups else df_filtered.iloc[0:0]

	new_samples = df_stratified.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
	output_df = pd.concat([existing_df, new_samples], ignore_index=True)
	output_df.to_csv(DEFAULT_CANDIDATE_PAIRS, index=False)

if __name__ == "__main__":
	main()
