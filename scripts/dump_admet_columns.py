"""Print ADMET_COLUMNS for paste into pipeline/admet/admet_columns.py.

Run after each admet_ai upgrade and update the canonical tuple.
"""
from admet_ai import ADMETModel


def main() -> None:
    cols = tuple(ADMETModel().predict(smiles=["CCO"]).columns)
    print(f"# admet_ai produced {len(cols)} columns")
    print("ADMET_COLUMNS: tuple[str, ...] = (")
    for c in cols:
        print(f"    {c!r},")
    print(")")


if __name__ == "__main__":
    main()
