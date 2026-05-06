"""Convenience wrapper — `python run_modeling.py train …` / `… ablate …`.

Mirrors `src/run_pipeline.py`. Equivalent to `python -m model …`.
"""

from model.cli import main

if __name__ == "__main__":
    main()
