# Golden Notebooks

Canonical, section-organized notebooks that reproduce **every number and figure in the thesis**
from raw data. One notebook per thesis subsection. To refresh anything in the thesis: find the
section in [`INDEX.md`](INDEX.md), open the one notebook that owns it, rerun it, copy the printed
number into the `.tex` (figures update automatically — see below).

## Layout

- **Build script is the source of truth:** `scripts/golden/build_<NN>_<name>.py` generates the
  notebook programmatically (`nbformat`). Never hand-edit the `.ipynb`; edit the build script and
  regenerate.
- **Notebook** `notebooks/golden/<NN>_<name>.ipynb` is the executed artifact (numbers in outputs).
- **Figures** are written by `fig.savefig(...)` to the exact `thesis/plots/<subdir>/figN_*.png`
  path the LaTeX already `\includegraphics`. The path is fixed, so recompiling the thesis picks up
  the new PNG with **no `.tex` change**.
- **Numbers** are printed in blocks headed by a comment naming the thesis `\label` / line, e.g.
  `--- sec:results:perf L55 GPQA baselines ---`, for manual copy into the `.tex`.

## Regenerating one section

```bash
# 1. (edit scripts/golden/build_NN_name.py if the analysis changed)
python scripts/golden/build_NN_name.py                 # writes the .ipynb
# 2. execute it (jupyter launcher has a stale shebang -> use python -m nbconvert)
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=2400 "notebooks/golden/NN_name.ipynb"
# 3. recompile the thesis; copy any changed printed numbers into the .tex
```

Regenerate everything: run the two loops in `scripts/golden/run_all.sh` (light notebooks first,
heavy ones — 06 dominance, 07 LC, 11 robustness, 10 linguistic — take several minutes each).

## Conventions (if you add or port a notebook)

- Notebooks live one level below `notebooks/`, so inside every code cell relative paths are
  `Path('../..') / 'results' / ...` and `Path('../..') / 'thesis' / 'plots' / ...`, and
  `sys.path.insert(0, '../..')`.
- Shared styling: `from src.viz.thesis_style import apply_style, ...; apply_style()`.
- Metrics come from `src/metrics/*.py` (never reimplement a detector).
- First cell is a markdown provenance header: thesis `\label`, data path, figures, status.
- Where a recomputed value differs from the current `.tex`, print
  `computed=…  thesis_says=…  MISMATCH` — the notebook is authoritative; the `.tex` is updated by
  hand.

## Data sources

- `results/mas/final_dataset_new_system` — main corpus (420 JSON = 35 tasks × 2 datasets × 6
  configs, R=50). Used by 03, 04, 05, 06, 07, 10.
- `results/mas/high_repetition` — R=1000 (q84/q125/q144). Used by 08.
- `results/mas/seeded_init_0`, `seeded_init_1` — causal seeding. Used by 09.
- `results/linguistic_mistral/message`, `results/linguistic_nli/message` — linguistic scores (10).
- `results/mas/*_subset_scan_nothink_small` — Tier-3 screening (01, repointed; partial).
- `results/mas/final_dataset_depricated` — old-prompt arm (11, repointed).

See [`INDEX.md`](INDEX.md) for the full section → notebook → numbers/figures map and the
reproducibility status of each (including the frozen artifacts whose source data was deleted).

## Superseded originals

These golden notebooks supersede the old scattered analysis. The originals are left on disk
(untouched) for history but should no longer be used as the thesis source:
`scripts/build_nb_031–036.py`, `scripts/plot_lc_new.py`, and notebooks
`020/023/024/025` (perf), `026v2/027/028` (linguistic), `008v2/030` (task selection),
`029` (robustness).
