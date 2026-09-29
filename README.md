# Self-Organization Patterns in LLM-based Multi-Agent Systems

Master's thesis of Eric Echtermeyer. Huge thanks to Julius Eckhardt and Christoph Kommer at SAP for supporting me throughout this work — thank you both very much.

📄 **[Read the thesis (PDF)](thesis/thesis.pdf)**

## Overview

Multi-agent debate — several language-model agents discussing a problem over several
rounds — is a popular way to improve reasoning. But debate is almost always judged only by
its *final accuracy*, which cannot distinguish two debates that reach the same answer: one
because the group corrected an early mistake, the other because it merely amplified the
answer most agents already held.

This thesis treats debate as a **dynamical system** and introduces **Self-Organization Motif
Analysis (SOMA)**, a method that detects four self-organization patterns in the interaction
logs, measured against principled null models:

- **Self-reinforcement** — positions harden over rounds
- **Limit cycles** — the discussion oscillates instead of settling
- **Bistability** — multiple stable outcomes for the same problem
- **Dominance** — an emergent leader steers the group

Applied to **~30,000 debates** (GPQA and HiddenBench, N=4 agents, one model family), the four
motifs turn out to be **four faces of one mechanism**: debate mostly *amplifies* the answer
distribution the agents start with rather than reasoning a new answer into existence. A causal
seeding experiment confirms it — each additional agent that starts with the correct answer
raises group accuracy sharply (~13× from one seed), while the group almost never reaches a
correct answer that none of its agents started with. Structural choices (topology, memory
window) shape the *dynamics* but not accuracy, and internal signals like confidence and
unanimity are misleading. Because SOMA works from information debate systems already log, the
*path* to an answer becomes something practitioners can measure rather than assume.

> **Scope.** Findings are for homogeneous, cooperative agents, N=4, a single model family, and
> two benchmarks. Only the seeding result is causal; the rest are observational. See the thesis
> for full caveats.

## Repository layout

| Path | Contents |
|---|---|
| `thesis/` | LaTeX source, figures (`thesis/plots/`), and the compiled `thesis.pdf` |
| `notebooks/golden/` | **Canonical** notebooks — one per thesis subsection; reproduce every number/figure. Start at [`INDEX.md`](notebooks/golden/INDEX.md) |
| `notebooks/archive/` | Earlier exploratory notebooks (superseded, kept for history) |
| `scripts/golden/` | Build scripts that generate the golden notebooks (`build_*.py`, `run_all.sh`) |
| `src/` | Library: MAS engine (`src/mas`), motif detectors (`src/metrics`), benchmark loaders, plotting style |
| `results/` | Raw debate logs (JSON) used by the analysis |
| `dataset/` | Benchmark inputs (GPQA, HiddenBench) and debate seeds (`dataset/seeds/`) |
| `run_mas.py`, `run_seeded.py`, `run_benchmark.py` | Simulation drivers that produced the logs |

## Setup

```bash
git clone https://github.com/echtermeyer/LLM-Based-MAS.git
cd LLM-Based-MAS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Read the thesis:** open [`thesis/thesis.pdf`](thesis/thesis.pdf). To rebuild it from source:

```bash
./thesis/build.sh          # runs pdflatex; figures come from thesis/plots/
```

**Reproduce a number or figure:** every thesis result maps to exactly one golden notebook.
Find it in [`notebooks/golden/INDEX.md`](notebooks/golden/INDEX.md), then:

```bash
python scripts/golden/build_NN_name.py                                   # (re)generate the notebook
python -m nbconvert --to notebook --execute --inplace "notebooks/golden/NN_name.ipynb"
```

Figures are written straight to their fixed `thesis/plots/...` path, so recompiling the thesis
picks them up with no `.tex` change. Regenerate everything with `scripts/golden/run_all.sh`.

**Run new debates** (optional): the `run_*.py` drivers produce the JSON logs in `results/`.
They require model API credentials (a `gen_ai_credential*.json`, not included) and are not
needed to read the thesis or reproduce its figures.

```bash
python run_mas.py --dataset gpqa --index 56 --n 4 --t 15 --w 2 --model mistral-medium --verbose
```

## Reproducibility notes

The tracked `results/` data regenerates every live figure and number in the thesis. Two
appendix/side sections (`10_linguistic`, `11_appendix_robustness`) are **frozen** — their
source data was intentionally not committed, so those specific values do not recompute on a
clone. See [`notebooks/golden/INDEX.md`](notebooks/golden/INDEX.md) for the full section →
notebook → status map and [`thesis/STATISTICAL_REVIEW_LEDGER.md`](thesis/STATISTICAL_REVIEW_LEDGER.md)
for the claim-by-claim statistical audit.
