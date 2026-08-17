# Golden Notebooks — Master Index

Every number and figure in the thesis maps to exactly one golden notebook here. To refresh
anything: find its section below → open that notebook → rerun (`python scripts/golden/build_<nb>.py`
then execute) → copy the printed number into the `.tex`. Figures rewrite to their fixed
`thesis/plots/` path automatically. See [`README.md`](README.md) for the workflow.

Legend: **status** = reproducible (runs from committed data) · repointed (source data renamed,
pointed at nearest surviving) · frozen (source data deleted — value preserved but not
recomputable).

## Master map

| Golden notebook | Thesis section(s) | Tables owned | Figures owned (→ path) | Data | Status |
|---|---|---|---|---|---|
| `01_setup_and_datasets` | ch4 `sec:setup:*` | `tab:notation`, `tab:knob-summary` (design) | — | `final_dataset_new_system` + `*_subset_scan_nothink_small` | repointed (Tier-3 frozen) |
| `02_detection_worked_examples` | ch5 `sec:method:*` worked examples | — | — | none (synthetic) | reproducible |
| `03_system_performance` | `sec:results:perf`; `app:perf:configs`, `app:perf:rounds_acc` | `tab:perf:baseline`, `tab:perf:topology`, `tab:app:perf:configs` | `plots/general/fig1_single_vs_multi`, `fig2_topology`, `fig3_memory_window`, `fig4_round_distribution`, `fig5_rounds_vs_accuracy` | `final_dataset_new_system` | reproducible |
| `04_self_reinforcement` | `sec:results:sr` | `tab:sr:overview`, `tab:sr:subgroups` | `plots/sr/fig1_psr_violin`, `fig2_mean_slope_W`, `fig3_heatmap`, `fig4_sr_efficiency`, `fig5_sr_accuracy`, `fig6_subgroups` | `final_dataset_new_system` | reproducible |
| `05_bistability` | `sec:results:bistability`; `app:bistab` | `tab:bistab:overview` | `plots/bistab/fig1_neff_v_violin`, `fig2_label_bars`, `fig3_neff_accuracy`, `fig4_config_heatmaps` | `final_dataset_new_system` | reproducible |
| `06_dominance` | `sec:results:dominance` | `tab:dom:overview` | `plots/dominance/fig1_D_distribution`, `fig2_excess_fc`, `fig3_hub_stability`, `fig4_D_accuracy` | `final_dataset_new_system` | reproducible |
| `07_limit_cycles` | `sec:results:lc`; `sec:method:lc` figures | `tab:lc:funnel` | `plots/lc/lc_timelines_new`, `lc_rec_new`, `lc_onset_new` | `final_dataset_new_system` | reproducible |
| `08_high_repetition` | `sec:results:evid` (distributions); `app:highrep` | `tab:highrep:init_votes` | `plots/highrep/fig1_init_votes_split`, `fig2_h0_accuracy_efficiency`, `fig3_rounds_accuracy_flips`, `figA_full_feature_breakdown`, `figA_correlation_heatmap`, `figA_distributions_by_outcome` | `high_repetition` (R=1000) | reproducible |
| `09_seeded_causal` | `sec:results:seeded` | — | `plots/highrep/fig4_seeded_causal`, `fig5_seeded_dynamics`, `fig6_seeded_mechanism`, `fig7_recovery_difficulty` | `seeded_init_0/1` + `high_repetition` | reproducible |
| `10_linguistic` | `sec:results:linguistic`; `sec:method:linguistic` | `tab:ling:diversity` | — (table-only section) | `linguistic_mistral`, `linguistic_nli` + `final_dataset_new_system` | repointed (see mismatch note) |
| `11_appendix_robustness` | `app:sysprompt` | `tab:app:sysprompt` | — (table-only) | `final_dataset_depricated` (old arm) + `final_dataset_new_system` | repointed |

Cross-chapter re-quotes (no recomputation — they cite the owning notebook): ch1 `sec:intro:*`
headline "thirteen-fold" → `09`; ch7 `sec:results:synthesis` / `sec:discussion:*` restate results
from `03`–`10`; ch2/ch3 numbers are external citations (not thesis data).

## Per-notebook numbers (verified reproduction)

**01 setup** — corpus = 420 files, 50 reps/file = 21,000 (MATCH); 35 GPQA + 35 HiddenBench qids
(printed); message counts fc N(N-1)=12, star 2(N-1)=6 (MATCH); HiddenBench M: 31×M=3, 4×M=4.
GPQA 4-tier screening reproduced under repointed scan. *Frozen:* exact 35-task Tier-3 selection
(screening dirs deleted, HB scan empty), runtime estimates.

**02 worked examples** — all illustrative ch5 values MATCH: SR slopes +1.0/−0.5/0; N_eff
1.14/2.0/3.0; dominance σ=(61/6,25/6,5/3,0), shares (0.635,0.260,0.104,0), D≈0.31, hub 64%,
lone-dissenter D=1/9; RQA DET=1.0, P̂=2, nulls 0.10 / ~2e-4 / floor 1e-3; macrostates 15 (M=3),
35 (M=4).

**03 system performance** — baselines exact: GPQA 36.7/39.4/40.7/68.8 %, HB 15.5/8.3/29.4/46.6 %;
independence 83.9 % / 49.0 %; recovered 0.55; per-M 30.3 %/22.7 %; topology table (40.7/40.7,
3.77/4.23; 30.2/28.6, 5.12/6.96); cap-reach ranges; rounds-vs-accuracy buckets. **Ledger
corrections surfaced:** debate gain +1.4 pp (thesis +1.3); memory-window Friedman HB/fc p=0.019
(thesis had "all p≥0.10"); topology task-paired Wilcoxon.

**04 self-reinforcement** — p_SR 0.742–0.875 across all 12 cells, 35/35 sign test each; β̄ per
cell; Page trend, topology/dataset comparisons; subgroup table; efficiency ρ; mediation.

**05 bistability** — label counts monostable 90 / multistable 176 / stochastic 154 (multistable
41.9 %); N_eff means; basin Cramér's V; correct-basin.

**06 dominance** — emergent-excess z>0 all cells (p ≤ 1e-24); flagged fractions GPQA 7.3 % / HB
11.3 %; hub shares fc≈0.42–0.45, star≈0.76–0.79; D-accuracy; hub-stability HHI.

**07 limit cycles** — funnel counts; agent-level flagged 29, **genuine period-2 = 20/29 (69 %)**;
system-level flagged 23; taxonomy. Figures for the worked cases.

**08 high-repetition** — per-q accuracy (q84 0.269, q125 0.273, q144 0.509); GT-in-init-votes
near-deterministic (present 0.36–0.58 vs absent 0.004–0.07); H0/rounds/flips.

**09 seeded causal** — pooled init_0 0.0182 vs init_1 0.2371 → **13.0× lift**; paired Wilcoxon
34/34 tasks **p=3.62e-07**; de-novo recovery 15/34; overtake round; recovery-vs-difficulty.

**10 linguistic** — linguistic-only MATCH: 420 cells, means conviction 3.55/3.30, dialogic
3.01/2.51, NLI 221 files. *See mismatch note.*

**11 appendix robustness** — `tab:app:sysprompt`: 12 motif rows (accuracy, rounds, ceiling,
convergence, dominance D, hub capitulation, SR p_SR/slope, LC agent/system, N_eff, Cramér's V),
milder vs critical prompt, "# of 6 configs significant". *Old-prompt arm repointed.*

## ⚠ Known mismatches & frozen artifacts (action needed in the `.tex`)

1. **`tab:ling:diversity` (10)** — the thesis table was computed on the now-**deleted** old-prompt
   `results/mas/final_dataset` (correlations use MAS metrics from that corpus). Recomputed on the
   current `final_dataset_new_system` the MAS-linked correlations shift materially (e.g. anchoring
   diversity → Rounds ρ −0.39 vs thesis −0.55; Acc(HB) −0.11 vs −0.24). Linguistic-only numbers
   (means, distributions, counts) are unaffected. **Decide:** update the table to new-dataset
   values, or keep and footnote the provenance.
2. **LC genuine period-2 (07)** — notebook computes **20/29 agent (69 %)** with the DET≥0.70 gate;
   thesis text says 22 (76 %). (Ledger M1/M2.) Update text or drop the DET wording.
3. **Perf ledger items (03)** — debate gain **+1.4 pp** (not +1.3); memory-window **Friedman
   HB/fc p=0.019** (not "all p≥0.10"); topology comparisons restated at task level. Printed
   `computed vs thesis_says` in the notebook.
4. **Tier-3 selection (01)** — screening source (`gpqa_subset_scan`, `hiddenbench_subset_scan`)
   deleted; HB `*_nothink_small` scan is empty. The exact 35-task selection is a **frozen
   artifact**; the notebook prints the current corpus task IDs instead.
5. **Pilot figures** — `figures/time_savings.png`, `early_stopping*.png` and the ch4
   transition-matrix numbers (0.97–1.00 self-loop, "1 of 20") come from deleted pilot data →
   **frozen**, no golden notebook regenerates them.
6. **Robustness old arm (11)** — old-prompt data repointed to `final_dataset_depricated` (R=30,
   old persona); if that differs from the corpus the thesis table was built on, some `tab:app:sysprompt`
   rows may print MISMATCH.

## Not notebook-derived (documented, not audited as data)

Design constants: N=4, T=15, u=3, W∈{1,2,5}, temperature 0.7, model mistral-medium (config).
Dataset facts: GPQA 198 questions / M=4, HiddenBench M∈{3,4} (dataset properties).
External citations (ch2/ch3): 80.7 %/30.1 % HiddenBench, "nine benchmarks", etc. — from cited
papers, not this work.

## Superseded originals (kept for history, do not use as thesis source)

`scripts/build_nb_031–036.py`, `scripts/plot_lc_new.py`; notebooks `020/023/024/025`,
`026v2/027/028`, `008v2/030`, `029`.
