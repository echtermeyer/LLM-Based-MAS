# Statistical Review Ledger — Verified Claim-by-Claim

Generated 2026-08-15 from a claim-verification pass: every quantitative claim in the results/setup
was reproduced from the build-script code AND independently recomputed from the raw JSON data, with
task-clustered corrected values. This is the working checklist for the statistical revision.

Conventions being applied (simplest defensible, see thesis methods paragraph):
1. Pseudo-replication -> TASK-LEVEL aggregation (unit = task, n~35), then ordinary test.
2. Multiplicity -> Benjamini-Hochberg FDR within each family.
3. "Doesn't matter" negatives -> effect size + task-level CI, no formal TOST.

---

## 1. Reproduction status per analysis

| Analysis | Build script | Code reproduces? | LaTeX matches code? | Verdict |
|---|---|---|---|---|
| Self-reinforcement (`sec:results:sr`) | build_nb_032.py | Yes | minor-mismatch | Core prevalence/CI/Page/topology correct & task-level. 2 text errors; Simpson-paradox para has NO source in code; interaction +0.11 needs z-scoring; some p's pseudoreplicated. |
| Bistability (`sec:results:bistability`) | build_nb_033.py | Yes | minor-mismatch | Descriptives+consistency clean. 2 table transcription errors, 1 test-count error; N_eff<=M FIXED; B=1000 ok. 3-4 HB significance claims collapse at task level. |
| Dominance (`sec:results:dominance`) | build_nb_034.py | Yes | minor-mismatch | Transcribed numbers reproduce. B=200 vs text B=1000; hub-correct tautology; clustering OVERTURNS two D-accuracy conclusions. |
| Limit cycles (`sec:results:lc`) | build_nb_035.py | Yes | minor-mismatch | Funnel/counts/topology/taxonomy reproduce. DET>=0.70 gate described but NOT implemented; "all six HB/star" wrong; null-floor p's pseudoreplicated. |
| Seeded-causal (`sec:results:seeded`) | build_nb_036.py | Yes | all-match | Every number matches. Primary paired-Wilcoxon correct. Divided-opposition chi2 cites wrong comparison + fails clustering; logistic p's collapse under GEE. |
| High-rep (`sec:results:highrep`) | build_nb_031.py | Yes | all-match | Every number matches. Within-task tests VALID (independent RNG). Only n=3 tasks + one mild overstatement. |
| System-performance (`sec:results:perf`) | nb 020/023-025 hand-authored; build_nb_030 for Tier-3 | Yes | minor-mismatch | Descriptives reproduce. Rank tests on binary outcome; "all p>=0.10" false (HB/fc); Tier-3 provenance mismatch + source data missing. |

---

## 2. WRONG / MISMATCHED claims (fix first) — file:line -> old -> correct

1. LC DET>=0.70 gate NOT in code (`06_results.tex:709-710,729-730`; `05_detection.tex`). classify_flag checks only P==2 & longest_run<=L/2. FIX: add `and det>=0.70` (agent genuine 22->20, system 12->7) OR strip DET wording.
2. "22 (76%) agent genuine" (`06_results.tex:728-730`) -> 20 (69%) if gate enforced.
3. "12 of 23 system genuine" (`06_results.tex:733`) -> 7 of 23 if gate enforced.
4. "all six cross-level in HB/star" (`06_results.tex:798-800`) -> 5 HB/star + 1 GPQA/star (qid90/rep43, the collective cycle highlighted at L723-725). SELF-CONTRADICTORY.
5. W-accuracy "all p>=0.10 (ns)" (`06_results.tex:224,232`) -> FALSE: HB/fc KW p=0.096; task-blocked Friedman HB/fc p=0.019 (SIGNIFICANT), others 0.175/0.281/0.679.
6. GPQA/fc & HB/star D-accuracy "not significant" (`06_results.tex:1544`) -> task-level GPQA/fc d=+0.025 p=0.044; HB/star d=-0.051 p=0.010 (NEGATIVE).
7. "No config shows dominance -> lower accuracy" (`06_results.tex:1545-1546`) -> CONTRADICTED by HB/star (d=-0.051, p=0.010). Delete/soften.
8. Topology-accuracy MWU p=1.00/0.075 (`06_results.tex:176,179,187`) -> wrong test on 0/1. Task: GPQA Wilcoxon p=0.903, HB p=0.151 (LMM 0.050). Null holds; add effect-size CI.
9. Divided-opposition chi2=14 p=2.2e-4 (`06_results.tex:1844-1846`) -> chi2=14 is wrong comparison (3-0-0 vs ALL-divided); cited 19.5% vs 32.3% is chi2=7.9 p=4.9e-3. Task-clustered Wilcoxon p=0.053 (ns), CI[-0.004,0.150].
10. Dataset star "GPQA>HB all p<0.005" (`06_results.tex:414-416`) -> star/W5 p=0.047. "p<0.005 at W1,W2; p<0.05 at W5".
11. Interaction SR x GT "+0.11 both" (`06_results.tex:606-611`) -> only z-scored; raw LPM +0.42/+0.48. State "per 1 SD of beta".
12. Marginal rpb max "+0.049" (`06_results.tex:580-582`) -> +0.052.
13. Table GPQA Cramer V mean "0.47" (`06_results.tex:1264`) -> 0.46.
14. Table HB N_eff mean "2.19" (`06_results.tex:1265`) -> 2.20 (2.19 = median; possible mean/median mixup).
15. "all 24 MWU tests" (`06_results.tex:1293`) -> 12 (design 2x3x2; code runs 12).
16. Dominance "B=1000 surrogates" (`05_detection.tex:633`) -> code B=200. Rerun at B=1000 (chosen) or change text.
17. GPQA debate gain "+1.3 pp" (`06_results.tex:78`) -> +1.4 pp.
18. GPQA rounds W-effect "all p>=0.21" (`06_results.tex:245`) -> GPQA/fc p=0.2055; ">=0.20".
19. LC worked example "p~2e-4" (`05_detection.tex:363-364`) -> floor 1/(B+1)~1e-3.

## Overstated / pseudoreplicated (restate at task level)

20. GPQA/star D higher-for-correct +0.039 p<0.001 (`06_results.tex:1542-1543`) -> task d=+0.003 p=0.81. VANISHES.
21. HB N_eff-accuracy rho=0.15 p=0.027 (`06_results.tex:1324`) -> task rho=0.14 p=0.43. NO.
22. HB partial N_eff-acc rho=0.18 p=0.009 (`06_results.tex:1325`) -> task rho=0.175 p=0.32. NO.
23. HB stochastic<multistable acc p=0.003 (`06_results.tex:1333`) -> bootstrap p=0.066. NO (marginal).
24. HB N_eff-vs-SR rho=-0.195 p=0.004 (`06_results.tex:1375`) -> task rho=-0.258 p=0.134. NO.
25. HB GT-absent SR-acc rpb=-0.04 p<0.003 (`06_results.tex:590-594`) -> cluster CI incl 0. NO (GPQA -0.11 survives).
26. Confidence gain 1.66 vs 1.64 GPQA MWU p=0.001 (`06_results.tex:624-627`) -> task d=-0.055 p=0.10 (sign FLIPS). Supports "negligible".
27. conf_advantage logistic p<0.01 (`06_results.tex:1849-1850`) -> GEE p=0.031.
28. n_correct_pool logistic p<1e-6 "helps most" (`06_results.tex:1850-1851`) -> GEE p=0.029; drop "helps most".
29. KW init-dist H=7914 p<1e-15 (`06_results.tex:483`) -> task-paired Wilcoxon p<1e-4 (effect real).
30. Efficiency rho -0.65..-0.81 p<1e-10 (`06_results.tex:560-561`) -> task -0.75..-0.89 p<1e-6 (STRONGER).
31. Mediation CI [0.194,0.214] 51% (`06_results.tex:508-538`) -> point est ok; CI too tight, needs task-stratified bootstrap.
32. Subgroup MWU all p<0.0001 (`06_results.tex:446`) -> pseudoreplicated; direction robust.
33. Correct-basin HB V 0.430 vs 0.325 "fixed dim" (`06_results.tex:1355`) -> mixes M=3+M=4. M=3-only 0.435 vs 0.333; bootstrap diff 0.102 CI[0.026,0.177] p=0.005. SURVIVES.
34. Emergent-excess z>0 p<1e-22 (`06_results.tex:1460-1461`) -> task 35/35 p=2.9e-11. SURVIVES.
35. Flag fraction GPQA p<1e-6 / HB p<0.001 (`06_results.tex:1462-1464`) -> task GPQA p=0.009, HB p=1.6e-5.
36. Hub final-vote correct 42.2% vs 25% p<0.001 (`06_results.tex:1504-1507`) -> TAUTOLOGY (hub=consensus). Relabel as group accuracy (task p=0.003).
37. HB/fc D correct 0.219 vs 0.150 p<1e-30 (`06_results.tex:1539-1541`) -> task d=+0.059 p=1e-4. SURVIVES.
38. D vs max-source rho 0.94-0.95 (`06_results.tex:1558-1560`) -> task 0.85/0.84. SURVIVES.
41. Seeded pooled chi2=364 p<0.001 (`06_results.tex:1766`) -> paired Wilcoxon p=3.6e-7 (already primary). Fig4b subtitle prints p<1e-80 -> fix.
42. Agent null-floor p<1e-17 (`06_results.tex:675`) -> rep-level p~1.7e-7.
43. System null-floor p<1e-15 (`06_results.tex:676`) -> 23/291 reps but 17 tasks; still sig.
40. GPQA "<=40% from round 5 onwards" (`06_results.tex:301-302`) -> violated by sparse late buckets; soften.

---

## 3. Method bugs to fix in code (build scripts / src)

- M1 DET>=0.70 gate: add to classify_flag (build_nb_035.py:152-164 / src/metrics/limit_cycles.py). -> 20 agent/7 system genuine.
- M2 Binary-outcome rank tests: nb 020 -> replace MWU/KW on 0/1 with two-proportion/chi2 + task-paired Wilcoxon / task-blocked Friedman.
- M3 Dominance B=200 -> 1000 (build_nb_034.py:78,294); rerun z/flag.
- M4 Hub-correctness tautology (build_nb_034.py:270-282): relabel as group accuracy; keep dominant-hub-stability as the non-tautological test.
- M5 Simpson-paradox accuracy paragraph absent from code (build_nb_032.py): add Part-4b computing GT-split rpb, interaction, amplification, confidence-gain.
- M6 Interaction +0.11 z-scored: state units or report raw +0.42/+0.48.
- M7 Divided-opposition + logistic pseudoreplicated (build_nb_036.py sec 3): refit GEE (groups=qid); replace pooled chi2 with task-paired Wilcoxon.
- M8 Cited chi2 != accompanying proportions (build_nb_036 / 06_results.tex:1844-1846).
- M9 Cramer V uncorrected: apply Bergsma or restrict to identical (M,shape) strata.
- M10 "within HB dimension fixed" false: restrict correct-basin to M=3.
- M11 Null-floor tests non-independent candidates: report rep-level.
- M12-M14 Pseudoreplication in SR/bistability/dominance inferential tests: task-level aggregation.
- M15 "all p>=0.10" false: reword + Friedman.
- M16 [CORRECTED - FALSE ALARM]: verifier compared ch04 against build_nb_030.py, but that is the HIGH-REP 3-task selection, NOT the Tier-3 35-task selection. The real Tier-3 code is notebook 008v2 (assign_tier), and it MATCHES ch04 exactly: screening fc/W1/R3 (confirmed by run_subset_scans_nothink.sh --w 1 --topology fc --r 3), Tier0=unanimous+noflips, Tier1<=4 rounds, Tier2=no outcome variance, Tier3=outcome varies. NO REWRITE NEEDED. Real residual issues: (a) screening source data results/mas/gpqa_subset_scan + hiddenbench_subset_scan DELETED -> figure task_difficulty_4tiers.png and exact 35-task list not reproducible (reproducibility caveat); (b) "35 highest-ranked by outcome variance" is imprecise since outcome_variance is BOOLEAN in code (no continuous rank) -> minor wording softening; (c) screening model ambiguous (run script has mistral-medium commented, mistral-small active).
- M17 nb 020 prints "R=30" while data R=50 (cosmetic).
- M18 Seeded Fig 4b subtitle prints p<1e-80 -> clustered value.

Non-stat: SR identifying signature (beta INCREASING in W, 05_detection.tex:207-212) is REFUTED (beta decreasing). Already disclosed at 06:395-401; ensure discussion frames SR by prevalence only.

---

## 4. Unverifiable / provenance gaps
1. Tier-3 selection (04_system_design.tex:588-624): build_nb_030 describes a different procedure (W2/R30 6-component composite, GPQA-only, no tiers) than text (fc/W1/R3, 4-tier, outcome-variance). Path results/mas/final_dataset/ missing; script won't run. Source data appears deleted. NEEDS AUTHOR.
2. SR Simpson-paradox paragraph (06_results.tex:589-627): numbers reproduce from raw JSON, correct, but computed nowhere in committed code. Add Part-4b cell (M5).
3. LC figure-caption DET values (fig:lc:rec 0.80-0.98; fig:lc:onset 0.86/1.00): plausible, within observed range, not individually re-extracted.

## Headlines that SURVIVE (do not touch)
Four motif prevalences (task-level template); seeded causal 13x / +22pp (paired Wilcoxon p=3.6e-7);
high-rep GT-presence near-deterministic; SR efficiency (stronger under clustering); emergent-excess;
bistability multistable/consistency; N_eff<=M fixed; correct-basin (M=3-only).
