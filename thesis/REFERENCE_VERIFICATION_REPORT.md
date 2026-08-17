# Reference Verification Report

Scope: all 85 BibTeX entries in `thesis/bibliography.bib`, checked against ground
truth (Semantic Scholar batch API for 43 ID-bearing entries + web verification via
arXiv / DBLP / ACL Anthology / OpenReview / publisher pages for all 85). For every
one of the 77 cited keys, the paper's actual content was compared against the exact
thesis sentence(s) that cite it (185 citation sites, contexts extracted from all
chapters).

## Headline
- **1 fabricated citation removed** (`tian2025madm2` — no such paper exists).
- **3 wrong-paper / mismatched entries removed** (their arXiv IDs point to different papers).
- **1 duplicate removed** (`openai2023gpt4` = `achiam2023gpt4`).
- **9 metadata errors fixed** on cited entries (titles, author lists, years, ordering).
- **Claim support: 76/77 cited references genuinely support their claims.** The 1
  exception was the fabricated entry; its claim was re-anchored to real sources.
- **Venues: all published papers already carry correct venues.** 30 entries are
  genuinely arXiv-only (recent 2025/2026 MAD papers with no peer-reviewed version yet);
  no arXiv entry had a missed published version.
- Bib went from 85 -> 80 entries; 0 undefined references; all entries parse cleanly.

## HIGH severity — fabricated citation (removed + claim re-anchored)
| key | problem | fix |
|---|---|---|
| `tian2025madm2` | "MAD-M2: Mitigating Memory-Induced Errors ... Selective Masking", claimed TMLR 2025, author "Tian and others". **No such paper exists** on arXiv, OpenReview/TMLR, or Semantic Scholar. Used once in appendix to support "MAD depends on memory quality and erroneous memories propagate." | Entry deleted. Appendix sentence re-anchored to `wynn2025talk` (documents error persistence/propagation across debate rounds) + `liu2024lostmiddle` (uneven long-context use). |

## Wrong-paper / mismatched entries (removed — all were unused)
| key | bib claimed | arXiv ID actually is |
|---|---|---|
| `wang2025confmad` | "ConfMAD: Confidence-Aware MAD", Wang et al. | 2509.14034 = "Enhancing MAD System Performance via Confidence Expression", Lin & Hooi |
| `sun2025deliberative` | "Deliberative Dynamics and Value Alignment in LLM Debates", Sun et al. | 2510.10002 = "Interaction Protocol Shapes Moral Judgment in MAD", Sachdeva & Nuenen |
| `wu2026conformity` | author "Wu and others" | 2601.05606 first author is Chen Han et al. |

## Duplicate (removed)
- `openai2023gpt4` (@misc) was a redundant duplicate of `achiam2023gpt4` (@article), same arXiv:2303.08774. Unused.

## Metadata fixes on CITED entries
| key | field | before -> after |
|---|---|---|
| `li2026hiddenbench` | year + title | 2026 -> **2025**; title -> authoritative "Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs". (Paper *does* name its benchmark "HiddenBench" — prose nickname kept, verified.) |
| `zhu2025multiagentbench` | authors | fabricated 23-name list -> correct 11 authors (Zhu, Du, Hong, Yang, Guo, Wang, Wang, Qian, Tang, Ji, You); added pages + DOI. Venue ACL 2025 confirmed. |
| `chang2024partnr` (PARTNR) | authors | "Patel"->"Patki"; fabricated tail (Savva/Spiridonov/Straub/Szot/Wijmans/Yokoyama/Yenamandra/Zhao/Batra) replaced with official 20-author list. |
| `geng2025realmbench` | title | -> "REALM-Bench: A Benchmark for Evaluating Multi-Agent Systems on Real-World, Dynamic Planning and Scheduling Tasks". |
| `prasad2025certain` | title | "Two LLMs Debate, Both Are Certain They've Won" -> "When Two LLMs Debate, Both Think They'll Win". |
| `zhu2026demystifying` | authors + title | "Zhu and others" -> full 6 authors; added subtitle "The Role of Confidence and Diversity". |
| `li2024sparsetopology` | authors | "Li, Yunxuan and others" -> full 7 authors. |
| `liang2024encouraging` | author order | last two swapped to correct order (Shi before Tu). |
| `chuang2025debate` | author order | "Vasani, Li" -> "Li, You; Vasani, Smit" (correct order). |
| `ouyang2022instructgpt` | type | @article -> @inproceedings (NeurIPS 2022) + volume/pages. |

## Other corrections
- `wang2025dar` (unused): was placeholder "Wang, others" / "arXiv preprint" with no ID.
  It is a **real** paper — corrected to Nguyen et al. 2026, arXiv:2603.20640.
- `du2024improving`: added ICML 2024 pages + PMLR publisher.
- `alon2007network`: added DOI 10.1038/nrg2102.
- Removed stray non-BibTeX line ("------------ Knobs to tweak") -> commented out.

## Semantic Scholar glitches — bib was CORRECT, deliberately NOT changed
- `marwan2007recurrence`: S2 lists year 2025; correct is 2007 (Physics Reports 438). Kept 2007.
- `theiler1992surrogate`: S2 OCR-garbled author names ("Theder","Ludank"); bib names correct. Kept.
- `arthur1989competing`: S2 title truncated; bib full title correct. Kept.

## Verified correct (no change) — highlights
Foundational/method refs all confirmed: `milo2002network`, `alon2007network`,
`strogatz2015nonlinear`, `efron1979bootstrap`, `eckmann1987recurrence`,
`zbilut1992embeddings`, `webber1994dynamical`, `cocodale2014crqa`,
`theiler1992surrogate`, `driscoll2024dynamical`, `freeman1978centrality`,
`degroot1974consensus`, `jia2015opinion`, `bonacich1972factoring`,
`martin2014localization`, `newman2010networks`, `ashby1947principles`,
`cramer1946mathematical`, `baron1986moderator`, `hill1973diversity`,
`menck2013basin`, `feudel2008complex`, `pisarchik2014control`,
`watanabe2014energy`, `ezaki2017energy`, `martins2008coda`, `klemm2005stableunstable`,
`hirschman1964paternity`.
MAD/LLM refs confirmed with correct venues: `du2024improving` (ICML 2024),
`chan2024chateval` (ICLR 2024), `chen2024reconcile` (ACL 2024),
`zhu2025multiagentbench` (ACL 2025), `zhou2024sotopia` (ICLR 2024),
`chen2024agentverse` (ICLR 2024), `hendrycks2021mmlu` (ICLR 2021),
`hendrycks2021math` (NeurIPS 2021 D&B), `geva2021strategyqa` (TACL 2021),
`wei2022chainofthought` (NeurIPS 2022), `kojima2022zeroshot` (NeurIPS 2022),
`rein2024gpqa` (COLM 2024), `kenton2024debate` (NeurIPS 2024),
`kaesberg2025voting` (Findings of ACL 2025), `li2024sparsetopology` (Findings of EMNLP 2024),
`yin2023eot` (EMNLP 2023), `zhang2024socialpsych` (ACL 2024),
`abdelnabi2024cooperation` (NeurIPS 2024 D&B), `zhang2025debateorvote` (NeurIPS 2025).

## Notes / residual
- 4 entries are defined but currently uncited (`christiano2017deep`, `eo2025downsample`,
  `klemm2005stableunstable`, `wang2025dar`). All are real, correct, and harmless
  (uncited entries do not appear in the compiled reference list). Left in place.
- `achiam2023gpt4` uses "and others" for authors — standard/acceptable for the
  100+ author GPT-4 report.

---

# Part 2: Missing-citation audit (claims that needed a source)

A second pass fanned out one agent per chapter to find claims that a reader would
expect to be sourced but were uncited (named methods, benchmarks, established
findings). 31 candidate gaps were triaged; every recommended reference was
web/S2-verified as real before use. **16 new references added** (all cited once at
first use); several gaps were filled by **reusing existing entries**; ubiquitous
textbook operations and the thesis's own contributions were left uncited by design.

## New references added (16) and where cited
| ref | claim it now supports | location |
|---|---|---|
| `vaswani2017attention` | "an LLM is ... a Transformer" | 02 background |
| `grofman1983thirteen` | error-correction / Condorcet-jury intuition | 02 background |
| `wang2023selfconsistency` | self-consistency baseline | 03 related work |
| `wei2022chainofthought` (reuse) | chain-of-thought baseline (first methodological mention) | 03 related work |
| `zheng2024selectors` | LLMs show position bias -> randomised peer order | 04 system design |
| `benjamini1995fdr` | Benjamini-Hochberg FDR control | 05 detection (methods) |
| `wilcoxon1945signedrank` | Wilcoxon signed-rank test | 05 detection (methods) |
| `friedman1937test` | task-blocked Friedman test | 05 detection (methods) |
| `phipson2010permutation` | add-one permutation p-value | 05 detection |
| `becker1997disconnectivity` | disconnectivity graph | 05 detection |
| `zheng2023mtbench` | LLM-as-a-judge scoring | 05 detection |
| `page1963trend` | Page trend test | 06 results |
| `kruskal1952` | Kruskal-Wallis test | 06 results |
| `wilson1927ci` | Wilson score CI | 06 results |
| `liang1986gee` | GEE (exchangeable working correlation) | 06 results |
| `holm1979` | Holm multiple-comparison correction | appendix |
| `sun2025collaboverco` | Collab-Overcooked benchmark | appendix |

## Filled by reusing existing entries
| claim | reused key | location |
|---|---|---|
| bistability definition (parallel to the other 3 motifs) | `feudel2008complex` | 01 intro |
| "MAD is often presented as a way to reason better" | `du2024improving`, `liang2024encouraging` | 08 conclusion |
| "topology-tuning results in the literature are mixed" | `yang2025revisiting` | 07 discussion |

## Deliberately NOT added (avoid over-citing)
- Re-mentions of methods already cited at first use: BH-FDR, Friedman, Wilcoxon,
  bootstrap (`efron1979bootstrap`), surrogate tests (`theiler1992surrogate`),
  chain-of-thought, self-consistency — sourced once at first use, not on every recurrence.
- Ubiquitous textbook operations (Spearman correlation, sign test) and the thesis's
  own results/definitions.

## Net bibliography state
- 85 -> 80 (Part 1 cleanup) -> **96 entries** (Part 2 additions).
- 0 undefined references, 0 duplicate keys, all entries brace-balanced and parse cleanly.

