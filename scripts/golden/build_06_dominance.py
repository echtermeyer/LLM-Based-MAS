import os
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def co(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# GOLDEN 06_dominance

**Thesis section:** `sec:results:dominance`  
**Provenance:** Dominance. Ported verbatim from scripts/build_nb_034.py (superseded).  
**Data:** results/mas/final_dataset_new_system  
**Figures exported to:** thesis/plots/dominance/ (fig1_D_distribution .. fig4_D_accuracy)  
**Status:** reproducible from committed data.

This is a GOLDEN notebook: it recomputes every number in the named thesis section from raw data and exports every figure to the fixed `thesis/plots/` path the LaTeX already includes. Numbers are printed in labelled blocks for manual copy into the thesis. See `notebooks/golden/INDEX.md` for the full section->notebook map.
""")

md(r"""# 034 — Dominance Analysis (new_system dataset, R=50)

Full dominance analysis on `results/mas/final_dataset_new_system`.
50 tasks × 6 configs × 2 datasets = 600 cells, R=50 reps each.

Figures exported to `thesis/plots/dominance/`.

Key design decisions carried forward:
- FC only for z/flag (star null is degenerate for >80% of reps due to forced leaf→centre conversions)
- Raw D reported for both topologies
- Task-level aggregation for cluster-robust inference
""")

co(r"""import sys
sys.path.insert(0, '../..')

import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from src.metrics.dominance import analyse_dominance, _influence_graph
from src.metrics.stats_utils import (
    task_paired_wilcoxon, task_one_sample_wilcoxon, task_sign_test,
    task_spearman, bh_fdr,
)

BASE   = Path('../..') / 'results' / 'mas' / 'final_dataset_new_system'
EXPORT = Path('../..') / 'thesis' / 'plots' / 'dominance'
EXPORT.mkdir(parents=True, exist_ok=True)

W_VALUES = [1, 2, 5]
TOPOS    = ['fc', 'star']
DATASETS = ['gpqa', 'hiddenbench']
Z_FLAG_TOPO = 'fc'

from src.viz.thesis_style import (apply_style, W_COLORS, DS_COLORS, T_COLORS,
                                   DS_LABELS, T_LABELS, ABBR)

apply_style()
print('setup ok')""")

co(r"""# ── Load and run dominance analysis ──────────────────────────────────────────
cell_rows = []
rep_rows  = []

t0 = time.time()
for f in sorted(BASE.glob('**/*.json')):
    d    = json.loads(f.read_text())
    W    = d['W']
    ds   = d['dataset']
    topo = d.get('topology_name', 'fc')
    qid  = d['question_id']
    gt   = d['ground_truth']
    reps = d['repetitions']
    M    = len(d['options'])
    N    = reps[0]['N']
    accuracy = float(np.mean([rep['correct'] for rep in reps]))

    rep_results = [analyse_dominance(rep, B=1000, seed=42) for rep in reps]

    valid    = [r for r in rep_results if r is not None]
    testable = [r for r in valid if r['testable']]
    Ds       = [r['D'] for r in valid]
    zs       = [r['z'] for r in testable]
    flags    = [r['flagged'] for r in testable]

    cell_rows.append({
        'dataset': ds, 'topology': topo, 'W': W, 'qid': qid,
        'ground_truth': gt, 'M': M, 'N': N, 'accuracy': accuracy,
        'n_reps': len(rep_results), 'n_valid': len(valid),
        'n_excluded': len(rep_results) - len(valid),
        'n_testable': len(testable), 'n_degenerate': len(valid) - len(testable),
        'D_mean': float(np.mean(Ds)) if Ds else np.nan,
        'D_median': float(np.median(Ds)) if Ds else np.nan,
        'z_mean': float(np.mean(zs)) if zs else np.nan,
        'flagged_fraction': float(np.mean(flags)) if flags else np.nan,
    })

    for rep_idx, (rep, r) in enumerate(zip(reps, rep_results)):
        if r is None:
            continue
        rep_rows.append({
            'dataset': ds, 'topology': topo, 'W': W, 'qid': qid, 'rep_idx': rep_idx,
            'M': M, 'correct': rep.get('correct', None),
            'D': r['D'],
            'z': r['z'],
            'p_value': r['p_value'],
            'testable': r['testable'],
            'flagged': r['flagged'],
            'hub': r['hub'],
            'hub_p': r['p'][r['hub']],
            'hub_is_consensus': r['hub_is_consensus'],
            'hub_capitulation': r['hub_capitulation'],
            'hub_correct': rep['trajectory'][-1]['phase_b'][r['hub']]['vote'] == gt,
        })

elapsed = time.time() - t0
df     = pd.DataFrame(cell_rows)
df_rep = pd.DataFrame(rep_rows)

df_task = (df_rep.groupby(['dataset','topology','W','qid'])
           .agg(D=('D','mean'),
                flagged=('flagged','mean'),
                z=('z','mean'),
                hub_p=('hub_p','mean'),
                hub_is_consensus=('hub_is_consensus','mean'),
                hub_capitulation=('hub_capitulation','mean'),
                n_testable=('testable','sum'))
           .reset_index())

print(f'Done in {elapsed:.1f}s  |  cells: {len(df)}  |  rep records: {len(df_rep)}')
print(f'Excluded (no conversions): {df["n_excluded"].sum()} / {df["n_reps"].sum()}')
print(f'Testable / valid         : {df["n_testable"].sum()} / {df["n_valid"].sum()}'
      f'  ({100*df["n_testable"].sum()/df["n_valid"].sum():.0f}%)')
print()
print('=== Accounting per (dataset, topology) ===')
for ds in DATASETS:
    for topo in TOPOS:
        s = df_rep[(df_rep['dataset']==ds) & (df_rep['topology']==topo)]
        c = df[(df['dataset']==ds) & (df['topology']==topo)]
        nv = len(s); nt = int(s['testable'].sum())
        nex = int(c['n_excluded'].sum())
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: valid={nv}  excl={nex}  testable={nt}'
              f'  ({100*nt/max(nv,1):.0f}%)')""")

md("---\n## Part 1 — Raw D distribution and topology comparison")

co(r"""print('=== Mean raw D by (dataset, topology, W) ===')
pivot_D = df_rep.groupby(['dataset','topology','W'])['D'].mean().unstack()
print(pivot_D.round(3).to_string())

print()
print('=== TOPOLOGY effect: star vs fc (paired Wilcoxon, task-level) ===')
def _sig(p): return '***' if p<1e-3 else '**' if p<1e-2 else '*' if p<0.05 else 'ns'
for ds in DATASETS:
    for w in W_VALUES:
        fc = df_task[(df_task.dataset==ds)&(df_task.topology=='fc')  &(df_task.W==w)].set_index('qid')['D']
        st = df_task[(df_task.dataset==ds)&(df_task.topology=='star')&(df_task.W==w)].set_index('qid')['D']
        c = fc.index.intersection(st.index)
        stat, p = sp_stats.wilcoxon(st[c], fc[c])
        ratio = st[c].mean() / fc[c].mean() if fc[c].mean() > 0 else float('nan')
        print(f'  {DS_LABELS[ds]:12s} W={w}: fc={fc[c].mean():.3f}  star={st[c].mean():.3f}'
              f'  (x{ratio:.1f})  {_sig(p)} (p={p:.1e})')""")

co(r"""# Fig 1 — D distribution by topology + dataset
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
for row, ds in enumerate(DATASETS):
    for col, topo in enumerate(TOPOS):
        ax = axes[row, col]
        sub = df_rep[(df_rep['dataset']==ds) & (df_rep['topology']==topo)]['D']
        ax.hist(sub, bins=40, color=T_COLORS[topo], edgecolor='white', alpha=0.8)
        ax.axvline(sub.median(), color='black', linestyle='--', lw=1.5,
                   label=f'median={sub.median():.3f}')
        ax.axvline(sub.mean(),   color='red',   linestyle=':',  lw=1.2,
                   label=f'mean={sub.mean():.3f}')
        ax.set_xlabel('D (Herfindahl dominance)')
        ax.set_ylabel('Count')
        ax.set_title(f'{DS_LABELS[ds]} / {T_LABELS[topo]}')
        ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(EXPORT / 'fig1_D_distribution.png')
plt.show()
print('fig1 saved')""")

md("---\n## Part 2 — Emergent excess in FC (z-score test)")

co(r"""print(f'=== Emergent excess in {Z_FLAG_TOPO.upper()}: z > 0? ===')
print('(star null degenerate for >80% reps -> FC only)')
print()
for ds in DATASETS:
    for w in W_VALUES:
        z = df_rep[(df_rep.dataset==ds)&(df_rep.topology==Z_FLAG_TOPO)
                   &(df_rep.W==w)&(df_rep.testable)]['z'].dropna()
        stat, p = sp_stats.wilcoxon(z, alternative='greater')
        print(f'  {DS_LABELS[ds]:12s} W={w}: mean_z={z.mean():+.3f}  '
              f'median={z.median():+.3f}  {_sig(p)} (p={p:.1e})  n={len(z)}')

print()
print('=== Flag rate vs 5% baseline (FC, testable) ===')
for ds in DATASETS:
    s = df_rep[(df_rep.dataset==ds)&(df_rep.topology==Z_FLAG_TOPO)&(df_rep.testable)]['flagged']
    k, n = int(s.sum()), len(s)
    p = sp_stats.binomtest(k, n, 0.05, alternative='greater').pvalue
    print(f'  {DS_LABELS[ds]:12s}: flagged={k}/{n}={k/n:.3f}  vs 5%  {_sig(p)} (p={p:.1e})')""")

co(r"""# Fig 2 — z violin FC + flag rate by W
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
data = [df_rep[(df_rep.dataset==ds)&(df_rep.topology==Z_FLAG_TOPO)&(df_rep.testable)]['z'].dropna()
        for ds in DATASETS]
parts = ax.violinplot(data, positions=[1,2], widths=0.6, showmeans=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(list(DS_COLORS.values())[i]); pc.set_alpha(0.55)
ax.axhline(0, color='black', linestyle='--', lw=1.2, label='z=0 (null)')
ax.set_xticks([1,2]); ax.set_xticklabels([DS_LABELS[d] for d in DATASETS])
ax.set_ylabel('z (D vs surrogate null)')
ax.set_title(f'FC emergent excess z')
ax.legend(fontsize=9)

ax = axes[1]
for ds in DATASETS:
    grp = (df_rep[(df_rep.dataset==ds)&(df_rep.topology==Z_FLAG_TOPO)&(df_rep.testable)]
           .groupby('W')['flagged'].mean().reset_index())
    ax.plot(grp['W'], grp['flagged'], marker='o', color=DS_COLORS[ds],
            label=DS_LABELS[ds], linewidth=2)
ax.axhline(0.05, color='black', linestyle='--', lw=1.2, label='5% chance')
ax.set_xlabel('W'); ax.set_ylabel('Flagged fraction'); ax.set_xticks(W_VALUES)
ax.set_title('Flagged rate by W (FC, testable)')
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(EXPORT / 'fig2_excess_fc.png')
plt.show()
print('fig2 saved')""")

md("---\n## Part 3 — W effect")

co(r"""print('=== W effect: Spearman rho(W,D) task-level, one-sided greater ===')
for ds in DATASETS:
    for topo in TOPOS:
        s = df_task[(df_task.dataset==ds)&(df_task.topology==topo)]
        rho, p = sp_stats.spearmanr(s['W'], s['D'])
        p1 = p/2 if rho>0 else 1-p/2
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: rho={rho:+.3f}  {_sig(p1)} (p1={p1:.3f})')

print()
print('=== Paired Wilcoxon W1->W5 ===')
for ds in DATASETS:
    for topo in TOPOS:
        w1 = df_task[(df_task.dataset==ds)&(df_task.topology==topo)&(df_task.W==1)].set_index('qid')['D']
        w5 = df_task[(df_task.dataset==ds)&(df_task.topology==topo)&(df_task.W==5)].set_index('qid')['D']
        c = w1.index.intersection(w5.index)
        stat, p = sp_stats.wilcoxon(w5[c], w1[c], alternative='greater')
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: W1={w1[c].mean():.3f}  W5={w5[c].mean():.3f}'
              f'  delta={w5[c].mean()-w1[c].mean():+.3f}  {_sig(p)} (p={p:.4f})')""")

md("---\n## Part 4 — Hub analysis")

co(r"""print('=== Hub influence share p_hub ===')
for ds in DATASETS:
    for topo in TOPOS:
        sub = df_rep[(df_rep['dataset']==ds)&(df_rep['topology']==topo)]
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: hub_p mean={sub["hub_p"].mean():.3f}'
              f'  p(hub>=50%)={(sub["hub_p"]>=0.5).mean():.3f}'
              f'  hub_is_consensus={sub["hub_is_consensus"].mean():.3f}'
              f'  hub_capitulation={sub["hub_capitulation"].mean():.3f}')

print()
print('=== Group accuracy vs 1/M chance (rep-level; == hub final-vote correctness because hub_is_consensus~1.0 -> tautological relabel; task-level test in Part 9) ===')
for ds in DATASETS:
    for topo in TOPOS:
        sub = df_rep[(df_rep['dataset']==ds)&(df_rep['topology']==topo)]
        frac = sub['hub_correct'].mean()
        # baseline: 1/M
        M = int(sub['M'].mode()[0])
        baseline = 1/M
        n = len(sub)
        k = int(sub['hub_correct'].sum())
        p = sp_stats.binomtest(k, n, baseline, alternative='greater').pvalue
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: hub_correct={frac:.3f}'
              f'  (baseline=1/{M}={baseline:.3f})  {_sig(p)} (p={p:.1e})')""")

md("---\n## Part 5 — Hub stability across repetitions (novel analysis)\n\nDoes the same agent keep winning the hub role, or is it randomly assigned each rep?")

co(r"""# Hub stability: for each (task, config), compute Herfindahl concentration of hub identities
hub_stab_rows = []
for f in sorted(BASE.glob('**/*.json')):
    d    = json.loads(f.read_text())
    W, ds, topo, qid = d['W'], d['dataset'], d.get('topology_name','fc'), d['question_id']
    gt   = d['ground_truth']
    reps = d['repetitions']

    rep_results = [analyse_dominance(rep, B=0, seed=0) for rep in reps]  # B=0: skip surrogates
    hubs = [r['hub'] for r in rep_results if r is not None]
    if not hubs:
        continue

    N = reps[0]['N']
    cnts = Counter(hubs)
    p = np.array([cnts.get(i, 0)/len(hubs) for i in range(N)])
    hub_hhi = float(np.dot(p, p))  # raw HHI (not normalized)
    hub_hhi_norm = (hub_hhi - 1/N) / (1 - 1/N)  # normalized like Herfindahl

    # fraction of reps where hub is agent 0 (centre in star) vs distributed
    most_common_hub = cnts.most_common(1)[0]
    dominant_hub_frac = most_common_hub[1] / len(hubs)
    dominant_hub_id   = most_common_hub[0]

    # hub correct: does the most common hub hold the correct answer most often?
    hub_vote_rows = [(r['hub'], reps[i]['trajectory'][-1]['phase_b'][r['hub']]['vote'])
                     for i, r in enumerate(rep_results) if r is not None]
    dominant_hub_votes = [v for (h, v) in hub_vote_rows if h == dominant_hub_id]
    dom_hub_correct_frac = sum(v==gt for v in dominant_hub_votes)/len(dominant_hub_votes) if dominant_hub_votes else np.nan

    hub_stab_rows.append({
        'dataset': ds, 'topology': topo, 'W': W, 'qid': qid,
        'hub_hhi': hub_hhi,
        'hub_hhi_norm': hub_hhi_norm,
        'dominant_hub_frac': dominant_hub_frac,
        'dominant_hub_id': dominant_hub_id,
        'dom_hub_correct_frac': dom_hub_correct_frac,
        'n_reps': len(hubs),
    })

df_stab = pd.DataFrame(hub_stab_rows)
print('=== Hub stability (HHI over hub identities per cell) ===')
print('  HHI_norm=0 -> random; HHI_norm=1 -> same hub always')
for ds in DATASETS:
    for topo in TOPOS:
        sub = df_stab[(df_stab.dataset==ds)&(df_stab.topology==topo)]
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: hub_hhi_norm mean={sub["hub_hhi_norm"].mean():.3f}'
              f'  dominant_frac mean={sub["dominant_hub_frac"].mean():.3f}')
print()

print('=== Dominant hub holds correct answer fraction ===')
for ds in DATASETS:
    for topo in TOPOS:
        sub = df_stab[(df_stab.dataset==ds)&(df_stab.topology==topo)].dropna(subset=['dom_hub_correct_frac'])
        M = 4 if ds == 'gpqa' else 3
        baseline = 1/M
        t, p = sp_stats.ttest_1samp(sub['dom_hub_correct_frac'], baseline)
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: dom_hub_correct={sub["dom_hub_correct_frac"].mean():.3f}'
              f'  (baseline={baseline:.3f})  t={t:.2f}  p={p:.4f}')""")

co(r"""# Fig 3 — hub stability violin + dominant hub correct fraction
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for topo in TOPOS:
    data = [df_stab[(df_stab.dataset==ds)&(df_stab.topology==topo)]['dominant_hub_frac'].values
            for ds in DATASETS]
    positions = [1,2] if topo == 'fc' else [3,4]
    parts = ax.violinplot(data, positions=positions, widths=0.5, showmeans=True)
    for pc, ds in zip(parts['bodies'], DATASETS):
        pc.set_facecolor(DS_COLORS[ds]); pc.set_alpha(0.55)

N = 4
ax.axhline(1/N, color='black', linestyle='--', lw=1.2, label=f'uniform=1/N={1/N:.2f}')
ax.set_xticks([1,2,3,4])
ax.set_xticklabels([f'fc/{ABBR[d]}' for d in DATASETS] +
                   [f'star/{ABBR[d]}' for d in DATASETS], fontsize=9)
ax.set_ylabel('Fraction of reps where dominant hub wins')
ax.set_title('Hub identity stability per cell')
ax.legend(fontsize=9)

ax = axes[1]
for topo in TOPOS:
    for ds in DATASETS:
        sub = df_stab[(df_stab.dataset==ds)&(df_stab.topology==topo)].dropna(subset=['dom_hub_correct_frac'])
        ax.scatter(sub['dominant_hub_frac'], sub['dom_hub_correct_frac'],
                   color=DS_COLORS[ds], marker=('o' if topo=='fc' else '^'),
                   s=22, alpha=0.55,
                   label=f'{T_LABELS[topo]} / {ABBR[ds]}', edgecolors='none')

ax.axhline(1/4, color='grey', lw=0.8, linestyle=':', label='chance (M=4)')
ax.axhline(1/3, color='orange', lw=0.8, linestyle=':', label='chance (M=3)')
ax.set_xlabel('Dominant hub fraction')
ax.set_ylabel('Dominant hub correct vote fraction')
ax.set_title('Hub stability vs correctness')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(EXPORT / 'fig3_hub_stability.png')
plt.show()
print('fig3 saved')""")

md("---\n## Part 6 — Dominance vs Accuracy")

co(r"""print('=== Spearman: D_mean vs task accuracy (cell-level) ===')
for ds in DATASETS:
    for topo in TOPOS:
        sub = df[(df.dataset==ds)&(df.topology==topo)].dropna(subset=['D_mean','accuracy'])
        r, p = sp_stats.spearmanr(sub['D_mean'], sub['accuracy'])
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: rho={r:+.3f}  {_sig(p)} (p={p:.4f})  n={len(sub)}')

print()
print('=== Per-rep D: correct vs incorrect (MWU) ===')
for ds in DATASETS:
    for topo in TOPOS:
        s = df_rep[(df_rep.dataset==ds)&(df_rep.topology==topo)].dropna(subset=['correct'])
        dc = s[s.correct==True]['D']; di = s[s.correct==False]['D']
        if len(dc)<2 or len(di)<2: continue
        stat, p = sp_stats.mannwhitneyu(dc, di, alternative='two-sided')
        sign = '+' if dc.mean()>di.mean() else '-'
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: D|correct={dc.mean():.3f}'
              f'  D|incorrect={di.mean():.3f}  (delta={dc.mean()-di.mean():+.3f})'
              f'  {_sig(p)} (p={p:.1e})')

print()
print('=== hub_is_consensus vs hub_correct: does leadership predict correctness? ===')
for ds in DATASETS:
    for topo in TOPOS:
        sub = df_rep[(df_rep.dataset==ds)&(df_rep.topology==topo)]
        # When hub IS consensus (its vote became group answer), is it correct more often?
        is_con = sub[sub['hub_is_consensus']==True]['hub_correct'].mean()
        not_con = sub[sub['hub_is_consensus']==False]['hub_correct'].mean()
        ct = pd.crosstab(sub['hub_is_consensus'].astype(int), sub['hub_correct'].astype(int))
        if ct.shape == (2,2):
            chi2, p, _, _ = sp_stats.chi2_contingency(ct)
        else:
            p = float('nan')
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: hub_is_consensus->hub_correct={is_con:.3f}'
              f'  vs not={not_con:.3f}  {_sig(p)} (p={p:.4f})')""")

co(r"""# Fig 4 — D vs accuracy scatter + correct vs incorrect D
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for ds in DATASETS:
    sub = df[df['dataset']==ds].dropna(subset=['D_mean','accuracy'])
    for topo in TOPOS:
        s = sub[sub['topology']==topo]
        rho, p = sp_stats.spearmanr(s['D_mean'], s['accuracy'])
        ax.scatter(s['D_mean'], s['accuracy'], color=T_COLORS[topo],
                   s=25, alpha=0.5, edgecolors='none',
                   label=f'{T_LABELS[topo][:4]} ({DS_LABELS[ds][:4]}) rho={rho:+.2f}')
ax.set_xlabel('Mean D per cell')
ax.set_ylabel('Task accuracy')
ax.set_title('Dominance vs accuracy')
ax.legend(fontsize=7, ncol=2)

ax = axes[1]
width = 0.35
x = np.arange(len(DATASETS))
for i, topo in enumerate(TOPOS):
    dc_means = [df_rep[(df_rep.dataset==ds)&(df_rep.topology==topo)&(df_rep.correct==True)]['D'].mean()
                for ds in DATASETS]
    di_means = [df_rep[(df_rep.dataset==ds)&(df_rep.topology==topo)&(df_rep.correct==False)]['D'].mean()
                for ds in DATASETS]
    ax.bar(x + (i-0.5)*width, dc_means, width/2, label=f'{T_LABELS[topo][:4]} correct',
           color=T_COLORS[topo], alpha=0.9)
    ax.bar(x + i*width,       di_means, width/2, label=f'{T_LABELS[topo][:4]} wrong',
           color=T_COLORS[topo], alpha=0.4)
ax.set_xticks(x); ax.set_xticklabels([DS_LABELS[d] for d in DATASETS])
ax.set_ylabel('Mean D'); ax.set_title('D: correct vs incorrect reps')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(EXPORT / 'fig4_D_accuracy.png')
plt.show()
print('fig4 saved')""")

md("---\n## Part 7 — Cross-motif: Dominance × Bistability")

co(r"""from src.metrics.bistability import analyse_bistability

bist_rows = []
for f in sorted(BASE.glob('**/*.json')):
    d = json.loads(f.read_text())
    W, ds, topo, qid, gt = d['W'], d['dataset'], d.get('topology_name','fc'), d['question_id'], d['ground_truth']
    r = analyse_bistability(d['repetitions'], seed=42)
    bist_rows.append({'dataset':ds,'topology':topo,'W':W,'qid':qid,
                      'n_eff':r['n_eff'],'cramers_v':r['cramers_v'],'label':r['label']})

df_bist = pd.DataFrame(bist_rows)
df_cross = df.merge(df_bist, on=['dataset','topology','W','qid'])

print('=== Spearman: D_mean vs N_eff (per dataset) ===')
for ds in DATASETS:
    sub = df_cross[df_cross['dataset']==ds].dropna(subset=['D_mean','n_eff'])
    r, p = sp_stats.spearmanr(sub['D_mean'], sub['n_eff'])
    print(f'  {DS_LABELS[ds]:12s}: rho={r:+.3f}  {_sig(p)} (p={p:.4f})  n={len(sub)}')

print()
print('=== D_mean by bistability label ===')
for lbl in ['monostable','multistable','stochastic']:
    sub = df_cross[df_cross['label']==lbl]['D_mean'].dropna()
    print(f'  {lbl:14s}: mean={sub.mean():.3f}  n={len(sub)}')

# MWU
mono_d = df_cross[df_cross['label']=='monostable']['D_mean'].dropna()
mult_d = df_cross[df_cross['label']=='multistable']['D_mean'].dropna()
stoc_d = df_cross[df_cross['label']=='stochastic']['D_mean'].dropna()
print()
for a_lbl, a_v, b_lbl, b_v in [
    ('mono', mono_d, 'multi', mult_d),
    ('mono', mono_d, 'stoch', stoc_d),
    ('multi', mult_d, 'stoch', stoc_d),
]:
    if len(a_v)<2 or len(b_v)<2: continue
    stat, p = sp_stats.mannwhitneyu(a_v, b_v, alternative='two-sided')
    print(f'  {a_lbl} vs {b_lbl}: {_sig(p)} (p={p:.4f})')""")

md("---\n## Part 8 — Conversion structure: do high-D reps have concentrated conversions?")

co(r"""# For each rep, compute: n_conversions, total_weight, max_single_source_weight/total
conv_struct_rows = []
for f in sorted(BASE.glob('**/*.json')):
    d = json.loads(f.read_text())
    W, ds, topo, qid, gt = d['W'], d['dataset'], d.get('topology_name','fc'), d['question_id'], d['ground_truth']
    for rep_idx, rep in enumerate(d['repetitions']):
        G = _influence_graph(rep)
        total = G.sum()
        if total == 0:
            continue
        per_source = G.sum(axis=1)  # influence generated by each source agent
        n_sources_active = int((per_source > 0).sum())
        max_source_frac = float(per_source.max() / total)
        conv_struct_rows.append({
            'dataset': ds, 'topology': topo, 'W': W, 'qid': qid, 'rep_idx': rep_idx,
            'total_conv_weight': float(total),
            'n_sources_active': n_sources_active,
            'max_source_frac': max_source_frac,
            'correct': rep.get('correct'),
        })

df_conv = pd.DataFrame(conv_struct_rows)
df_rep_aug = df_rep.merge(df_conv, on=['dataset','topology','W','qid','rep_idx'], how='left')

print('=== D vs max_source_frac (concentration of conversions on one source) ===')
for ds in DATASETS:
    sub = df_rep_aug[df_rep_aug['dataset']==ds].dropna(subset=['D','max_source_frac'])
    r, p = sp_stats.spearmanr(sub['D'], sub['max_source_frac'])
    print(f'  {DS_LABELS[ds]:12s}: rho={r:+.3f}  {_sig(p)} (p={p:.4f})  n={len(sub)}')

print()
print('=== D vs total_conv_weight ===')
for ds in DATASETS:
    for topo in TOPOS:
        sub = df_rep_aug[(df_rep_aug.dataset==ds)&(df_rep_aug.topology==topo)].dropna(subset=['D','total_conv_weight'])
        r, p = sp_stats.spearmanr(sub['D'], sub['total_conv_weight'])
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: rho={r:+.3f}  {_sig(p)} (p={p:.4f})')

print()
print('=== n_sources_active by topology (fewer in star?) ===')
for topo in TOPOS:
    sub = df_rep_aug[df_rep_aug.topology==topo]['n_sources_active']
    print(f'  {topo:5s}: mean={sub.mean():.2f}  median={sub.median():.1f}')""")

co(r"""# Summary table
print('=== DOMINANCE SUMMARY TABLE ===')
print()
print(f'Total reps: {df["n_reps"].sum():,}  |  valid: {df["n_valid"].sum():,}'
      f'  |  testable FC: {df[df.topology==Z_FLAG_TOPO]["n_testable"].sum():,}')
print()
print('Mean raw D:')
for ds in DATASETS:
    for topo in TOPOS:
        sub = df_rep[(df_rep.dataset==ds)&(df_rep.topology==topo)]['D']
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: {sub.mean():.3f}  (star/fc ratio: '
              + (f'{df_rep[(df_rep.dataset==ds)&(df_rep.topology=="star")]["D"].mean()/df_rep[(df_rep.dataset==ds)&(df_rep.topology=="fc")]["D"].mean():.1f}x)' if topo=='star' else 'see star)'))
print()
print('FC emergent excess z (pooled):')
for ds in DATASETS:
    z = df_rep[(df_rep.dataset==ds)&(df_rep.topology==Z_FLAG_TOPO)&(df_rep.testable)]['z'].dropna()
    stat, p = sp_stats.wilcoxon(z, alternative='greater')
    print(f'  {DS_LABELS[ds]:12s}: mean_z={z.mean():+.3f}  flagged={z[z>0].shape[0]/len(z):.3f}  {_sig(p)}')""")

md("---\n## Part 9 — Task-level corrections (n~50) and BH-FDR\n\nAll tests below aggregate reps to one value per task (unit = task) to avoid pseudo-replication, then apply Benjamini-Hochberg FDR within each family of related tests.")

co(r"""fam = {}

print('=== TASK-LEVEL (n~50): FC emergent excess z>0 (one-sample across tasks) ===')
for ds in DATASETS:
    s = df_rep[(df_rep.dataset==ds)&(df_rep.topology=='fc')&(df_rep.testable)]
    per_task = s.groupby('qid')['z'].mean().values
    stat, pw, n = task_one_sample_wilcoxon(per_task, null=0.0)
    k, ns, ps = task_sign_test(per_task, null=0.0, alternative='greater')
    print(f'  {DS_LABELS[ds]:12s}: mean_task_z={np.nanmean(per_task):+.3f}  pos={k}/{ns}'
          f'  sign p={ps:.2e}  wilcoxon p={pw:.2e}  n={n}')

print()
print('=== TASK-LEVEL (n~50): FC flagged fraction vs 5% [flag-rate family] ===')
for ds in DATASETS:
    s = df_rep[(df_rep.dataset==ds)&(df_rep.topology=='fc')&(df_rep.testable)]
    per_task = s.groupby('qid')['flagged'].mean().values
    stat, p, n = task_one_sample_wilcoxon(per_task, null=0.05)
    fam.setdefault('flag_rate', []).append((DS_LABELS[ds], p))
    print(f'  {DS_LABELS[ds]:12s}: mean_task_flag={np.nanmean(per_task):.3f}  vs 0.05  p={p:.2e}  n={n}')""")

co(r"""print('=== TASK-LEVEL (n~50): group accuracy vs 1/M chance ===')
print('(relabels tautological hub-correctness; hub_is_consensus~1.0 -> hub-correct == group-correct)')
for ds in DATASETS:
    for topo in TOPOS:
        s = df[(df.dataset==ds)&(df.topology==topo)]
        per_task = s.groupby('qid').apply(lambda x:(x['accuracy']-1.0/x['M']).mean(),
                                          include_groups=False).values
        stat_g, pg = sp_stats.wilcoxon(per_task, alternative='greater')
        stat, p2, n = task_one_sample_wilcoxon(per_task, null=0.0)
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: mean(acc-1/M)={np.mean(per_task):+.3f}'
              f'  one-sided p={pg:.4f}  two-sided p={p2:.4f}  n={n}')""")

co(r"""print('=== TASK-LEVEL (n~50): dominant-hub-correct vs 1/M [dominant-hub-correct family] ===')
print('(non-tautological analogue: does the recurring dominant hub hold the correct answer?)')
Mq = df.groupby('qid')['M'].first()
for ds in DATASETS:
    for topo in TOPOS:
        s = df_stab[(df_stab.dataset==ds)&(df_stab.topology==topo)].dropna(subset=['dom_hub_correct_frac'])
        g = s.groupby('qid')['dom_hub_correct_frac'].mean()
        diff = (g - 1.0/Mq.loc[g.index]).values
        stat, p, n = task_one_sample_wilcoxon(diff, null=0.0)
        fam.setdefault('dom_hub_correct', []).append((f'{DS_LABELS[ds]}/{topo}', p))
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: dom_hub_correct={g.mean():.3f}'
              f'  mean(diff)={np.mean(diff):+.3f}  p={p:.4f}  n={n}')""")

co(r"""print('=== TASK-LEVEL (n~50): W1->W5 paired Wilcoxon (one-sided greater) [W-effect family] ===')
for ds in DATASETS:
    for topo in TOPOS:
        w1 = df_task[(df_task.dataset==ds)&(df_task.topology==topo)&(df_task.W==1)].set_index('qid')['D']
        w5 = df_task[(df_task.dataset==ds)&(df_task.topology==topo)&(df_task.W==5)].set_index('qid')['D']
        c = w1.index.intersection(w5.index)
        stat, p = sp_stats.wilcoxon(w5[c], w1[c], alternative='greater')
        fam.setdefault('W_effect', []).append((f'{DS_LABELS[ds]}/{topo}', p))
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: delta={w5[c].mean()-w1[c].mean():+.3f}  p={p:.4f}  n={len(c)}')""")

co(r"""print('=== TASK-LEVEL (n~50): D correct vs incorrect (task-paired Wilcoxon) [D-corr-vs-incorrect family] ===')
for ds in DATASETS:
    for topo in TOPOS:
        s = df_rep[(df_rep.dataset==ds)&(df_rep.topology==topo)].dropna(subset=['correct'])
        corr = s[s.correct==True].groupby('qid')['D'].mean()
        inc  = s[s.correct==False].groupby('qid')['D'].mean()
        c = corr.index.intersection(inc.index)
        stat, p, md, n = task_paired_wilcoxon(corr[c].values, inc[c].values)
        d = float((corr[c]-inc[c]).mean())
        fam.setdefault('D_corr_inc', []).append((f'{DS_LABELS[ds]}/{topo}', p))
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: d={d:+.3f}  median_diff={md:+.3f}  p={p:.4f}  n={n}')""")

co(r"""print('=== TASK-LEVEL (n~50): D vs accuracy Spearman [D-accuracy family] ===')
THESIS_D_ACC = {
    ('hiddenbench', 'fc'):   (+0.556, '<0.001', 'SURVIVES BH-FDR'),
    ('hiddenbench', 'star'): (-0.286, '0.044',  'nominal only'),
    ('gpqa',        'fc'):   (-0.148, 'ns',      'flat'),
    ('gpqa',        'star'): (+0.078, 'ns',      'flat'),
}
for ds in DATASETS:
    for topo in TOPOS:
        dt = df_rep[(df_rep.dataset==ds)&(df_rep.topology==topo)].groupby('qid')['D'].mean()
        at = df[(df.dataset==ds)&(df.topology==topo)].groupby('qid')['accuracy'].mean()
        c = dt.index.intersection(at.index)
        rho, p, n = task_spearman(dt[c].values, at[c].values)
        fam.setdefault('D_accuracy', []).append((f'{DS_LABELS[ds]}/{topo}', p))
        ts_rho = THESIS_D_ACC.get((ds, topo), (None, None, None))[0]
        match = 'MATCH' if ts_rho is None or abs(rho - ts_rho) < 0.01 else f'MISMATCH thesis_says={ts_rho}'
        print(f'  {DS_LABELS[ds]:12s} {topo:5s}: rho={rho:+.3f}  p={p:.4f}  n={n}  [{match}]')

print()
print('=== TASK-LEVEL (n~50): D vs max_source_frac Spearman (conversion structure) ===')
for ds in DATASETS:
    m = df_rep_aug[df_rep_aug.dataset==ds].dropna(subset=['D','max_source_frac'])
    dt = m.groupby('qid')['D'].mean()
    st = m.groupby('qid')['max_source_frac'].mean()
    rho, p, n = task_spearman(dt.values, st.values)
    print(f'  {DS_LABELS[ds]:12s}: rho={rho:+.3f}  p={p:.4f}  n={n}')

print()
print('=== FC hub influence share (~44%) ===')
for ds in DATASETS:
    v = df_rep[(df_rep.dataset==ds)&(df_rep.topology=='fc')]['hub_p'].mean()
    print(f'  {DS_LABELS[ds]:12s} fc: hub_p mean={v:.3f}')""")

co(r"""print('=== BH-FDR within families (q<0.05 survives) ===')
for name in ['flag_rate','W_effect','dom_hub_correct','D_accuracy','D_corr_inc']:
    items = fam.get(name, [])
    if not items:
        continue
    labels = [l for l, _ in items]
    ps = [p for _, p in items]
    q, rej = bh_fdr(ps)
    print(f'-- {name} (n={len(ps)}) --')
    for l, p, qq, rj in zip(labels, ps, q, rej):
        print(f'   {l:20s} p={p:.2e}  q={qq:.2e}  {"SURVIVES" if rj else "ns"}')""")

nb['cells'] = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
from pathlib import Path as _P
_out = _P(__file__).resolve().parents[2] / 'notebooks' / 'golden' / '06_dominance.ipynb'
_out.parent.mkdir(parents=True, exist_ok=True)
import nbformat as _nbf
with open(_out, 'w') as _fh:
    _nbf.write(nb, _fh)
print('wrote', _out, 'with', len(cells), 'cells')
