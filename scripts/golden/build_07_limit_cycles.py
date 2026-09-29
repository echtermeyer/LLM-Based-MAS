import os
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def co(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# GOLDEN 07_limit_cycles

**Thesis section:** `sec:results:lc`  
**Provenance:** Limit Cycles (analysis). Ported from scripts/build_nb_035.py; LC figures appended from scripts/plot_lc_new.py.  
**Data:** results/mas/final_dataset_new_system  
**Figures exported to:** thesis/plots/lc/ (lc_timelines_new, lc_rec_new, lc_onset_new)  
**Status:** reproducible from committed data.

This is a GOLDEN notebook: it recomputes every number in the named thesis section from raw data and exports every figure to the fixed `thesis/plots/` path the LaTeX already includes. Numbers are printed in labelled blocks for manual copy into the thesis. See `notebooks/golden/INDEX.md` for the full section->notebook map.
""")

md(r"""# 035 — Limit Cycle Analysis (new_system dataset, R=50)

Full limit cycle RQA analysis on `results/mas/final_dataset_new_system`.
50 tasks × 6 configs × 2 datasets = 600 cells, R=50 reps each.

Figures exported to `thesis/plots/lc/` (reuses existing plot subdirectory).
""")

co(r"""import sys
sys.path.insert(0, '../..')

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as sp_stats

from src.metrics.limit_cycles import (
    detect_agent_limit_cycles,
    detect_system_limit_cycles,
    summarise_lc,
    _recurrence_matrix,
    _det,
    _diagonalwise_rr,
    _trailing_k,
)

BASE   = Path('../..') / 'results' / 'mas' / 'final_dataset_new_system'
EXPORT = Path('../..') / 'thesis' / 'plots' / 'lc'
EXPORT.mkdir(parents=True, exist_ok=True)

W_VALUES = [1, 2, 5]
TOPOS    = ['fc', 'star']
DATASETS = ['gpqa', 'hiddenbench']

from src.viz.thesis_style import (apply_style, W_COLORS, DS_COLORS, T_COLORS,
                                   DS_LABELS, T_LABELS, VOTE_COLORS, CMAP_SEQ, no_grid)

apply_style()
print('setup ok')""")

co(r"""# Load and run LC detection
agent_rows = []
sys_rows   = []
skipped    = []

t0 = time.time()

for f in sorted(BASE.glob('**/*.json')):
    d    = json.loads(f.read_text())
    W    = d['W']
    ds   = d['dataset']
    topo = d.get('topology_name', 'fc')
    qid  = d['question_id']
    gt   = d['ground_truth']
    reps = d['repetitions']
    opts = tuple(reps[0]['options'].keys())

    for row in detect_agent_limit_cycles(reps, B=1000, seed=0):
        traj = reps[row['rep_idx']]['trajectory']
        seq  = [traj[t]['phase_b'][row['agent_idx']]['vote'] for t in range(len(traj))]
        agent_rows.append({'W': W, 'dataset': ds, 'topology': topo,
                           'qid': qid, 'ground_truth': gt, 'seq': seq, **row})

    try:
        sys_out = detect_system_limit_cycles(reps, B=1000, seed=0)
    except ValueError as e:
        skipped.append((f.name, str(e)))
        continue
    for row in sys_out:
        traj = reps[row['rep_idx']]['trajectory']
        seq  = [tuple(sum(ag['vote'] == o for ag in traj[t]['phase_b']) for o in opts)
                for t in range(len(traj))]
        sys_rows.append({'W': W, 'dataset': ds, 'topology': topo,
                         'qid': qid, 'ground_truth': gt, 'seq': seq, **row})

elapsed = time.time() - t0
agent_df = pd.DataFrame(agent_rows)
sys_df   = pd.DataFrame(sys_rows)
print(f'Done in {elapsed:.1f}s')
print(f'Agent rows : {len(agent_df):,}')
print(f'System rows: {len(sys_df):,}')
if skipped:
    print(f'[warning] {len(skipped)} file(s) skipped: ' + '; '.join(s[0] for s in skipped))""")

md("---\n## Part 1 — Accounting Funnel")

co(r"""def funnel(df, label):
    n_total = len(df)
    n_fp    = int(df['fixed_point'].sum())
    n_cand  = int((~df['fixed_point'] & (df['L'] >= 4)).sum())
    n_flag  = int(df['lc'].sum())
    p_lc    = n_flag / n_cand if n_cand > 0 else float('nan')
    print(f'{label}')
    print(f'  Total     : {n_total:>7,}')
    print(f'  Fixed pts : {n_fp:>7,}  ({100*n_fp/n_total:.2f}%)')
    print(f'  Candidates: {n_cand:>7,}  ({100*n_cand/n_total:.2f}%)')
    print(f'  Flagged   : {n_flag:>7,}  (p_LC={p_lc:.4f} among cands)' if n_cand else f'  Flagged: {n_flag}')
    print()

funnel(agent_df, 'AGENT-LEVEL')
funnel(sys_df,   'SYSTEM-LEVEL')""")

md("---\n## Part 2 — Config table")

co(r"""def config_table(df):
    rows = []
    for ds in DATASETS:
        for topo in TOPOS:
            for w in W_VALUES:
                sub = df[(df['dataset']==ds)&(df['topology']==topo)&(df['W']==w)]
                s = summarise_lc(sub.to_dict('records'))
                rows.append({
                    'Dataset': DS_LABELS[ds], 'Topology': T_LABELS[topo], 'W': w,
                    'n_total': s['n_total'], 'n_fp': s['n_fixed_point'],
                    'n_cand': s['n_candidates'], 'n_flagged': s['n_flagged'],
                    'p_LC': round(s['p_lc'], 4) if not np.isnan(s['p_lc']) else np.nan,
                })
    return pd.DataFrame(rows)

print('=== AGENT-LEVEL ===')
atab = config_table(agent_df)
print(atab.to_string(index=False))
print()
print('=== SYSTEM-LEVEL ===')
stab = config_table(sys_df)
print(stab.to_string(index=False))""")

md("---\n## Part 3 — Flagged case taxonomy and detail")

co(r"""def classify_flag(row):
    seq = row['seq']
    states = [tuple(s) for s in seq] if isinstance(seq[0], tuple) else list(seq)
    longest = cur = 1
    for i in range(1, len(states)):
        cur = cur + 1 if states[i] == states[i-1] else 1
        longest = max(longest, cur)
    P = row['period']
    det = row['det'] if row['det'] is not None else 0.0
    if P == 2 and longest <= len(states) // 2 and det >= 0.70:
        return 'genuine period-2 oscillation'
    if P == 2:
        return 'period-2 with long hold'
    return f'period-{int(P)} / stable-revisit'

for df, label in [(agent_df, 'AGENT'), (sys_df, 'SYSTEM')]:
    fl = df[df['lc']].copy()
    if len(fl) == 0:
        print(f'{label}: no flagged cases'); continue
    fl['kind'] = fl.apply(classify_flag, axis=1)
    print(f'=== {label}-LEVEL flagged (n={len(fl)}) ===')
    print(fl['kind'].value_counts().to_string())
    n_gen = int((fl['kind']=='genuine period-2 oscillation').sum())
    print(f'>>> genuine period-2 (DET>=0.70): {n_gen} of {len(fl)} ({100*n_gen/len(fl):.0f}%)')
    print()
    print(fl.groupby(['kind','topology']).size().to_string())
    print()
    cols = ['dataset','topology','W','qid','rep_idx','L','det','period','kind']
    if label == 'AGENT':
        cols.insert(5, 'agent_idx')
    print(fl[cols].sort_values(['kind','dataset','topology']).to_string(index=False))
    print()""")

md("---\n## Part 4 — Cross-level overlap")

co(r"""agent_flag_keys = set(
    agent_df[agent_df['lc']]
    .apply(lambda r: (r['dataset'],r['topology'],r['W'],r['qid'],r['rep_idx']), axis=1)
)
sys_flag_keys = set(
    sys_df[sys_df['lc']]
    .apply(lambda r: (r['dataset'],r['topology'],r['W'],r['qid'],r['rep_idx']), axis=1)
)
both = agent_flag_keys & sys_flag_keys
print(f'Reps with agent-level LC flag : {len(agent_flag_keys)}')
print(f'Reps with system-level LC flag: {len(sys_flag_keys)}')
print(f'Reps with BOTH flags          : {len(both)}')
if both:
    print()
    print('=== CROSS-LEVEL CO-OCCURRENCE BREAKDOWN ===')
    both_sorted = sorted(both, key=lambda k: (k[0], k[1], k[2], k[3], k[4]))
    for ds, topo, w, qid, rep in both_sorted:
        print(f'  {DS_LABELS[ds]:12s} / {T_LABELS[topo]:14s}  W={w}  qid={qid}  rep={rep}')
    ds_topo = pd.Series([(DS_LABELS[k[0]], T_LABELS[k[1]]) for k in both_sorted]).value_counts()
    print()
    print('  breakdown by (dataset, topology):')
    for (dsl, tl), c in ds_topo.items():
        print(f'    {dsl} / {tl}: {c}')""")

md("---\n## Part 5 — Null floor calibration (is the signal real?)")

co(r"""from src.metrics.limit_cycles import _dominant_period as _dp

def _flag_seq_fast(seq, B, rng):
    L = len(seq)
    if _trailing_k(seq) >= 3 or L < 4:
        return None
    R = _recurrence_matrix(seq); det = _det(R)
    cnt = sum(1 for _ in range(B)
              if _det(_recurrence_matrix([seq[i] for i in rng.permutation(L)])) >= det)
    p = (1 + cnt) / (1 + B)
    P = _dp(_diagonalwise_rr(R))
    return bool(p < 0.05 and 2 <= P <= L // 2)

def null_flag_rate(M, L_pool, n_trials, B, seed):
    rng = np.random.default_rng(seed)
    n_c = n_f = 0
    for _ in range(n_trials):
        L = int(rng.choice(L_pool))
        seq = list(rng.integers(0, M, size=L))
        res = _flag_seq_fast(seq, B, rng)
        if res is None: continue
        n_c += 1; n_f += int(res)
    return (n_f/n_c if n_c else float('nan')), n_c

print('Null-floor calibration (Monte-Carlo)...')
for label, df, Ms in [
    ('AGENT',  agent_df, {'gpqa': 4, 'hiddenbench': 3}),
    ('SYSTEM', sys_df,   {'gpqa': 4, 'hiddenbench': 3}),
]:
    cands = df[~df['fixed_point'] & (df['L'] >= 4)]
    n_obs = len(cands); k_obs = int(cands['lc'].sum())
    floors = []
    for ds, M in Ms.items():
        sub = cands[cands['dataset']==ds]
        if len(sub) == 0: continue
        fr, _ = null_flag_rate(M, sub['L'].tolist(), n_trials=3000, B=300, seed=42)
        floors.append((len(sub), fr))
    p0 = sum(w*f for w,f in floors) / sum(w for w,_ in floors)
    pval = sp_stats.binom.sf(k_obs - 1, n_obs, p0)
    print(f'\n{label}-level: flagged={k_obs}/{n_obs}  null-floor={p0:.4f}'
          f'  expected={n_obs*p0:.1f}  p={pval:.2e}'
          f'  --> {"REAL STRUCTURE" if pval<0.05 else "NOT ABOVE CHANCE"}')""")

md("---\n## Part 5b — Null floor at the REP level (M11: pseudoreplication-corrected)")

co(r"""rep_keys = ['dataset','topology','W','qid','rep_idx']

def rep_level_nullfloor(df, level, Ms):
    if level == 'AGENT':
        cand_ag = df[~df['fixed_point'] & (df['L'] >= 4)]
        cand = (cand_ag.groupby(rep_keys)
                .agg(m=('lc','size'), any_flag=('lc','any'), L=('L','max'))
                .reset_index())
    else:
        cand = df[~df['fixed_point'] & (df['L'] >= 4)].copy()
        cand['m'] = 1
        cand['any_flag'] = cand['lc']
    n_reps = len(cand)
    k_reps = int(cand['any_flag'].sum())
    n_tasks = cand.loc[cand['any_flag'], ['dataset','qid']].drop_duplicates().shape[0]
    p0_ds = {}
    for ds, M in Ms.items():
        sub = cand[cand['dataset']==ds]
        if len(sub) == 0: continue
        fr, _ = null_flag_rate(M, sub['L'].tolist(), n_trials=3000, B=300, seed=42)
        p0_ds[ds] = fr
    cand = cand.copy()
    cand['pflag'] = cand.apply(lambda r: 1 - (1 - p0_ds[r['dataset']]) ** r['m'], axis=1)
    p0 = float(cand['pflag'].mean())
    lam = float(cand['pflag'].sum())
    pval = sp_stats.binom.sf(k_reps - 1, n_reps, p0)
    return n_reps, k_reps, n_tasks, p0, lam, pval

print('=== REP-LEVEL NULL-FLOOR (M11) ===')
print('unit = rep; observed count = reps (star spokes slaved to hub, not double-counted);')
print('null = rep flags if any of its m candidate agents false-positives (agents iid under null)')
for label, df, Ms in [
    ('AGENT',  agent_df, {'gpqa': 4, 'hiddenbench': 3}),
    ('SYSTEM', sys_df,   {'gpqa': 4, 'hiddenbench': 3}),
]:
    n_reps, k_reps, n_tasks, p0, lam, pval = rep_level_nullfloor(df, label, Ms)
    print(f'\n{label}-level: flagged reps={k_reps}/{n_reps}'
          f'  (spanning {n_tasks} tasks)  rep-null-floor={p0:.4f}'
          f'  expected={lam:.2f}  p={pval:.2e}'
          f'  --> {"REAL STRUCTURE" if pval<0.05 else "NOT ABOVE CHANCE"}')""")

md("---\n## Part 6 — Rep-level prevalence (pseudoreplication-corrected)")

co(r"""rep_keys = ['dataset','topology','W','qid','rep_idx']
agent_rep = (agent_df.groupby(rep_keys)
             .agg(any_flag=('lc','any'), any_cand=('fixed_point', lambda s: (~s).any()))
             .reset_index())

rows = []
for ds in DATASETS:
    for topo in TOPOS:
        for w in W_VALUES:
            ar = agent_rep[(agent_rep.dataset==ds)&(agent_rep.topology==topo)&(agent_rep.W==w)]
            sr = sys_df[(sys_df.dataset==ds)&(sys_df.topology==topo)&(sys_df.W==w)]
            n  = len(ar)
            rows.append({
                'Dataset': DS_LABELS[ds], 'Topology': T_LABELS[topo], 'W': w,
                'n_reps': n,
                'agent_flag_reps': int(ar['any_flag'].sum()),
                'sys_flag_reps': int(sr['lc'].sum()),
            })
rtab = pd.DataFrame(rows)
print('=== REP-LEVEL PREVALENCE ===')
print(rtab.to_string(index=False))
print()
for col in ['agent_flag_reps','sys_flag_reps']:
    total = rtab[col].sum()
    by_topo  = rtab.groupby('Topology')[col].sum()
    by_ds    = rtab.groupby('Dataset')[col].sum()
    print(f'{col}: total={total}')
    print('  by topology:', by_topo.to_dict())
    print('  by dataset: ', by_ds.to_dict())
    print()""")

md("---\n## Part 7 — Novel analysis: non-convergence rate by question difficulty")

co(r"""# Non-convergence rate by task (does the same task consistently fail to converge?)
# This tests whether non-convergence is a task property or random

agent_rep2 = (agent_df.groupby(['dataset','topology','W','qid','rep_idx'])
              .agg(any_cand=('fixed_point', lambda s: (~s).any()))
              .reset_index())

task_cand_rate = (agent_rep2.groupby(['dataset','qid'])['any_cand'].mean().reset_index())

print('=== Task-level non-convergence rate stats ===')
for ds in DATASETS:
    sub = task_cand_rate[task_cand_rate['dataset']==ds]['any_cand']
    print(f'  {DS_LABELS[ds]:12s}: mean={sub.mean():.4f}  max={sub.max():.4f}'
          f'  tasks with >5% non-conv: {(sub>0.05).sum()}/35')

print()
print('=== Top non-converging tasks (all datasets) ===')
top = task_cand_rate.sort_values('any_cand', ascending=False).head(15)
print(top.to_string(index=False))

# Cross-config consistency: does the same task fail to converge across different configs?
print()
print('=== Cross-config Spearman of per-task non-conv rate ===')
for ds in DATASETS:
    sub = agent_rep2[agent_rep2['dataset']==ds]
    sub['config'] = sub['topology'] + '_W' + sub['W'].astype(str)
    piv = sub.groupby(['qid','config'])['any_cand'].mean().unstack('config')
    configs = list(piv.columns)
    rhos = []
    for i in range(len(configs)):
        for j in range(i+1, len(configs)):
            a = piv[configs[i]]; b = piv[configs[j]]
            mask = a.notna() & b.notna()
            if mask.sum() < 5: continue
            rho, _ = sp_stats.spearmanr(a[mask], b[mask])
            rhos.append(rho)
    print(f'  {DS_LABELS[ds]:12s}: pairwise Spearman mean={np.mean(rhos):.3f}'
          f'  min={np.min(rhos):.3f}  max={np.max(rhos):.3f}')""")

md("---\n## Part 8 — Recurrence plots for all flagged cases")

co(r"""def plot_recurrence(seq, title, ax_seq, ax_rec, ax_rr):
    L = len(seq)
    R = _recurrence_matrix(seq)
    rr = _diagonalwise_rr(R)
    det_val = _det(R)

    vote_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    if isinstance(seq[0], tuple):
        M = len(seq[0]); opts = ['A','B','C','D'][:M]
        bottoms = np.zeros(L)
        colors = ['#4C72B0','#DD8452','#55A868','#c44e52'][:M]
        for i, (opt, col) in enumerate(zip(opts, colors)):
            vals = [s[i] for s in seq]
            ax_seq.bar(range(L), vals, bottom=bottoms, color=col, label=opt, width=0.8)
            bottoms += np.array(vals, dtype=float)
        ax_seq.legend(fontsize=7, loc='upper right')
        ax_seq.set_ylabel('Agent count')
    else:
        cmap = plt.cm.Set1
        codes = [vote_map.get(v, 0) for v in seq]
        ax_seq.bar(range(L), [1]*L, color=[cmap(c/4) for c in codes], width=0.8)
        handles = [mpatches.Patch(color=cmap(i/4), label=v) for i,v in enumerate(['A','B','C','D'])]
        ax_seq.legend(handles=handles, fontsize=7, loc='upper right')
        ax_seq.set_yticks([])
    ax_seq.set_xlabel('Round')
    ax_seq.set_title(f'{title}\n{list(seq) if not isinstance(seq[0],tuple) else seq}', fontsize=8)

    ax_rec.imshow(R.astype(int), cmap='Greys', origin='upper', aspect='equal')
    ax_rec.set_title(f'Recurrence  DET={det_val:.3f}')

    taus = np.arange(1, L)
    ax_rr.bar(taus, rr, color='steelblue', alpha=0.8, width=0.6)
    if len(rr):
        P_hat = int(np.argmax(rr)) + 1
        ax_rr.axvline(P_hat, color='crimson', lw=1.8, linestyle='--', label=f'P={P_hat}')
        ax_rr.legend(fontsize=8)
    ax_rr.set_xlabel('Lag tau'); ax_rr.set_title('Diagonalwise RR'); ax_rr.set_xticks(taus)

for df, label in [(agent_df, 'AGENT'), (sys_df, 'SYSTEM')]:
    fl = df[df['lc']].reset_index(drop=True)
    if len(fl) == 0:
        print(f'{label}: no flagged cases')
        continue
    print(f'{label}-level flagged cases ({len(fl)}):')
    for _, row in fl.iterrows():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        level_info = f"agent={row['agent_idx']}" if 'agent_idx' in row else 'system'
        ttl = (f"{label} | {DS_LABELS[row['dataset']]} / {T_LABELS[row['topology']]} "
               f"W={row['W']} | qid={row['qid']} rep={row['rep_idx']} {level_info} "
               f"| p={row['p_value']:.4f} P={row['period']}")
        plot_recurrence(row['seq'], ttl, axes[0], axes[1], axes[2])
        plt.tight_layout()
        plt.show()""")

md("---\n## Part 9 — Summary and comparison with old dataset")

co(r"""print('=== SUMMARY ===')
for df, label in [(agent_df,'AGENT'),(sys_df,'SYSTEM')]:
    n = len(df); fp = int(df['fixed_point'].sum())
    c = int((~df['fixed_point']&(df['L']>=4)).sum()); fl = int(df['lc'].sum())
    print(f'{label}: total={n:,}  fixed_pts={fp:,} ({100*fp/n:.1f}%)'
          f'  candidates={c}  flagged={fl}'
          + (f'  p_LC={fl/c:.4f}' if c else ''))
print()
print('OLD dataset (R=30, no anti-conformity prompt): 14 agent, 16 system LCs')
print('NEW dataset (R=50, anti-conformity prompt):')
print(f'  agent:  {int(agent_df["lc"].sum())}')
print(f'  system: {int(sys_df["lc"].sum())}')
print()
# Dataset breakdown
for ds in DATASETS:
    a = int(agent_df[(agent_df.dataset==ds)&agent_df.lc].shape[0])
    s = int(sys_df[(sys_df.dataset==ds)&sys_df.lc].shape[0])
    print(f'  {DS_LABELS[ds]:12s}: agent={a}  system={s}')
print()
# Topology breakdown
for topo in TOPOS:
    a = int(agent_df[(agent_df.topology==topo)&agent_df.lc].shape[0])
    s = int(sys_df[(sys_df.topology==topo)&sys_df.lc].shape[0])
    print(f'  {topo:5s}: agent={a}  system={s}')""")

md(r"""---
## Limit-Cycle Figures (worked cases)

Ported from `scripts/plot_lc_new.py`. Regenerates the three LC figures used in `sec:results:lc` / `sec:method:lc`: `lc_timelines_new.png`, `lc_rec_new.png`, `lc_onset_new.png` in `thesis/plots/lc/`. Selected worked cases (GPQA/star/W2/q90/rep43; HB/star/W5/q20/rep45; HB/star/W5/q25/rep25).
""")

co(r"""import sys
sys.path.insert(0, '../..')

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.metrics.limit_cycles import _recurrence_matrix, _det, _diagonalwise_rr

BASE   = Path('../../results/mas/final_dataset_new_system')
OUTDIR = Path('../../thesis/plots/lc')
OUTDIR.mkdir(exist_ok=True)

from src.viz.thesis_style import apply_style, VOTE_COLORS, no_grid
apply_style()

def load_rep(ds, topo, W, qid, rep_idx):
    files = list(BASE.glob(f'**/*{ds}*W{W}*topo{topo}*q{qid}*.json'))
    assert files, f'Not found: {ds}/{topo}/W{W}/q{qid}'
    d = json.loads(files[0].read_text())
    rep = d['repetitions'][rep_idx]
    gt = d['ground_truth']
    opts = list(d['options'].keys())
    traj = rep['trajectory']
    N = rep['N']
    T = len(traj) - 1
    votes = [[traj[r]['phase_b'][i]['vote'] for i in range(N)] for r in range(T+1)]
    return {'votes': votes, 'gt': gt, 'opts': opts, 'N': N, 'T': T, 'traj': traj}

# ── FIGURE 1: Vote-sequence timelines ─────────────────────────────────────────
cases_timeline = [
    ('gpqa',        'star', 2, '90', 43,
     'GPQA / star / $W{=}2$ / q90 / rep~43\n(gt = B; collective cycle, all agents)',
     'a'),
    ('hiddenbench', 'star', 5, '20', 45,
     'HiddenBench / star / $W{=}5$ / q20 / rep~45\n(gt = A; hub alternates A$\\leftrightarrow$C)',
     'b'),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
legend_handles = [mpatches.Patch(facecolor=VOTE_COLORS[v], label=f'Vote {v}')
                  for v in 'ABCD']

for ax, (ds, topo, W, qid, rep_idx, title, panel) in zip(axes, cases_timeline):
    info = load_rep(ds, topo, W, qid, rep_idx)
    T = info['T']; N = info['N']; votes = info['votes']

    for r in range(T + 1):
        for i in range(N):
            v = votes[r][i]
            ax.scatter(r, i, color=VOTE_COLORS.get(v, '#999999'),
                       s=220, marker='s', zorder=3, edgecolors='white', lw=0.4)
            ax.text(r, i, v, ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

    ax.set_xlim(-0.6, T + 0.6)
    ax.set_ylim(-0.6, N - 0.4)
    ax.set_xticks(range(T + 1))
    ax.set_xticklabels([str(r) for r in range(T + 1)], fontsize=8)
    ax.set_yticks(range(N))
    ax.set_yticklabels([f'Agent {i}' for i in range(N)], fontsize=9)
    ax.set_xlabel('Round')
    ax.set_title(f'({panel}) {title}', loc='left', fontsize=10)
    ax.grid(False)
    ax.grid(axis='x', lw=0.4, alpha=0.4)

    # mark gt row background
    for i in range(N):
        seq = [votes[r][i] for r in range(T + 1)]
        if all(seq[r] == seq[r-1] for r in range(1, len(seq))):
            continue  # skip non-cycling

axes[0].legend(handles=legend_handles, loc='lower right', fontsize=8,
               framealpha=0.85, ncol=2)
plt.tight_layout()
out = OUTDIR / 'lc_timelines_new.png'
fig.savefig(out)
plt.show()
print(f'Saved {out}')


# ── FIGURE 2: Recurrence matrices ──────────────────────────────────────────────
cases_rec = [
    # (ds, topo, W, qid, rep_idx, agent_idx, label)
    ('gpqa',        'star', 2, '90', 43, 0, 'GPQA q90 rep43\nagent 0 (hub)'),
    ('gpqa',        'star', 2, '90', 43, 2, 'GPQA q90 rep43\nagent 2'),
    ('hiddenbench', 'star', 5, '20', 45, 0, 'HB q20 rep45\nagent 0'),
    ('hiddenbench', 'star', 5, '20', 45, 2, 'HB q20 rep45\nagent 2'),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))

for ax, (ds, topo, W, qid, rep_idx, ag, label) in zip(axes, cases_rec):
    info = load_rep(ds, topo, W, qid, rep_idx)
    seq = [info['votes'][r][ag] for r in range(info['T'] + 1)]
    R = _recurrence_matrix(seq)
    det_val = _det(R)
    rr = _diagonalwise_rr(R)
    P = int(np.argmax(rr)) + 1

    ax.imshow(R.astype(int), cmap='Blues', origin='upper', aspect='equal',
              vmin=0, vmax=1.2, interpolation='nearest')
    no_grid(ax)
    L = len(seq)
    ax.set_xticks(range(0, L, 2))
    ax.set_yticks(range(0, L, 2))
    ax.set_xticklabels(range(0, L, 2), fontsize=7)
    ax.set_yticklabels(range(0, L, 2), fontsize=7)
    ax.set_xlabel('Round $j$', fontsize=9)
    ax.set_ylabel('Round $i$', fontsize=9)
    ax.set_title(f'{label}\nDET={det_val:.2f}  $\\hat{{P}}={P}$', fontsize=9)

plt.tight_layout()
out = OUTDIR / 'lc_rec_new.png'
fig.savefig(out)
plt.show()
print(f'Saved {out}')


# ── FIGURE 3: Onset + mechanism for HB/q25/rep25 (hub drives lone dissenter) ──
# This case is especially clean: a single hub (agent 1, DET=1.0) vs three spokes
# Perfect period-2 from r5 onward, all at high confidence.

info3 = load_rep('hiddenbench', 'star', 5, '25', 25)
votes3 = info3['votes']; T3 = info3['T']; N3 = info3['N']

# Show timeline + recurrence side by side for this case only
fig, (ax_t, ax_r1, ax_r2) = plt.subplots(1, 3, figsize=(13, 4.2),
                                           gridspec_kw={'width_ratios': [2.2, 1, 1]})

# Timeline
for r in range(T3 + 1):
    for i in range(N3):
        v = votes3[r][i]
        ax_t.scatter(r, i, color=VOTE_COLORS.get(v, '#999999'),
                     s=220, marker='s', zorder=3, edgecolors='white', lw=0.4)
        ax_t.text(r, i, v, ha='center', va='center',
                  fontsize=7, color='white', fontweight='bold')

ax_t.set_xlim(-0.6, T3 + 0.6)
ax_t.set_ylim(-0.6, N3 - 0.4)
ax_t.set_xticks(range(T3 + 1))
ax_t.set_xticklabels([str(r) for r in range(T3 + 1)], fontsize=8)
ax_t.set_yticks(range(N3))
ax_t.set_yticklabels([f'Agent {i}' for i in range(N3)], fontsize=9)
ax_t.set_xlabel('Round')
ax_t.set_title('(a) Vote timeline: HB / star / $W{=}5$ / q25 / rep25\n(gt = B; hub = agent 1)',
                loc='left', fontsize=9)
ax_t.grid(False)
ax_t.grid(axis='x', lw=0.4, alpha=0.4)
ax_t.legend(handles=legend_handles, loc='upper right', fontsize=7,
            framealpha=0.85, ncol=2)

# Recurrence: hub agent (agent 1, DET=1.0)
for ax_r, ag, lbl in [(ax_r1, 1, '(b) Agent 1 (hub)'), (ax_r2, 0, '(c) Agent 0 (spoke)')]:
    seq = [votes3[r][ag] for r in range(T3 + 1)]
    R = _recurrence_matrix(seq)
    det_val = _det(R)
    rr = _diagonalwise_rr(R)
    P = int(np.argmax(rr)) + 1
    ax_r.imshow(R.astype(int), cmap='Blues', origin='upper', aspect='equal',
                vmin=0, vmax=1.2, interpolation='nearest')
    no_grid(ax_r)
    L = len(seq)
    ax_r.set_xticks(range(0, L, 3))
    ax_r.set_yticks(range(0, L, 3))
    ax_r.set_xticklabels(range(0, L, 3), fontsize=7)
    ax_r.set_yticklabels(range(0, L, 3), fontsize=7)
    ax_r.set_xlabel('Round $j$', fontsize=9)
    ax_r.set_ylabel('Round $i$', fontsize=9)
    ax_r.set_title(f'{lbl}\nDET={det_val:.2f}  $\\hat{{P}}={P}$', fontsize=9)

plt.tight_layout()
out = OUTDIR / 'lc_onset_new.png'
fig.savefig(out)
plt.show()
print(f'Saved {out}')

print('\nAll LC plots generated.')
""")

nb['cells'] = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
from pathlib import Path as _P
_out = _P(__file__).resolve().parents[2] / 'notebooks' / 'golden' / '07_limit_cycles.ipynb'
_out.parent.mkdir(parents=True, exist_ok=True)
import nbformat as _nbf
with open(_out, 'w') as _fh:
    _nbf.write(nb, _fh)
print('wrote', _out, 'with', len(cells), 'cells')
