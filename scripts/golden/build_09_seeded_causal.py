import os
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def co(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# GOLDEN 09_seeded_causal

**Thesis section:** `sec:results:seeded`  
**Provenance:** Seeded Initial-Condition causal experiment. Ported verbatim from scripts/build_nb_036.py (superseded).  
**Data:** results/mas/seeded_init_0, results/mas/seeded_init_1, results/mas/high_repetition  
**Figures exported to:** thesis/plots/highrep/ (fig4_seeded_causal .. fig7_recovery_difficulty)  
**Status:** reproducible from committed data.

This is a GOLDEN notebook: it recomputes every number in the named thesis section from raw data and exports every figure to the fixed `thesis/plots/` path the LaTeX already includes. Numbers are printed in labelled blocks for manual copy into the thesis. See `notebooks/golden/INDEX.md` for the full section->notebook map.
""")

md(r"""# 036 — Seeded Initial-Condition Analysis (causal test of GT-presence)

The R=1000 high-repetition analysis (nb 031) established *observationally* that
whether the ground-truth (GT) option is present in the initial vote distribution
is the dominant predictor of final accuracy. That is a correlation: initial
composition co-varies with task difficulty, agent confidence, and everything else.

This notebook analyses a **controlled intervention** that fixes the initial vote
composition directly, holding everything else at the real debate dynamics:

- **seeded_init_0**: exactly **0** of 4 agents start on the correct answer (GT absent).
- **seeded_init_1**: exactly **1** of 4 agents starts on the correct answer
  (GT present, as a 1/4 minority).

The wrong votes are drawn from a pool of genuine independent agent responses to the
same task, so only the *initial composition* is manipulated; the debate itself is
real. Comparing the two conditions isolates the causal effect of seeding the ground
truth into the starting distribution, and the init_0 arm additionally measures the
system's capacity to *generate* the correct answer de novo when no agent starts with it.

34 GPQA tasks are common to both conditions, $R=50$ each ($W=2$, FC, $N=4$).

Figures exported to `thesis/plots/highrep/`.
""")

co(r"""import sys
sys.path.insert(0, '../..')

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import (chi2_contingency, wilcoxon, mannwhitneyu,
                         pearsonr, spearmanr, entropy as scipy_entropy)
import statsmodels.api as sm

BASE   = Path('../..') / 'results' / 'mas'
EXPORT = Path('../..') / 'thesis'  / 'plots' / 'highrep'
EXPORT.mkdir(parents=True, exist_ok=True)

C0 = '#7F7F7F'   # init_0  (GT absent)  -- grey baseline
C1 = '#0072B2'   # init_1  (GT seeded, 1/4)
CACC = '#D55E00'

from src.viz.thesis_style import apply_style, no_grid
apply_style()
print('setup ok')""")

co(r"""def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; denom = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return float(max(0, centre-half)), float(min(1, centre+half))

def load_condition(cond):
    out = {}
    for f in sorted((BASE / f'seeded_init_{cond}').glob('*.json')):
        d = json.loads(f.read_text())
        out[str(d['question_id'])] = d
    return out

def vote_profile(rep, t):
    return [ag['vote'] for ag in rep['trajectory'][t]['phase_b']]

def gt_share_trajectory(rep, gt):
    return [vote_profile(rep, t).count(gt) / rep['N']
            for t in range(len(rep['trajectory']))]

d0 = load_condition(0)
d1 = load_condition(1)
COMMON = sorted(set(d0) & set(d1), key=int)
print(f'init_0 tasks={len(d0)}  init_1 tasks={len(d1)}  common={len(COMMON)}')
print('only in init_0:', sorted(set(d0)-set(d1), key=int))""")

co(r"""def build_rows(dd, cond):
    rows = []
    for q in COMMON:
        d = dd[q]; gt = d['ground_truth']; opts = list(d['options'].keys())
        for r in d['repetitions']:
            b0 = r['trajectory'][0]['phase_b']
            v0 = [ag['vote'] for ag in b0]
            vf = vote_profile(r, -1)
            wrong0 = [v for v in v0 if v != gt]
            opp_top = max(Counter(wrong0).values()) if wrong0 else 0
            conf_gt = np.mean([ag['confidence'] for ag in b0 if ag['vote'] == gt]) if gt in v0 else np.nan
            conf_wr = np.mean([ag['confidence'] for ag in b0 if ag['vote'] != gt]) if wrong0 else np.nan
            sh = gt_share_trajectory(r, gt)
            gt_first = next((i for i, s in enumerate(sh) if s > 0), -1)
            rows.append({
                'cond': cond, 'qid': q, 'gt': gt,
                'n_correct_pool': d['n_correct_pool'],
                'correct': int(r['correct']),
                'rounds': len(r['trajectory']) - 1,
                'final_unanimous': int(len(set(vf)) == 1),
                'gt_final_share': sh[-1],
                'opp_top_bloc': opp_top,
                'conf_advantage': (conf_gt - conf_wr) if not np.isnan(conf_gt) else np.nan,
                'gt_first_round': gt_first,
            })
    return pd.DataFrame(rows)

df0 = build_rows(d0, 0)
df1 = build_rows(d1, 1)
df  = pd.concat([df0, df1], ignore_index=True)
print(f'rows: init_0={len(df0)}  init_1={len(df1)}')
print(f"final unanimity: init_0={df0['final_unanimous'].mean():.3f}  init_1={df1['final_unanimous'].mean():.3f}")""")

# ── SECTION 1: core causal result ────────────────────────────────────────────
md(r"""## 1. Core causal result

Pooled and per-task accuracy under the two seeding conditions, and the paired
across-task test.
""")

co(r"""k0, n0 = df0['correct'].sum(), len(df0)
k1, n1 = df1['correct'].sum(), len(df1)
print(f'POOLED init_0 (GT absent) : {k0}/{n0} = {k0/n0:.4f}  CI={tuple(round(x,4) for x in wilson_ci(k0,n0))}')
print(f'POOLED init_1 (GT seeded) : {k1}/{n1} = {k1/n1:.4f}  CI={tuple(round(x,4) for x in wilson_ci(k1,n1))}')
ct = np.array([[k1, n1-k1],[k0, n0-k0]])
chi2, p, _, _ = chi2_contingency(ct)
print(f'pooled chi2={chi2:.1f}  p={p:.2e}  lift factor={(k1/n1)/(k0/n0):.1f}x')

a0 = df0.groupby('qid')['correct'].mean().reindex(COMMON)
a1 = df1.groupby('qid')['correct'].mean().reindex(COMMON)
w, pw = wilcoxon(a1.values, a0.values)
print(f'\npaired Wilcoxon (n={len(COMMON)} tasks): init_1 > init_0 in '
      f'{(a1.values>a0.values).sum()}/{len(COMMON)} tasks, W={w:.0f} p={pw:.2e}')
print(f'median per-task lift = {np.median(a1.values-a0.values):.3f}')

n_denovo_tasks = int((a0 > 0).sum())
print(f'\ninit_0 produced >=1 correct run in {n_denovo_tasks}/{len(COMMON)} tasks '
      f'(de-novo recovery is possible but rare)')""")

co(r"""fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                         gridspec_kw={'width_ratios': [2.4, 1]})

# --- (a) per-task paired dumbbell ---
ax = axes[0]
order = a1.sort_values().index.tolist()
y = np.arange(len(order))
a0o = a0.reindex(order).values
a1o = a1.reindex(order).values
for yi, x0, x1 in zip(y, a0o, a1o):
    ax.plot([x0, x1], [yi, yi], color='#cccccc', lw=1.5, zorder=1)
ax.scatter(a0o, y, color=C0, s=42, zorder=2, label='init_0 (GT absent)', edgecolor='k', lw=0.4)
ax.scatter(a1o, y, color=C1, s=42, zorder=3, label='init_1 (GT seeded, 1/4)', edgecolor='k', lw=0.4)
ax.set_yticks(y); ax.set_yticklabels([f'q{q}' for q in order], fontsize=7)
ax.set_xlabel('System accuracy'); ax.set_xlim(-0.02, 1.0)
ax.set_title('(a) Per-task accuracy, 34 tasks (paired)')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.25)

# --- (b) pooled bars ---
ax = axes[1]
accs = [k0/n0, k1/n1]
cis  = [wilson_ci(k0, n0), wilson_ci(k1, n1)]
bars = ax.bar(['init_0\n(GT absent)', 'init_1\n(GT seeded)'], accs,
              color=[C0, C1], width=0.6, edgecolor='k', lw=0.5)
ax.errorbar(['init_0\n(GT absent)', 'init_1\n(GT seeded)'], accs,
            yerr=[[a-c[0] for a,c in zip(accs,cis)],
                  [c[1]-a for a,c in zip(accs,cis)]],
            fmt='none', color='k', capsize=6, lw=1.4)
for b, a, c, n in zip(bars, accs, cis, [n0, n1]):
    ax.text(b.get_x()+b.get_width()/2, c[1]+0.008, f'{a:.3f}\n$n={n}$',
            ha='center', va='bottom', fontsize=9)
ax.set_ylim(0, 0.32); ax.set_ylabel('Pooled accuracy')
pw_mant, pw_exp = f'{pw:.1e}'.split('e')
ax.set_title(f'(b) Pooled\n$\\chi^2={chi2:.0f}$, paired $p={pw_mant}\\times10^{{{int(pw_exp)}}}$')

plt.tight_layout()
fig.savefig(EXPORT / 'fig4_seeded_causal.png')
plt.show()
print('saved fig4_seeded_causal.png')""")

# ── SECTION 2: binary outcomes & dynamics ────────────────────────────────────
md(r"""## 2. Convergence dynamics: convert-all or capitulate-all

Both conditions reach 100% unanimous final states, so the seeded correct agent has
a strictly binary fate: either it converts the whole group (win) or it fully
capitulates (loss). We also characterise how the correct answer re-enters the
debate when it was absent at $t=0$ (init_0 de-novo recovery).
""")

co(r"""# init_1 seed fate
cap  = int((df1['gt_final_share'] == 0).sum())
conv = int((df1['gt_final_share'] >= 0.5).sum())
part = len(df1) - cap - conv
print('init_1 seed fate (n=%d):' % len(df1))
print(f'  capitulated (GT share -> 0):    {cap}  ({cap/len(df1):.1%})')
print(f'  partial (survives as minority): {part} ({part/len(df1):.1%})')
print(f'  converted (GT reaches >=50%):    {conv} ({conv/len(df1):.1%})')

# init_1 overtake round (won runs)
won = df1[df1['correct'] == 1]
overtake = []
for q in COMMON:
    d = d1[q]; gt = d['ground_truth']; opts = list(d['options'].keys())
    for r in d['repetitions']:
        if not r['correct']: continue
        for i in range(len(r['trajectory'])):
            v = vote_profile(r, i); maj = max(opts, key=lambda o: v.count(o))
            if maj == gt and v.count(gt) >= 2:
                overtake.append(i); break
print(f'\ninit_1 WON runs: mean overtake round={np.mean(overtake):.2f} median={np.median(overtake):.0f}')

# init_0 de-novo recovery
recov = df0[df0['correct'] == 1]['gt_first_round'].values
print(f'\ninit_0 de-novo correct runs: {len(recov)}')
print(f'  GT first-appearance round dist: {dict(sorted(Counter(recov).items()))}')
print(f'  mean first-appearance round: {np.mean(recov):.2f}')
rs, ps = spearmanr(df0.groupby('qid')['n_correct_pool'].first().reindex(COMMON).values,
                   a0.values)
print(f'  spearman(n_correct_pool, init_0 acc)={rs:.3f} p={ps:.3e}')""")

co(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- (a) seed fate stacked bar ---
ax = axes[0]
ax.bar(['init_1 seed fate'], [conv], color=C1, label=f'converts group ({conv/len(df1):.0%})',
       edgecolor='k', lw=0.5, width=0.5)
ax.bar(['init_1 seed fate'], [cap], bottom=[conv], color=C0,
       label=f'capitulates ({cap/len(df1):.0%})', edgecolor='k', lw=0.5, width=0.5)
ax.set_ylabel('Number of runs'); ax.set_ylim(0, len(df1))
ax.set_title('(a) Fate of the single correct agent\n(100% of runs end unanimous)')
ax.legend(fontsize=9, loc='center right')

# --- (b) overtake / recovery round histograms ---
ax = axes[1]
bins = np.arange(0, 11) - 0.5
ax.hist(overtake, bins=bins, density=True, color=C1, alpha=0.75,
        label=f'init_1: GT reaches plurality\n(won runs, mean={np.mean(overtake):.1f})',
        edgecolor='k', lw=0.4)
ax.hist(recov, bins=bins, density=True, color=CACC, alpha=0.6,
        label=f'init_0: GT first appears\n(recovered runs, mean={np.mean(recov):.1f})',
        edgecolor='k', lw=0.4)
ax.set_xlabel('Debate round'); ax.set_ylabel('Fraction of runs')
ax.set_xlim(-0.5, 8.5)
ax.set_title('(b) When the correct answer takes hold')
ax.legend(fontsize=8.5)

plt.tight_layout()
fig.savefig(EXPORT / 'fig5_seeded_dynamics.png')
plt.show()
print('saved fig5_seeded_dynamics.png')""")

# ── SECTION 3: mechanism — divided opposition & external validity ────────────
md(r"""## 3. What lets the minority win, and does the intervention match observation?

Within init_1, the correct agent's odds depend on how the three opposing agents are
arranged: a unified wrong bloc (3-0-0) resists conversion far more than a fragmented
opposition. We fit a logistic model and cross-check against the observational
R=1000 data (the "exactly-one-GT minority" subset of nb 031), which the intervention
should reproduce if it is externally valid.
""")

co(r"""# logistic model within init_1
mdl_df = df1.dropna(subset=['conf_advantage']).copy()
Xcols = ['opp_top_bloc', 'conf_advantage', 'n_correct_pool']
X = mdl_df[Xcols].values.astype(float)
X = (X - X.mean(0)) / X.std(0)
X = sm.add_constant(X)
res = sm.Logit(mdl_df['correct'].values, X).fit(disp=0)
print('init_1 logistic: GT-win ~ standardised predictors')
for name, b, pv in zip(['const'] + Xcols, res.params, res.pvalues):
    print(f'  {name:16s} beta={b:+.3f}  p={pv:.2e}')
print(f'  pseudo-R2={res.prsquared:.3f}')

# divided-opposition contingency
uni = df1[df1['opp_top_bloc'] == 3]; div = df1[df1['opp_top_bloc'] < 3]
ct = np.array([[uni['correct'].sum(), len(uni)-uni['correct'].sum()],
               [div['correct'].sum(), len(div)-div['correct'].sum()]])
chi2d, pd_, _, _ = chi2_contingency(ct)
print(f'\nunified opposition (3-0-0): acc={uni["correct"].mean():.3f} (n={len(uni)})')
print(f'divided opposition (<3):    acc={div["correct"].mean():.3f} (n={len(div)})')
print(f'chi2={chi2d:.1f} p={pd_:.2e}')""")

co(r"""from src.metrics.stats_utils import task_paired_wilcoxon

print('=== TASK-CLUSTERED (GEE, groups=qid) win predictors ===')
gee_res = sm.GEE(mdl_df['correct'].values, X, groups=mdl_df['qid'].values,
                 family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Exchangeable()).fit()
for name, b, pv in zip(['const'] + Xcols, gee_res.params, gee_res.pvalues):
    print(f'  {name:16s} beta={b:+.3f}  p={pv:.2e}')""")

co(r"""print('=== 3-0-0 vs 1-1-1 (matches LaTeX 19.5% vs 32.3% proportions) ===')
uni3 = df1[df1['opp_top_bloc'] == 3]
split111 = df1[df1['opp_top_bloc'] == 1]
ct311 = np.array([[uni3['correct'].sum(), len(uni3) - uni3['correct'].sum()],
                  [split111['correct'].sum(), len(split111) - split111['correct'].sum()]])
chi2_311, p_311, _, _ = chi2_contingency(ct311)
print(f'3-0-0 (unified): acc={uni3["correct"].mean():.4f} (n={len(uni3)})')
print(f'1-1-1 (split):   acc={split111["correct"].mean():.4f} (n={len(split111)})')
print(f'chi2={chi2_311:.1f} p={p_311:.2e}')""")

co(r"""print('=== TASK-LEVEL (n~31): unified vs divided opposition ===')
uni_task, div_task, both_qids = [], [], []
for q in COMMON:
    g = df1[df1['qid'] == q]
    gu = g[g['opp_top_bloc'] == 3]; gd = g[g['opp_top_bloc'] < 3]
    if len(gu) and len(gd):
        uni_task.append(gu['correct'].mean()); div_task.append(gd['correct'].mean())
        both_qids.append(q)
uni_task = np.array(uni_task); div_task = np.array(div_task)
stat_w, p_w, med_w, n_w = task_paired_wilcoxon(div_task, uni_task)
print(f'tasks with both bloc types: n={n_w}')
print(f'paired Wilcoxon (divided vs unified): W={stat_w:.1f} p={p_w:.3f} '
      f'median per-task diff={med_w:+.4f}')

def cluster_pooled_diff(qids, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    tab = np.array([[df1[(df1.qid == q) & (df1.opp_top_bloc < 3)]['correct'].sum(),
                     len(df1[(df1.qid == q) & (df1.opp_top_bloc < 3)]),
                     df1[(df1.qid == q) & (df1.opp_top_bloc == 3)]['correct'].sum(),
                     len(df1[(df1.qid == q) & (df1.opp_top_bloc == 3)])] for q in qids], float)
    pooled = lambda t: t[:, 0].sum() / t[:, 1].sum() - t[:, 2].sum() / t[:, 3].sum()
    n = len(tab)
    boots = np.array([pooled(tab[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(pooled(tab)), float(lo), float(hi)

diff_c, lo_c, hi_c = cluster_pooled_diff(COMMON)
print(f'cluster bootstrap pooled diff (divided-unified): {diff_c:+.4f} '
      f'CI[{lo_c:.4f},{hi_c:.4f}] (includes 0: {lo_c < 0 < hi_c})')""")

co(r"""# observational comparison for q84,q125,q144
HIGHREP = BASE / 'high_repetition'
def obs_exactly_one_gt(qid):
    f = next(HIGHREP.glob(f'*_q{qid}_*.json'))
    d = json.loads(f.read_text()); gt = d['ground_truth']; opts = list(d['options'].keys())
    acc = []
    for r in d['repetitions']:
        v0 = [ag['vote'] for ag in r['trajectory'][0]['phase_b']]
        maj = max(opts, key=lambda o: v0.count(o))
        if v0.count(gt) == 1 and maj != gt:
            acc.append(int(r['correct']))
    return np.mean(acc), len(acc)

VAL = ['84', '125', '144']
val_rows = []
for q in VAL:
    obs, nobs = obs_exactly_one_gt(q)
    seed = df1[df1['qid'] == q]['correct'].mean()
    val_rows.append((q, obs, nobs, seed))
    print(f'q{q}: observational(1-GT-minority)={obs:.3f} (n={nobs})  seeded_init_1={seed:.3f} (n=50)')""")

co(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- (a) win rate by opposition concentration ---
ax = axes[0]
blocs = [3, 2, 1]
labels = ['3-0-0\n(unified)', '2-1-0', '1-1-1\n(split)']
accs, los, his, ns = [], [], [], []
for b in blocs:
    g = df1[df1['opp_top_bloc'] == b]
    k, n = g['correct'].sum(), len(g)
    accs.append(k/n); lo, hi = wilson_ci(k, n)
    los.append(lo); his.append(hi); ns.append(n)
xb = np.arange(len(blocs))
bars = ax.bar(xb, accs, color=C1, width=0.6, edgecolor='k', lw=0.5)
ax.errorbar(xb, accs, yerr=[np.array(accs)-np.array(los), np.array(his)-np.array(accs)],
            fmt='none', color='k', capsize=5, lw=1.2)
for x, a, n in zip(xb, accs, ns):
    ax.text(x, a+0.012, f'{a:.2f}\n$n={n}$', ha='center', fontsize=9)
ax.set_xticks(xb); ax.set_xticklabels(labels)
ax.set_ylabel('P(GT wins)'); ax.set_ylim(0, 0.42)
ax.set_xlabel('Arrangement of the three opposing votes at $t=0$')
ax.set_title(f'(a) A divided opposition helps the minority\n($\\chi^2={chi2d:.0f}$, $p<10^{{-3}}$)')

# --- (b) observational vs interventional ---
ax = axes[1]
xv = np.arange(len(VAL)); w = 0.36
obs_vals  = [r[1] for r in val_rows]
seed_vals = [r[3] for r in val_rows]
obs_ci  = [wilson_ci(int(r[1]*r[2]), r[2]) for r in val_rows]
seed_ci = [wilson_ci(int(r[3]*50), 50) for r in val_rows]
b1 = ax.bar(xv-w/2, obs_vals, w, color='#8c8c8c', edgecolor='k', lw=0.5,
            label='observational (R=1000,\n1-GT minority subset)')
b2 = ax.bar(xv+w/2, seed_vals, w, color=C1, edgecolor='k', lw=0.5,
            label='interventional (seeded\\_init_1)')
ax.errorbar(xv-w/2, obs_vals, yerr=[[o-c[0] for o,c in zip(obs_vals,obs_ci)],
                                    [c[1]-o for o,c in zip(obs_vals,obs_ci)]],
            fmt='none', color='k', capsize=4, lw=1.0)
ax.errorbar(xv+w/2, seed_vals, yerr=[[o-c[0] for o,c in zip(seed_vals,seed_ci)],
                                     [c[1]-o for o,c in zip(seed_vals,seed_ci)]],
            fmt='none', color='k', capsize=4, lw=1.0)
ax.set_xticks(xv); ax.set_xticklabels([f'q{q}' for q in VAL])
ax.set_ylabel('P(correct)'); ax.set_ylim(0, 0.6)
ax.set_title('(b) Intervention reproduces observation')
ax.legend(fontsize=8.5)

plt.tight_layout()
fig.savefig(EXPORT / 'fig6_seeded_mechanism.png')
plt.show()
print('saved fig6_seeded_mechanism.png')""")

# ── SECTION 4: recovery vs task difficulty ───────────────────────────────────
md(r"""## 4. Is de-novo recovery bounded by task difficulty?

Each task carries a fixed pool of 200 independent single-agent responses (identical
across both conditions); the initial votes are sampled from it. The correct fraction
of that pool, $\mathrm{acc}_1 = n_\text{correct}/200$, is the task's single-agent
accuracy and a clean, debate-independent measure of difficulty. It matches the
natural $t=0$ per-agent accuracy in the separate $R=1000$ runs almost exactly
(q84 0.295 vs 0.280, q125 0.280 vs 0.261, q144 0.390 vs 0.398), confirming it is a
faithful difficulty index rather than an artifact of the seeding procedure. We ask
whether the system's ability to recover the ground truth de novo (init_0) depends on
this difficulty.
""")

co(r"""base = np.array([d0[q]['n_correct_pool'] / (d0[q]['n_correct_pool'] + d0[q]['n_wrong_pool'])
                 for q in COMMON])
rec  = a0.reindex(COMMON).values          # init_0 recovery rate per task
rec_runs = df0.groupby('qid')['correct'].sum().reindex(COMMON).values

rs_d, ps_d = spearmanr(base, rec)
rp_d, pp_d = pearsonr(base, rec)
print(f'recovery vs single-agent accuracy: Spearman rho={rs_d:.3f} p={ps_d:.4f}  '
      f'Pearson r={rp_d:.3f} p={pp_d:.4f}')

# quartiles by difficulty
order = np.argsort(base); n = len(order)
qlabels = ['Q1 hardest', 'Q2', 'Q3', 'Q4 easiest']
quarts = [order[i*n//4:(i+1)*n//4] for i in range(4)]
print('\nrecovery by single-agent-accuracy quartile:')
for lab, idx in zip(qlabels, quarts):
    print(f'  {lab:12s}: base_acc={base[idx].mean():.3f}  '
          f'mean recovery={rec[idx].mean():.4f}  recovered={int(rec_runs[idx].sum())}/{len(idx)*50}')""")

co(r"""fig, ax = plt.subplots(figsize=(8, 6))

sizes = 40 + rec_runs * 60
ax.scatter(base, rec, s=sizes, color=C1, edgecolor='k', lw=0.5, alpha=0.85, zorder=3)
# label only the notable high-recovery tasks (hard-task cluster sits on y=0, self-evident)
for q, x, y, k in zip(COMMON, base, rec, rec_runs):
    if y > 0.055:
        ax.annotate(f'q{q}', (x, y), fontsize=8, xytext=(5, 2),
                    textcoords='offset points', color='#333333')
# LOWESS-free simple trend: bin means
order = np.argsort(base); n = len(order)
for i in range(4):
    idx = order[i*n//4:(i+1)*n//4]
    ax.plot(base[idx].mean(), rec[idx].mean(), 'D', color=CACC, ms=9,
            zorder=4, mec='k', mew=0.6)
ax.plot([], [], 'D', color=CACC, ms=9, mec='k', mew=0.6, label='quartile mean')
ax.axhline(0, color='gray', lw=0.7, ls=':')
q1_hi = base[order[:n//4]].max()
ax.axvspan(-0.02, q1_hi, color='#D65F5F', alpha=0.07, zorder=0)
ax.text((q1_hi-0.02)/2, max(rec)*1.05, 'hardest quartile\n(0 recoveries / 400 runs)',
        ha='center', va='top', fontsize=8.5, color=CACC)
ax.set_xlabel('Single-agent accuracy $n_\\text{correct}/200$ (task difficulty, higher = easier)')
ax.set_ylabel('init_0 de-novo recovery rate')
ax.set_title(f'De-novo recovery vs task difficulty\n'
             f'Spearman $\\rho={rs_d:.2f}$ ($p={ps_d:.3f}$); the hardest quartile recovers 0/{len(quarts[0])*50}')
ax.legend(fontsize=9)
ax.set_ylim(-0.01, max(rec)*1.15)

plt.tight_layout()
fig.savefig(EXPORT / 'fig7_recovery_difficulty.png')
plt.show()
print('saved fig7_recovery_difficulty.png')""")

# ── SECTION 5: summary table ─────────────────────────────────────────────────
md(r"""## 5. Summary statistics for LaTeX""")

co(r"""print('=== headline numbers for LaTeX ===\n')
print(f'init_0 pooled acc = {k0/n0*100:.1f}% ({k0}/{n0})')
print(f'init_1 pooled acc = {k1/n1*100:.1f}% ({k1}/{n1})')
print(f'lift factor = {(k1/n1)/(k0/n0):.1f}x   chi2={chi2:.0f}')
print(f'paired Wilcoxon: {(a1.values>a0.values).sum()}/{len(COMMON)} tasks, p={pw:.1e}')
print(f'seed converts group: {conv/len(df1)*100:.1f}%  capitulates: {cap/len(df1)*100:.1f}%  partial: {part}')
print(f'init_1 mean overtake round: {np.mean(overtake):.2f}')
print(f'init_0 de-novo runs: {len(recov)}  mean first-appearance round: {np.mean(recov):.2f}')
print(f'init_0 tasks with >=1 recovery: {n_denovo_tasks}/{len(COMMON)}')
print(f'unified vs divided opposition: {uni["correct"].mean():.3f} vs {div["correct"].mean():.3f} (p={pd_:.1e})')
print(f'spearman(pool, init_0 acc): rs={rs:.3f} p={ps:.1e}')
print(f'recovery vs difficulty: Spearman rho={rs_d:.3f} p={ps_d:.4f}  hardest-quartile recovery={rec[order[:n//4]].mean():.4f} ({int(rec_runs[order[:n//4]].sum())}/{len(order[:n//4])*50})')
print()
print('mean rounds: init_0=%.2f  init_1=%.2f' % (df0['rounds'].mean(), df1['rounds'].mean()))""")

nb['cells'] = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
from pathlib import Path as _P
_out = _P(__file__).resolve().parents[2] / 'notebooks' / 'golden' / '09_seeded_causal.ipynb'
_out.parent.mkdir(parents=True, exist_ok=True)
import nbformat as _nbf
with open(_out, 'w') as _fh:
    _nbf.write(nb, _fh)
print('wrote', _out, 'with', len(cells), 'cells')
