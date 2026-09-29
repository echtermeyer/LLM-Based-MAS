import os
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def co(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# GOLDEN 11_appendix_robustness

**Thesis section:** `app:sysprompt` — *System-Prompt Robustness Check* (thesis/chapters/appendix.tex, table `tab:app:sysprompt`).
**Provenance:** Ported from scripts/build_nb_029.py (notebook *029 New System Prompt Drift*, superseded).
**Data (old / milder prompt):** `results/mas/final_dataset_depricated` — REPOINTED. The path used by nb 029 (`results/mas/final_dataset`) is missing locally; `final_dataset_depricated` is the surviving old-prompt HiddenBench arm (verified: persona ends at "…and so may you", R=30, carries the stray 0-confidence votes).
**Data (new / critical prompt):** `results/mas/final_dataset_new_system` (present).
**Figures exported to:** none — the appendix subsection `app:sysprompt` is table-only (`tab:app:sysprompt`); diagnostic plots are shown inline only.
**Status:** reproducible from committed data (old arm via repoint). If the old-prompt dir is ever removed, the old-arm cells self-guard and the notebook still runs new-only.

This is a GOLDEN notebook: it recomputes every number in `tab:app:sysprompt` from raw data and prints each in a labelled block for manual copy into the thesis. The final cell prints computed vs thesis-table values with MATCH/MISMATCH. See `notebooks/golden/INDEX.md` for the full section->notebook map.
""")

md(r"""# 029 — New System-Prompt Drift (HiddenBench)

Compares the **old / milder** system-prompt runs against the **new, more critical** system-prompt
runs on the **same 35 HiddenBench questions × 6 configs** (W∈{1,2,5} × {fc,star}).

**Data repoint (golden).** nb 029 read `results/mas/final_dataset` for the old arm; that directory is
missing locally. The surviving old-prompt arm is `results/mas/final_dataset_depricated` (verified old
persona, R=30, stray 0-confidence votes present). This notebook repoints the old arm there. If even
that directory is absent, the old-arm cells are guarded (`OLD_AVAILABLE`) so the notebook runs
new-only and prints a frozen-artifact note instead of crashing.

**The prompt change.** The new persona appends one explicit anti-conformity instruction:

> *Be critical — your neighbors may be wrong, and so may you. **Do not change your position
> simply because others disagree; only update if you encounter a specific argument you cannot
> counter.***

(The old / milder persona ends at "…and so may you.")

**Known confounds, reported as-is (no data modification):**

1. **Repetitions differ.** Old = R30 (downsampled), new = R50. We compare at the paired-task level,
   so the different rep counts only affect the precision of each task's estimate, not the pairing.
2. **Confidence scale.** The old data contains stray **0-confidence** values (a known artifact); the
   new data has none. Confidence enters only the *confidence-weighted* metrics (dominance credit,
   self-reinforcement slopes); we flag this where relevant but do **not** alter either dataset.
3. **No GPQA in the old run** → this analysis is HiddenBench-only.

**Statistics.** Paired unit = shared **question id (qid)**, paired **within each config**. For every
metric we aggregate to one value per (config, qid) per dataset, run a **Wilcoxon signed-rank** on the
paired deltas per config, **Holm-correct across the 6 configs**, and report a **bootstrap 95% CI on
the mean paired delta**. Task-level aggregation avoids rep-level pseudoreplication.
""")

co(r"""import sys
sys.path.insert(0, '../..')

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from src.metrics.dominance import analyse_dominance
from src.metrics.self_reinforcement import extract_runs, summarise_runs
from src.metrics.limit_cycles import detect_agent_limit_cycles
from src.metrics.bistability import analyse_bistability
from src.viz.thesis_style import apply_style
apply_style()

OLD_BASE = Path('../..') / 'results' / 'mas' / 'final_dataset_depricated'   # REPOINT (was final_dataset)
NEW_BASE = Path('../..') / 'results' / 'mas' / 'final_dataset_new_system'

OLD_AVAILABLE = OLD_BASE.exists()

W_VALUES = [1, 2, 5]
TOPOS    = ['fc', 'star']
CONFIG_ORDER = [f'{t}/W={w}' for t in TOPOS for w in W_VALUES]
FC_CONFIGS = [c for c in CONFIG_ORDER if c.startswith('fc')]
DS_ORDER = ['old', 'new']
DS_LABEL = {'old': 'Milder prompt', 'new': 'Critical prompt'}
DS_COLOR = {'old': '#7f7f7f', 'new': '#EE6666'}
T_CEIL   = 15
SEED     = 42
B_DOM    = 200   # surrogate draws for dominance (speed; appendix robustness only)
B_LC     = 1000   # shuffle surrogates for limit cycles
B_BIST   = 1000   # permutation draws for bistability basin test

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 40)
np.set_printoptions(suppress=True)
print('setup ok  OLD_AVAILABLE =', OLD_AVAILABLE, '  OLD_BASE =', OLD_BASE)
if not OLD_AVAILABLE:
    print('FROZEN ARTIFACT: old-prompt arm not reproducible locally -> running new-only.')""")

md(r"""### Old-prompt arm status

If `OLD_AVAILABLE` is `True`, the old / milder arm is read live from
`results/mas/final_dataset_depricated` (repointed). If `False`, that directory is missing too and
this is a **frozen artifact** for the old arm — the paired tests degrade gracefully (old columns are
all-NaN, so every `sig_configs` reads `0/6` and the pooled old means read `nan`); the table numbers
below then cannot be recomputed and should be treated as previously-committed values.
""")

co(r"""def iter_files(base):
    for f in sorted(base.glob('**/*.json')):
        if 'hiddenbench' not in f.name:
            continue
        yield f

def _bases():
    b = [(NEW_BASE, 'new')]
    if OLD_AVAILABLE:
        b = [(OLD_BASE, 'old')] + b
    return b

def load_meta(base, ds_tag):
    rows = []
    for f in iter_files(base):
        d = json.loads(f.read_text())
        reps = d['repetitions']
        rounds = [len(r['trajectory']) - 1 for r in reps]
        confs = [pb.get('confidence') for r in reps
                 for rd in r['trajectory'] for pb in rd['phase_b']]
        confs = [c for c in confs if c is not None]
        rows.append({
            'ds': ds_tag,
            'W': d['W'],
            'topology': d.get('topology_name', 'fc'),
            'config': d.get('topology_name', 'fc') + '/W=' + str(d['W']),
            'qid': d['question_id'],
            'M': len(d['options']),
            'n_reps': len(reps),
            'mean_rounds': float(np.mean(rounds)),
            'conf_min': int(min(confs)) if confs else -1,
            'conf_max': int(max(confs)) if confs else -1,
            'n_conf_zero': int(sum(c == 0 for c in confs)),
        })
    return pd.DataFrame(rows)

meta = pd.concat([load_meta(b, t) for b, t in _bases()], ignore_index=True)
print(f'files: old={ (meta.ds=="old").sum() }  new={ (meta.ds=="new").sum() }')
print()
print('per-dataset provenance:')
prov = meta.groupby('ds').agg(
    files=('qid', 'size'),
    n_reps_mean=('n_reps', 'mean'),
    conf_min=('conf_min', 'min'),
    conf_max=('conf_max', 'max'),
    n_conf_zero=('n_conf_zero', 'sum'),
).round(2)
print(prov.to_string())""")

co(r"""# --- assert the two datasets are a clean paired design (guarded) --------
if OLD_AVAILABLE:
    old_keys = set(map(tuple, meta[meta.ds=='old'][['config','qid']].values))
    new_keys = set(map(tuple, meta[meta.ds=='new'][['config','qid']].values))
    missing_from_new = old_keys - new_keys
    assert not missing_from_new, f'old tasks missing from new: {missing_from_new}'
    extra_in_new = new_keys - old_keys
    if extra_in_new:
        print(f'INFO: {len(extra_in_new)} (config,qid) cells in new only (added tasks) — restricting to shared qids for paired analysis')
        shared_qids = sorted(meta[meta.ds=='old']['qid'].unique())
        meta = meta[meta['qid'].isin(shared_qids)].copy()
    n_qids = meta[meta.ds=='new']['qid'].nunique()
    print(f'PAIRED design: {len(old_keys)} (config,qid) cells, {n_qids} qids x {len(CONFIG_ORDER)} configs.')
    print('shared qids:', sorted(meta[meta.ds=="old"].qid.unique()))
else:
    print('old arm absent -> paired-design assertion skipped (new-only).')

# stray out-of-option votes (data-quality drift signal): known transient 'D' artifact
def scan_stray(base):
    n_cells = n_rounds = n_votes = 0
    for f in iter_files(base):
        d = json.loads(f.read_text()); opts = set(d['options'].keys()); hit = False
        for rep in d['repetitions']:
            for rd in rep['trajectory']:
                extra = [pb['vote'] for pb in rd['phase_b'] if pb['vote'] not in opts]
                if extra:
                    n_rounds += 1; n_votes += len(extra); hit = True
        n_cells += hit
    return n_cells, n_rounds, n_votes
print()
print('stray out-of-option votes (cells, rounds, votes):')
if OLD_AVAILABLE:
    print('  old:', scan_stray(OLD_BASE), '  new:', scan_stray(NEW_BASE))
else:
    print('  new:', scan_stray(NEW_BASE), '  (old skipped)')""")

md(r"""### Paired-test helpers

`paired_wilcoxon` takes a long dataframe with columns `[config, qid, ds, <value>]`, pivots to
old/new per (config, qid), drops unpaired/NaN pairs, and for each config returns the paired mean
delta (new − old), a bootstrap 95% CI, the Wilcoxon signed-rank p, and n pairs. p-values are
**Holm-corrected across the 6 configs**. If the old arm is absent the pivot has no `old` column, so
every row degrades to NaN / `n_pairs=0` rather than crashing.
""")

co(r"""def _boot_ci(delta, B=5000, seed=SEED):
    d = np.asarray(delta, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def _holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj

def paired_wilcoxon(df_long, value, configs=CONFIG_ORDER):
    rows = []
    for cfg in configs:
        sub = df_long[df_long.config == cfg]
        piv = sub.pivot_table(index='qid', columns='ds', values=value, aggfunc='mean')
        piv = piv.reindex(columns=['old', 'new'])   # guarantees both cols even if old absent
        piv = piv.dropna(subset=['old', 'new'])
        old_v, new_v = piv['old'].values, piv['new'].values
        delta = new_v - old_v
        n = len(delta)
        mean_delta = float(np.mean(delta)) if n else np.nan
        lo, hi = _boot_ci(delta)
        if n >= 1 and np.any(delta != 0):
            try:
                stat, p = sp_stats.wilcoxon(new_v, old_v, zero_method='wilcox')
            except ValueError:
                p = np.nan
        else:
            p = np.nan
        rows.append({
            'config': cfg, 'n_pairs': n,
            'old_mean': float(np.mean(old_v)) if n else np.nan,
            'new_mean': float(np.mean(new_v)) if n else np.nan,
            'delta': mean_delta, 'ci_lo': lo, 'ci_hi': hi,
            'p_raw': p,
        })
    out = pd.DataFrame(rows)
    out['p_holm'] = _holm(out['p_raw'].fillna(1.0).values)
    out['sig'] = out['p_holm'].apply(lambda p: '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns')
    return out

def show(res, title, fmt='{:+.4f}'):
    print(f'=== {title} — paired new vs old, Wilcoxon (Holm across configs) ===')
    disp = res.copy()
    for c in ['old_mean', 'new_mean', 'delta', 'ci_lo', 'ci_hi']:
        disp[c] = disp[c].map(lambda x: fmt.format(x) if pd.notna(x) else 'nan')
    disp['p_holm'] = disp['p_holm'].map(lambda x: f'{x:.3g}')
    print(disp[['config','n_pairs','old_mean','new_mean','delta','ci_lo','ci_hi','p_holm','sig']].to_string(index=False))
print('helpers ok')""")

# ---------------- ACCURACY & EFFICIENCY ----------------
md(r"""## 1. Accuracy & efficiency

Task-level values per (config, qid): **accuracy** = mean of `rep['correct']`; **mean_rounds** =
mean trajectory length − 1 (rounds to consensus); **ceil_frac** = fraction of reps hitting the
T=15 ceiling (where high, mean_rounds is a lower bound). These feed the *Accuracy*, *Mean rounds*
and *Ceiling-hit fraction* rows of `tab:app:sysprompt`.
""")

co(r"""def load_taskvals(base, ds_tag):
    rows = []
    for f in iter_files(base):
        d = json.loads(f.read_text())
        reps = d['repetitions']
        rounds = np.array([len(r['trajectory']) - 1 for r in reps])
        rows.append({
            'ds': ds_tag,
            'config': d.get('topology_name','fc') + '/W=' + str(d['W']),
            'topology': d.get('topology_name','fc'),
            'W': d['W'], 'qid': d['question_id'],
            'accuracy': float(np.mean([r['correct'] for r in reps])),
            'mean_rounds': float(rounds.mean()),
            'ceil_frac': float(np.mean(rounds >= T_CEIL)),
        })
    return pd.DataFrame(rows)

taskvals = pd.concat([load_taskvals(b, t) for b, t in _bases()], ignore_index=True)
print('overall (pooled across configs):')
print(taskvals.groupby('ds')[['accuracy','mean_rounds','ceil_frac']].mean().round(4).to_string())""")

co(r"""acc_res = paired_wilcoxon(taskvals, 'accuracy')
show(acc_res, 'Accuracy')""")

co(r"""rnd_res = paired_wilcoxon(taskvals, 'mean_rounds')
show(rnd_res, 'Mean rounds-to-consensus', fmt='{:+.3f}')
print()
ceil_res = paired_wilcoxon(taskvals, 'ceil_frac')
show(ceil_res, 'Ceiling-hit fraction (reps reaching T=15)')""")

co(r"""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (col, ttl) in zip(axes, [('accuracy','Accuracy'),('mean_rounds','Mean rounds'),('ceil_frac','Ceiling-hit frac')]):
    x = np.arange(len(CONFIG_ORDER)); w = 0.38
    for k, ds in enumerate(DS_ORDER):
        means = [taskvals[(taskvals.ds==ds)&(taskvals.config==c)][col].mean() for c in CONFIG_ORDER]
        ax.bar(x + (k-0.5)*w, means, w, label=DS_LABEL[ds], color=DS_COLOR[ds])
    ax.set_xticks(x); ax.set_xticklabels(CONFIG_ORDER, rotation=45, ha='right')
    ax.set_title(ttl); ax.legend(fontsize=8)
fig.suptitle('Accuracy & efficiency: milder vs critical prompt (per config, mean over 35 shared qids)')
plt.tight_layout()
plt.show()""")

# ---------------- DOMINANCE ----------------
md(r"""## 2. Dominance

Per-rep influence-concentration `D` (Herfindahl of confidence-weighted influence shares) via
`analyse_dominance` (B=1000 surrogates, Theiler-0 null). Task-level value = mean `D` over reps in a
cell. Feeds the *Dominance D* and *Hub capitulation* rows of `tab:app:sysprompt`. z/flag are
identifiable in **fc only** (star null is a point mass for most reps).

**Confidence caveat:** `D` and hub capitulation are confidence-weighted; the old data's stray
0-confidence votes contribute zero credit, so part of any shift reflects that artifact, not only the
prompt.
""")

co(r"""def load_dominance(base, ds_tag):
    cell_rows = []
    for f in iter_files(base):
        d = json.loads(f.read_text())
        reps = d['repetitions']
        cfg = d.get('topology_name','fc') + '/W=' + str(d['W'])
        topo = d.get('topology_name','fc')
        res = [analyse_dominance(rep, B=B_DOM, seed=SEED) for rep in reps]
        valid = [r for r in res if r is not None]
        testable = [r for r in valid if r['testable']]
        Ds = [r['D'] for r in valid]
        zs = [r['z'] for r in testable]
        flags = [r['flagged'] for r in testable]
        caps = [r['hub_capitulation'] for r in valid]
        cell_rows.append({
            'ds': ds_tag, 'config': cfg, 'topology': topo, 'W': d['W'], 'qid': d['question_id'],
            'D_mean': float(np.mean(Ds)) if Ds else np.nan,
            'z_mean': float(np.mean(zs)) if zs else np.nan,
            'flagged_fraction': float(np.mean(flags)) if flags else np.nan,
            'hub_capitulation': float(np.mean(caps)) if caps else np.nan,
            'n_valid': len(valid), 'n_testable': len(testable),
        })
    return pd.DataFrame(cell_rows)

print('computing dominance (B=%d) ... this is the slow step (~10 min total)' % B_DOM)
dom = pd.concat([load_dominance(b, t) for b, t in _bases()], ignore_index=True)
print('done. overall D_mean by ds:')
print(dom.groupby('ds')[['D_mean','hub_capitulation']].mean().round(4).to_string())""")

co(r"""dom_res = paired_wilcoxon(dom, 'D_mean')
show(dom_res, 'Dominance D (mean per cell)')
print()
cap_res = paired_wilcoxon(dom, 'hub_capitulation')
show(cap_res, 'Hub capitulation (confidence-weighted)')""")

co(r"""fc_dom = dom[dom.topology=='fc']
zres = paired_wilcoxon(fc_dom, 'z_mean', configs=FC_CONFIGS)
show(zres, 'Emergent excess z (fc only)', fmt='{:+.3f}')
print()
fres = paired_wilcoxon(fc_dom, 'flagged_fraction', configs=FC_CONFIGS)
show(fres, 'Flagged fraction — reps with significant emergent concentration (fc only)')""")

# ---------------- SELF-REINFORCEMENT ----------------
md(r"""## 3. Self-reinforcement

Over vote-stable runs (length ≥ 3), the OLS slope of confidence vs round (`extract_runs` /
`summarise_runs`). `p_sr` = fraction of runs with slope > 0 (baseline 0.5 = none); `mean_slope` =
mean confidence gain per round while holding a position. Feeds the *Self-reinf. p_SR* and
*Self-reinf. slope* rows of `tab:app:sysprompt`.

**Confidence caveat:** slopes use raw confidence integers; old-data 0-values can distort individual
slopes, so cross-dataset slope *magnitude* is caveated. `p_sr` (sign only) is more robust.
""")

co(r"""def load_selfreinf(base, ds_tag):
    rows = []
    for f in iter_files(base):
        d = json.loads(f.read_text())
        s = summarise_runs(extract_runs(d['repetitions']))
        rows.append({
            'ds': ds_tag,
            'config': d.get('topology_name','fc') + '/W=' + str(d['W']),
            'qid': d['question_id'],
            'p_sr': s['p_sr'], 'mean_slope': s['mean_slope'],
            'p_sr_terminal': s['p_sr_terminal'], 'n_runs': s['n_runs'],
        })
    return pd.DataFrame(rows)

sr = pd.concat([load_selfreinf(b, t) for b, t in _bases()], ignore_index=True)
print('overall by ds:')
print(sr.groupby('ds')[['p_sr','mean_slope','p_sr_terminal','n_runs']].mean().round(4).to_string())""")

co(r"""psr_res = paired_wilcoxon(sr, 'p_sr')
show(psr_res, 'Self-reinforcement prevalence p_sr (fraction of runs slope>0; 0.5=none)')
print()
slp_res = paired_wilcoxon(sr, 'mean_slope')
show(slp_res, 'Mean confidence slope per round', fmt='{:+.4f}')""")

# ---------------- LIMIT CYCLES ----------------
md(r"""## 4. Limit cycles

RQA DET shuffle-surrogate test (B=1000) on vote sequences. **Agent-level**: per-agent Phase-B vote
sequence. **System-level**: vote-composition macrostate. `p_lc` = flagged / candidates (candidate =
not a fixed point and L ≥ 4). Task-level value = candidate-weighted flag rate per cell. Feeds the
*Limit cycle (agent)* and *Limit cycle (system)* rows of `tab:app:sysprompt`.

**Data-quality note.** The old data contains exactly **one** stray out-of-option vote (transient
`'D'`, q45/W2/star); the new data has **zero**. `detect_system_limit_cycles` in `src` raises on it,
so the system-level loader here reimplements the tiny detection loop over imported RQA internals and
drops the stray vote for that one round (matching `analyse_bistability`'s tolerant behaviour).
""")

co(r"""from src.metrics.limit_cycles import (
    _recurrence_matrix, _det, _shuffle_p, _diagonalwise_rr, _dominant_period, _trailing_k,
)

def _comp_seq_lenient(traj, options):
    seq = []
    for rd in traj:
        votes = [ag['vote'] for ag in rd['phase_b']]
        seq.append(tuple(votes.count(o) for o in options))
    return seq

def _system_lc_lenient(reps, B, seed):
    options = tuple(reps[0]['options'].keys())
    rng = np.random.default_rng(seed)
    rows = []
    for rep in reps:
        seq = _comp_seq_lenient(rep['trajectory'], options)
        L = len(seq); is_fp = _trailing_k(seq) >= 3
        row = {'fixed_point': is_fp, 'L': L, 'lc': False}
        if not is_fp and L >= 4:
            R = _recurrence_matrix(seq); det_obs = _det(R)
            p = _shuffle_p(seq, det_obs, B, rng)
            P_hat = _dominant_period(_diagonalwise_rr(R))
            row['lc'] = bool(p < 0.05 and 2 <= P_hat <= L // 2)
        rows.append(row)
    return rows

def load_limitcycles(base, ds_tag):
    rows = []
    for f in iter_files(base):
        d = json.loads(f.read_text())
        reps = d['repetitions']
        ag = detect_agent_limit_cycles(reps, B=B_LC, seed=SEED)
        sy = _system_lc_lenient(reps, B=B_LC, seed=SEED)
        def rate(rec):
            cand = [r for r in rec if not r['fixed_point'] and r['L'] >= 4]
            return (sum(r['lc'] for r in cand) / len(cand)) if cand else np.nan, len(cand)
        a_p, a_n = rate(ag); s_p, s_n = rate(sy)
        rows.append({
            'ds': ds_tag,
            'config': d.get('topology_name','fc') + '/W=' + str(d['W']),
            'qid': d['question_id'],
            'p_lc_agent': a_p, 'n_cand_agent': a_n,
            'p_lc_system': s_p, 'n_cand_system': s_n,
        })
    return pd.DataFrame(rows)

print('computing limit cycles (B=%d) ...' % B_LC)
lc = pd.concat([load_limitcycles(b, t) for b, t in _bases()], ignore_index=True)
print('done. overall candidate-weighted flag rates + candidate counts by ds:')
agg = lc.groupby('ds').agg(
    p_lc_agent=('p_lc_agent','mean'), n_cand_agent=('n_cand_agent','sum'),
    p_lc_system=('p_lc_system','mean'), n_cand_system=('n_cand_system','sum')).round(4)
print(agg.to_string())""")

co(r"""lca_res = paired_wilcoxon(lc, 'p_lc_agent')
show(lca_res, 'Agent-level limit-cycle rate p_lc', fmt='{:+.4f}')
print()
lcs_res = paired_wilcoxon(lc, 'p_lc_system')
show(lcs_res, 'System-level limit-cycle rate p_lc', fmt='{:+.4f}')""")

# ---------------- BISTABILITY ----------------
md(r"""## 5. Bistability / multistability

Per (config, qid) cell via `analyse_bistability` (B=1000 permutation basin test, B_boot=500 for the
N_eff CI). `N_eff` = inverse-Simpson number of distinct consensus attractors over converged reps;
`cramers_v` = association between initial composition and final attractor. `conv_frac` = fraction of
reps settling to unanimous consensus (R-independent). Feeds the *Convergence rate*,
*Bistability N_eff* and *Bistability Cramér's V* rows of `tab:app:sysprompt`.
""")

co(r"""def load_bistability(base, ds_tag):
    rows = []
    for f in iter_files(base):
        d = json.loads(f.read_text())
        r = analyse_bistability(d['repetitions'], B=B_BIST, B_boot=500, seed=SEED)
        rows.append({
            'ds': ds_tag,
            'config': d.get('topology_name','fc') + '/W=' + str(d['W']),
            'qid': d['question_id'], 'M': r['M'],
            'label': r['label'], 'n_eff': r['n_eff'], 'cramers_v': r['cramers_v'],
            'n_converged': r['n_converged'], 'n_reps': r['n_reps'],
            'conv_frac': r['n_converged'] / r['n_reps'] if r['n_reps'] else np.nan,
        })
    return pd.DataFrame(rows)

print('computing bistability (B=%d) ...' % B_BIST)
bist = pd.concat([load_bistability(b, t) for b, t in _bases()], ignore_index=True)
print('label mix by ds:')
print(pd.crosstab(bist.ds, bist.label).to_string())
print()
print('convergence: mean converged reps per cell and mean convergence RATE by ds:')
print('  (rate is R-independent; raw count is inflated for new by R50 vs old R30)')
print(bist.groupby('ds')[['n_converged','conv_frac']].mean().round(3).to_string())""")

co(r"""neff_res = paired_wilcoxon(bist, 'n_eff')
show(neff_res, 'N_eff (distinct consensus attractors; lower = more consistent)', fmt='{:+.4f}')
print()
cv_res = paired_wilcoxon(bist, 'cramers_v')
show(cv_res, "Cramer's V (initial->final basin association)", fmt='{:+.4f}')
print()
conv_res = paired_wilcoxon(bist, 'conv_frac')
show(conv_res, 'Convergence rate (fraction of reps settling to unanimous consensus; R-independent)', fmt='{:+.4f}')""")

co(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 4))
lab_order = ['monostable','multistable','stochastic','insufficient']
for ax, ds in zip(axes, DS_ORDER):
    ct = (bist[bist.ds==ds].groupby(['config','label']).size()
          .unstack(fill_value=0).reindex(index=CONFIG_ORDER, columns=lab_order, fill_value=0))
    ct.plot(kind='bar', stacked=True, ax=ax, legend=(ds=='new'),
            color=['#55A868','#EE6666','#DD8452','#cccccc'])
    ax.set_title(f'{DS_LABEL[ds]} — bistability labels'); ax.set_ylabel('# cells (of 35)')
    ax.set_xticklabels(CONFIG_ORDER, rotation=45, ha='right')
plt.tight_layout()
plt.show()""")

# ---------------- SUMMARY ----------------
md(r"""## 6. Summary — reproduces `tab:app:sysprompt`

One row per measure: pooled mean old (Milder), pooled mean new (Critical), pooled paired Δ, and the
**number of the 6 configs** with a Holm-significant paired shift plus its shared sign. This is the
compact readout the appendix table prints.
""")

co(r"""def pooled_delta(res):
    return res['delta'].mean()
def sig_count(res):
    s = res[res.p_holm < 0.05]
    if len(s) == 0:
        return '0/%d' % len(res)
    signs = np.sign(s['delta'])
    tag = '+' if (signs > 0).all() else ('-' if (signs < 0).all() else '+/-')
    return '%d/%d (%s)' % (len(s), len(res), tag)

summary_rows = [
    ('Accuracy',                    taskvals, 'accuracy',        acc_res),
    ('Mean rounds',                 taskvals, 'mean_rounds',     rnd_res),
    ('Ceiling-hit frac',            taskvals, 'ceil_frac',       ceil_res),
    ('Convergence rate',            bist,     'conv_frac',       conv_res),
    ('Dominance D',                 dom,      'D_mean',          dom_res),
    ('Hub capitulation',            dom,      'hub_capitulation',cap_res),
    ('Self-reinf p_sr',             sr,       'p_sr',            psr_res),
    ('Self-reinf slope',            sr,       'mean_slope',      slp_res),
    ('Limit cycle (agent)',         lc,       'p_lc_agent',      lca_res),
    ('Limit cycle (system)',        lc,       'p_lc_system',     lcs_res),
    ('Bistability N_eff',           bist,     'n_eff',           neff_res),
    ("Bistability Cramer's V",      bist,     'cramers_v',       cv_res),
]
rows = []
for name, df_, col, res in summary_rows:
    rows.append({
        'metric': name,
        'old_mean': df_[df_.ds=='old'][col].mean(),
        'new_mean': df_[df_.ds=='new'][col].mean(),
        'mean_delta': pooled_delta(res),
        'sig_configs': sig_count(res),
    })
summary = pd.DataFrame(rows)
disp = summary.copy()
for c in ['old_mean','new_mean','mean_delta']:
    disp[c] = disp[c].map(lambda x: f'{x:+.4f}' if pd.notna(x) else 'nan')
print(disp.to_string(index=False))""")

md(r"""## 7. Number traceability — computed vs `tab:app:sysprompt`

Prints each table cell as `computed  thesis_says=<tex value>  MATCH|MISMATCH`. Means use a 0.01
absolute tolerance (the table rounds to 2–3 dp); `sig_configs` is compared as a string. If the old
arm is a frozen artifact (`OLD_AVAILABLE=False`) the Milder column and `sig_configs` cannot be
recomputed — those are flagged `FROZEN` rather than MISMATCH.
""")

co(r"""# --- app:sysprompt  tab:app:sysprompt (thesis/chapters/appendix.tex ~L165-190) ---
# thesis values: (milder_old, critical_new, sig_configs_string)
THESIS = {
    'Accuracy':                (0.316, 0.261, '0/6'),
    'Mean rounds':             (6.85,  5.76,  '3/6 (-)'),
    'Ceiling-hit frac':        (0.146, 0.068, '3/6 (-)'),
    'Convergence rate':        (0.862, 0.940, '3/6 (+)'),
    'Dominance D':             (0.375, 0.371, '0/6'),
    'Hub capitulation':        (0.146, 0.157, '4/6 (+)'),
    'Self-reinf p_sr':         (0.830, 0.844, '0/6'),
    'Self-reinf slope':        (0.372, 0.409, '1/6 (+)'),
    'Limit cycle (agent)':     (0.019, 0.043, '0/6'),
    'Limit cycle (system)':    (0.040, 0.096, '0/6'),
    'Bistability N_eff':       (2.04,  2.06,  '1/6 (+)'),
    "Bistability Cramer's V":  (0.394, 0.346, '0/6'),
}
TOL = 0.01
srow = summary.set_index('metric')
print('--- tab:app:sysprompt : milder(old) column ---')
for m, (t_old, t_new, t_sig) in THESIS.items():
    c = srow.loc[m, 'old_mean']
    if pd.isna(c):
        print(f'  {m:24s} computed=nan (FROZEN)  thesis_says={t_old}')
    else:
        tag = 'MATCH' if abs(c - t_old) <= TOL else 'MISMATCH'
        print(f'  {m:24s} computed={c:.3f}  thesis_says={t_old}  {tag}')
print()
print('--- tab:app:sysprompt : critical(new) column ---')
for m, (t_old, t_new, t_sig) in THESIS.items():
    c = srow.loc[m, 'new_mean']
    tag = 'MATCH' if pd.notna(c) and abs(c - t_new) <= TOL else 'MISMATCH'
    print(f'  {m:24s} computed={c:.3f}  thesis_says={t_new}  {tag}')
print()
print('--- tab:app:sysprompt : Sig. configs column ---')
for m, (t_old, t_new, t_sig) in THESIS.items():
    c = srow.loc[m, 'sig_configs']
    if not OLD_AVAILABLE:
        print(f'  {m:24s} computed={c} (FROZEN old arm)  thesis_says={t_sig}')
    else:
        tag = 'MATCH' if c == t_sig else 'MISMATCH'
        print(f'  {m:24s} computed={c:8s}  thesis_says={t_sig}  {tag}')""")

md(r"""### Reading guide

- **`sig_configs`** = how many of the 6 configs show a Holm-significant paired shift, with the shared
  sign (`+` critical higher, `−` critical lower, `+/-` mixed). `0/6` = no reliable drift.
- **Confidence-weighted metrics** (Dominance D, Hub capitulation, Self-reinf slope) carry the
  old-data 0-confidence artifact — treat cross-dataset *magnitude* with the caveat noted per section;
  sign-based summaries (p_sr, flag rates) are more robust.
- **Bottom line (as written in the thesis):** the critical prompt changes the *dynamics*
  (faster fc convergence, ceiling-hits halved, higher unanimous-convergence rate) but leaves accuracy
  unchanged (0/6) and does not suppress self-reinforcement (0/6). The N_eff / limit-cycle uptick is a
  by-product of more runs converging, not reduced multistability (Cramér's V trends down). Single
  HiddenBench comparison — not evidence of robustness across models or group sizes.
""")

nb['cells'] = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
from pathlib import Path as _P
_out = _P(__file__).resolve().parents[2] / 'notebooks' / 'golden' / '11_appendix_robustness.ipynb'
_out.parent.mkdir(parents=True, exist_ok=True)
import nbformat as _nbf
with open(_out, 'w') as _fh:
    _nbf.write(nb, _fh)
print('wrote', _out, 'with', len(cells), 'cells')
