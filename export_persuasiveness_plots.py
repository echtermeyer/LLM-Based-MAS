import sys, json, re
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as sp_stats
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent))
from src.metrics.persuasiveness import Persuasiveness, PersuasivenessPostFlip

OUT   = Path('plots/persuasiveness')
BASE  = Path('results/mas/gpqa_full_sim_tier3')
OUT.mkdir(parents=True, exist_ok=True)

W_VALUES   = [1, 2, 5]
W_COLORS   = {1: '#4C72B0', 2: '#DD8452', 5: '#55A868'}
DS_COLORS  = {'gpqa': '#5470C6', 'hiddenbench': '#EE6666'}
CEILING    = 10.0
datasets   = ['gpqa', 'hiddenbench']

pers_metric      = Persuasiveness()
pers_plus_metric = PersuasivenessPostFlip(ceiling=CEILING)

def load_window_data(w):
    folder = BASE / f'W{w}_fc'
    by_qid = {}
    for f in sorted(folder.glob('*.json')):
        d = json.loads(f.read_text())
        qid = d['question_id']
        if qid not in by_qid or str(f) > by_qid[qid]['_path']:
            d['_path'] = str(f)
            by_qid[qid] = d
    return by_qid

raw = {w: load_window_data(w) for w in W_VALUES}
common_qids = sorted(set(raw[1].keys()) & set(raw[2].keys()) & set(raw[5].keys()))
print(f'Common qids: {len(common_qids)}')

records = []
for w, data in raw.items():
    for qid, d in data.items():
        if qid not in common_qids:
            continue
        gt = d['ground_truth']
        dataset = d.get('dataset', 'unknown')
        for rep_idx, rep in enumerate(d['repetitions']):
            traj = rep['trajectory']
            T = len(traj)
            N = len(traj[0]['phase_b'])
            p_scores  = pers_metric.score(rep)
            pp_scores = pers_plus_metric.score(rep)
            votes_final = [traj[-1]['phase_b'][a]['vote'] for a in range(N)]
            majority_final = max(set(votes_final), key=votes_final.count)
            rep_correct   = majority_final == gt
            rep_converged = len(set(votes_final)) == 1
            for a in range(N):
                v0 = traj[0]['phase_b'][a]['vote']
                vf = traj[-1]['phase_b'][a]['vote']
                confs = [traj[t]['phase_b'][a].get('confidence') or 0 for t in range(T)]
                msgs  = [traj[t]['phase_b'][a].get('message','') for t in range(T)]
                reas  = [traj[t]['phase_b'][a].get('reasoning','') for t in range(T)]
                pt = sum(traj[t]['phase_b'][a].get('prompt_tokens', 0) for t in range(T))
                ct = sum(traj[t]['phase_b'][a].get('completion_tokens', 0) for t in range(T))
                own_flips = sum(1 for t in range(1, T) if traj[t]['phase_b'][a]['vote'] != traj[t-1]['phase_b'][a]['vote'])
                votes_r0 = [traj[0]['phase_b'][b]['vote'] for b in range(N)]
                in_majority_r0 = votes_r0.count(v0) > N / 2
                combined_text = ' '.join(msgs)
                math_count    = len(re.findall(r'\d+\.?\d*', combined_text))
                formula_count = len(re.findall(r'[=<>≈±×÷∑∫√]', combined_text))
                hedge_count   = len(re.findall(r'\b(however|although|but|yet|while|though|uncertain|maybe|perhaps|possibly)\b', combined_text, re.I))
                assert_count  = len(re.findall(r'\b(therefore|thus|clearly|obviously|must|certainly|definitely|conclude)\b', combined_text, re.I))
                avg_msg_len   = np.mean([len(m.split()) for m in msgs]) if msgs else 0
                avg_reas_len  = np.mean([len(r.split()) for r in reas]) if reas else 0
                records.append({
                    'W': w, 'qid': qid, 'dataset': dataset, 'rep': rep_idx, 'agent': a,
                    'pers': p_scores[a], 'pers_plus': pp_scores[a],
                    'T': T, 'N': N, 'own_flips': own_flips, 'in_majority_r0': in_majority_r0,
                    'conf_r0': confs[0], 'conf_final': confs[-1], 'conf_mean': np.mean(confs),
                    'conf_delta': confs[-1] - confs[0],
                    'prompt_tokens': pt, 'completion_tokens': ct, 'total_tokens': pt+ct,
                    'initial_correct': v0==gt, 'final_correct': vf==gt,
                    'rep_correct': rep_correct, 'rep_converged': rep_converged,
                    'math_count': math_count, 'formula_count': formula_count,
                    'hedge_count': hedge_count, 'assert_count': assert_count,
                    'avg_msg_len': avg_msg_len, 'avg_reas_len': avg_reas_len,
                })

df = pd.DataFrame(records)
print(f'Rows: {len(df)}')

rep_df = df.groupby(['dataset', 'W', 'qid', 'rep']).agg(
    pers_mean=('pers', 'mean'),
    pers_max=('pers', 'max'),
    pers_plus_mean=('pers_plus', 'mean'),
    pers_spread=('pers', lambda x: x.max() - x.min()),
    rep_correct=('rep_correct', 'first'),
    rep_converged=('rep_converged', 'first'),
    T=('T', 'first'),
).reset_index()

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}.png')

def kde_line(ax, vals, color, label=None, bw=0.08):
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return
    kde = gaussian_kde(vals, bw_method=bw)
    xs = np.linspace(-1.1, 1.1, 400)
    ax.plot(xs, kde(xs), color=color, linewidth=2, label=label)

print('--- Part 1: Distributions ---')
for score_col, score_label in [('pers', 'Pers'), ('pers_plus', 'Pers+')]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ds in zip(axes, datasets):
        sub = df[df['dataset'] == ds]
        ax.hist(sub[score_col], bins=40, density=True, color=DS_COLORS[ds], alpha=0.35, edgecolor='white')
        kde_line(ax, sub[score_col].values, DS_COLORS[ds], label='all W')
        for w in W_VALUES:
            kde_line(ax, sub[sub['W']==w][score_col].values, W_COLORS[w], label=f'W={w}', bw=0.1)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axvline(sub[score_col].mean(), color=DS_COLORS[ds], linewidth=1.5, linestyle=':',
                   label=f'mean={sub[score_col].mean():.3f}')
        ax.set_xlabel(score_label, fontsize=11); ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{score_label} — {ds}  (n={len(sub)})', fontsize=12)
        ax.legend(fontsize=9); ax.set_xlim(-1.1, 1.1); ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle(f'{score_label} distribution by dataset and W', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save(fig, f'01_distribution_{score_col}')

qdf = df.groupby(['dataset', 'qid', 'W'])[['pers', 'pers_plus']].mean().reset_index()
for score_col, score_label in [('pers', 'Pers'), ('pers_plus', 'Pers+')]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, ds in zip(axes, datasets):
        sub = qdf[qdf['dataset'] == ds]
        qids = sorted(sub['qid'].unique())
        x = np.arange(len(qids))
        for i, w in enumerate(W_VALUES):
            vals = [sub[(sub['qid']==q)&(sub['W']==w)][score_col].values[0]
                    if len(sub[(sub['qid']==q)&(sub['W']==w)]) > 0 else np.nan for q in qids]
            ax.bar(x+(i-1)*0.25, vals, 0.25, color=W_COLORS[w], alpha=0.85, edgecolor='white', label=f'W={w}')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels([f'q{q}' for q in qids], rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(f'Mean {score_label}', fontsize=10)
        ax.set_title(f'{score_label} per question — {ds}', fontsize=12)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle(f'{score_label}: mean per question × W', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save(fig, f'02_per_question_{score_col}')

print('--- Part 2: Outcome ---')
def binned_outcome(ax, sub, x_col, y_col, color, n_bins=8, ylabel='', title=''):
    bins = np.percentile(sub[x_col], np.linspace(0, 100, n_bins+1))
    bins = np.unique(bins)
    if len(bins) < 3:
        return
    labels = (bins[:-1]+bins[1:])/2
    cut = pd.cut(sub[x_col], bins=bins, labels=labels)
    grp = sub.groupby(cut)[y_col]
    means, sems, ns = grp.mean(), grp.sem(), grp.count()
    valid = ns >= 3
    xs = means.index.astype(float)[valid]
    ax.plot(xs, means[valid], marker='o', color=color, linewidth=2, markersize=6)
    ax.fill_between(xs, (means-sems*1.96)[valid], (means+sems*1.96)[valid], color=color, alpha=0.15)
    r, p = sp_stats.spearmanr(sub[x_col], sub[y_col])
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    ax.set_xlabel(x_col.replace('_',' '), fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f'{title}\nSpearman r={r:.2f} {sig}', fontsize=11)
    ax.grid(alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for row, ds in enumerate(datasets):
    sub = rep_df[rep_df['dataset']==ds]
    color = DS_COLORS[ds]
    binned_outcome(axes[row,0], sub, 'pers_mean', 'rep_correct',   color, ylabel='P(correct)',   title=f'{ds}: mean Pers → correct')
    binned_outcome(axes[row,1], sub, 'pers_mean', 'rep_converged', color, ylabel='P(converged)', title=f'{ds}: mean Pers → converged')
    binned_outcome(axes[row,2], sub, 'pers_max',  'rep_correct',   color, ylabel='P(correct)',   title=f'{ds}: max Pers → correct')
    binned_outcome(axes[row,3], sub, 'pers_spread','rep_correct',  color, ylabel='P(correct)',   title=f'{ds}: Pers spread → correct')
fig.suptitle('Persuasiveness → system outcome', fontsize=14, fontweight='bold')
plt.tight_layout()
save(fig, '03_outcome_binned')

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for row, ds in enumerate(datasets):
    sub = rep_df[rep_df['dataset']==ds].copy()
    med = sub['pers_mean'].median()
    sub['pers_group'] = (sub['pers_mean']>med).map({True:'High Pers', False:'Low Pers'})
    for col_idx, (y_col, ylabel) in enumerate([('rep_correct','P(correct)'),('rep_converged','P(converged)'),('T','Mean rounds')]):
        ax = axes[row, col_idx]
        for grp, color in [('High Pers','#55A868'),('Low Pers','#C44E52')]:
            vals = sub[sub['pers_group']==grp][y_col]
            ax.bar(grp, vals.mean(), color=color, alpha=0.8, edgecolor='white')
            ax.errorbar(grp, vals.mean(), yerr=vals.sem()*1.96, fmt='none', color='black', capsize=5)
            ax.text(grp, vals.mean()+0.01, f'{vals.mean():.3f}', ha='center', fontsize=10, fontweight='bold')
        t, p = sp_stats.ttest_ind(sub[sub['pers_group']=='High Pers'][y_col], sub[sub['pers_group']=='Low Pers'][y_col])
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else f'ns p={p:.2f}'
        ax.set_title(f'{ds}: {ylabel}\nhigh vs low Pers ({sig})', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('High vs Low Pers — outcome comparison (median split)', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '04_high_vs_low_pers')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ds in zip(axes, datasets):
    sub = rep_df[rep_df['dataset']==ds]
    for col, label, color in [('pers_mean','Pers','#4C72B0'),('pers_plus_mean','Pers+','#C44E52')]:
        bins = np.percentile(sub[col], np.linspace(0,100,9)); bins = np.unique(bins)
        if len(bins) < 3: continue
        lbls = (bins[:-1]+bins[1:])/2
        cut = pd.cut(sub[col], bins=bins, labels=lbls)
        grp = sub.groupby(cut)['rep_correct']
        means, sems = grp.mean(), grp.sem()
        xs = means.index.astype(float)
        ax.plot(xs, means, marker='o', color=color, linewidth=2, markersize=5, label=label)
        ax.fill_between(xs, means-sems*1.96, means+sems*1.96, color=color, alpha=0.15)
    ax.set_xlabel('Persuasiveness (mean over agents)', fontsize=10); ax.set_ylabel('P(correct)', fontsize=10)
    ax.set_title(f'{ds}: Pers vs Pers+ predicting correctness', fontsize=11)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Pers vs Pers+: correctness prediction', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '05_pers_vs_persplus_outcome')

print('--- Part 3: Drivers ---')
feature_pairs = [
    ('prompt_tokens','Prompt tokens'),('completion_tokens','Completion tokens'),
    ('conf_r0','Initial confidence'),('conf_delta','Confidence change'),
    ('avg_msg_len','Avg message length (words)'),('avg_reas_len','Avg reasoning length (words)'),
    ('math_count','Numeric expressions'),('assert_count','Assertive language'),
]
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, (feat, flabel) in zip(axes.flat, feature_pairs):
    for ds, color in DS_COLORS.items():
        sub = df[df['dataset']==ds].copy()
        bins = np.percentile(sub[feat], np.linspace(0,100,9)); bins = np.unique(bins)
        if len(bins) < 3: continue
        lbls = (bins[:-1]+bins[1:])/2
        cut = pd.cut(sub[feat], bins=bins, labels=lbls)
        grp = sub.groupby(cut)['pers']
        means, sems, ns = grp.mean(), grp.sem(), grp.count()
        valid = ns >= 5
        xs = means.index.astype(float)[valid]
        ax.plot(xs, means[valid], marker='o', color=color, linewidth=2, markersize=5, label=ds)
        ax.fill_between(xs, (means-sems*1.96)[valid], (means+sems*1.96)[valid], color=color, alpha=0.12)
    r_g, p_g = sp_stats.spearmanr(df[df['dataset']=='gpqa'][feat], df[df['dataset']=='gpqa']['pers'])
    r_h, p_h = sp_stats.spearmanr(df[df['dataset']=='hiddenbench'][feat], df[df['dataset']=='hiddenbench']['pers'])
    sig_g = '***' if p_g<0.001 else '**' if p_g<0.01 else '*' if p_g<0.05 else 'ns'
    sig_h = '***' if p_h<0.001 else '**' if p_h<0.01 else '*' if p_h<0.05 else 'ns'
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
    ax.set_xlabel(flabel, fontsize=9); ax.set_ylabel('Mean Pers', fontsize=9)
    ax.set_title(f'{flabel}\ngpqa r={r_g:.2f}{sig_g}  hb r={r_h:.2f}{sig_h}', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Features vs Pers — binned mean ± 95% CI', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '06_feature_vs_pers')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ds in zip(axes, datasets):
    sub = df[df['dataset']==ds]
    for maj, label, color in [(True,'Majority r0','#4C72B0'),(False,'Minority r0','#C44E52')]:
        vals = sub[sub['in_majority_r0']==maj]['pers'].values
        bp = ax.boxplot(vals, positions=[0 if maj else 1], widths=0.4, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        boxprops=dict(facecolor=color, alpha=0.6),
                        whiskerprops=dict(color=color), capprops=dict(color=color),
                        flierprops=dict(marker='', alpha=0))
        jitter = np.random.uniform(-0.1, 0.1, min(len(vals), 300))
        sample = np.random.choice(vals, min(len(vals), 300), replace=False)
        ax.scatter([0 if maj else 1]+jitter, sample, s=5, alpha=0.2, color=color, edgecolors='none')
        ax.text(0 if maj else 1, np.median(vals)+0.04, f'{np.median(vals):.3f}',
                ha='center', fontsize=10, fontweight='bold')
    t, p = sp_stats.ttest_ind(sub[sub['in_majority_r0']==True]['pers'], sub[sub['in_majority_r0']==False]['pers'])
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else f'ns p={p:.2f}'
    ax.set_xticks([0,1]); ax.set_xticklabels(['Majority r0','Minority r0'], fontsize=11)
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
    ax.set_ylabel('Pers', fontsize=10)
    ax.set_title(f'{ds}: majority vs minority\nt-test: {sig}', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Initial position → Persuasiveness', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '07_majority_minority')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
corr_feats = ['W','T','conf_r0','conf_final','conf_delta','conf_mean',
              'prompt_tokens','completion_tokens','math_count','formula_count',
              'hedge_count','assert_count','avg_msg_len','avg_reas_len','pers','pers_plus']
for ax, ds in zip(axes, datasets):
    sub = df[df['dataset']==ds][corr_feats].dropna()
    corr = sub.corr(method='spearman')
    pers_row = corr['pers'].drop(['pers','pers_plus']).sort_values()
    colors = ['#C44E52' if v<0 else '#55A868' for v in pers_row]
    ax.barh(range(len(pers_row)), pers_row.values, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(pers_row))); ax.set_yticklabels(pers_row.index, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    for i, v in enumerate(pers_row.values):
        ax.text(v+(0.005 if v>=0 else -0.005), i, f'{v:.3f}', va='center',
                ha='left' if v>=0 else 'right', fontsize=8)
    ax.set_xlabel('Spearman r with Pers', fontsize=10)
    ax.set_title(f'{ds}: feature correlations with Pers', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Spearman correlation with Pers — ranked', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '08_correlations')

pre_flip_records = []
for w, data in raw.items():
    for qid, d in data.items():
        if qid not in common_qids: continue
        dataset = d.get('dataset','unknown')
        for rep in d['repetitions']:
            traj = rep['trajectory']
            T = len(traj); N = len(traj[0]['phase_b'])
            for a in range(N):
                for t in range(T-1):
                    ag = traj[t]['phase_b'][a]; ag_nx = traj[t+1]['phase_b'][a]
                    msg = ag.get('message','')
                    did_flip = ag['vote'] != ag_nx['vote']
                    pre_flip_records.append({
                        'W': w, 'qid': qid, 'dataset': dataset, 'round': t,
                        'did_flip_next': did_flip,
                        'msg_len': len(msg.split()),
                        'math_count': len(re.findall(r'\d+\.?\d*', msg)),
                        'formula_count': len(re.findall(r'[=<>≈±×÷∑∫√]', msg)),
                        'hedge_count': len(re.findall(r'\b(however|although|but|yet|while|though|uncertain|maybe|perhaps|possibly)\b', msg, re.I)),
                        'assert_count': len(re.findall(r'\b(therefore|thus|clearly|obviously|must|certainly|definitely|conclude)\b', msg, re.I)),
                        'conf': ag.get('confidence') or 0,
                    })

pf = pd.DataFrame(pre_flip_records)
text_feats = ['msg_len','math_count','formula_count','hedge_count','assert_count','conf']

fig, axes = plt.subplots(2, len(text_feats), figsize=(22, 9))
for row, ds in enumerate(datasets):
    sub = pf[pf['dataset']==ds]
    for ax, feat in zip(axes[row], text_feats):
        flip_vals  = sub[sub['did_flip_next']==True][feat]
        nflip_vals = sub[sub['did_flip_next']==False][feat]
        t, p = sp_stats.ttest_ind(flip_vals, nflip_vals)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        for vals, label, color in [(nflip_vals,'stayed','#4C72B0'),(flip_vals,'flipped','#C44E52')]:
            ax.bar(label, vals.mean(), color=color, alpha=0.8, edgecolor='white')
            ax.errorbar(label, vals.mean(), yerr=vals.sem()*1.96, fmt='none', color='black', capsize=4)
        ax.set_title(f'{feat}\n{sig}', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[row,0].set_ylabel(f'{ds}\nMean', fontsize=10)
fig.suptitle('Message features before flip vs stay', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '09_preflip_text_features')

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for row, ds in enumerate(datasets):
    sub = df[df['dataset']==ds]
    for col_idx, (score_col, score_label) in enumerate([('pers','Pers'),('pers_plus','Pers+')]):
        ax = axes[row, col_idx]
        data_by_w = [sub[sub['W']==w][score_col].dropna().values for w in W_VALUES]
        parts = ax.violinplot(data_by_w, positions=W_VALUES, showmedians=True, showextrema=False)
        for pc, w in zip(parts['bodies'], W_VALUES):
            pc.set_facecolor(W_COLORS[w]); pc.set_alpha(0.7)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        for w, vals in zip(W_VALUES, data_by_w):
            jitter = np.random.uniform(-0.12, 0.12, min(len(vals), 300))
            sample = np.random.choice(vals, min(len(vals), 300), replace=False)
            ax.scatter(w+jitter, sample, s=5, alpha=0.2, color=W_COLORS[w], edgecolors='none')
            ax.text(w, np.median(vals)+0.04, f'{np.median(vals):.3f}',
                    ha='center', fontsize=9, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
        ax.set_xticks(W_VALUES); ax.set_xticklabels([f'W={w}' for w in W_VALUES])
        ax.set_ylabel(score_label, fontsize=10)
        ax.set_title(f'{ds}: {score_label} by W', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax = axes[row, 2]
    for w in W_VALUES:
        subw = sub[sub['W']==w]
        r, p = sp_stats.spearmanr(subw['pers'], subw['pers_plus'])
        ax.scatter(subw['pers'], subw['pers_plus'], color=W_COLORS[w], s=5, alpha=0.2,
                   edgecolors='none', label=f'W={w} r={r:.2f}')
    ax.set_xlabel('Pers', fontsize=10); ax.set_ylabel('Pers+', fontsize=10)
    ax.set_title(f'{ds}: Pers vs Pers+', fontsize=11)
    ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Persuasiveness by W', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, '10_pers_by_W')

print(f'\nAll plots saved to {OUT}/')
print('\n--- Summary stats ---')
print(df.groupby(['dataset','W'])[['pers','pers_plus']].agg(['mean','median','std']).round(3).to_string())
