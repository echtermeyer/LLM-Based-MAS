import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest


def task_means(df, task_col, value_col, cell_cols=None, agg='mean'):
    keys = ([] if cell_cols is None else list(cell_cols)) + [task_col]
    return df.groupby(keys)[value_col].agg(agg).reset_index()


def cluster_bootstrap_ci(values, statistic=np.mean, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    boot = np.array([statistic(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(arr)), float(lo), float(hi)


def cluster_bootstrap_diff(a, b, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    diffs = np.array([rng.choice(a, na, True).mean() - rng.choice(b, nb, True).mean()
                      for _ in range(n_boot)])
    obs = a.mean() - b.mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(p, 1.0))


def task_paired_wilcoxon(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if np.all(a - b == 0):
        return np.nan, 1.0, 0.0, int(mask.sum())
    stat, p = stats.wilcoxon(a, b)
    return float(stat), float(p), float(np.median(a - b)), int(mask.sum())


def task_one_sample_wilcoxon(values, null=0.0):
    arr = np.asarray(values, float)
    arr = arr[~np.isnan(arr)] - null
    if np.all(arr == 0):
        return np.nan, 1.0, int(len(arr))
    stat, p = stats.wilcoxon(arr)
    return float(stat), float(p), int(len(arr))


def task_sign_test(values, null=0.5, alternative='two-sided'):
    arr = np.asarray(values, float)
    arr = arr[~np.isnan(arr)]
    k = int((arr > null).sum())
    n = int((arr != null).sum())
    p = stats.binomtest(k, n, 0.5, alternative=alternative).pvalue if n else np.nan
    return k, n, float(p)


def task_spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), int(mask.sum())


def task_friedman(*groups):
    groups = [np.asarray(g, float) for g in groups]
    stat, p = stats.friedmanchisquare(*groups)
    return float(stat), float(p), int(len(groups[0]))


def two_proportion_test(k1, n1, k2, n2):
    stat, p = proportions_ztest([k1, k2], [n1, n2])
    return float(stat), float(p)


def bh_fdr(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    rej, q, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    return q.tolist(), rej.tolist()


def cramers_v_bias_corrected(table):
    table = np.asarray(table, float)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else np.nan
