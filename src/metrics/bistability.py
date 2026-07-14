import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def _composition_seq(traj: List[Dict], options: tuple) -> List[Tuple[int, ...]]:
    result = []
    for t in range(len(traj)):
        votes = [ag['vote'] for ag in traj[t]['phase_b']]
        result.append(tuple(votes.count(o) for o in options))
    return result



def _attractor_label(composition: Tuple[int, ...], options: tuple) -> Any:
    total = sum(composition)
    for cnt, opt in zip(composition, options):
        if cnt == total:
            return opt
    return composition


def _trailing_k(seq: list) -> int:
    if not seq:
        return 0
    last = seq[-1]
    k = 0
    for x in reversed(seq):
        if x == last:
            k += 1
        else:
            break
    return k


def _n_eff(counter: Dict) -> float:
    total = sum(counter.values())
    if total == 0:
        return 1.0
    return 1.0 / sum((v / total) ** 2 for v in counter.values())


def _sort_key(x: Any):
    return (0, x) if isinstance(x, str) else (1, str(x))


def _cramers_v(g0s: List, endpoints: List) -> float:
    g0_cats = sorted(set(g0s))
    ep_cats = sorted(set(endpoints), key=_sort_key)
    R, C = len(g0_cats), len(ep_cats)
    if R < 2 or C < 2:
        return 0.0
    g0_idx = {v: i for i, v in enumerate(g0_cats)}
    ep_idx = {v: i for i, v in enumerate(ep_cats)}
    n = len(g0s)
    table = np.zeros((R, C), dtype=float)
    for g, e in zip(g0s, endpoints):
        table[g0_idx[g], ep_idx[e]] += 1
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    expected = row_sums * col_sums / n
    chi2 = float(np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)))
    denom = min(R - 1, C - 1)
    if denom == 0:
        return 0.0
    return float(min(np.sqrt(chi2 / n / denom), 1.0))


def _permutation_p(
    g0s: List,
    endpoints: List,
    v_obs: float,
    B: int,
    rng: np.random.Generator,
) -> float:
    n = len(g0s)
    count = 0
    for _ in range(B):
        perm = rng.permutation(n)
        g0_perm = [g0s[i] for i in perm]
        if _cramers_v(g0_perm, endpoints) >= v_obs:
            count += 1
    return (1 + count) / (1 + B)


def _bootstrap_n_eff_ci(
    endpoints: List,
    B_boot: int,
    rng: np.random.Generator,
    alpha: float,
) -> Tuple[float, float]:
    n = len(endpoints)
    boots = []
    for _ in range(B_boot):
        idx = rng.integers(0, n, size=n)
        counter: Dict = {}
        for i in idx:
            e = endpoints[i]
            counter[e] = counter.get(e, 0) + 1
        boots.append(_n_eff(counter))
    arr = np.array(boots)
    return float(np.percentile(arr, 100 * alpha / 2)), float(np.percentile(arr, 100 * (1 - alpha / 2)))


def analyse_bistability(
    repetitions: List[Dict],
    u: int = 3,
    B: int = 1000,
    B_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
    options: Optional[tuple] = None,
) -> Dict:
    """
    Bistability / multistability analysis for one (task, config) cell.

    Signature 1 — coexistence: N_eff (inverse Simpson over endpoint attractors).
    Signature 2 — basin selection: Cramér's V + permutation p.

    Label rule
    ----------
    monostable  : N_eff < 1.5
    multistable : N_eff >= 1.5  and  p_basin < alpha  and  ci_lo >= 1.5
    stochastic  : N_eff >= 1.5  and  (p_basin >= alpha  or  ci_lo < 1.5)
    insufficient: n_converged < 2

    Returns
    -------
    dict with:
      n_reps, n_converged, n_excluded,
      M            — number of options for this task
      p_hat        — {attractor: fraction}
      n_eff        — inverse Simpson  (nan if insufficient)
      n_eff_ci     — (lo, hi) bootstrap CI at 1-alpha coverage
      cramers_v    — Cramér's V  (nan if insufficient)
      p_basin      — permutation p  (nan if insufficient)
      label        — 'monostable' | 'multistable' | 'stochastic' | 'insufficient'
      degree       — 'bistable' | 'tristable' | 'quadstable' | None
      reps         — list of per-rep records {rep_idx, g0, g0_coarse, attractor, converged}
    """
    if options is None:
        options = tuple(repetitions[0]['options'].keys())
    M = len(options)
    rng = np.random.default_rng(seed)

    per_rep = []
    for rep_idx, rep in enumerate(repetitions):
        traj = rep['trajectory']
        comp_seq = _composition_seq(traj, options)
        k = _trailing_k(comp_seq)
        converged = k >= u
        g0 = comp_seq[0]
        attractor = _attractor_label(comp_seq[-1], options) if converged else None
        per_rep.append({'rep_idx': rep_idx, 'g0': g0, 'attractor': attractor, 'converged': converged})

    converged_reps = [r for r in per_rep if r['converged']]
    n_c = len(converged_reps)

    # Determine dominant endpoint attractor, then coarsen g0 to a scalar:
    # g0_coarse = #agents initially voting for the dominant option (in {1,2,3,4}).
    # Falls back to the full tuple when the dominant attractor is itself non-unanimous.
    # Computed for all reps so records stay inspectable; non-converged reps use None.
    if n_c >= 2:
        ep_counter: Dict = {}
        for r in converged_reps:
            e = r['attractor']
            ep_counter[e] = ep_counter.get(e, 0) + 1
        dom_att = max(ep_counter, key=ep_counter.__getitem__)
        for r in per_rep:
            if isinstance(dom_att, str):
                r['g0_coarse'] = r['g0'][options.index(dom_att)]
            else:
                r['g0_coarse'] = r['g0']
    else:
        dom_att = None
        for r in per_rep:
            r['g0_coarse'] = None

    result: Dict = {
        'n_reps': len(repetitions),
        'n_converged': n_c,
        'n_excluded': len(repetitions) - n_c,
        'M': M,
        'p_hat': {},
        'n_eff': float('nan'),
        'n_eff_ci': (float('nan'), float('nan')),
        'cramers_v': float('nan'),
        'p_basin': float('nan'),
        'label': 'insufficient',
        'degree': None,
        'reps': per_rep,
    }

    if n_c < 2:
        return result

    endpoints = [r['attractor'] for r in converged_reps]
    g0s = [r['g0_coarse'] for r in converged_reps]

    counter: Dict = ep_counter
    p_hat = {k: v / n_c for k, v in counter.items()}
    n_eff = _n_eff(counter)

    ci_lo, ci_hi = _bootstrap_n_eff_ci(endpoints, B_boot, rng, alpha)

    v = _cramers_v(g0s, endpoints)
    p_basin = _permutation_p(g0s, endpoints, v, B, rng)

    if n_eff < 1.5:
        label, degree = 'monostable', None
    elif p_basin < alpha and ci_lo >= 1.5:
        label = 'multistable'
        _degree_names = {2: 'bistable', 3: 'tristable', 4: 'quadstable'}
        degree = _degree_names.get(min(round(n_eff), M))
    else:
        label, degree = 'stochastic', None

    result.update({
        'p_hat': p_hat,
        'n_eff': n_eff,
        'n_eff_ci': (ci_lo, ci_hi),
        'cramers_v': v,
        'p_basin': p_basin,
        'label': label,
        'degree': degree,
    })
    return result


def summarise_bistability(cell_results: List[Dict]) -> Dict:
    """
    Aggregate a list of per-cell analyse_bistability results into summary scalars.
    """
    valid = [r for r in cell_results if r['label'] != 'insufficient']
    n_valid = len(valid)
    if n_valid == 0:
        return {
            'n_cells': len(cell_results),
            'n_insufficient': len(cell_results),
            'n_monostable': 0,
            'n_multistable': 0,
            'n_stochastic': 0,
            'p_multistable': float('nan'),
            'n_eff_mean': float('nan'),
            'n_eff_median': float('nan'),
            'cramers_v_mean': float('nan'),
            'cramers_v_median': float('nan'),
        }

    labels = [r['label'] for r in valid]
    n_effs = [r['n_eff'] for r in valid if not np.isnan(r['n_eff'])]
    vs = [r['cramers_v'] for r in valid if not np.isnan(r['cramers_v'])]

    return {
        'n_cells': len(cell_results),
        'n_insufficient': len(cell_results) - n_valid,
        'n_monostable': labels.count('monostable'),
        'n_multistable': labels.count('multistable'),
        'n_stochastic': labels.count('stochastic'),
        'p_multistable': labels.count('multistable') / n_valid,
        'n_eff_mean': float(np.mean(n_effs)) if n_effs else float('nan'),
        'n_eff_median': float(np.median(n_effs)) if n_effs else float('nan'),
        'cramers_v_mean': float(np.mean(vs)) if vs else float('nan'),
        'cramers_v_median': float(np.median(vs)) if vs else float('nan'),
    }
