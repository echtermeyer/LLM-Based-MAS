import numpy as np
from typing import Dict, List, Tuple


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


def _recurrence_matrix(seq: list) -> np.ndarray:
    L = len(seq)
    R = np.zeros((L, L), dtype=bool)
    for i in range(L):
        for j in range(i + 1, L):
            if seq[i] == seq[j]:
                R[i, j] = R[j, i] = True
    return R


def _diagonal_run_lengths(diag: np.ndarray) -> List[int]:
    runs, run = [], 0
    for v in diag:
        if v:
            run += 1
        else:
            if run:
                runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return runs


def _det(R: np.ndarray, l_min: int = 2) -> float:
    L = R.shape[0]
    total_rp = on_lines = 0
    for tau in range(1, L):
        diag = np.diagonal(R, offset=tau)
        total_rp += int(diag.sum())
        for rlen in _diagonal_run_lengths(diag):
            if rlen >= l_min:
                on_lines += rlen
    if total_rp == 0:
        return 0.0
    return on_lines / total_rp


def _diagonalwise_rr(R: np.ndarray) -> np.ndarray:
    L = R.shape[0]
    rr = np.zeros(L - 1)
    for tau in range(1, L):
        diag = np.diagonal(R, offset=tau)
        rr[tau - 1] = float(diag.mean())
    return rr


def _dominant_period(rr: np.ndarray) -> int:
    return int(np.argmax(rr)) + 1


def _shuffle_p(seq: list, det_obs: float, B: int, rng: np.random.Generator) -> float:
    n = len(seq)
    count = 0
    for _ in range(B):
        idx = rng.permutation(n)
        shuffled = [seq[i] for i in idx]
        det_b = _det(_recurrence_matrix(shuffled))
        if det_b >= det_obs:
            count += 1
    return (1 + count) / (1 + B)


def _agent_vote_seq(traj: List[Dict], agent_idx: int) -> List[str]:
    return [traj[t]['phase_b'][agent_idx]['vote'] for t in range(len(traj))]


def _composition_seq(traj: List[Dict], options: tuple) -> List[Tuple[int, ...]]:
    result = []
    for t in range(len(traj)):
        votes = [ag['vote'] for ag in traj[t]['phase_b']]
        result.append(tuple(votes.count(o) for o in options))
    return result


def detect_agent_limit_cycles(
    repetitions: List[Dict],
    B: int = 1000,
    u: int = 3,
    seed: int = 0,
) -> List[Dict]:
    """
    Agent-level RQA limit-cycle detection.

    For each (rep, agent): extract Phase B vote sequence, classify end-state,
    run shuffle-surrogate DET test on candidates.

    Returns one record per (rep_idx, agent_idx):
      fixed_point : trailing constant run >= u (excluded from test)
      lc          : True if flagged (p < 0.05)
      det         : observed DET (None if not tested)
      p_value     : surrogate p-value (None if not tested)
      period      : dominant period P_hat (None if not flagged)
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rep_idx, rep in enumerate(repetitions):
        traj = rep['trajectory']
        N = len(traj[0]['phase_b'])
        for agent_idx in range(N):
            seq = _agent_vote_seq(traj, agent_idx)
            L = len(seq)
            k = _trailing_k(seq)
            is_fp = k >= u
            row = {
                'rep_idx': rep_idx,
                'agent_idx': agent_idx,
                'L': L,
                'fixed_point': is_fp,
                'lc': False,
                'det': None,
                'p_value': None,
                'period': None,
            }
            if is_fp or L < 4:
                rows.append(row)
                continue
            R = _recurrence_matrix(seq)
            det_obs = _det(R)
            p = _shuffle_p(seq, det_obs, B, rng)
            rr = _diagonalwise_rr(R)
            P_hat = _dominant_period(rr)
            flagged = p < 0.05 and 2 <= P_hat <= L // 2
            row.update({
                'lc': flagged,
                'det': det_obs,
                'p_value': p,
                'period': P_hat if flagged else None,
            })
            rows.append(row)
    return rows


def detect_system_limit_cycles(
    repetitions: List[Dict],
    B: int = 1000,
    u: int = 3,
    seed: int = 0,
) -> List[Dict]:
    """
    System-level RQA limit-cycle detection on the vote-composition macrostate.

    For each repetition: build (n_A, n_B, ...) sequence over the task's real M options,
    classify end-state, run shuffle-surrogate DET test on candidates.

    Returns one record per rep_idx:
      fixed_point : trailing constant composition run >= u
      lc          : True if flagged (p < 0.05)
      det, p_value, period
    """
    options = tuple(repetitions[0]['options'].keys())
    rng = np.random.default_rng(seed)
    rows = []
    for rep_idx, rep in enumerate(repetitions):
        traj = rep['trajectory']
        seq = _composition_seq(traj, options)
        L = len(seq)
        k = _trailing_k(seq)
        is_fp = k >= u
        row = {
            'rep_idx': rep_idx,
            'L': L,
            'fixed_point': is_fp,
            'lc': False,
            'det': None,
            'p_value': None,
            'period': None,
        }
        if is_fp or L < 4:
            rows.append(row)
            continue
        R = _recurrence_matrix(seq)
        det_obs = _det(R)
        p = _shuffle_p(seq, det_obs, B, rng)
        rr = _diagonalwise_rr(R)
        P_hat = _dominant_period(rr)
        flagged = p < 0.05 and 2 <= P_hat <= L // 2
        row.update({
            'lc': flagged,
            'det': det_obs,
            'p_value': p,
            'period': P_hat if flagged else None,
        })
        rows.append(row)
    return rows


def summarise_lc(rows: List[Dict]) -> Dict:
    """
    Aggregate a list of limit-cycle records into prevalence scalars.
    Works for both agent-level and system-level records.
    """
    candidates = [r for r in rows if not r['fixed_point'] and r['L'] >= 4]
    flagged = [r for r in candidates if r['lc']]
    n_cand = len(candidates)
    return {
        'n_total': len(rows),
        'n_fixed_point': sum(r['fixed_point'] for r in rows),
        'n_candidates': n_cand,
        'n_flagged': len(flagged),
        'p_lc': len(flagged) / n_cand if n_cand > 0 else float('nan'),
    }
