from typing import Dict, List, Tuple

import numpy as np


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float)
    y = y.astype(float)
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()
    if denom == 0:
        return np.nan
    return float((x_c * y).sum() / denom)


def _stable_runs(votes: List[str]) -> List[Tuple[int, int]]:
    T = len(votes) - 1
    runs, s = [], 0
    for t in range(1, T + 1):
        if votes[t] != votes[s]:
            runs.append((s, t - 1))
            s = t
    runs.append((s, T))
    return [(s, e) for s, e in runs if e - s + 1 >= 3]


def _run_slope(confs: List[float], s: int, e: int) -> float:
    c_arr = np.array(confs[s : e + 1], dtype=float)
    t_arr = np.arange(len(c_arr), dtype=float)
    return _ols_slope(t_arr, c_arr)


def extract_runs(repetitions: List[Dict]) -> List[Dict]:
    """
    Returns one record per vote-stable run (length >= 3) across all (rep, agent) pairs.

    Each record:
      rep_idx     : index into repetitions list
      agent_idx   : agent index within N
      start, end  : inclusive round indices of the run
      run_length  : end - start + 1
      vote        : the vote held during the run
      slope       : OLS slope beta_run (confidence per round, relative position within run)
      terminal    : True iff run ends at T_r
      correct_vote: True iff vote == ground_truth of this repetition
    """
    rows = []
    for rep_idx, rep in enumerate(repetitions):
        traj = rep["trajectory"]
        T_r = len(traj) - 1
        N = len(traj[0]["phase_b"])
        gt = rep.get("ground_truth", None)

        for agent_idx in range(N):
            votes = [traj[t]["phase_b"][agent_idx]["vote"] for t in range(T_r + 1)]
            confs = [
                float(traj[t]["phase_b"][agent_idx].get("confidence") or 0.0)
                for t in range(T_r + 1)
            ]

            for s, e in _stable_runs(votes):
                slope = _run_slope(confs, s, e)
                if slope != slope:
                    continue
                run_confs = confs[s : e + 1]
                rows.append({
                    "rep_idx": rep_idx,
                    "agent_idx": agent_idx,
                    "start": s,
                    "end": e,
                    "run_length": e - s + 1,
                    "vote": votes[s],
                    "slope": slope,
                    "start_conf": run_confs[0],
                    "end_conf": run_confs[-1],
                    "ceiling_hit": run_confs[-1] >= 10.0,
                    "terminal": e == T_r,
                    "correct_vote": votes[s] == gt if gt is not None else None,
                })
    return rows


def count_all_runs(repetitions: List[Dict]) -> Dict:
    """
    Counts every stable run before any filter, broken down by what gets dropped.
    Used for descriptive reporting only.
    """
    n_total = n_short = n_kept = 0
    for rep in repetitions:
        traj = rep["trajectory"]
        T_r  = len(traj) - 1
        N    = len(traj[0]["phase_b"])
        for agent_idx in range(N):
            votes = [traj[t]["phase_b"][agent_idx]["vote"] for t in range(T_r + 1)]
            for s, e in _stable_runs_all(votes):
                n_total += 1
                if e - s + 1 < 3:
                    n_short += 1
                else:
                    n_kept += 1
    return {"n_total": n_total, "n_short": n_short, "n_kept": n_kept}


def _stable_runs_all(votes: List[str]) -> List[Tuple[int, int]]:
    T = len(votes) - 1
    runs, s = [], 0
    for t in range(1, T + 1):
        if votes[t] != votes[s]:
            runs.append((s, t - 1))
            s = t
    runs.append((s, T))
    return runs


def summarise_runs(runs: List[Dict]) -> Dict:
    """
    Aggregate a list of run records (from extract_runs) into scalar summaries.

    Returns:
      n_runs     : total runs
      p_sr       : fraction with slope > 0  (prevalence, baseline = 0.5)
      mean_slope : mean OLS slope (magnitude and sign)
      p_sr_terminal / mean_slope_terminal
      p_sr_nonterminal / mean_slope_nonterminal
      p_sr_correct / mean_slope_correct     (correct-vote runs)
      p_sr_incorrect / mean_slope_incorrect
    """
    def _stats(slopes):
        arr = np.array(slopes, dtype=float)
        if len(arr) == 0:
            return np.nan, np.nan
        return float((arr > 0).mean()), float(arr.mean())

    if not runs:
        return {k: np.nan for k in (
            "n_runs p_sr mean_slope "
            "p_sr_terminal mean_slope_terminal "
            "p_sr_nonterminal mean_slope_nonterminal "
            "p_sr_correct mean_slope_correct "
            "p_sr_incorrect mean_slope_incorrect"
        ).split()}

    slopes_all = [r["slope"] for r in runs]
    slopes_t   = [r["slope"] for r in runs if r["terminal"]]
    slopes_nt  = [r["slope"] for r in runs if not r["terminal"]]
    slopes_c   = [r["slope"] for r in runs if r.get("correct_vote")]
    slopes_ic  = [r["slope"] for r in runs if r.get("correct_vote") is False]

    p_sr,    ms    = _stats(slopes_all)
    p_sr_t,  ms_t  = _stats(slopes_t)
    p_sr_nt, ms_nt = _stats(slopes_nt)
    p_sr_c,  ms_c  = _stats(slopes_c)
    p_sr_ic, ms_ic = _stats(slopes_ic)

    return {
        "n_runs": len(runs),
        "p_sr": p_sr,
        "mean_slope": ms,
        "p_sr_terminal": p_sr_t,
        "mean_slope_terminal": ms_t,
        "p_sr_nonterminal": p_sr_nt,
        "mean_slope_nonterminal": ms_nt,
        "p_sr_correct": p_sr_c,
        "mean_slope_correct": ms_c,
        "p_sr_incorrect": p_sr_ic,
        "mean_slope_incorrect": ms_ic,
    }
