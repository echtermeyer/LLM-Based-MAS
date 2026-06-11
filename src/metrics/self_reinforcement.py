from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


class SelfReinforcementMetric(ABC):
    @abstractmethod
    def score(self, repetitions: List[Dict]) -> Dict: ...


def _polyfit1(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    beta, alpha = np.polyfit(x.astype(float), y.astype(float), 1)
    return float(beta), float(alpha)


class SystemSelfReinforcement(SelfReinforcementMetric):
    """
    Fits c̄_t = α + β_sys * t where c̄_t = (1 / R_t*N) * Σ_{r: T_r≥t} Σ_i c_{i,t}^(r).
    Returns β_sys and the full trajectory.
    """

    def score(self, repetitions: List[Dict]) -> Dict:
        if not repetitions:
            return {
                'beta': np.nan, 'alpha': np.nan,
                'rounds': [], 'mean_conf': [], 'n_reps_per_round': [],
            }

        t_max = max(len(rep['trajectory']) - 1 for rep in repetitions)
        N = len(repetitions[0]['trajectory'][0]['phase_b'])
        rounds = list(range(t_max + 1))
        mean_conf, n_reps_per_round = [], []

        for t in rounds:
            total, r_t = 0.0, 0
            for rep in repetitions:
                if len(rep['trajectory']) - 1 >= t:
                    r_t += 1
                    for ag in rep['trajectory'][t]['phase_b']:
                        total += ag.get('confidence') or 0.0
            n_reps_per_round.append(r_t)
            mean_conf.append(total / (r_t * N) if r_t > 0 else np.nan)

        valid = [(t, c) for t, c in zip(rounds, mean_conf) if not np.isnan(c)]
        if len(valid) < 2:
            return {
                'beta': np.nan, 'alpha': np.nan,
                'rounds': rounds, 'mean_conf': mean_conf,
                'n_reps_per_round': n_reps_per_round,
            }

        t_arr = np.array([v[0] for v in valid], dtype=float)
        c_arr = np.array([v[1] for v in valid], dtype=float)
        beta, alpha = _polyfit1(t_arr, c_arr)
        return {
            'beta': beta,
            'alpha': alpha,
            'rounds': rounds,
            'mean_conf': mean_conf,
            'n_reps_per_round': n_reps_per_round,
        }


class AgentSelfReinforcement(SelfReinforcementMetric):
    """
    Extracts maximal vote-stable runs of length ≥ 3 per (rep, agent) and fits
    c_{i,t}^(r) = α + β*(t-s) within each run.

    Each run is annotated with:
      terminal  = 1 iff e == T_r
      converged = 1 iff the agent held the terminal plurality vote throughout
    """

    def _stable_runs(self, votes: List[str], T_r: int) -> List[Tuple[int, int]]:
        runs, s = [], 0
        for t in range(1, T_r + 1):
            if votes[t] != votes[s]:
                runs.append((s, t - 1))
                s = t
        runs.append((s, T_r))
        return [(s, e) for s, e in runs if e - s + 1 >= 3]

    def _stratified(self, slopes: List[float]) -> Tuple[float, float]:
        if not slopes:
            return np.nan, np.nan
        arr = np.array(slopes, dtype=float)
        return float(np.mean(arr > 0)), float(np.mean(arr))

    def score(self, repetitions: List[Dict]) -> Dict:
        all_runs = []

        for rep_idx, rep in enumerate(repetitions):
            traj = rep['trajectory']
            T_r = len(traj) - 1
            N = len(traj[0]['phase_b'])
            terminal_votes = [traj[T_r]['phase_b'][i]['vote'] for i in range(N)]
            consensus = Counter(terminal_votes).most_common(1)[0][0]

            for agent_idx in range(N):
                votes = [traj[t]['phase_b'][agent_idx]['vote'] for t in range(T_r + 1)]
                confs = [
                    traj[t]['phase_b'][agent_idx].get('confidence') or 0.0
                    for t in range(T_r + 1)
                ]

                for s, e in self._stable_runs(votes, T_r):
                    t_arr = np.arange(e - s + 1, dtype=float)
                    c_arr = np.array(confs[s:e + 1], dtype=float)
                    beta, alpha = _polyfit1(t_arr, c_arr)
                    all_runs.append({
                        'rep_idx': rep_idx,
                        'agent_idx': agent_idx,
                        'start': s,
                        'end': e,
                        'run_length': e - s + 1,
                        'slope': beta,
                        'intercept': alpha,
                        'terminal': e == T_r,
                        'converged': votes[s] == consensus,
                        'vote': votes[s],
                    })

        slopes = [r['slope'] for r in all_runs]
        p_sr = float(np.mean(np.array(slopes) > 0)) if slopes else np.nan
        mean_beta = float(np.mean(slopes)) if slopes else np.nan

        terminal_s = [r['slope'] for r in all_runs if r['terminal']]
        nonterminal_s = [r['slope'] for r in all_runs if not r['terminal']]
        converged_s = [r['slope'] for r in all_runs if r['converged']]
        nonconverged_s = [r['slope'] for r in all_runs if not r['converged']]

        p_sr_t, mb_t = self._stratified(terminal_s)
        p_sr_nt, mb_nt = self._stratified(nonterminal_s)
        p_sr_c, mb_c = self._stratified(converged_s)
        p_sr_nc, mb_nc = self._stratified(nonconverged_s)

        return {
            'runs': all_runs,
            'n_runs': len(all_runs),
            'p_sr': p_sr,
            'mean_beta': mean_beta,
            'p_sr_terminal': p_sr_t,
            'mean_beta_terminal': mb_t,
            'p_sr_nonterminal': p_sr_nt,
            'mean_beta_nonterminal': mb_nt,
            'p_sr_converged': p_sr_c,
            'mean_beta_converged': mb_c,
            'p_sr_nonconverged': p_sr_nc,
            'mean_beta_nonconverged': mb_nc,
        }
