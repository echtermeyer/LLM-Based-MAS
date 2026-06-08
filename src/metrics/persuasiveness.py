from abc import ABC, abstractmethod
from typing import Dict, List


class PersuasivenessMetric(ABC):
    @abstractmethod
    def score(self, rep: Dict) -> List[float]: ...

    def score_all(self, repetitions: List[Dict]) -> List[List[float]]:
        return [self.score(rep) for rep in repetitions]


class Persuasiveness(PersuasivenessMetric):
    """
    Pers(A) = (W_A - L_A) / (W_A_max + L_A)

    W_A     = sum_t (1/m_A^t) * sum_{B!=A, v_B^t != v_A^t} 1[v_B^{t+1}=v_A^t] * c_B^t
    W_A_max = sum_t (1/m_A^t) * sum_{B!=A, v_B^t != v_A^t} c_B^t
    L_A     = sum_t 1[v_A^{t+1} != v_A^t] * c_A^t
    """

    def score(self, rep: Dict) -> List[float]:
        traj = rep["trajectory"]
        T = len(traj)
        N = len(traj[0]["phase_b"])
        results = []
        for a in range(N):
            W_a = W_a_max = L_a = 0.0
            for t in range(T - 1):
                phase = traj[t]["phase_b"]
                next_phase = traj[t + 1]["phase_b"]
                v_a = phase[a]["vote"]
                c_a = phase[a].get("confidence") or 0.0
                m_a = sum(1 for ag in phase if ag["vote"] == v_a)
                credit = 1.0 / m_a if m_a > 0 else 0.0
                for b in range(N):
                    if b == a:
                        continue
                    v_b = phase[b]["vote"]
                    c_b = phase[b].get("confidence") or 0.0
                    if v_b != v_a:
                        W_a_max += credit * c_b
                        if next_phase[b]["vote"] == v_a:
                            W_a += credit * c_b
                if next_phase[a]["vote"] != v_a:
                    L_a += c_a
            denom = W_a_max + L_a
            results.append((W_a - L_a) / denom if denom > 0 else 0.0)
        return results


class PersuasivenessPostFlip(PersuasivenessMetric):
    """
    Pers+(A) = (W_A+ - L_A+) / (W_A+_max + L_A+)

    W_A+     = sum_t (1/m_A^t) * sum_{B!=A, v_B^t != v_A^t} 1[v_B^{t+1}=v_A^t] * (c_B^t + c_B^{t+1})
    W_A+_max = sum_t (1/m_A^t) * sum_{B!=A, v_B^t != v_A^t} (c_B^t + C)
    L_A+     = sum_t 1[v_A^{t+1} != v_A^t] * (c_A^t + c_A^{t+1})
    C        = confidence ceiling (max observed across all agents and rounds)
    """

    def __init__(self, ceiling: float = 10.0):
        self.ceiling = ceiling

    def score(self, rep: Dict) -> List[float]:
        traj = rep["trajectory"]
        T = len(traj)
        N = len(traj[0]["phase_b"])
        C = self.ceiling
        results = []
        for a in range(N):
            W_a = W_a_max = L_a = 0.0
            for t in range(T - 1):
                phase = traj[t]["phase_b"]
                next_phase = traj[t + 1]["phase_b"]
                v_a = phase[a]["vote"]
                c_a = phase[a].get("confidence") or 0.0
                c_a_next = next_phase[a].get("confidence") or 0.0
                m_a = sum(1 for ag in phase if ag["vote"] == v_a)
                credit = 1.0 / m_a if m_a > 0 else 0.0
                for b in range(N):
                    if b == a:
                        continue
                    v_b = phase[b]["vote"]
                    c_b = phase[b].get("confidence") or 0.0
                    c_b_next = next_phase[b].get("confidence") or 0.0
                    if v_b != v_a:
                        W_a_max += credit * (c_b + C)
                        if next_phase[b]["vote"] == v_a:
                            W_a += credit * (c_b + c_b_next)
                if next_phase[a]["vote"] != v_a:
                    L_a += c_a + c_a_next
            denom = W_a_max + L_a
            results.append((W_a - L_a) / denom if denom > 0 else 0.0)
        return results
