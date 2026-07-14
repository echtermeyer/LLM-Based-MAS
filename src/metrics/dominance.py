from typing import Dict, List, Optional, Tuple

import numpy as np


def _neighbours(topology: List[List[int]], j: int) -> List[int]:
    return [k for k, v in enumerate(topology[j]) if v]


def _influence_graph(rep: Dict) -> np.ndarray:
    traj = rep["trajectory"]
    T = len(traj)
    N = len(traj[0]["phase_b"])
    topology = rep["topology"]
    W = np.zeros((N, N), dtype=float)

    for t in range(T - 1):
        phase = traj[t]["phase_b"]
        next_phase = traj[t + 1]["phase_b"]
        for j in range(N):
            v_j_next = next_phase[j]["vote"]
            v_j_cur = phase[j]["vote"]
            if v_j_next == v_j_cur:
                continue
            c_j = phase[j].get("confidence") or 0.0
            nbrs = _neighbours(topology, j)
            eligible = [k for k in nbrs if phase[k]["vote"] == v_j_next]
            m_j = len(eligible)
            if m_j == 0:
                continue
            credit = c_j / m_j
            for i in eligible:
                W[i, j] += credit

    return W


def _herfindahl(p: np.ndarray) -> float:
    N = len(p)
    if N <= 1:
        return float("nan")
    hhi = float(np.dot(p, p))
    return (hhi - 1.0 / N) / (1.0 - 1.0 / N)


def score_dominance(rep: Dict) -> Optional[Dict]:
    W = _influence_graph(rep)
    s = W.sum(axis=1)
    total = s.sum()
    if total == 0:
        return None
    p = s / total
    D = _herfindahl(p)
    hub = int(np.argmax(s))
    return {"D": D, "s": s.tolist(), "p": p.tolist(), "hub": hub}


def _surrogate_D(
    rep: Dict,
    rng: np.random.Generator,
) -> float:
    traj = rep["trajectory"]
    T = len(traj)
    N = len(traj[0]["phase_b"])
    topology = rep["topology"]
    W = np.zeros((N, N), dtype=float)

    for t in range(T - 1):
        phase = traj[t]["phase_b"]
        next_phase = traj[t + 1]["phase_b"]
        for j in range(N):
            v_j_next = next_phase[j]["vote"]
            v_j_cur = phase[j]["vote"]
            if v_j_next == v_j_cur:
                continue
            c_j = phase[j].get("confidence") or 0.0
            nbrs = _neighbours(topology, j)
            eligible = [k for k in nbrs if phase[k]["vote"] == v_j_next]
            m_j = len(eligible)
            if m_j == 0:
                continue
            credit = c_j / m_j
            # shuffle source identity among eligible neighbours
            sources = rng.choice(eligible, size=m_j, replace=False)
            for i in sources:
                W[i, j] += credit

    s = W.sum(axis=1)
    total = s.sum()
    if total == 0:
        return float("nan")
    p = s / total
    return _herfindahl(p)


def analyse_dominance(
    rep: Dict,
    B: int = 1000,
    seed: int = 0,
) -> Optional[Dict]:
    result = score_dominance(rep)
    if result is None:
        return None

    D = result["D"]
    rng = np.random.default_rng(seed)
    surrogates = []
    for _ in range(B):
        d_b = _surrogate_D(rep, rng)
        if not np.isnan(d_b):
            surrogates.append(d_b)

    if len(surrogates) == 0:
        return {**result, "mu_null": float("nan"), "sigma_null": float("nan"),
                "z": float("nan"), "p_value": float("nan"), "flagged": False}

    arr = np.array(surrogates)
    mu = float(arr.mean())
    sigma = float(arr.std())
    z = float((D - mu) / sigma) if sigma > 0 else float("nan")
    p_value = (1 + int((arr >= D).sum())) / (1 + len(arr))
    flagged = p_value < 0.05

    return {
        **result,
        "mu_null": mu,
        "sigma_null": sigma,
        "z": z,
        "p_value": p_value,
        "flagged": flagged,
    }


def score_dominance_all(
    repetitions: List[Dict],
    B: int = 1000,
    seed: int = 0,
) -> List[Optional[Dict]]:
    return [analyse_dominance(rep, B=B, seed=seed) for rep in repetitions]


def summarise_dominance(rep_results: List[Optional[Dict]]) -> Dict:
    valid = [r for r in rep_results if r is not None]
    n_total = len(rep_results)
    n_valid = len(valid)
    n_excluded = n_total - n_valid

    if n_valid == 0:
        return {
            "n_reps": n_total,
            "n_valid": 0,
            "n_excluded": n_excluded,
            "D_mean": float("nan"),
            "D_median": float("nan"),
            "z_mean": float("nan"),
            "flagged_fraction": float("nan"),
        }

    Ds = np.array([r["D"] for r in valid])
    zs = np.array([r["z"] for r in valid if not np.isnan(r["z"])])
    flagged = [r["flagged"] for r in valid]

    return {
        "n_reps": n_total,
        "n_valid": n_valid,
        "n_excluded": n_excluded,
        "D_mean": float(Ds.mean()),
        "D_median": float(np.median(Ds)),
        "z_mean": float(zs.mean()) if len(zs) else float("nan"),
        "flagged_fraction": float(np.mean(flagged)),
    }
