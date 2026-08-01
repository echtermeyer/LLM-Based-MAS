from typing import Dict, List, Optional

import numpy as np

_DEGENERATE_TOL = 1e-9


def _neighbours(topology: List[List[int]], j: int) -> List[int]:
    return [k for k, v in enumerate(topology[j]) if v]


def _influence_graph(rep: Dict, rng: Optional[np.random.Generator] = None) -> np.ndarray:
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
            if v_j_next == phase[j]["vote"]:
                continue
            c_j = phase[j].get("confidence") or 0.0
            nbrs = _neighbours(topology, j)
            eligible = [k for k in nbrs if phase[k]["vote"] == v_j_next]
            m_j = len(eligible)
            if m_j == 0:
                continue
            credit = c_j / m_j
            # real graph credits the eligible holders; the topology-keyed null
            # keeps j, t, weight and source count m_j but randomises the source
            # identity uniformly over the neighbourhood N(j) (Theiler Algorithm 0)
            sources = eligible if rng is None else rng.choice(nbrs, size=m_j, replace=False)
            for i in sources:
                W[i, j] += credit

    return W


def _herfindahl(p: np.ndarray) -> float:
    N = len(p)
    if N <= 1:
        return float("nan")
    hhi = float(np.dot(p, p))
    return (hhi - 1.0 / N) / (1.0 - 1.0 / N)


def _dominance_from_graph(W: np.ndarray) -> Optional[tuple]:
    s = W.sum(axis=1)
    total = s.sum()
    if total == 0:
        return None
    p = s / total
    return s, p, _herfindahl(p)


def _hub_descriptors(rep: Dict, hub: int) -> tuple:
    traj = rep["trajectory"]
    final_votes = [ag["vote"] for ag in traj[-1]["phase_b"]]
    consensus = max(set(final_votes), key=final_votes.count)
    hub_final = traj[-1]["phase_b"][hub]["vote"]

    # confidence-weighted capitulation of the hub, as in Pers (L_A / L_A_max)
    L = L_max = 0.0
    for t in range(len(traj) - 1):
        cur = traj[t]["phase_b"][hub]
        c = cur.get("confidence") or 0.0
        L_max += c
        if traj[t + 1]["phase_b"][hub]["vote"] != cur["vote"]:
            L += c
    capitulation = (L / L_max) if L_max > 0 else 0.0
    return bool(hub_final == consensus), capitulation


def score_dominance(rep: Dict) -> Optional[Dict]:
    out = _dominance_from_graph(_influence_graph(rep))
    if out is None:
        return None
    s, p, D = out
    hub = int(np.argmax(s))
    hub_is_consensus, hub_capitulation = _hub_descriptors(rep, hub)
    return {
        "D": D,
        "s": s.tolist(),
        "p": p.tolist(),
        "hub": hub,
        "hub_is_consensus": hub_is_consensus,
        "hub_capitulation": hub_capitulation,
    }


def _surrogate_D(rep: Dict, rng: np.random.Generator) -> float:
    out = _dominance_from_graph(_influence_graph(rep, rng))
    return float("nan") if out is None else out[2]


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
    surrogates = [d for d in (_surrogate_D(rep, rng) for _ in range(B)) if not np.isnan(d)]

    if len(surrogates) == 0:
        return {**result, "mu_null": float("nan"), "sigma_null": float("nan"),
                "z": float("nan"), "p_value": float("nan"),
                "testable": False, "flagged": False}

    arr = np.array(surrogates)
    mu = float(arr.mean())
    sigma = float(arr.std())
    # a degenerate null (point mass at D) carries no information: the topology
    # forces this exact concentration, so the repetition is not flaggable
    testable = sigma > _DEGENERATE_TOL
    z = float((D - mu) / sigma) if testable else float("nan")
    p_value = (1 + int((arr >= D).sum())) / (1 + len(arr))
    flagged = testable and p_value < 0.05

    return {
        **result,
        "mu_null": mu,
        "sigma_null": sigma,
        "z": z,
        "p_value": p_value,
        "testable": testable,
        "flagged": flagged,
    }


def score_dominance_all(
    repetitions: List[Dict],
    B: int = 1000,
    seed: int = 0,
) -> List[Optional[Dict]]:
    return [analyse_dominance(rep, B=B, seed=seed) for rep in repetitions]


def summarise_dominance(rep_results: List[Optional[Dict]]) -> Dict:
    n_total = len(rep_results)
    valid = [r for r in rep_results if r is not None]
    n_valid = len(valid)
    n_excluded = n_total - n_valid
    testable = [r for r in valid if r["testable"]]
    n_testable = len(testable)
    n_degenerate = n_valid - n_testable

    if n_valid == 0:
        return {
            "n_reps": n_total,
            "n_valid": 0,
            "n_excluded": n_excluded,
            "n_testable": 0,
            "n_degenerate": 0,
            "D_mean": float("nan"),
            "D_median": float("nan"),
            "z_mean": float("nan"),
            "flagged_fraction": float("nan"),
        }

    Ds = np.array([r["D"] for r in valid])
    zs = np.array([r["z"] for r in testable])
    flagged = [r["flagged"] for r in testable]

    return {
        "n_reps": n_total,
        "n_valid": n_valid,
        "n_excluded": n_excluded,
        "n_testable": n_testable,
        "n_degenerate": n_degenerate,
        "D_mean": float(Ds.mean()),
        "D_median": float(np.median(Ds)),
        "z_mean": float(zs.mean()) if len(zs) else float("nan"),
        "flagged_fraction": float(np.mean(flagged)) if flagged else float("nan"),
    }
