from typing import Dict, List, Optional

import numpy as np


def _get_vec(emb: Dict, r: int, i: int, field: str) -> Optional[np.ndarray]:
    for rb in emb["per_round"]:
        if rb["round"] == r:
            for a in rb["agents"]:
                if a["id"] == i:
                    v = a.get(field)
                    return np.array(v) if v is not None else None
    return None


def _blank(n: int, t: int) -> List:
    return [[None] * t for _ in range(n)]


def _blank_pair(n: int, t: int) -> List:
    return [[[None] * t for _ in range(n)] for _ in range(n)]


def _per_agent(
    rep: Dict,
    emb: Dict,
    gt_vec: Optional[np.ndarray],
    round_nums: List[int],
    ri_of: Dict[int, int],
    beliefs_by_round: Dict[int, Dict[int, str]],
) -> Dict:
    N, gt = rep["N"], rep["ground_truth"]
    R = len(round_nums)

    correct = _blank(N, R)
    flip = _blank(N, R)
    delta_pub = _blank(N, R)
    delta_priv = _blank(N, R)
    diss = _blank(N, R)
    correct_align_pub = _blank(N, R)
    correct_align_priv = _blank(N, R)
    toward_truth_pub = _blank(N, R)
    toward_truth_priv = _blank(N, R)

    for r in round_nums:
        ri = ri_of[r]
        prev_r = round_nums[ri - 1] if ri > 0 else None

        for i in range(N):
            pub_i = _get_vec(emb, r, i, "pub")
            priv_i = _get_vec(emb, r, i, "priv")

            correct[i][ri] = int(beliefs_by_round[r][i] == gt)

            if pub_i is not None and priv_i is not None:
                diss[i][ri] = float(1.0 - pub_i @ priv_i)

            if gt_vec is not None:
                if pub_i is not None:
                    correct_align_pub[i][ri] = float(pub_i @ gt_vec)
                if priv_i is not None:
                    correct_align_priv[i][ri] = float(priv_i @ gt_vec)

            if prev_r is not None:
                pub_prev = _get_vec(emb, prev_r, i, "pub")
                priv_prev = _get_vec(emb, prev_r, i, "priv")

                flip[i][ri] = int(beliefs_by_round[r][i] != beliefs_by_round[prev_r][i])

                if pub_i is not None and pub_prev is not None:
                    delta_pub[i][ri] = float(1.0 - pub_i @ pub_prev)
                if priv_i is not None and priv_prev is not None:
                    delta_priv[i][ri] = float(1.0 - priv_i @ priv_prev)

                if gt_vec is not None:
                    if (
                        correct_align_pub[i][ri] is not None
                        and correct_align_pub[i][ri - 1] is not None
                    ):
                        toward_truth_pub[i][ri] = (
                            correct_align_pub[i][ri] - correct_align_pub[i][ri - 1]
                        )
                    if (
                        correct_align_priv[i][ri] is not None
                        and correct_align_priv[i][ri - 1] is not None
                    ):
                        toward_truth_priv[i][ri] = (
                            correct_align_priv[i][ri] - correct_align_priv[i][ri - 1]
                        )

    return {
        "correct": correct,
        "flip": flip,
        "delta_pub": delta_pub,
        "delta_priv": delta_priv,
        "diss": diss,
        "correct_align_pub": correct_align_pub,
        "correct_align_priv": correct_align_priv,
        "toward_truth_pub": toward_truth_pub,
        "toward_truth_priv": toward_truth_priv,
    }


def _per_pair(
    rep: Dict,
    emb: Dict,
    round_nums: List[int],
    ri_of: Dict[int, int],
    beliefs_by_round: Dict[int, Dict[int, str]],
) -> Dict:
    N = rep["N"]
    R = len(round_nums)

    agree = _blank_pair(N, R)
    sim_pub = _blank_pair(N, R)
    sim_priv = _blank_pair(N, R)
    toward = _blank_pair(N, R)

    for r in round_nums:
        ri = ri_of[r]
        prev_r = round_nums[ri - 1] if ri > 0 else None

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                pub_i = _get_vec(emb, r, i, "pub")
                pub_j = _get_vec(emb, r, j, "pub")
                priv_i = _get_vec(emb, r, i, "priv")
                priv_j = _get_vec(emb, r, j, "priv")

                agree[i][j][ri] = int(beliefs_by_round[r][i] == beliefs_by_round[r][j])

                if pub_i is not None and pub_j is not None:
                    sim_pub[i][j][ri] = float(pub_i @ pub_j)
                if priv_i is not None and priv_j is not None:
                    sim_priv[i][j][ri] = float(priv_i @ priv_j)

                if prev_r is not None:
                    pub_i_prev = _get_vec(emb, prev_r, i, "pub")
                    pub_j_prev = _get_vec(emb, prev_r, j, "pub")
                    if (
                        pub_i is not None
                        and pub_i_prev is not None
                        and pub_j_prev is not None
                    ):
                        toward[i][j][ri] = float(
                            pub_i @ pub_j_prev - pub_i_prev @ pub_j_prev
                        )

    return {"agree": agree, "sim_pub": sim_pub, "sim_priv": sim_priv, "toward": toward}


def _influence(
    toward: List,
    N: int,
    round_nums: List[int],
    ri_of: Dict[int, int],
) -> List:
    influence = _blank(N, len(round_nums))
    for r in round_nums:
        ri = ri_of[r]
        if ri == 0:
            continue
        for i in range(N):
            vals = [
                toward[ip][i][ri]
                for ip in range(N)
                if ip != i and toward[ip][i][ri] is not None
            ]
            if vals:
                influence[i][ri] = float(np.mean(vals))
    return influence


def compute_scalars(rep: Dict, emb: Dict, gt_vec: Optional[np.ndarray]) -> Dict:
    round_nums = [e["round"] for e in rep["trajectory"]]
    ri_of = {r: ri for ri, r in enumerate(round_nums)}
    beliefs_by_round = {
        e["round"]: {pb["id"]: pb["belief"] for pb in e["phase_b"]}
        for e in rep["trajectory"]
    }

    agent_scalars = _per_agent(rep, emb, gt_vec, round_nums, ri_of, beliefs_by_round)
    pair_scalars = _per_pair(rep, emb, round_nums, ri_of, beliefs_by_round)
    agent_scalars["influence"] = _influence(
        pair_scalars["toward"], rep["N"], round_nums, ri_of
    )

    return {"per_agent_per_round": agent_scalars, "per_pair_per_round": pair_scalars}
