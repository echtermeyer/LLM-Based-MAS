from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedder import Embedder


def embed_shared(
    question: str,
    gt_text: Optional[str],
    embedder: Embedder,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    texts = [question.strip()]
    if gt_text is not None:
        texts.append(gt_text.strip())
    vecs = embedder.embed(texts)
    return vecs[0], (vecs[1] if gt_text is not None else None)


def embed_repetition(rep: Dict, embedder: Embedder) -> Dict:
    texts: List[str] = []
    slots: List[Tuple] = []

    for entry in rep["trajectory"]:
        r = entry["round"]
        for e in entry.get("phase_a", []):
            texts.append(e["defense"].strip())
            slots.append((r, e["id"], "draft"))
        for e in entry["phase_b"]:
            texts.append(e["message"].strip())
            slots.append((r, e["id"], "pub"))
            texts.append(e["reasoning"].strip())
            slots.append((r, e["id"], "priv"))

    vecs = embedder.embed(texts)
    bucket: Dict[int, Dict[int, Dict]] = {}
    for vec, (r, i, field) in zip(vecs, slots):
        bucket.setdefault(r, {}).setdefault(i, {})[field] = vec

    per_round = []
    for entry in rep["trajectory"]:
        r = entry["round"]
        agents = []
        for i in sorted(bucket.get(r, {})):
            agent: Dict = {
                "id": i,
                "pub": bucket[r][i]["pub"].tolist(),
                "priv": bucket[r][i]["priv"].tolist(),
            }
            if "draft" in bucket[r][i]:
                agent["draft"] = bucket[r][i]["draft"].tolist()
            agents.append(agent)
        per_round.append({"round": r, "agents": agents})

    return {"per_round": per_round}
