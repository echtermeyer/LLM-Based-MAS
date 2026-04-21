import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"
ROOT = Path(__file__).parent
DEFAULT_OUT_DIR = ROOT / "results/mas_eval"
DATASET_PATH = ROOT / "dataset" / "gpqa_diamond.csv"

_df_cache: Optional[pd.DataFrame] = None


def _get_explanation(question_id: str) -> Optional[str]:
    """Return the expert Explanation text for the given question_id (iloc index)."""
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(DATASET_PATH)
    val = _df_cache.iloc[int(question_id)]["Explanation"]
    return str(val) if pd.notna(val) else None


class Embedder:
    def __init__(self) -> None:
        torch.manual_seed(0)
        self._model = SentenceTransformer(MODEL_NAME)
        self._model.eval()

    @property
    def revision(self) -> str:
        import sentence_transformers as _st

        return _st.__version__

    def embed(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            return self._model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )


@dataclass
class RoundView:
    round: int
    beliefs: Dict[int, str]
    pub_texts: Dict[int, str]
    priv_texts: Dict[int, str]
    draft_texts: Dict[int, str]


@dataclass
class RunView:
    question: str
    ground_truth: str
    N: int
    rounds: List[RoundView]

    @staticmethod
    def from_dict(run: Dict) -> "RunView":
        rounds = []
        for entry in run["trajectory"]:
            drafts = {e["id"]: e["draft_message"] for e in entry.get("phase_a", [])}
            beliefs, pubs, privs = {}, {}, {}
            for e in entry["phase_b"]:
                beliefs[e["id"]] = e["belief"]
                pubs[e["id"]] = e["public_message"]
                privs[e["id"]] = e["belief_reasoning"]
            rounds.append(RoundView(entry["round"], beliefs, pubs, privs, drafts))
        return RunView(run["question"], run["ground_truth"], run["N"], rounds)


@dataclass
class AgentEmbeddings:
    id: int
    pub: np.ndarray
    priv: np.ndarray
    draft: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        d: Dict = {"id": self.id}
        if self.draft is not None:
            d["draft"] = self.draft.tolist()
        d["pub"] = self.pub.tolist()
        d["priv"] = self.priv.tolist()
        return d


@dataclass
class EmbeddingsBlock:
    model: str
    model_revision: str
    dim: int
    question: np.ndarray
    per_round: List[Dict]
    gt_reasoning: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self._index: Dict[int, Dict[int, AgentEmbeddings]] = {}
        for r_block in self.per_round:
            self._index[r_block["round"]] = {a["id"]: a for a in r_block["agents"]}

    def agent(self, r: int, i: int) -> Dict:
        return self._index[r][i]

    def to_dict(self) -> Dict:
        d: Dict = {
            "model": self.model,
            "model_revision": self.model_revision,
            "dim": self.dim,
            "normalized": True,
            "question": self.question.tolist(),
            "per_round": self.per_round,
        }
        if self.gt_reasoning is not None:
            d["gt_reasoning"] = self.gt_reasoning.tolist()
        return d


def compute_embeddings(
    run_view: RunView, embedder: Embedder, gt_reasoning_text: Optional[str] = None
) -> EmbeddingsBlock:
    texts: List[str] = [run_view.question.strip()]
    slots: List[tuple] = [("q", None, None)]

    for rv in run_view.rounds:
        for i, txt in rv.draft_texts.items():
            texts.append(txt.strip())
            slots.append((rv.round, i, "draft"))
        for i in sorted(rv.pub_texts):
            texts.append(rv.pub_texts[i].strip())
            slots.append((rv.round, i, "pub"))
            texts.append(rv.priv_texts[i].strip())
            slots.append((rv.round, i, "priv"))

    if gt_reasoning_text is not None:
        texts.append(gt_reasoning_text.strip())

    vecs = embedder.embed(texts)
    question_vec = vecs[0]
    gt_reasoning_vec = vecs[-1] if gt_reasoning_text is not None else None
    bucket: Dict[int, Dict[int, Dict]] = {rv.round: {} for rv in run_view.rounds}

    for idx, (r, i, f) in enumerate(slots[1:], 1):
        bucket[r].setdefault(i, {})[f] = vecs[idx]

    per_round = [
        {
            "round": rv.round,
            "agents": [
                AgentEmbeddings(
                    id=i,
                    pub=bucket[rv.round][i]["pub"],
                    priv=bucket[rv.round][i]["priv"],
                    draft=bucket[rv.round][i].get("draft"),
                ).to_dict()
                for i in sorted(bucket[rv.round])
            ],
        }
        for rv in run_view.rounds
    ]

    return EmbeddingsBlock(
        model=MODEL_NAME,
        model_revision=embedder.revision,
        dim=question_vec.shape[0],
        question=question_vec,
        per_round=per_round,
        gt_reasoning=gt_reasoning_vec,
    )


def _vec(emb: EmbeddingsBlock, r: int, i: int, field: str) -> Optional[np.ndarray]:
    v = emb.agent(r, i).get(field)
    return np.array(v) if v is not None else None


def compute_scalars(run_view: RunView, emb: EmbeddingsBlock) -> Dict:
    N = run_view.N
    gt = run_view.ground_truth
    rounds = run_view.rounds
    R = len(rounds)
    round_idx = {rv.round: ri for ri, rv in enumerate(rounds)}

    correct = [[None] * R for _ in range(N)]
    flip = [[None] * R for _ in range(N)]
    delta_pub = [[None] * R for _ in range(N)]
    delta_priv = [[None] * R for _ in range(N)]
    diss = [[None] * R for _ in range(N)]
    shift = [[None] * R for _ in range(N)]
    on_topic = [[None] * R for _ in range(N)]
    influence = [[None] * R for _ in range(N)]
    correct_align_pub = [[None] * R for _ in range(N)]
    correct_align_priv = [[None] * R for _ in range(N)]

    agree = [[[None] * R for _ in range(N)] for _ in range(N)]
    sim_pub = [[[None] * R for _ in range(N)] for _ in range(N)]
    sim_priv = [[[None] * R for _ in range(N)] for _ in range(N)]
    toward = [[[None] * R for _ in range(N)] for _ in range(N)]

    q_vec = np.array(emb.question)
    gt_r_vec = np.array(emb.gt_reasoning) if emb.gt_reasoning is not None else None

    for rv in rounds:
        r, ri = rv.round, round_idx[rv.round]
        prev_rv = rounds[ri - 1] if ri > 0 else None

        for i in range(N):
            pub_i = _vec(emb, r, i, "pub")
            priv_i = _vec(emb, r, i, "priv")

            # c_i(r) = 1[a_i(r) == a*]
            correct[i][ri] = int(rv.beliefs[i] == gt)

            # diss_i(r) = 1 - cos(pub, priv)
            if pub_i is not None and priv_i is not None:
                diss[i][ri] = float(1.0 - pub_i @ priv_i)

            # on_topic_i(r) = cos(pub, q)
            if pub_i is not None:
                on_topic[i][ri] = float(pub_i @ q_vec)

            # correct_align_{pub,priv}_i(r) = cos({pub,priv}, gt_reasoning)
            if gt_r_vec is not None:
                if pub_i is not None:
                    correct_align_pub[i][ri] = float(pub_i @ gt_r_vec)
                if priv_i is not None:
                    correct_align_priv[i][ri] = float(priv_i @ gt_r_vec)

            if prev_rv is not None:
                pub_prev = _vec(emb, prev_rv.round, i, "pub")
                priv_prev = _vec(emb, prev_rv.round, i, "priv")
                draft_i = _vec(emb, r, i, "draft")

                # flip_i(r) = 1[a_i(r) != a_i(r-1)]
                flip[i][ri] = int(rv.beliefs[i] != prev_rv.beliefs[i])

                # delta_pub_i(r) = 1 - cos(pub(r), pub(r-1))
                if pub_i is not None and pub_prev is not None:
                    delta_pub[i][ri] = float(1.0 - pub_i @ pub_prev)

                # delta_priv_i(r) = 1 - cos(priv(r), priv(r-1))
                if priv_i is not None and priv_prev is not None:
                    delta_priv[i][ri] = float(1.0 - priv_i @ priv_prev)

                # shift_i(r) = 1 - cos(draft(r), pub(r))
                if draft_i is not None and pub_i is not None:
                    shift[i][ri] = float(1.0 - draft_i @ pub_i)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                pub_i = _vec(emb, r, i, "pub")
                pub_j = _vec(emb, r, j, "pub")
                priv_i = _vec(emb, r, i, "priv")
                priv_j = _vec(emb, r, j, "priv")

                # agree_ij(r) = 1[a_i(r) == a_j(r)]
                agree[i][j][ri] = int(rv.beliefs[i] == rv.beliefs[j])

                # sim_pub_ij(r) = cos(pub_i(r), pub_j(r))
                if pub_i is not None and pub_j is not None:
                    sim_pub[i][j][ri] = float(pub_i @ pub_j)

                # sim_priv_ij(r) = cos(priv_i(r), priv_j(r))
                if priv_i is not None and priv_j is not None:
                    sim_priv[i][j][ri] = float(priv_i @ priv_j)

                if prev_rv is not None:
                    pub_j_prev = _vec(emb, prev_rv.round, j, "pub")
                    pub_i_prev = _vec(emb, prev_rv.round, i, "pub")

                    # toward_ij(r) = cos(pub_i(r), pub_j(r-1)) - cos(pub_i(r-1), pub_j(r-1))
                    if (
                        pub_i is not None
                        and pub_j_prev is not None
                        and pub_i_prev is not None
                    ):
                        toward[i][j][ri] = float(
                            pub_i @ pub_j_prev - pub_i_prev @ pub_j_prev
                        )

    for rv in rounds:
        r, ri = rv.round, round_idx[rv.round]
        if ri == 0:
            continue
        for i in range(N):
            # influence_i(r) = mean_{i'!=i} toward_{i'->i}(r)
            vals = [
                toward[ip][i][ri]
                for ip in range(N)
                if ip != i and toward[ip][i][ri] is not None
            ]
            if vals:
                influence[i][ri] = float(np.mean(vals))

    mean_correct, majority_correct, majority_tie = [], [], []
    unanimous_correct, fraction_agreeing_pairs = [], []
    entropy, mean_diss, mean_sim_pub, mean_sim_priv = [], [], [], []
    mean_correct_align_pub, mean_correct_align_priv = [], []

    for rv in rounds:
        r, ri = rv.round, round_idx[rv.round]
        beliefs = [rv.beliefs[i] for i in range(N)]

        mean_correct.append(sum(b == gt for b in beliefs) / N)
        unanimous_correct.append(int(all(b == gt for b in beliefs)))

        counts: Dict[str, int] = {}
        for b in beliefs:
            counts[b] = counts.get(b, 0) + 1
        max_count = max(counts.values())
        winners = sorted(k for k, v in counts.items() if v == max_count)
        majority_tie.append(int(len(winners) > 1))
        majority_correct.append(int(winners[0] == gt))

        n_pairs = N * (N - 1) // 2
        n_agree = sum(
            1
            for ii in range(N)
            for jj in range(ii + 1, N)
            if beliefs[ii] == beliefs[jj]
        )
        fraction_agreeing_pairs.append(n_agree / n_pairs if n_pairs > 0 else 1.0)

        # H(r) = -sum_a p_a(r) * log(p_a(r))
        ps = np.array([sum(1 for b in beliefs if b == opt) / N for opt in "ABCD"])
        entropy.append(float(-np.sum(ps[ps > 0] * np.log(ps[ps > 0]))))

        diss_vals = [diss[i][ri] for i in range(N) if diss[i][ri] is not None]
        mean_diss.append(float(np.mean(diss_vals)) if diss_vals else None)

        sp = [
            sim_pub[i][j][ri]
            for i in range(N)
            for j in range(i + 1, N)
            if sim_pub[i][j][ri] is not None
        ]
        mean_sim_pub.append(float(np.mean(sp)) if sp else None)

        sv = [
            sim_priv[i][j][ri]
            for i in range(N)
            for j in range(i + 1, N)
            if sim_priv[i][j][ri] is not None
        ]
        mean_sim_priv.append(float(np.mean(sv)) if sv else None)

        cap = [correct_align_pub[i][ri] for i in range(N) if correct_align_pub[i][ri] is not None]
        mean_correct_align_pub.append(float(np.mean(cap)) if cap else None)

        cap2 = [correct_align_priv[i][ri] for i in range(N) if correct_align_priv[i][ri] is not None]
        mean_correct_align_priv.append(float(np.mean(cap2)) if cap2 else None)

    return {
        "per_agent_per_round": {
            "correct": correct,
            "flip": flip,
            "delta_pub": delta_pub,
            "delta_priv": delta_priv,
            "diss": diss,
            "shift": shift,
            "on_topic": on_topic,
            "influence": influence,
            "correct_align_pub": correct_align_pub,
            "correct_align_priv": correct_align_priv,
        },
        "per_pair_per_round": {
            "agree": agree,
            "sim_pub": sim_pub,
            "sim_priv": sim_priv,
            "toward": toward,
        },
        "per_round": {
            "mean_correct": mean_correct,
            "majority_correct": majority_correct,
            "majority_tie": majority_tie,
            "unanimous_correct": unanimous_correct,
            "fraction_agreeing_pairs": fraction_agreeing_pairs,
            "entropy": entropy,
            "mean_diss": mean_diss,
            "mean_sim_pub": mean_sim_pub,
            "mean_sim_priv": mean_sim_priv,
            "mean_correct_align_pub": mean_correct_align_pub,
            "mean_correct_align_priv": mean_correct_align_priv,
        },
    }


def augment(run: Dict, embedder: Optional[Embedder] = None) -> Dict:
    if embedder is None:
        embedder = Embedder()
    run_view = RunView.from_dict(run)
    gt_reasoning_text = _get_explanation(run["question_id"])
    emb = compute_embeddings(run_view, embedder, gt_reasoning_text)
    return {
        **run,
        "embeddings": emb.to_dict(),
        "scalars": compute_scalars(run_view, emb),
    }


def augment_file(
    path: Path, embedder: Optional[Embedder] = None, out: Optional[Path] = None
) -> Path:
    run = json.loads(path.read_text())
    dest = out or path
    dest.write_text(json.dumps(augment(run, embedder), indent=2))
    return dest


def augment_files(paths: List[Path], out_dir: Path = DEFAULT_OUT_DIR) -> List[Path]:
    embedder = Embedder()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in paths:
        out = (out_dir / path.name) if out_dir else None
        dest = augment_file(path, embedder=embedder, out=out)
        written.append(dest)
        print(f"  {dest}")
    return written



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_NAME} …")
    augment_files(args.inputs, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
