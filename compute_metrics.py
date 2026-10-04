import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.metrics import Embedder, compute_scalars, embed_repetition, embed_shared

ROOT = Path(__file__).parent
DEFAULT_OUT_DIR = ROOT / "results/mas_eval"
DATASET_PATH = ROOT / "dataset" / "gpqa_diamond.csv"

_df: Optional[pd.DataFrame] = None


def _load_explanation(question_id: str) -> Optional[str]:
    global _df
    if _df is None:
        _df = pd.read_csv(DATASET_PATH)
    val = _df.iloc[int(question_id)]["Explanation"]
    return str(val) if pd.notna(val) else None


def process_file(path: Path, embedder: Embedder, out_dir: Path) -> Path:
    wrapper = json.loads(path.read_text())

    explanation = (
        _load_explanation(wrapper["question_id"])
        if wrapper.get("dataset", "gpqa") == "gpqa"
        else None
    )
    question_vec, gt_vec = embed_shared(wrapper["question"], explanation, embedder)

    rep_metrics = []
    for rep in wrapper["repetitions"]:
        emb = embed_repetition(rep, embedder)
        scalars = compute_scalars(rep, emb, gt_vec)
        rep_metrics.append(
            {"repetition": rep["repetition"], "embeddings": emb, **scalars}
        )

    wrapper["metrics"] = {
        "embedding_model": embedder.model_name,
        "question_vec": question_vec.tolist(),
        "gt_reasoning_vec": gt_vec.tolist() if gt_vec is not None else None,
        "repetitions": rep_metrics,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / path.name
    dest.write_text(json.dumps(wrapper, indent=2))
    return dest


def augment_files(paths: list, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    embedder = Embedder()
    for path in paths:
        process_file(Path(path), embedder, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    embedder = Embedder()
    print(f"Loading {embedder.model_name} …")
    for path in args.inputs:
        dest = process_file(path, embedder, args.out_dir)
        print(f"  {dest}")


if __name__ == "__main__":
    main()
