from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"


class Embedder:
    def __init__(self) -> None:
        torch.manual_seed(0)
        self._model = SentenceTransformer(MODEL_NAME)
        self._model.eval()

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    def embed(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            return self._model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
