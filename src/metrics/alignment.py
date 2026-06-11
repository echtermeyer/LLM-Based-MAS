import re
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = {
    'the','a','an','is','it','in','of','to','and','or','that','this',
    'for','with','are','was','be','as','by','at','from','on','have',
    'has','its','we','i','you','they','their','our','which','but','not',
    'if','so','can','will','would','could','should','may','also','more',
    'than','each','other','all','any','one','two','there','these','those',
}


def _preprocess(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]


def _bigram_jaccard(a: str, b: str) -> float:
    ta, tb = _preprocess(a), _preprocess(b)
    bg_a = set(zip(ta, ta[1:]))
    bg_b = set(zip(tb, tb[1:]))
    if not bg_a and not bg_b:
        return 1.0
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


def _mean_pairwise_jaccard(msgs: List[str]) -> float:
    pairs = list(combinations(range(len(msgs)), 2))
    if not pairs:
        return np.nan
    return float(np.mean([_bigram_jaccard(msgs[i], msgs[j]) for i, j in pairs]))


def _mean_pairwise_cosine(vecs: np.ndarray) -> float:
    pairs = list(combinations(range(len(vecs)), 2))
    if not pairs:
        return np.nan
    sims = [
        float(cosine_similarity(vecs[i].reshape(1, -1), vecs[j].reshape(1, -1))[0, 0])
        for i, j in pairs
    ]
    return float(np.mean(sims))


class AlignmentMetric(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]) -> None: ...

    @abstractmethod
    def score_round(self, msgs: List[str]) -> float: ...

    def score(self, rep: Dict) -> Dict:
        """
        Returns per-round and mean alignment for a single repetition.
        {
          'per_round': [float, ...],   # one value per round t
          'mean': float,               # (1/T) * sum_t alpha^t
        }
        """
        per_round = []
        for rnd in rep['trajectory']:
            msgs = [a.get('message', '') for a in rnd['phase_b'] if a.get('message', '')]
            if len(msgs) < 2:
                per_round.append(np.nan)
            else:
                per_round.append(self.score_round(msgs))
        valid = [v for v in per_round if not np.isnan(v)]
        return {
            'per_round': per_round,
            'mean': float(np.mean(valid)) if valid else np.nan,
        }

    def score_all(self, repetitions: List[Dict]) -> List[Dict]:
        return [self.score(rep) for rep in repetitions]


class TFIDFAlignment(AlignmentMetric):
    """
    alpha^t = mean pairwise TF-IDF cosine similarity of messages at round t.
    Fit the TF-IDF on a corpus first (ideally all messages in the dataset).
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 3, max_features: int = 8000):
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            stop_words=list(STOPWORDS),
        )
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def score_round(self, msgs: List[str]) -> float:
        if not self._fitted:
            raise RuntimeError('Call fit() before score_round()')
        vecs = self._vectorizer.transform(msgs).toarray()
        return _mean_pairwise_cosine(vecs)


class BigramJaccardAlignment(AlignmentMetric):
    """
    alpha^t = mean pairwise bigram Jaccard similarity of messages at round t.
    Parameter-free — fit() is a no-op.
    """

    def fit(self, corpus: List[str]) -> None:
        pass

    def score_round(self, msgs: List[str]) -> float:
        return _mean_pairwise_jaccard(msgs)


def build_corpus(raw: Dict, common_keys: Optional[List] = None) -> List[str]:
    msgs = []
    for w, data in raw.items():
        for key, d in data.items():
            if common_keys is not None and key not in common_keys:
                continue
            for rep in d['repetitions']:
                for rnd in rep['trajectory']:
                    for a in rnd['phase_b']:
                        m = a.get('message', '')
                        if m:
                            msgs.append(m)
    return msgs
