"""In-memory cosine-similarity search over the imported embeddings (numpy only)."""
from __future__ import annotations

import re

import numpy as np

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "which", "what", "are", "was", "were", "has",
    "have", "how", "does", "into", "than", "their", "there", "about", "while", "when", "below", "above",
    "under", "over", "each", "all", "any", "can", "show", "between", "using", "used", "paper", "papers",
}


def _terms(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9\-\.]{3,}", text.lower()) if t not in _STOP}


class VectorStore:
    def __init__(self, chunks: list, embeddings: np.ndarray):
        self.chunks = chunks
        self.emb = embeddings
        self._sources = np.array([c["source"] for c in chunks])
        self._types = np.array([c.get("chunk_type", "text") for c in chunks])
        self.sources = sorted(set(self._sources.tolist()))

    def search(self, qvec, k=6, source=None, chunk_type=None, max_per_source=None, query_text=""):
        q = np.asarray(qvec, dtype=np.float32)
        scores = self.emb @ q
        mask = np.ones(len(self.chunks), dtype=bool)
        if source:
            mask &= self._sources == source
        if chunk_type:
            mask &= self._types == chunk_type
        scores = np.where(mask, scores, -np.inf)

        n_cand = min(len(scores), max(200, k * 25))
        if n_cand < len(scores):
            cand = np.argpartition(-scores, n_cand - 1)[:n_cand]
        else:
            cand = np.arange(len(scores))
        q_terms = _terms(query_text) if query_text else set()

        rescored = []
        for i in cand:
            if not np.isfinite(scores[i]):
                continue
            bonus = 0.0
            if q_terms:
                overlap = len(q_terms & _terms(self.chunks[i]["text"]))
                bonus = 0.08 * overlap / len(q_terms)  # light keyword boost (hybrid search)
            rescored.append((float(scores[i]) + bonus, int(i)))
        rescored.sort(reverse=True)

        results, per_source = [], {}
        for score, i in rescored:
            src = self.chunks[i]["source"]
            if max_per_source and per_source.get(src, 0) >= max_per_source:
                continue
            per_source[src] = per_source.get(src, 0) + 1
            hit = dict(self.chunks[i])
            hit["score"] = round(score, 4)
            results.append(hit)
            if len(results) >= k:
                break
        return results
