"""Retrievers: BM25 (lexical), dense (embeddings), and hybrid (reciprocal rank fusion)."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Paper

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by can for from has have in into is it its of on or our that the "
    "their these this to was we were which with via using".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    paper: Paper
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[Hit]: ...


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top], kind="stable")]


class BM25Retriever:
    name = "bm25"

    def __init__(self, papers: Sequence[Paper]):
        self.papers = list(papers)
        self._bm25 = BM25Okapi([tokenize(p.text) for p in self.papers])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        return [Hit(self.papers[i], float(scores[i])) for i in _top_k(scores, k)]


class DenseRetriever:
    """Cosine similarity over sentence-transformer embeddings (needs the `dense` extra)."""

    name = "dense"

    def __init__(
        self,
        papers: Sequence[Paper],
        model_name: str,
        cache_dir: Path | None = None,
        query_prefix: str | None = None,
        batch_size: int = 64,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                'Dense retrieval needs the optional extra: pip install -e ".[dense]"'
            ) from exc
        self.papers = list(papers)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # BGE models are trained with an instruction prefix on queries (not on documents).
        if query_prefix is None and "bge" in model_name.lower():
            query_prefix = "Represent this sentence for searching relevant passages: "
        self.query_prefix = query_prefix or ""
        self.embeddings = self._load_or_encode(cache_dir, batch_size)

    def _load_or_encode(self, cache_dir: Path | None, batch_size: int) -> np.ndarray:
        ids = np.array([p.arxiv_id for p in self.papers])
        cache = None
        if cache_dir is not None:
            cache = cache_dir / f"dense_{re.sub(r'[^A-Za-z0-9]+', '_', self.model_name)}.npz"
            if cache.exists():
                data = np.load(cache, allow_pickle=False)
                if np.array_equal(data["ids"], ids):
                    return data["embeddings"]
        embeddings = self.model.encode(
            [p.text for p in self.papers],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, ids=ids, embeddings=embeddings)
        return embeddings

    def search(self, query: str, k: int = 10) -> list[Hit]:
        query_vec = self.model.encode(
            [self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        scores = self.embeddings @ query_vec
        return [Hit(self.papers[i], float(scores[i])) for i in _top_k(scores, k)]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank). Ties break by id."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], depth: int = 50, rrf_k: int = 60):
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = list(retrievers)
        self.depth = depth
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 10) -> list[Hit]:
        by_id: dict[str, Paper] = {}
        rankings = []
        for retriever in self.retrievers:
            hits = retriever.search(query, self.depth)
            by_id.update((h.paper.arxiv_id, h.paper) for h in hits)
            rankings.append([h.paper.arxiv_id for h in hits])
        fused = reciprocal_rank_fusion(rankings, self.rrf_k)[:k]
        return [Hit(by_id[doc_id], score) for doc_id, score in fused]
