"""Cross-encoder reranking over hybrid retrieval candidates.

A bi-encoder embeds the query and the paper separately: fast, but the two texts never
interact, so the ranking is coarse. A cross-encoder reads the (query, paper) pair
jointly and ranks far more accurately, at a cost that rules out scoring a whole corpus.
The standard compromise is two stages -- retrieve a deep candidate list with
HybridRetriever, then rerank only those candidates and keep the top k.

Planned model: "cross-encoder/ms-marco-MiniLM-L-6-v2" via
sentence_transformers.CrossEncoder.

Not implemented yet.
"""

from __future__ import annotations

from .retrieval import Hit, Retriever


class RerankingRetriever:
    name = "rerank"

    def __init__(self, base: Retriever, model_name: str, depth: int = 50):
        self.base = base
        self.model_name = model_name
        self.depth = depth

    def search(self, query: str, k: int = 10) -> list[Hit]:
        # 1. candidates = self.base.search(query, self.depth)
        # 2. score each (query, candidate.paper.text) pair with the cross-encoder
        # 3. return the top k as Hit objects carrying the cross-encoder score
        raise NotImplementedError("cross-encoder reranking is not implemented yet")
