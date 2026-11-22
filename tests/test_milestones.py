"""Tests for components that are not implemented yet.

Each skip is lifted when the corresponding component lands.
"""

import pytest

from arxiv_rag.retrieval import BM25Retriever


@pytest.mark.skip(reason="RerankingRetriever is not implemented yet")
def test_reranker_returns_k_hits_from_candidates(papers):
    from arxiv_rag.rerank import RerankingRetriever

    reranker = RerankingRetriever(
        BM25Retriever(papers), "cross-encoder/ms-marco-MiniLM-L-6-v2", depth=3
    )
    hits = reranker.search("speculative decoding with draft models", k=2)
    assert len(hits) == 2
    assert hits[0].paper.arxiv_id == "2401.00003"


@pytest.mark.skip(reason="run_agent is not implemented yet")
def test_agent_rewrites_query_when_first_round_is_insufficient(papers):
    # Drives the agent with a stub LLM that grades the first round "insufficient",
    # then asserts a second, different query was issued before the answer.
    raise NotImplementedError
