from arxiv_rag.retrieval import (
    BM25Retriever,
    Hit,
    HybridRetriever,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Vehicle-Routing problem with Time Windows") == [
        "vehicle-routing", "problem", "time", "windows",
    ]


def test_bm25_ranks_matching_paper_first(papers):
    hits = BM25Retriever(papers).search("variable neighborhood search heuristic", k=2)
    assert hits[0].paper.arxiv_id == "2401.00002"
    assert len(hits) == 2


def test_bm25_k_larger_than_corpus(papers):
    assert len(BM25Retriever(papers).search("decoding", k=50)) == len(papers)


def test_rrf_rewards_agreement_and_breaks_ties_by_id():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids[:2] == ["a", "b"]  # equal scores, tie broken by id
    assert set(ids) == {"a", "b", "c", "d"}
    assert fused[0][1] > fused[2][1]


class _FixedRetriever:
    def __init__(self, name, papers):
        self.name = name
        self.papers = papers

    def search(self, query, k=10):
        return [Hit(p, 1.0) for p in self.papers[:k]]


def test_hybrid_fuses_two_rankings(papers):
    a = _FixedRetriever("a", [papers[0], papers[1]])
    b = _FixedRetriever("b", [papers[1], papers[2]])
    hits = HybridRetriever([a, b]).search("anything", k=3)
    assert hits[0].paper.arxiv_id == "2401.00002"  # ranked by both
    assert {h.paper.arxiv_id for h in hits} == {"2401.00001", "2401.00002", "2401.00003"}
