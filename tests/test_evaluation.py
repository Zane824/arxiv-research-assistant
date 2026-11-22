import json

import pytest

from arxiv_rag.evaluation import (
    EvalItem,
    evaluate_retriever,
    format_table,
    load_eval_set,
    make_template,
    recall_at_k,
    reciprocal_rank,
)
from arxiv_rag.retrieval import BM25Retriever


def test_recall_and_rr():
    relevant = frozenset({"b", "x"})
    assert recall_at_k(["a", "b", "c"], relevant, 1) == 0.0
    assert recall_at_k(["a", "b", "c"], relevant, 3) == 0.5
    assert reciprocal_rank(["a", "b", "c"], relevant) == 0.5
    assert reciprocal_rank(["a"], relevant) == 0.0


def test_load_eval_set_skips_unlabeled_rows(tmp_path):
    path = tmp_path / "questions.jsonl"
    rows = [
        {"qid": "q1", "question": "What is RAG?", "relevant_ids": ["2401.00001"]},
        {"qid": "q2", "question": "", "relevant_ids": ["2401.00002"]},
        {"qid": "q3", "question": "No label", "relevant_ids": []},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert [i.qid for i in load_eval_set(path)] == ["q1"]


def test_evaluate_retriever_end_to_end(papers):
    items = [
        EvalItem("q1", "mixed-integer program for routing", frozenset({"2401.00002"})),
        EvalItem("q2", "draft models verify tokens", frozenset({"2401.00003"})),
    ]
    scores = evaluate_retriever(BM25Retriever(papers), items, ks=(1, 3))
    assert scores == {"recall@1": 1.0, "recall@3": 1.0, "mrr": 1.0}
    assert "| bm25 | 1.000 | 1.000 | 1.000 |" in format_table({"bm25": scores}, 2)


def test_evaluate_empty_set_raises(papers):
    with pytest.raises(ValueError):
        evaluate_retriever(BM25Retriever(papers), [])


def test_make_template_is_deterministic(papers):
    first = make_template(papers, n=2, seed=1)
    assert first == make_template(papers, n=2, seed=1)
    assert all(row["question"] == "" and len(row["relevant_ids"]) == 1 for row in first)
