"""Retrieval evaluation: Recall@k and MRR over a hand-labeled question set."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .corpus import Paper
from .retrieval import Retriever


@dataclass(frozen=True)
class EvalItem:
    qid: str
    question: str
    relevant_ids: frozenset[str]


def load_eval_set(path: Path) -> list[EvalItem]:
    """Read labeled questions; rows with an empty question or no relevant ids are skipped."""
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question", "").strip()
            relevant = frozenset(row.get("relevant_ids", []))
            if question and relevant:
                items.append(EvalItem(row["qid"], question, relevant))
    return items


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    retriever: Retriever, items: Sequence[EvalItem], ks: Sequence[int] = (1, 5, 10)
) -> dict[str, float]:
    if not items:
        raise ValueError("Eval set is empty -- label some questions first")
    depth = max(ks)
    totals = {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0}
    for item in items:
        ranked = [h.paper.arxiv_id for h in retriever.search(item.question, depth)]
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(ranked, item.relevant_ids, k)
        totals["mrr"] += reciprocal_rank(ranked, item.relevant_ids)
    return {name: value / len(items) for name, value in totals.items()}


def format_table(results: dict[str, dict[str, float]], n_questions: int) -> str:
    metrics = list(next(iter(results.values())))
    lines = [
        f"Retrieval results on {n_questions} labeled questions",
        "",
        "| retriever | " + " | ".join(metrics) + " |",
        "|---" * (len(metrics) + 1) + "|",
    ]
    for name, scores in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{scores[m]:.3f}" for m in metrics) + " |")
    return "\n".join(lines)


def make_template(papers: Sequence[Paper], n: int, seed: int = 0) -> list[dict]:
    """Sample papers to label: you write a question each sampled paper answers."""
    sample = random.Random(seed).sample(list(papers), min(n, len(papers)))
    return [
        {
            "qid": f"q{i:03d}",
            "question": "",
            "relevant_ids": [p.arxiv_id],
            "source_title": p.title,
            "source_abstract": p.abstract,
        }
        for i, p in enumerate(sample, start=1)
    ]
