"""Single-pass RAG: retrieve papers, ask the LLM to answer with [arxiv_id] citations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .llm import ChatModel
from .retrieval import Hit, Retriever

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions about arXiv papers.\n"
    "Use ONLY the papers provided in the context. After every claim, cite the supporting "
    "paper id in square brackets, e.g. [2401.01234]. If the papers do not contain the answer, "
    "say that you cannot answer from the given papers."
)

# New-style ids (2401.01234) and old-style ids (hep-th/9901001, math.AG/0101001).
_ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7}")
_BRACKETS = re.compile(r"\[([^\[\]]+)\]")


def format_context(hits: Sequence[Hit]) -> str:
    return "\n\n".join(
        f"[{h.paper.arxiv_id}] {h.paper.title} ({h.paper.published})\n{h.paper.abstract}"
        for h in hits
    )


def build_messages(question: str, hits: Sequence[Hit]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Papers:\n\n{format_context(hits)}\n\nQuestion: {question}"},
    ]


def extract_citations(text: str) -> list[str]:
    """Ids cited in square brackets, in first-seen order; handles '[id1, id2]' groups."""
    seen: dict[str, None] = {}
    for group in _BRACKETS.findall(text):
        for arxiv_id in _ARXIV_ID.findall(group):
            seen.setdefault(arxiv_id, None)
    return list(seen)


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    hits: tuple[Hit, ...]
    citations: tuple[str, ...]

    @property
    def unsupported_citations(self) -> tuple[str, ...]:
        """Cited ids that were not in the retrieved context, i.e. likely hallucinated."""
        retrieved = {h.paper.arxiv_id for h in self.hits}
        return tuple(c for c in self.citations if c not in retrieved)


def answer_question(question: str, retriever: Retriever, llm: ChatModel, k: int = 5) -> Answer:
    hits = retriever.search(question, k)
    text = llm.complete(build_messages(question, hits))
    return Answer(question, text, tuple(hits), tuple(extract_citations(text)))
