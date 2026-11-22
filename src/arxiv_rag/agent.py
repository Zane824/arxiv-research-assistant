"""Multi-step research agent over the retrieval layer.

Single-pass RAG (rag.answer_question) fails when the first query misses the right
papers -- vague questions, or questions whose answer needs evidence from two papers.
A retrieve/grade/rewrite loop addresses that:

    retrieve --> grade: "are these papers enough to answer?"
       ^              | no (and rounds left)       | yes / out of rounds
       |              v                            v
       +------ rewrite the query               answer with citations

Planned implementation: LangGraph, with one node per step, a conditional edge after
the grade node, and a state dict holding the question, the current query, the
accumulated hits and the round count.

Not implemented yet.
"""

from __future__ import annotations

from .llm import ChatModel
from .rag import Answer
from .retrieval import Retriever


def run_agent(
    question: str, retriever: Retriever, llm: ChatModel, k: int = 5, max_rounds: int = 2
) -> Answer:
    raise NotImplementedError("the retrieve/grade/rewrite agent is not implemented yet")
