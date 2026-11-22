from arxiv_rag.rag import answer_question, build_messages, extract_citations
from arxiv_rag.retrieval import BM25Retriever


def test_extract_citations_handles_groups_and_duplicates():
    text = "RAG helps [2401.00001]. Routing too [2401.00002, hep-th/9901001]. Again [2401.00001]."
    assert extract_citations(text) == ["2401.00001", "2401.00002", "hep-th/9901001"]


def test_extract_citations_ignores_non_ids():
    assert extract_citations("See [1] and [Table 2].") == []


def test_build_messages_puts_ids_in_context(papers):
    hits = BM25Retriever(papers).search("routing", k=1)
    messages = build_messages("How are routes optimized?", hits)
    assert messages[0]["role"] == "system"
    assert "[2401.00002]" in messages[1]["content"]
    assert messages[1]["content"].endswith("Question: How are routes optimized?")


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, messages, temperature=0.0):
        self.calls.append(messages)
        return self.reply


def test_answer_flags_citations_outside_context(papers):
    llm = _FakeLLM("Use a VNS heuristic [2401.00002]. Also see [2501.99999].")
    answer = answer_question("vehicle routing heuristic", BM25Retriever(papers), llm, k=1)
    assert answer.citations == ("2401.00002", "2501.99999")
    assert answer.unsupported_citations == ("2501.99999",)
    assert len(llm.calls) == 1
