import pytest

from arxiv_rag.corpus import Paper


def make_paper(arxiv_id: str, title: str, abstract: str = "") -> Paper:
    return Paper(arxiv_id, title, abstract, ("A. Author",), ("cs.CL",), "2024-01-01")


@pytest.fixture
def papers() -> list[Paper]:
    return [
        make_paper("2401.00001", "Retrieval-augmented generation for question answering",
                   "We combine a dense retriever with a generator for open-domain questions."),
        make_paper("2401.00002", "Vehicle routing with time windows",
                   "A mixed-integer program and a variable neighborhood search heuristic."),
        make_paper("2401.00003", "Speculative decoding for faster inference",
                   "Draft models propose tokens that a large model verifies in parallel."),
    ]
