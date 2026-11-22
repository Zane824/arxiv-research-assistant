from pathlib import Path

from arxiv_rag.corpus import load_papers, merge_papers, save_papers
from arxiv_rag.ingest import normalize_id, parse_feed

FIXTURE = Path(__file__).parent / "fixtures" / "sample_feed.xml"


def test_normalize_id_strips_version_and_url():
    assert normalize_id("http://arxiv.org/abs/2410.01234v2") == "2410.01234"
    assert normalize_id("http://arxiv.org/abs/hep-th/9901001v1") == "hep-th/9901001"


def test_parse_feed_extracts_clean_fields():
    papers, total = parse_feed(FIXTURE.read_text(encoding="utf-8"))
    assert total == 2
    first = papers[0]
    assert first.arxiv_id == "2410.01234"
    assert first.title == "Dense Retrieval for Long Documents"
    assert first.abstract == "We study dense retrieval over long scientific documents."
    assert first.authors == ("Ada Lovelace", "Alan Turing")
    assert first.categories == ("cs.CL", "cs.IR")
    assert first.published == "2024-10-02"
    assert papers[1].arxiv_id == "hep-th/9901001"


def test_save_load_roundtrip_and_merge(tmp_path, papers):
    path = tmp_path / "corpus" / "papers.jsonl"
    assert save_papers(path, papers) == 3
    assert load_papers(path) == papers

    updated = papers[0].__class__(**{**papers[0].to_dict(), "title": "New title"})
    merged = merge_papers(papers, [updated])
    assert len(merged) == 3
    assert merged[0].title == "New title"


def test_load_missing_file_returns_empty(tmp_path):
    assert load_papers(tmp_path / "nope.jsonl") == []


def test_fetch_retries_after_429_then_parses():
    import httpx

    from arxiv_rag.ingest import fetch_category

    feed = FIXTURE.read_text(encoding="utf-8")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, text=feed)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    papers = list(fetch_category("cs.CL", max_results=10, client=client, delay_s=0))
    assert [p.arxiv_id for p in papers] == ["2410.01234", "hep-th/9901001"]
    assert len(calls) == 2
