"""Fetch paper metadata from the public arXiv API (https://info.arxiv.org/help/api/)."""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator

import httpx

from .corpus import Paper

API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
# arXiv asks clients to wait 3 seconds between consecutive API calls.
REQUEST_DELAY_S = 3.0
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
USER_AGENT = "arxiv-research-assistant/0.1 (personal research project)"

_VERSION_SUFFIX = re.compile(r"v\d+$")
_WHITESPACE = re.compile(r"\s+")


def normalize_id(raw: str) -> str:
    """'http://arxiv.org/abs/2410.01234v2' -> '2410.01234'; old-style ids keep their archive."""
    return _VERSION_SUFFIX.sub("", raw.rsplit("/abs/", 1)[-1])


def _clean(text: str | None) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def parse_feed(xml_text: str) -> tuple[list[Paper], int]:
    """Parse one Atom response into papers, plus the total result count the API reports."""
    root = ET.fromstring(xml_text)
    total = int(root.findtext(f"{OPENSEARCH}totalResults", default="0"))
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = entry.findtext(f"{ATOM}id")
        if not raw_id or "/abs/" not in raw_id:  # error feeds contain a non-paper entry
            continue
        papers.append(
            Paper(
                arxiv_id=normalize_id(raw_id),
                title=_clean(entry.findtext(f"{ATOM}title")),
                abstract=_clean(entry.findtext(f"{ATOM}summary")),
                authors=tuple(
                    _clean(a.findtext(f"{ATOM}name")) for a in entry.findall(f"{ATOM}author")
                ),
                categories=tuple(c.get("term", "") for c in entry.findall(f"{ATOM}category")),
                published=(entry.findtext(f"{ATOM}published") or "")[:10],
            )
        )
    return papers, total


def fetch_category(
    category: str,
    max_results: int,
    start: int = 0,
    batch_size: int = 50,  # arXiv throttled 100-paper pages from our network during testing
    client: httpx.Client | None = None,
    delay_s: float = REQUEST_DELAY_S,
    retries: int = 5,
) -> Iterator[Paper]:
    """Yield up to `max_results` papers in an arXiv category (e.g. 'cs.CL'), newest first,
    skipping the `start` most recent ones."""
    owns_client = client is None
    client = client or httpx.Client(
        timeout=60, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )
    try:
        fetched = 0
        while fetched < max_results:
            params = {
                "search_query": f"cat:{category}",
                "start": start + fetched,
                "max_results": min(batch_size, max_results - fetched),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            papers, total = _get_batch(client, params, retries, delay_s)
            if not papers:
                break
            yield from papers
            fetched += len(papers)
            if start + fetched >= total:
                break
            time.sleep(delay_s)
    finally:
        if owns_client:
            client.close()


def _retry_wait(response: httpx.Response, attempt: int, delay_s: float) -> float:
    retry_after = response.headers.get("Retry-After", "")
    if retry_after.isdigit():
        return float(retry_after)
    return delay_s * 2 ** (attempt + 1)


def _get_batch(
    client: httpx.Client, params: dict, retries: int, delay_s: float
) -> tuple[list[Paper], int]:
    for attempt in range(retries):
        response = client.get(API_URL, params=params)
        # 429 / 5xx: arXiv is throttling or busy; back off and try again.
        if response.status_code in RETRY_STATUS and attempt < retries - 1:
            wait = _retry_wait(response, attempt, delay_s)
            print(f"  arXiv returned {response.status_code}; retrying in {wait:.0f}s")
            time.sleep(wait)
            continue
        response.raise_for_status()
        papers, total = parse_feed(response.text)
        # The API occasionally returns an empty page mid-pagination; retrying usually fixes it.
        if papers or params["start"] >= total:
            return papers, total
        time.sleep(delay_s * (attempt + 1))
    return [], 0
