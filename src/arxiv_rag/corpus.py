"""Paper records and JSONL persistence."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    published: str  # YYYY-MM-DD

    @property
    def text(self) -> str:
        """The text that gets indexed: title plus abstract."""
        return f"{self.title}\n\n{self.abstract}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> Paper:
        return cls(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            abstract=row["abstract"],
            authors=tuple(row.get("authors", ())),
            categories=tuple(row.get("categories", ())),
            published=row.get("published", ""),
        )


def load_papers(path: Path) -> list[Paper]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [Paper.from_dict(json.loads(line)) for line in f if line.strip()]


def save_papers(path: Path, papers: Iterable[Paper]) -> int:
    """Write papers atomically (temp file + rename); returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def merge_papers(existing: Iterable[Paper], new: Iterable[Paper]) -> list[Paper]:
    """Deduplicate by arXiv id; newer records replace older ones, first-seen order is kept."""
    merged: dict[str, Paper] = {p.arxiv_id: p for p in existing}
    for paper in new:
        merged[paper.arxiv_id] = paper
    return list(merged.values())
