"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("ARXIV_RAG_DATA_DIR", "data")))
    results_dir: Path = field(
        default_factory=lambda: Path(_env("ARXIV_RAG_RESULTS_DIR", "results"))
    )
    embedding_model: str = field(
        default_factory=lambda: _env("ARXIV_RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    # Any OpenAI-compatible chat endpoint works; the default is a local Ollama server.
    llm_base_url: str = field(
        default_factory=lambda: _env("ARXIV_RAG_LLM_BASE_URL", "http://localhost:11434/v1")
    )
    llm_model: str = field(default_factory=lambda: _env("ARXIV_RAG_LLM_MODEL", "qwen2.5:7b"))
    llm_api_key: str | None = field(default_factory=lambda: _env("ARXIV_RAG_LLM_API_KEY"))

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus" / "papers.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def eval_path(self) -> Path:
        return self.data_dir / "eval" / "questions.jsonl"
