"""Minimal client for any OpenAI-compatible chat endpoint (Ollama, vLLM, LM Studio, ...)."""

from __future__ import annotations

from typing import Protocol

import httpx


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str: ...


class OpenAICompatibleChat:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 120):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        response = self._client.post(
            self.url,
            json={"model": self.model, "messages": messages, "temperature": temperature},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
