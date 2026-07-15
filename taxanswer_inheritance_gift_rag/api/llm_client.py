from __future__ import annotations

from typing import Any, Dict, List

import requests

from api.config import get_required_url, load_settings


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        url_key: str = "llm_base_url",
    ):
        settings = load_settings().get("llm", {})
        self.base_url = (base_url or get_required_url(url_key)).rstrip("/")
        self.api_key = api_key or settings.get("api_key", "dummy")
        self.model = model or settings.get("model", "local-model")
        self.temperature = temperature if temperature is not None else float(settings.get("temperature", 0.1))
        self.max_tokens = max_tokens if max_tokens is not None else int(settings.get("max_tokens", 2048))
        self.timeout_sec = int(settings.get("timeout_sec", 180))

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        res = requests.post(url, json=payload, headers=headers, timeout=self.timeout_sec)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]


class OpenAICompatibleEmbedding:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        settings = load_settings().get("embedding", {})
        self.base_url = (base_url or get_required_url("embedding_base_url")).rstrip("/")
        self.api_key = api_key or settings.get("api_key", "dummy")
        self.model = model or settings.get("model", "BAAI/bge-m3")
        self.timeout_sec = int(settings.get("timeout_sec", 180))

    def embed(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        res = requests.post(url, json=payload, headers=headers, timeout=self.timeout_sec)
        res.raise_for_status()
        data = res.json()
        return [item["embedding"] for item in data["data"]]
