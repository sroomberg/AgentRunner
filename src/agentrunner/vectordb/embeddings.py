"""Embedding client for vLLM's OpenAI-compatible /v1/embeddings endpoint."""

from __future__ import annotations

import json
import urllib.request


def embed(endpoint: str, model_id: str, texts: list[str]) -> list[list[float]]:
    """Return embeddings for *texts* using the vLLM embedding endpoint."""
    payload = json.dumps({"model": model_id, "input": texts}).encode()
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


def embed_one(endpoint: str, model_id: str, text: str) -> list[float]:
    return embed(endpoint, model_id, [text])[0]
