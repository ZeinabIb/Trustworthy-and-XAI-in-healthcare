"""LLM client (Phase 5): a LOCAL OpenAI-compatible server (Ollama on the PC, vLLM on the server) or a mock, behind
a cache. Every call is cached in artifacts/<profile>/arbiter_runs/calls.jsonl keyed by
sha256(model_id, revision, prompt_text, seed, temperature, top_p); a cache hit costs nothing and returns the same
text, so reruns are deterministic. Logged per call: key, packet hash, messages, raw output, latency, token
counts, sampling params, model id and revision. The files hold patient-derived text: gitignored, never shared.
HARD RULE: the endpoint must be a loopback address (no patient-derived text to a third-party API).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class NonLocalEndpointError(ValueError):
    """The configured endpoint is not on this machine."""


class Backend(Protocol):
    model_id: str
    revision: str

    def complete(self, messages: list[dict[str, str]], seed: int, temperature: float, top_p: float,
                 max_tokens: int) -> tuple[str, dict[str, Any]]: ...


def check_local(endpoint: str) -> None:
    host = urlparse(endpoint).hostname or ""
    if host not in LOOPBACK:
        raise NonLocalEndpointError(f"arbiter endpoint {endpoint!r} is not local; patient-derived text must not "
                                    "leave the machine")


@dataclass
class OpenAICompatBackend:
    """POST {endpoint}/chat/completions with JSON output mode; seed, temperature, top_p passed through."""

    endpoint: str
    model_id: str
    revision: str
    timeout_s: float

    def __post_init__(self) -> None:
        check_local(self.endpoint)

    def complete(self, messages: list[dict[str, str]], seed: int, temperature: float, top_p: float,
                 max_tokens: int) -> tuple[str, dict[str, Any]]:
        import requests

        body = {"model": self.model_id, "messages": messages, "seed": seed, "temperature": temperature,
                "top_p": top_p, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
                "stream": False}
        r = requests.post(self.endpoint.rstrip("/") + "/chat/completions", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"], data.get("usage", {})


@dataclass
class MockBackend:
    """Scripted backend for tests and the smoke profile: fn(messages, seed) -> raw text."""

    fn: Callable[[list[dict[str, str]], int], str]
    model_id: str = "mock"
    revision: str = "mock-v1"

    def complete(self, messages: list[dict[str, str]], seed: int, temperature: float, top_p: float,
                 max_tokens: int) -> tuple[str, dict[str, Any]]:
        return self.fn(messages, seed), {}


def call_key(model_id: str, revision: str, messages: list[dict[str, str]], seed: int, temperature: float,
             top_p: float) -> str:
    blob = json.dumps([model_id, revision, json.dumps(messages, sort_keys=True), seed, temperature, top_p])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CachedClient:
    def __init__(self, backend: Backend, cache_dir: Path, temperature: float, top_p: float,
                 max_tokens: int) -> None:
        self.backend, self.temperature, self.top_p, self.max_tokens = backend, temperature, top_p, max_tokens
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir / "calls.jsonl"
        self.cache: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec
        self.n_new = self.n_hit = 0

    def complete(self, messages: list[dict[str, str]], seed: int, packet_hash: str) -> dict[str, Any]:
        key = call_key(self.backend.model_id, self.backend.revision, messages, seed, self.temperature, self.top_p)
        if key in self.cache:
            self.n_hit += 1
            return self.cache[key]
        t0 = time.perf_counter()
        raw, usage = self.backend.complete(messages, seed, self.temperature, self.top_p, self.max_tokens)
        rec = {"key": key, "packet_hash": packet_hash, "model_id": self.backend.model_id,
               "revision": self.backend.revision, "seed": seed, "temperature": self.temperature,
               "top_p": self.top_p, "max_tokens": self.max_tokens, "messages": messages, "raw": raw,
               "latency_s": time.perf_counter() - t0, "usage": usage,
               "utc": datetime.datetime.now(datetime.UTC).isoformat()}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.cache[key] = rec
        self.n_new += 1
        return rec


class PinError(RuntimeError):
    """The served model is not the pinned one (or no pin is set)."""


def served_digest(endpoint: str, model_id: str, timeout_s: float = 10.0) -> str | None:
    """Digest of model_id on a local Ollama server (GET <base>/api/tags); None if the server does not say."""
    import requests

    base = endpoint.rstrip("/").removesuffix("/v1")
    try:
        r = requests.get(base + "/api/tags", timeout=timeout_s)
        r.raise_for_status()
    except requests.RequestException:
        return None
    for m in r.json().get("models", []):
        if m.get("name") == model_id or m.get("model") == model_id:
            return m.get("digest")
    return None


def make_client(cfg: dict[str, Any], cache_dir: Path) -> CachedClient:
    """Mock for model 'mock' (smoke); otherwise the local OpenAI-compatible server, refusing unless the served
    weights match arbiter.model_digest (checked on Ollama; on servers that do not report a digest the pinned
    revision string is required and recorded)."""
    from mfsi_cxr.arbiter.mock import policy

    a = cfg["arbiter"]
    if a["model"] == "mock":
        backend: Backend = MockBackend(policy)
    else:
        if not a["endpoint"]:
            raise PinError("arbiter.endpoint is not set")
        check_local(a["endpoint"])
        found = served_digest(a["endpoint"], a["model"])
        if not a["model_digest"]:
            raise PinError(f"arbiter.model_digest is not set (the server reports {found!r} for {a['model']}); "
                           "pin it in the profile first")
        if found is not None and found != a["model_digest"]:
            raise PinError(f"served {a['model']} has digest {found}, pinned {a['model_digest']}")
        backend = OpenAICompatBackend(a["endpoint"], a["model"], a["model_digest"], a["timeout_s"])
    return CachedClient(backend, cache_dir, a["temperature"], a["top_p"], a["max_tokens"])
