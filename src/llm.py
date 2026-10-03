"""Thin OpenAI-compatible chat-completions client.

Works with any provider that speaks the OpenAI chat-completions API (Gemini's
OpenAI-compatible endpoint, OpenAI, Groq, a local Ollama server, ...). Everything
is configured through ``.env`` - no key or model name is hard-coded:

    LLM_BASE_URL   e.g. https://generativelanguage.googleapis.com/v1beta/openai/
    LLM_API_KEY    provider key
    LLM_MODEL      provider model id

Three things matter for free-tier providers:

* a minimum delay between requests,
* exponential backoff (with ``Retry-After`` support) on 429/5xx,
* a disk cache keyed by (model, prompt, temperature, max_tokens) so repeated
  evaluation runs do not burn quota.

Mock mode (``LLM_MOCK=1``, or no key/base URL/model configured) keeps the whole
pipeline runnable without any API key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.utils import CACHE_DIR, env_flag, get_logger, load_env

LLM_CACHE_DIR = CACHE_DIR / "llm"

logger = get_logger("llm")


class LLMError(RuntimeError):
    """Raised when the provider cannot be reached or returns an error."""


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 900
    timeout: float = 90.0
    min_interval_s: float = 1.0
    max_retries: int = 4
    force_mock: bool = False

    @property
    def mock(self) -> bool:
        """Mock when asked, or when the provider is not configured."""

        return self.force_mock or not (self.base_url and self.api_key and self.model)

    @classmethod
    def from_env(cls, *, force_mock: bool | None = None, **overrides) -> "LLMConfig":
        load_env()
        values = {
            "base_url": os.environ.get("LLM_BASE_URL", "").strip(),
            "api_key": os.environ.get("LLM_API_KEY", "").strip(),
            "model": os.environ.get("LLM_MODEL", "").strip(),
        }
        if force_mock is None:
            force_mock = env_flag("LLM_MOCK", False)
        values["force_mock"] = bool(force_mock)
        if os.environ.get("LLM_TEMPERATURE"):
            values["temperature"] = float(os.environ["LLM_TEMPERATURE"])
        if os.environ.get("LLM_MAX_TOKENS"):
            values["max_tokens"] = int(os.environ["LLM_MAX_TOKENS"])
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    cached: bool = False
    mock: bool = False
    raw: dict = field(default_factory=dict)


def cache_key(prompt: str, model: str, temperature: float, max_tokens: int, system: str = "") -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class LLMClient:
    """Chat-completions client with rate limiting, retries and a disk cache."""

    def __init__(self, config: LLMConfig | None = None, *, cache_dir: Path | None = None):
        self.config = config or LLMConfig.from_env()
        self.cache_dir = Path(cache_dir) if cache_dir else LLM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_ts = 0.0
        self._session = requests.Session()

    # -- public API -------------------------------------------------------
    @property
    def mock(self) -> bool:
        return self.config.mock

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
        model: str | None = None,
    ) -> LLMResponse:
        """One chat completion. Raises ``LLMError`` in mock mode."""

        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        model = model or self.config.model

        if self.mock:
            raise LLMError(
                "LLM is in mock mode (no LLM_BASE_URL/LLM_API_KEY/LLM_MODEL configured)"
            )

        key = cache_key(prompt, model, temperature, max_tokens, system)
        cache_path = self.cache_dir / f"{key}.json"
        if use_cache and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                logger.info("cache hit %s (model=%s)", key, model)
                return LLMResponse(
                    text=cached["text"],
                    model=cached.get("model", model),
                    prompt_tokens=cached.get("prompt_tokens"),
                    completion_tokens=cached.get("completion_tokens"),
                    latency_ms=0.0,
                    cached=True,
                    raw=cached,
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning("ignoring corrupt cache file %s", cache_path)

        payload = {
            "model": model,
            "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": prompt}]
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        last_error = ""
        for attempt in range(1, self.config.max_retries + 1):
            self._respect_min_interval()
            try:
                response = self._session.post(
                    url, headers=headers, json=payload, timeout=self.config.timeout
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("request error (%s), attempt %d/%d", last_error, attempt, self.config.max_retries)
                self._backoff(attempt)
                continue

            if response.status_code == 200:
                data = response.json()
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                usage = data.get("usage", {}) or {}
                result = LLMResponse(
                    text=(text or "").strip(),
                    model=data.get("model", model),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    raw=data,
                )
                if use_cache:
                    self._write_cache(cache_path, result, prompt, system, temperature, max_tokens)
                return result

            body = response.text[:300]
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                last_error = f"HTTP {response.status_code}: {body}"
                logger.warning(
                    "provider throttled/error (%s) attempt %d/%d%s",
                    response.status_code,
                    attempt,
                    self.config.max_retries,
                    f", Retry-After={retry_after}" if retry_after else "",
                )
                self._backoff(attempt, retry_after)
                continue

            raise LLMError(f"HTTP {response.status_code} from {url}: {body}")

        raise LLMError(f"giving up after {self.config.max_retries} attempts: {last_error}")

    # -- internals --------------------------------------------------------
    def _respect_min_interval(self) -> None:
        wait = self.config.min_interval_s - (time.monotonic() - self._last_request_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        delay = min(60.0, 2.0 ** attempt) + random.uniform(0, 1)
        if retry_after:
            try:
                delay = max(delay, min(120.0, float(retry_after)))
            except ValueError:
                pass
        time.sleep(delay)

    def _write_cache(
        self,
        path: Path,
        result: LLMResponse,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        payload = {
            "text": result.text,
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": result.latency_ms,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_preview": prompt[:400],
            "system_preview": system[:200],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--mock", action="store_true", help="force mock mode")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = LLMConfig.from_env(force_mock=True if args.mock else None)
    client = LLMClient(config)
    print(f"model={config.model or '(none)'} base_url={config.base_url or '(none)'} mock={client.mock}")
    if client.mock:
        print("mock mode: set LLM_BASE_URL/LLM_API_KEY/LLM_MODEL in .env for real answers")
        return 0
    response = client.complete(args.prompt, use_cache=not args.no_cache)
    print("-" * 70)
    print(response.text)
    print("-" * 70)
    print(
        f"model={response.model} cached={response.cached} latency_ms={response.latency_ms:.0f} "
        f"prompt_tokens={response.prompt_tokens} completion_tokens={response.completion_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
