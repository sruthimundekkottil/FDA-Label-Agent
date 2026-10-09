"""Record/replay cache around any adapter.

Modes (``LLM_CACHE_MODE``):

* ``auto``    serve a recorded response if there is one, otherwise call and record
* ``record``  always call the provider and overwrite the recording
* ``replay``  never touch the network; a missing recording is an error
* ``off``     no caching at all

Recordings live in two places:

* the working cache ``data/cache/llm_v2/<key>.json`` (gitignored, one file per call)
* an optional *cassette* ``results/cassettes/<run_id>.jsonl`` (committed) holding
  every response one evaluation run used, so a fresh clone can replay the run
  offline and get identical numbers.

The key is a SHA-256 of the full request fingerprint (provider, model, system
prompt, every message including tool results and thought signatures, tool
schemas, generation settings) plus ``CACHE_SCHEMA_VERSION``. The API key is never
part of the request, so it can never reach a cache file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from agent.llm.base import (
    CacheMissError,
    GenerationSettings,
    LLMAdapter,
    LLMResult,
    Message,
    ToolSpec,
    request_fingerprint,
)
from src.utils import CACHE_DIR, get_logger

CACHE_SCHEMA_VERSION = 1
CACHE_MODES = ("auto", "record", "replay", "off")
DEFAULT_CACHE_DIR = CACHE_DIR / "llm_v2"

logger = get_logger("agent.llm.cache")


def cache_key(fingerprint: dict) -> str:
    payload = json.dumps(
        {"v": CACHE_SCHEMA_VERSION, **fingerprint},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedAdapter:
    """Wraps an adapter with record/replay. Same ``generate`` signature."""

    def __init__(
        self,
        inner: LLMAdapter,
        *,
        mode: str = "auto",
        cache_dir: Path | None = None,
        cassette: Path | None = None,
    ):
        if mode not in CACHE_MODES:
            raise ValueError(f"unknown cache mode {mode!r}; expected one of {CACHE_MODES}")
        self.inner = inner
        self.mode = mode
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cassette = Path(cassette) if cassette else None
        self._cassette_entries: dict[str, dict] = {}
        if self.cassette and self.cassette.exists():
            for line in self.cassette.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._cassette_entries[entry["key"]] = entry["result"]
        self.hits = 0
        self.misses = 0

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    # -- public ------------------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        settings: GenerationSettings | None = None,
    ) -> LLMResult:
        settings = settings or GenerationSettings()
        if self.mode == "off":
            return self.inner.generate(messages, system=system, tools=tools, settings=settings)

        fingerprint = request_fingerprint(
            provider=self.provider,
            model=self.model,
            messages=messages,
            system=system,
            tools=tools,
            settings=settings,
        )
        key = cache_key(fingerprint)

        if self.mode in ("auto", "replay"):
            recorded = self._lookup(key)
            if recorded is not None:
                self.hits += 1
                result = LLMResult.from_dict(recorded)
                result.cached = True
                result.latency_ms = 0.0
                self._append_cassette(key, recorded)
                return result
            if self.mode == "replay":
                raise CacheMissError(
                    f"replay mode: no recording for request {key[:16]} "
                    f"(model={self.model}); re-run with LLM_CACHE_MODE=auto or record"
                )

        self.misses += 1
        result = self.inner.generate(messages, system=system, tools=tools, settings=settings)
        recorded = result.to_dict()
        recorded["cached"] = False
        self._store(key, recorded)
        self._append_cassette(key, recorded)
        return result

    # -- internals ---------------------------------------------------------
    def _path(self, key: str) -> Path:
        # 128 bits of the hash keep paths short (Windows MAX_PATH); the full
        # key is stored inside the file and checked on read.
        return self.cache_dir / f"{key[:32]}.json"

    def _lookup(self, key: str) -> dict | None:
        if key in self._cassette_entries:
            return self._cassette_entries[key]
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload["result"] if payload.get("key") == key else None
        except (json.JSONDecodeError, KeyError):
            logger.warning("ignoring corrupt cache file %s", path)
            return None

    def _store(self, key: str, recorded: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "result": recorded,
        }
        self._path(key).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    def _append_cassette(self, key: str, recorded: dict) -> None:
        """Every response a run uses (hit or miss) lands in its cassette once."""

        if self.cassette is None or key in self._cassette_entries:
            return
        self.cassette.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "result": recorded,
        }
        with self.cassette.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._cassette_entries[key] = recorded
