"""Shared test setup.

Tests must run with no API key, no network and no data: retrievers are stubbed
and the model is scripted. A test that genuinely needs the built indexes is
marked ``@pytest.mark.needs_data`` and is skipped when ``data/index/`` is
missing (always the case in CI, since data/ is not in git).
"""

from __future__ import annotations

import pytest

from src.utils import INDEX_DIR

_REQUIRED_INDEXES = ("section", "table_aware")


def _indexes_present() -> bool:
    return all((INDEX_DIR / name / "index.faiss").exists() for name in _REQUIRED_INDEXES)


def pytest_collection_modifyitems(config, items) -> None:
    if _indexes_present():
        return
    skip = pytest.mark.skip(reason="needs data/index/ (build it: python -m src.chunking && python -m src.indexing)")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_real_llm_credentials(monkeypatch) -> None:
    """Make sure no test can reach a real provider with a key from the environment."""

    for name in ("GEMINI_API_KEY", "LLM_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
