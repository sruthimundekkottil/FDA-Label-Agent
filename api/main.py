"""FastAPI service for the FDA label agent.

    uvicorn api.main:app --host 0.0.0.0 --port 8000      # one worker: limits are in-memory

    GET  /health   liveness + model name + today's usage (no key needed)
    POST /ask      {"question": "..."} -> answer, citations, stop_reason

Indexes and models are loaded once at startup (lifespan), not per request.

Environment (all optional):

    AGENT_API_KEY          if set, /ask requires the header X-API-Key with this value
    RATE_LIMIT_PER_MINUTE  questions per client IP per minute (default 10, 0 = off)
    DAILY_CALL_CAP         /ask runs per UTC day, all clients together (default 200, 0 = off);
                           each run uses at most 5 model calls
    AGENT_CONCURRENCY      agent runs at the same time (default 1)
    TRUST_PROXY_HEADERS=1  take the client IP from X-Forwarded-For (only behind your own proxy)
    API_LOG_PATH           JSONL agent log (default results/runs/api/agent_log.jsonl)

plus the LLM_* settings read by ``agent.llm``.
"""

from __future__ import annotations

import hmac
import logging
import math
import os
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from agent.llm import get_adapter, settings_from_env
from agent.loop import AgentGraph, JsonlLog
from agent.tools import ToolBox
from api.limits import DailyCap, SlidingWindowLimiter
from src.utils import env_flag, load_env

DISCLAIMER = "For information only. Not medical advice."
logger = logging.getLogger("api")


# --------------------------------------------------------------------------- #
# Settings and schemas
# --------------------------------------------------------------------------- #
@dataclass
class ApiSettings:
    api_key: str = ""
    rate_limit_per_minute: int = 10
    daily_call_cap: int = 200
    concurrency: int = 1
    trust_proxy_headers: bool = False
    log_path: str = "results/runs/api/agent_log.jsonl"

    @classmethod
    def from_env(cls) -> "ApiSettings":
        load_env()
        return cls(
            api_key=os.environ.get("AGENT_API_KEY", "").strip(),
            rate_limit_per_minute=int(os.environ.get("RATE_LIMIT_PER_MINUTE", 10)),
            daily_call_cap=int(os.environ.get("DAILY_CALL_CAP", 200)),
            concurrency=max(1, int(os.environ.get("AGENT_CONCURRENCY", 1))),
            trust_proxy_headers=env_flag("TRUST_PROXY_HEADERS", False),
            log_path=os.environ.get("API_LOG_PATH", "results/runs/api/agent_log.jsonl"),
        )


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, description="A question about one of the 30 drug labels.")


class Citation(BaseModel):
    drug: str
    section: str
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    stop_reason: str = Field(description="answered | abstained | no_answer")
    n_steps: int
    model: str
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    model: str
    provider: str
    auth_required: bool
    calls_today: int
    daily_call_cap: int


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def create_app(*, llm=None, toolbox: ToolBox | None = None, settings: ApiSettings | None = None) -> FastAPI:
    """Build the app. Tests inject a mock ``llm`` and a stub ``toolbox``."""

    settings = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        box = toolbox or ToolBox()
        box.load()  # indexes + models, once
        model = llm or get_adapter()
        app.state.graph = AgentGraph(
            llm=model,
            toolbox=box,
            settings=settings_from_env(),
            log=JsonlLog(Path(settings.log_path)) if settings.log_path else None,
        )
        app.state.model = model
        app.state.box = box
        logger.info("agent ready: provider=%s model=%s", model.provider, model.model)
        yield

    app = FastAPI(title="FDA-Label-Agent", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.limiter = SlidingWindowLimiter(settings.rate_limit_per_minute, 60.0)
    app.state.daily = DailyCap(settings.daily_call_cap)
    app.state.slots = threading.BoundedSemaphore(settings.concurrency)

    def client_ip(request: Request) -> str:
        if settings.trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def require_key(request: Request) -> None:
        if not settings.api_key:
            return
        given = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(given.encode(), settings.api_key.encode()):
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    def rate_limit(request: Request) -> None:
        wait = app.state.limiter.check(client_ip(request))
        if wait is not None:
            raise HTTPException(
                status_code=429,
                detail=f"rate limit: {settings.rate_limit_per_minute} questions per minute",
                headers={"Retry-After": str(max(1, math.ceil(wait)))},
            )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        model = request.app.state.model
        return HealthResponse(
            status="ok",
            model=model.model,
            provider=model.provider,
            auth_required=bool(settings.api_key),
            calls_today=app.state.daily.used(),
            daily_call_cap=settings.daily_call_cap,
        )

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_key), Depends(rate_limit)])
    def ask(body: AskRequest, request: Request) -> AskResponse:
        if not app.state.daily.try_take():
            raise HTTPException(
                status_code=429,
                detail=f"daily limit of {settings.daily_call_cap} questions reached; resets at 00:00 UTC",
                headers={"Retry-After": str(DailyCap.seconds_until_reset())},
            )
        with app.state.slots:  # FastAPI runs sync endpoints in a thread pool
            result = request.app.state.graph.run(
                body.question.strip(), run_id="api", question_id=uuid.uuid4().hex[:12]
            )
        if result.stop_reason == "llm_error":
            quota = "quota" in result.answer.lower()
            logger.warning("agent run failed: %s", result.answer[:300])
            raise HTTPException(
                status_code=503,
                detail=(
                    "the language model's daily quota is used up; try again later"
                    if quota
                    else "the language model is unavailable right now; try again later"
                ),
            )
        return AskResponse(
            answer=result.answer,
            citations=[Citation(**citation) for citation in result.citations],
            stop_reason=result.stop_reason,
            n_steps=result.n_steps,
            model=result.model_version or result.model,
        )

    return app


app = create_app()
