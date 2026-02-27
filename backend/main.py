"""FastAPI application entry point.

Startup sequence: load data → init Qdrant → init LLM → create agent → init DB.
"""

import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from backend.agent.agent import create_agent
from backend.api.routes import router
from backend.core.database import init_db
from backend.core.llm_adapter import LLMAdapter
from backend.core.qdrant_manager import QdrantManager
from backend.data.loader import get_schema_metadata, load_dataframe

load_dotenv()

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("startup.begin")

    # Load and engineer the Titanic dataset
    csv_path = os.environ.get("TITANIC_CSV_PATH", "backend/data/titanic.csv")
    df = load_dataframe(csv_path)
    app.state.df = df
    app.state.schema_info = get_schema_metadata(df)
    logger.info("startup.data_loaded", rows=len(df), columns=len(df.columns))

    # Initialize Qdrant
    qdrant = QdrantManager()
    app.state.qdrant = qdrant
    logger.info("startup.qdrant_initialized", healthy=qdrant.is_healthy())

    # Initialize LLM adapter
    llm_adapter = LLMAdapter()
    app.state.llm_adapter = llm_adapter
    logger.info("startup.llm_initialized", healthy=llm_adapter.is_healthy())

    # Create the LangGraph agent (may fail if no LLM keys configured)
    try:
        agent = create_agent(llm_adapter, df)
        app.state.agent = agent
        logger.info("startup.agent_created")
    except Exception as e:
        app.state.agent = None
        logger.error("startup.agent_failed", error=str(e),
                     hint="Set CEREBRAS_API_KEY or GROQ_API_KEY in .env")

    # Initialize SQLite database
    init_db()
    logger.info("startup.db_initialized")

    logger.info("startup.complete")
    yield
    logger.info("shutdown.complete")


app = FastAPI(
    title="IcBerg API",
    description="Titanic Dataset Conversational Analysis Agent",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter: per-session request throttling
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "10"))
_rate_buckets: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce per-session rate limiting on /chat."""
    if request.url.path != "/chat" or request.method != "POST":
        return await call_next(request)

    # Read request body to extract session_id
    body = await request.body()
    try:
        import json
        data = json.loads(body)
        session_id = data.get("session_id", "unknown")
    except Exception:
        session_id = "unknown"

    now = time.monotonic()
    window = _rate_buckets[session_id]

    # Prune timestamps older than 60s
    _rate_buckets[session_id] = [t for t in window if now - t < 60]
    window = _rate_buckets[session_id]

    if len(window) >= RATE_LIMIT:
        logger.warning("rate_limit.exceeded", session_id=session_id)
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please wait a moment."},
        )

    window.append(now)

    # Reconstruct the request with the already-read body
    from starlette.requests import Request as StarletteRequest
    scope = request.scope
    receive = request.receive

    async def receive_body():
        return {"type": "http.request", "body": body}

    request = StarletteRequest(scope, receive_body)
    return await call_next(request)


app.include_router(router)
