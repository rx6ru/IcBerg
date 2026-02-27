"""FastAPI application entry point.

Startup sequence: load data → init Qdrant → init LLM → create agent → init DB.
"""

import os
from contextlib import asynccontextmanager

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    # 1. Load and engineer the Titanic dataset
    csv_path = os.environ.get("TITANIC_CSV_PATH", "backend/data/titanic.csv")
    df = load_dataframe(csv_path)
    app.state.df = df
    app.state.schema_info = get_schema_metadata(df)
    logger.info("startup.data_loaded", rows=len(df), columns=len(df.columns))

    # 2. Initialize Qdrant
    qdrant = QdrantManager()
    app.state.qdrant = qdrant
    logger.info("startup.qdrant_initialized", healthy=qdrant.healthy)

    # 3. Initialize LLM adapter
    llm_adapter = LLMAdapter()
    app.state.llm_adapter = llm_adapter
    logger.info("startup.llm_initialized", healthy=llm_adapter.is_healthy())

    # 4. Create the LangGraph agent
    agent = create_agent(llm_adapter, df)
    app.state.agent = agent

    # 5. Initialize SQLite database
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

app.include_router(router)
