"""Context assembly for agent invocation.

Fetches data from Qdrant (cache + semantic history) and SQLite (recent messages)
in parallel, then assembles a ContextBundle with token budgeting.

Uses a module-level thread pool (bounded at 20 workers) to prevent
thread exhaustion under concurrent request load.
"""

import atexit
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import structlog

from backend.api.schemas import MessageRecord
from backend.core.database import get_recent_messages
from backend.core.qdrant_manager import QdrantManager

logger = structlog.get_logger(__name__)

# Global thread pool — shared across all requests, bounded to prevent thread exhaustion.
_MAX_CONTEXT_WORKERS = int(os.environ.get("CONTEXT_POOL_MAX_WORKERS", "20"))
_pool = ThreadPoolExecutor(max_workers=_MAX_CONTEXT_WORKERS)
atexit.register(_pool.shutdown, wait=False)


@dataclass
class ContextBundle:
    """Assembled context for agent invocation.

    Attributes:
        schema_info: Dataset schema description.
        recent_messages: Last K messages from SQLite (chronological).
        semantic_messages: Top-K semantically similar past exchanges from Qdrant.
        cached_execution: Cached execution result dict, or None.
        cached_visualization: Cached base64 chart string, or None.
        cache_type_hit: Which cache was hit: "execution", "visualization", or "none".
    """
    schema_info: str = ""
    recent_messages: list[MessageRecord] = field(default_factory=list)
    semantic_messages: list[dict] = field(default_factory=list)
    cached_execution: dict | None = None
    cached_visualization: str | None = None
    cache_type_hit: str = "none"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def build_context(
    session_id: str,
    embedding: list[float],
    qdrant: QdrantManager,
    schema_info: str,
) -> ContextBundle:
    """Assemble context from cache, semantic history, and recent messages.

    Runs Qdrant lookups and SQLite fetch in parallel. Applies token budgeting
    to keep total context under 3000 tokens.

    Args:
        session_id: Current conversation session UUID.
        embedding: 3072-dim embedding of the user's message.
        qdrant: Initialized QdrantManager instance.
        schema_info: Pre-computed dataset schema string.

    Returns:
        Fully assembled ContextBundle.
    """
    recent_k = int(os.environ.get("HISTORY_RECENT_K", "4"))
    semantic_k = int(os.environ.get("HISTORY_SEMANTIC_K", "3"))
    exec_threshold = float(os.environ.get("EXECUTION_CACHE_THRESHOLD", "0.88"))
    viz_threshold = float(os.environ.get("VISUALIZATION_CACHE_THRESHOLD", "0.90"))

    bundle = ContextBundle(schema_info=schema_info)

    # Parallel lookups using global thread pool
    results = {}
    futures = {
        _pool.submit(_fetch_execution_cache, qdrant, embedding, exec_threshold): "exec_cache",
        _pool.submit(_fetch_visualization_cache, qdrant, embedding, viz_threshold): "viz_cache",
        _pool.submit(_fetch_semantic_history, qdrant, session_id, embedding, semantic_k): "semantic",
        _pool.submit(_fetch_recent_messages, session_id, recent_k): "recent",
    }
    for future in as_completed(futures):
        key = futures[future]
        try:
            results[key] = future.result()
        except Exception as e:
            logger.error("context.lookup_failed", key=key, error=str(e))
            results[key] = None

    # Unpack results
    exec_result = results.get("exec_cache")
    if exec_result and exec_result.hit:
        bundle.cached_execution = exec_result.payload
        bundle.cache_type_hit = "execution"
        logger.info("context.cache_hit", type="execution", score=exec_result.score)

    viz_result = results.get("viz_cache")
    if viz_result and viz_result.hit and bundle.cache_type_hit == "none":
        bundle.cached_visualization = viz_result.payload.get("image_base64") if viz_result.payload else None
        bundle.cache_type_hit = "visualization"
        logger.info("context.cache_hit", type="visualization", score=viz_result.score)

    bundle.recent_messages = results.get("recent") or []
    bundle.semantic_messages = results.get("semantic") or []

    # Token budgeting: if total context > 3000 tokens, reduce semantic history
    total_tokens = _estimate_context_tokens(bundle)
    if total_tokens > 3000 and len(bundle.semantic_messages) > 1:
        bundle.semantic_messages = bundle.semantic_messages[:1]
        logger.warning("context.token_budget_exceeded", original=total_tokens,
                       reduced_semantic_to=1)

    return bundle


def _estimate_context_tokens(bundle: ContextBundle) -> int:
    """Estimate total token count for the assembled context."""
    total = _estimate_tokens(bundle.schema_info)

    for msg in bundle.recent_messages:
        total += _estimate_tokens(msg.content)

    for msg in bundle.semantic_messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        total += _estimate_tokens(content)

    if bundle.cached_execution:
        total += _estimate_tokens(str(bundle.cached_execution))
    if bundle.cached_visualization:
        total += 50  # Flat estimate for "chart available" context injection

    return total


# Parallel fetch helpers (each handles its own errors gracefully)

def _fetch_execution_cache(qdrant, embedding, threshold):
    """Search execution cache in Qdrant."""
    try:
        return qdrant.search_cache("execution_cache", embedding, threshold)
    except Exception as e:
        logger.error("context.exec_cache_failed", error=str(e))
        return None


def _fetch_visualization_cache(qdrant, embedding, threshold):
    """Search visualization cache in Qdrant."""
    try:
        return qdrant.search_cache("visualization_cache", embedding, threshold)
    except Exception as e:
        logger.error("context.viz_cache_failed", error=str(e))
        return None


def _fetch_semantic_history(qdrant, session_id, embedding, limit):
    """Search semantically similar past exchanges."""
    try:
        return qdrant.search_history(session_id, embedding, limit=limit)
    except Exception as e:
        logger.error("context.semantic_history_failed", error=str(e))
        return []


def _fetch_recent_messages(session_id, limit):
    """Fetch recent messages from SQLite."""
    try:
        return get_recent_messages(session_id, limit=limit)
    except Exception as e:
        logger.error("context.recent_messages_failed", error=str(e))
        return []
