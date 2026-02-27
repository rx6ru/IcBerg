"""Qdrant vector store manager.

Manages three collections (execution_cache, visualization_cache, chat_history),
provides semantic search with threshold-based hit/miss, and degrades gracefully
if the database is unreachable.
"""

import os
import uuid
from dataclasses import dataclass

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = structlog.get_logger(__name__)


class QdrantConnectionError(Exception):
    pass


@dataclass
class CacheResult:
    """Result of a semantic cache search.

    Attributes:
        hit: Whether a matching entry was found above the threshold.
        score: Cosine similarity score of the top match (0.0 if miss).
        payload: The stored payload dict, or None on miss.
    """
    hit: bool
    score: float = 0.0
    payload: dict | None = None


COLLECTIONS = ["execution_cache", "visualization_cache", "chat_history"]
VECTOR_SIZE = 3072


class QdrantManager:
    """Wraps QdrantClient with auto-collection setup and graceful degradation."""

    def __init__(self):
        self._url = os.environ.get("QDRANT_URL")
        self._api_key = os.environ.get("QDRANT_API_KEY")
        self._client = None

        if not self._url:
            logger.warning("qdrant.no_url")
            return

        try:
            self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=10)
            self._ensure_collections()
            logger.info("qdrant.connected")
        except Exception as e:
            logger.error("qdrant.init_failed", error=str(e))
            self._client = None

    def _ensure_collections(self):
        """Create missing collections on startup."""
        if not self._client:
            return

        for name in COLLECTIONS:
            if not self._client.collection_exists(name):
                logger.info("qdrant.create_collection", name=name)
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    ),
                )
                if name == "chat_history":
                    self._client.create_payload_index(
                        collection_name=name,
                        field_name="session_id",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )

    def is_healthy(self) -> bool:
        """Check if the Qdrant client connection is active.

        Returns:
            True if the client was successfully initialized.
        """
        return self._client is not None

    def search_cache(self, collection: str, embedding: list[float], threshold: float) -> CacheResult:
        """Semantic search — returns a hit if the top result's score >= threshold.

        Args:
            collection: Name of the Qdrant collection to search.
            embedding: 3072-dim query vector.
            threshold: Minimum cosine similarity to count as a hit.

        Returns:
            CacheResult with hit status, score, and payload.
        """
        if not self._client:
            return CacheResult(hit=False)

        try:
            response = self._client.query_points(
                collection_name=collection,
                query=embedding,
                limit=1,
            )
            results = response.points if response else []

            if not results:
                return CacheResult(hit=False)

            top = results[0]
            if top.score >= threshold:
                logger.info("cache.hit", collection=collection, score=top.score)
                return CacheResult(hit=True, score=top.score, payload=top.payload)

            return CacheResult(hit=False, score=top.score)

        except Exception as e:
            logger.error("cache.search_err", collection=collection, error=str(e))
            return CacheResult(hit=False)

    def upsert_cache(self, collection: str, embedding: list[float], payload: dict) -> None:
        """Store a new cache entry with a random UUID as point ID.

        Args:
            collection: Target collection name.
            embedding: 3072-dim vector for this entry.
            payload: JSON-serializable dict to store alongside the vector.
        """
        if not self._client:
            return

        point_id = str(uuid.uuid4())
        try:
            self._client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(id=point_id, vector=embedding, payload=payload)
                ],
            )
        except Exception as e:
            logger.error("cache.upsert_err", collection=collection, error=str(e))

    def search_history(self, session_id: str, embedding: list[float], limit: int = 5) -> list[dict]:
        """Find semantically similar past messages, filtered by session_id.

        Args:
            session_id: Chat session UUID used as a hard filter.
            embedding: Query vector for similarity search.
            limit: Maximum number of results to return.

        Returns:
            List of dicts with 'score' and 'payload' keys.
        """
        if not self._client:
            return []

        try:
            response = self._client.query_points(
                collection_name="chat_history",
                query=embedding,
                limit=limit,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=session_id),
                        )
                    ]
                ),
            )
            results = response.points if response else []
            return [{"score": r.score, "payload": r.payload} for r in results]

        except Exception as e:
            logger.error("history.search_err", session_id=session_id, error=str(e))
            return []
