"""Gemini embedding adapter.

Wraps the google-genai SDK to produce 3072-dim vectors
for semantic cache lookups.
"""

import os

import structlog

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = structlog.get_logger(__name__)


class EmbeddingError(Exception):
    pass


_client = None


def _init_client():
    global _client
    if genai is None:
        raise EmbeddingError("google-genai package is not installed.")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    _client = genai.Client(api_key=api_key)


def embed_text(text: str) -> list[float]:
    """Generate a 3072-dim embedding via Gemini (SEMANTIC_SIMILARITY task type).

    Args:
        text: The text to embed.

    Returns:
        List of 3072 floats representing the embedding vector.

    Raises:
        EmbeddingError: If the API key is missing or the API call fails.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EmbeddingError("GEMINI_API_KEY environment variable is not set.")

    if _client is None:
        _init_client()

    model_name = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")

    try:
        response = _client.models.embed_content(
            model=model_name,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        embedding = response.embeddings[0].values

        logger.debug("embed.ok", model=model_name, dims=len(embedding))
        return embedding

    except Exception as e:
        logger.error("embed.failed", error=str(e))
        raise EmbeddingError(f"Gemini API Error: {e}")
