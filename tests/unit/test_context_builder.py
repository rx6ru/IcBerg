"""Unit tests for context assembly."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from backend.api.schemas import MessageRecord
from backend.core.context_builder import _estimate_tokens, build_context
from backend.core.qdrant_manager import CacheResult


@pytest.fixture
def mock_qdrant():
    qdrant = MagicMock()
    qdrant.healthy = True
    qdrant.search_cache.return_value = CacheResult(hit=False, payload=None, score=0.0)
    qdrant.search_history.return_value = []
    return qdrant


@pytest.fixture
def mock_embedding():
    return [0.1] * 3072


class TestBuildContext:

    @patch("backend.core.context_builder.get_recent_messages", return_value=[])
    def test_no_cache_hit(self, mock_recent, mock_qdrant, mock_embedding):
        bundle = build_context("s1", mock_embedding, mock_qdrant, "schema info")
        assert bundle.cache_type_hit == "none"
        assert bundle.cached_execution is None
        assert bundle.cached_visualization is None

    @patch("backend.core.context_builder.get_recent_messages", return_value=[])
    def test_execution_cache_hit(self, mock_recent, mock_qdrant, mock_embedding):
        mock_qdrant.search_cache.side_effect = [
            CacheResult(hit=True, payload={"result": "42"}, score=0.92),
            CacheResult(hit=False, payload=None, score=0.5),
        ]
        bundle = build_context("s1", mock_embedding, mock_qdrant, "schema")
        assert bundle.cache_type_hit == "execution"
        assert bundle.cached_execution == {"result": "42"}

    @patch("backend.core.context_builder.get_recent_messages", return_value=[])
    def test_visualization_cache_hit(self, mock_recent, mock_qdrant, mock_embedding):
        mock_qdrant.search_cache.side_effect = [
            CacheResult(hit=False, payload=None, score=0.5),
            CacheResult(hit=True, payload={"image_base64": "abc"}, score=0.95),
        ]
        bundle = build_context("s1", mock_embedding, mock_qdrant, "schema")
        assert bundle.cache_type_hit == "visualization"
        assert bundle.cached_visualization == "abc"

    @patch("backend.core.context_builder.get_recent_messages")
    def test_recent_messages_included(self, mock_recent, mock_qdrant, mock_embedding):
        records = [
            MessageRecord(role="user", content="hello", timestamp=datetime.now()),
            MessageRecord(role="assistant", content="hi", timestamp=datetime.now()),
        ]
        mock_recent.return_value = records
        bundle = build_context("s1", mock_embedding, mock_qdrant, "schema")
        assert len(bundle.recent_messages) == 2

    @patch("backend.core.context_builder.get_recent_messages", return_value=[])
    def test_qdrant_failure_graceful(self, mock_recent, mock_qdrant, mock_embedding):
        mock_qdrant.search_cache.side_effect = Exception("connection refused")
        mock_qdrant.search_history.side_effect = Exception("connection refused")
        bundle = build_context("s1", mock_embedding, mock_qdrant, "schema")
        assert bundle.cache_type_hit == "none"
        assert bundle.semantic_messages == []


class TestTokenBudgeting:

    def test_estimate_tokens(self):
        assert _estimate_tokens("hello world") == 2  # 11 chars // 4

    @patch("backend.core.context_builder.get_recent_messages", return_value=[])
    def test_semantic_reduction_on_overflow(self, mock_recent, mock_qdrant, mock_embedding):
        # Fill semantic history with lots of content to exceed 3000 tokens
        mock_qdrant.search_history.return_value = [
            {"content": "x" * 4000},
            {"content": "y" * 4000},
            {"content": "z" * 4000},
        ]
        bundle = build_context("s1", mock_embedding, mock_qdrant, "a" * 400)
        # Should have been reduced from 3 to 1
        assert len(bundle.semantic_messages) == 1
