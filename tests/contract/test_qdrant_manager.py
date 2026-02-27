"""Contract tests for the Qdrant vector store manager (mocked client)."""

import pytest
from backend.core.qdrant_manager import QdrantManager, CacheResult


@pytest.fixture
def mock_qdrant(mocker):
    mock_client = mocker.patch("backend.core.qdrant_manager.QdrantClient")
    instance = mocker.MagicMock()
    mock_client.return_value = instance
    instance.collection_exists.return_value = True
    return instance


@pytest.fixture
def manager(mock_qdrant, monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://test-qdrant")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    return QdrantManager()


class TestInit:

    def test_creates_missing_collections(self, mock_qdrant, monkeypatch):
        mock_qdrant.collection_exists.return_value = False
        monkeypatch.setenv("QDRANT_URL", "http://test-qdrant")
        QdrantManager()
        assert mock_qdrant.create_collection.call_count == 3

    def test_connection_failure_degrades_gracefully(self, mocker, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://bad-url")
        mocker.patch("backend.core.qdrant_manager.QdrantClient", side_effect=Exception("refused"))
        mgr = QdrantManager()
        assert not mgr.is_healthy()


class TestSearchCache:

    def test_miss_below_threshold(self, manager, mock_qdrant):
        from qdrant_client.http.models import ScoredPoint
        mock_qdrant.search.return_value = [
            ScoredPoint(id=1, version=1, score=0.5, payload={"result": "data"}, vector=None)
        ]
        result = manager.search_cache("execution_cache", [0.1] * 3072, threshold=0.88)
        assert not result.hit
        assert result.score == 0.5

    def test_hit_above_threshold(self, manager, mock_qdrant):
        from qdrant_client.http.models import ScoredPoint
        mock_qdrant.search.return_value = [
            ScoredPoint(id=1, version=1, score=0.95, payload={"result": "data"}, vector=None)
        ]
        result = manager.search_cache("execution_cache", [0.1] * 3072, threshold=0.88)
        assert result.hit
        assert result.payload == {"result": "data"}
        assert result.score == 0.95

    def test_empty_results(self, manager, mock_qdrant):
        mock_qdrant.search.return_value = []
        assert not manager.search_cache("execution_cache", [0.1] * 3072, 0.88).hit


class TestUpsertCache:

    def test_upsert_calls_qdrant(self, manager, mock_qdrant):
        manager.upsert_cache("execution_cache", [0.1] * 3072, {"query": "test"})
        mock_qdrant.upsert.assert_called_once()
        kwargs = mock_qdrant.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "execution_cache"
        assert len(kwargs["points"]) == 1


class TestSearchHistory:

    def test_filters_by_session(self, manager, mock_qdrant):
        from qdrant_client.http.models import ScoredPoint
        mock_qdrant.search.return_value = [
            ScoredPoint(id=1, version=1, score=0.8, payload={"role": "user", "content": "hi"}, vector=None)
        ]
        results = manager.search_history("session-123", [0.1] * 3072, limit=5)
        assert len(results) == 1

        # verify the session_id filter was passed
        kwargs = mock_qdrant.search.call_args.kwargs
        assert kwargs["query_filter"].must[0].key == "session_id"
        assert kwargs["query_filter"].must[0].match.value == "session-123"

    def test_unhealthy_returns_empty(self, manager):
        manager._client = None
        result = manager.search_cache("execution_cache", [0.1] * 3072, 0.88)
        assert not result.hit
