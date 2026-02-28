"""Contract tests for Gemini embeddings (mocked SDK)."""

import pytest

from backend.core.embeddings import EmbeddingError, embed_text


@pytest.fixture
def mock_genai(monkeypatch):
    """Swap out google.genai with a fake that returns 3072-dim vectors."""
    class MockEmbedResult:
        def __init__(self, values):
            self.values = values

    class MockEmbedResponse:
        def __init__(self, values):
            self.embeddings = [MockEmbedResult(values)]

    class MockModels:
        def embed_content(self, model, contents, config):
            if contents == "error":
                raise Exception("API Error")
            return MockEmbedResponse([0.1] * 3072)

    class MockClient:
        def __init__(self, api_key=None):
            self.models = MockModels()

    class MockEmbedContentConfig:
        def __init__(self, task_type):
            self.task_type = task_type

    class MockTypes:
        EmbedContentConfig = MockEmbedContentConfig

    import google.genai
    monkeypatch.setattr(google.genai, "Client", MockClient)
    monkeypatch.setattr(google.genai, "types", MockTypes())


def test_embed_returns_3072_floats(mock_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    result = embed_text("Hello world")
    assert isinstance(result, list)
    assert len(result) == 3072


def test_api_error_raises(mock_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(EmbeddingError, match="Gemini API Error"):
        embed_text("error")


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        embed_text("Hello")
