"""Unit tests for Pydantic API schemas."""

import pytest
from datetime import datetime
from pydantic import ValidationError
from backend.api.schemas import ChatRequest, ChatResponse, MessageRecord, HistoryResponse


class TestChatRequest:

    def test_valid_request(self):
        req = ChatRequest(session_id="abc-123", message="What is the survival rate?")
        assert req.session_id == "abc-123"
        assert req.message == "What is the survival rate?"

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="abc", message="")

    def test_empty_session_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="", message="hello")

    def test_message_max_length(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="abc", message="x" * 2001)

    def test_message_at_max_length(self):
        req = ChatRequest(session_id="abc", message="x" * 2000)
        assert len(req.message) == 2000


class TestChatResponse:

    def test_serialization(self):
        resp = ChatResponse(
            session_id="abc",
            text="The survival rate is 38.38%.",
            latency_ms=1200,
            tools_called=["query_data"],
        )
        data = resp.model_dump()
        assert data["text"] == "The survival rate is 38.38%."
        assert data["cached"] is False
        assert data["image_base64"] is None

    def test_with_image(self):
        resp = ChatResponse(
            session_id="abc", text="Here is the chart.", latency_ms=500,
            image_base64="iVBOR...", cached=False,
        )
        assert resp.image_base64 == "iVBOR..."


class TestMessageRecord:

    def test_valid_record(self):
        rec = MessageRecord(
            role="user", content="hello", timestamp=datetime.now()
        )
        assert rec.role == "user"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            MessageRecord(role="system", content="hello", timestamp=datetime.now())
