"""Unit tests for SQLite database operations (in-memory)."""

import pytest
from datetime import datetime, timezone
from backend.core.database import init_db, save_message, get_recent_messages, get_session_history


@pytest.fixture(autouse=True)
def setup_db():
    """Create a fresh in-memory database for each test."""
    init_db("sqlite:///:memory:")
    yield


class TestSaveAndRetrieve:

    def test_save_and_get_history(self):
        save_message("s1", "user", "hello")
        save_message("s1", "assistant", "hi there")
        messages = get_session_history("s1")
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_messages_have_timestamps(self):
        save_message("s1", "user", "test")
        messages = get_session_history("s1")
        assert messages[0].timestamp is not None

    def test_image_persisted(self):
        save_message("s1", "assistant", "chart", image_base64="abc123")
        messages = get_session_history("s1")
        assert messages[0].image_base64 == "abc123"


class TestRecentMessages:

    def test_respects_limit(self):
        for i in range(10):
            save_message("s1", "user", f"msg {i}")
        recent = get_recent_messages("s1", limit=4)
        assert len(recent) == 4

    def test_returns_most_recent(self):
        for i in range(6):
            save_message("s1", "user", f"msg {i}")
        recent = get_recent_messages("s1", limit=3)
        assert recent[-1].content == "msg 5"
        assert recent[0].content == "msg 3"

    def test_chronological_order(self):
        save_message("s1", "user", "first")
        save_message("s1", "assistant", "second")
        recent = get_recent_messages("s1", limit=4)
        assert recent[0].content == "first"
        assert recent[1].content == "second"


class TestSessionIsolation:

    def test_sessions_isolated(self):
        save_message("s1", "user", "session 1")
        save_message("s2", "user", "session 2")
        assert len(get_session_history("s1")) == 1
        assert len(get_session_history("s2")) == 1

    def test_empty_session(self):
        assert len(get_session_history("nonexistent")) == 0
