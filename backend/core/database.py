"""SQLAlchemy + SQLite persistence for chat messages and agent traces.

Provides CRUD operations for message storage and session history retrieval.
"""

import json
import os
from datetime import datetime, timezone

import structlog
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from backend.api.schemas import MessageRecord

logger = structlog.get_logger(__name__)

Base = declarative_base()


class Message(Base):
    """Persistent chat message row."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    image_base64 = Column(Text, nullable=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    agent_trace = Column(Text, nullable=True)  # JSON string


_engine = None
_SessionLocal = None


def init_db(database_url: str | None = None) -> None:
    """Create engine + tables. Call once at startup.

    Args:
        database_url: SQLAlchemy connection string. Defaults to DATABASE_URL env var.
    """
    global _engine, _SessionLocal

    url = database_url or os.environ.get("DATABASE_URL", "sqlite:///data/icberg.sqlite")
    _engine = create_engine(url, echo=False)
    _SessionLocal = sessionmaker(bind=_engine)

    Base.metadata.create_all(_engine)
    logger.info("db.initialized", url=url.split("///")[0] + "///***")


def get_session() -> Session:
    """Get a new database session."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal()


def save_message(
    session_id: str,
    role: str,
    content: str,
    image_base64: str | None = None,
    agent_trace: dict | None = None,
) -> None:
    """Persist a single message to SQLite.

    Args:
        session_id: UUID4 conversation session.
        role: "user" or "assistant".
        content: Message text content.
        image_base64: Optional base64-encoded chart image.
        agent_trace: Optional agent trace dict (stored as JSON string).
    """
    with get_session() as session:
        msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            image_base64=image_base64,
            timestamp=datetime.now(timezone.utc),
            agent_trace=json.dumps(agent_trace) if agent_trace else None,
        )
        session.add(msg)
        session.commit()
        logger.debug("db.message_saved", session_id=session_id, role=role)


def get_recent_messages(session_id: str, limit: int = 4) -> list[MessageRecord]:
    """Fetch the most recent messages for a session.

    Args:
        session_id: UUID4 conversation session.
        limit: Max number of messages to return (default 4).

    Returns:
        List of MessageRecord ordered oldest-first.
    """
    with get_session() as session:
        rows = (
            session.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.timestamp.desc())
            .limit(limit)
            .all()
        )
        # Reverse to get chronological order (oldest first)
        rows.reverse()
        return [_row_to_record(r) for r in rows]


def get_session_history(session_id: str) -> list[MessageRecord]:
    """Fetch full conversation history for a session.

    Args:
        session_id: UUID4 conversation session.

    Returns:
        List of MessageRecord ordered chronologically.
    """
    with get_session() as session:
        rows = (
            session.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.timestamp.asc())
            .all()
        )
        return [_row_to_record(r) for r in rows]


def _row_to_record(row: Message) -> MessageRecord:
    """Convert a SQLAlchemy row to a Pydantic MessageRecord."""
    return MessageRecord(
        role=row.role,
        content=row.content,
        image_base64=row.image_base64,
        timestamp=row.timestamp,
    )
