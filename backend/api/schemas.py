"""Pydantic models for the API layer.

Defines request/response schemas for all endpoints.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message from the frontend."""
    session_id: str = Field(..., min_length=1, description="UUID4 session identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User question")


class MessageRecord(BaseModel):
    """Single message in a conversation history."""
    role: Literal["user", "assistant"]
    content: str
    image_base64: str | None = None
    timestamp: datetime


class ChatResponse(BaseModel):
    """Outgoing response to the frontend."""
    session_id: str
    text: str
    image_base64: str | None = None
    cached: bool = False
    latency_ms: int
    tools_called: list[str] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """Full conversation history for a session."""
    session_id: str
    messages: list[MessageRecord]
