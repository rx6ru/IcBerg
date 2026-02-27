"""FastAPI endpoints for the IcBerg API.

POST /chat — process a user message through the agent
GET /history/{session_id} — fetch conversation history
GET /health — component health check
"""

import time
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.agent.prompts import build_system_prompt
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    MessageRecord,
)
from backend.core.context_builder import build_context
from backend.core.database import get_session_history, save_message
from backend.core.embeddings import EmbeddingError, embed_text
from backend.core.guardrails import InputGuard, OutputGuard

logger = structlog.get_logger(__name__)

router = APIRouter()
_input_guard = InputGuard()
_output_guard = OutputGuard()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, req: Request):
    """Process a user message: embed → context → agent → persist → respond."""
    start = time.monotonic()
    session_id = request.session_id
    message = request.message

    logger.info("chat.request", session_id=session_id, msg_len=len(message))

    agent = req.app.state.agent
    qdrant = req.app.state.qdrant
    schema_info = req.app.state.schema_info

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available. Configure LLM API keys in .env and restart.")

    # Input guardrail — block prompt injection before any processing
    guard_result = _input_guard.check(message)
    if not guard_result.passed:
        latency_ms = int((time.monotonic() - start) * 1000)
        return ChatResponse(
            session_id=session_id,
            text="I can only help with Titanic dataset analysis. Please rephrase your question.",
            guardrail_triggered=True,
            latency_ms=latency_ms,
        )

    # Embed the message for cache lookup
    embedding = None
    try:
        embedding = embed_text(message)
    except EmbeddingError as e:
        logger.error("chat.embedding_failed", error=str(e))

    # Build context (cache + history)
    context = None
    if embedding:
        try:
            context = build_context(session_id, embedding, qdrant, schema_info)
        except Exception as e:
            logger.error("chat.context_failed", error=str(e))

    # Build cache-aware prompt
    cache_context = "No cached data available."
    if context and context.cache_type_hit == "execution":
        cache_context = (
            f"CACHED DATA: {context.cached_execution}\n"
            "CRITICAL: Compare the cached 'query' to the user's CURRENT question. "
            "If demographics, columns, filters, or conditions differ in ANY way, "
            "IGNORE this cache entirely and use query_data to compute fresh results."
        )
    elif context and context.cache_type_hit == "visualization":
        cache_context = (
            "A CHART was previously generated for a similar query. "
            "If the user's current question matches the cached chart's intent exactly, "
            "explain it and skip visualize_data. Otherwise, generate a new chart."
        )

    # Invoke the agent with conversation history for multi-turn memory
    try:
        messages = []
        if cache_context != "No cached data available.":
            messages.append(SystemMessage(content=cache_context))

        # Inject recent conversation history so the agent remembers prior turns
        if context and context.recent_messages:
            for msg in context.recent_messages:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))

        messages.append(HumanMessage(content=message))
        agent_input = {"messages": messages}
        config = {"recursion_limit": 10}
        result = agent.invoke(agent_input, config=config)
    except Exception as e:
        logger.error("chat.agent_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service temporarily unavailable. Please try again in a moment.")

    # Extract response and run output guardrail
    text, image_base64, tools_called, trace = _extract_response(result)
    text, output_result = _output_guard.check(text)
    guardrail_triggered = not output_result.passed

    # Persist messages
    try:
        save_message(session_id, "user", message)
        save_message(session_id, "assistant", text, image_base64=image_base64, agent_trace=trace)
    except Exception as e:
        logger.error("chat.persist_failed", error=str(e))

    # Upsert to Qdrant caches
    if embedding and qdrant:
        try:
            # Always upsert to chat_history
            qdrant.upsert_cache("chat_history", embedding, {
                "session_id": session_id,
                "content": f"User: {message}\nAssistant: {text}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Upsert execution/visualization cache only if fresh computation
            if context and context.cache_type_hit == "none":
                if image_base64:
                    qdrant.upsert_cache("visualization_cache", embedding, {
                        "image_base64": image_base64,
                        "query": message,
                    })
                elif tools_called:
                    qdrant.upsert_cache("execution_cache", embedding, {
                        "result": text,
                        "query": message,
                    })
        except Exception as e:
            logger.error("chat.cache_upsert_failed", error=str(e))

    latency_ms = int((time.monotonic() - start) * 1000)
    cached = context.cache_type_hit != "none" if context else False

    logger.info("chat.response", session_id=session_id, latency_ms=latency_ms,
                cached=cached, tools=tools_called)

    return ChatResponse(
        session_id=session_id,
        text=text,
        image_base64=image_base64,
        cached=cached,
        guardrail_triggered=guardrail_triggered,
        latency_ms=latency_ms,
        tools_called=tools_called,
    )


@router.get("/history/{session_id}", response_model=HistoryResponse)
def history(session_id: str):
    """Fetch full conversation history for a session."""
    messages = get_session_history(session_id)
    return HistoryResponse(session_id=session_id, messages=messages)


@router.get("/health")
def health(req: Request):
    """Check health of all backend components."""
    components = {}

    # LLM adapter
    llm = req.app.state.llm_adapter
    components["cerebras"] = "ok" if llm.cerebras_key else "error"
    components["groq"] = "ok" if llm.groq_key else "error"

    # Qdrant
    qdrant = req.app.state.qdrant
    components["qdrant"] = "ok" if qdrant.is_healthy() else "error"

    # SQLite
    try:
        from backend.core.database import get_session
        get_session()
        components["sqlite"] = "ok"
    except Exception:
        components["sqlite"] = "error"

    # Overall status
    errors = [k for k, v in components.items() if v == "error"]
    if not errors:
        status = "healthy"
    elif len(errors) == len(components):
        status = "unhealthy"
    else:
        status = "degraded"

    return {"status": status, "components": components}


def _extract_response(result: dict) -> tuple[str, str | None, list[str], dict]:
    """Extract text, image, tool names, and trace from agent result.

    Args:
        result: LangGraph agent invoke result (dict with 'messages' key).

    Returns:
        Tuple of (text, image_base64, tools_called, trace_dict).
    """
    messages = result.get("messages", [])
    text = ""
    image_base64 = None
    tools_called = []
    trace_steps = []

    for msg in messages:
        msg_type = getattr(msg, "type", "")

        if msg_type == "ai" and not getattr(msg, "tool_calls", None):
            # Final AI response (no tool calls = it's the answer)
            text = msg.content or ""

        elif msg_type == "ai" and getattr(msg, "tool_calls", None):
            # AI decided to call tools
            for tc in msg.tool_calls:
                tools_called.append(tc.get("name", "unknown"))
                trace_steps.append({
                    "type": "tool_call",
                    "tool": tc.get("name"),
                    "input": tc.get("args", {}),
                })

        elif msg_type == "tool":
            # Tool result
            content = msg.content or ""
            trace_steps.append({
                "type": "tool_result",
                "tool": getattr(msg, "name", ""),
                "output_preview": content[:200],
            })

            # Extract base64 image if present
            if content.startswith("BASE64:") and not image_base64:
                image_base64 = content[7:]  # Strip "BASE64:" prefix

    # Validate response per FAILURE_HANDLING.md
    if not text or not text.strip():
        text = "I wasn't able to generate a response. Try rephrasing your question."
    if "Traceback (most recent call last)" in text:
        text = "An internal error occurred. Please try again."

    trace = {"steps": trace_steps, "tools_called": tools_called}
    return text, image_base64, tools_called, trace
