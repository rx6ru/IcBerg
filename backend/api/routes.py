"""FastAPI endpoints for the IcBerg API.

POST /chat - process a user message through the agent
GET /history/{session_id} - fetch conversation history
GET /health - component health check
"""

import json
import time
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
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
    """Process a user message: embed -> context -> agent -> persist -> respond."""
    start = time.monotonic()
    session_id = request.session_id
    message = request.message

    logger.info("chat.request", session_id=session_id, msg_len=len(message))

    agent = req.app.state.agent
    qdrant = req.app.state.qdrant
    schema_info = req.app.state.schema_info

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available. Configure LLM API keys in .env and restart.")

    # Input guardrail - block prompt injection before any processing
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


@router.post("/chat/stream")
def chat_stream(request: ChatRequest, req: Request):
    """Process a user message and stream agent intermediate steps via SSE."""
    start = time.monotonic()
    session_id = request.session_id
    message = request.message

    logger.info("chat_stream.request", session_id=session_id, msg_len=len(message))

    agent = req.app.state.agent
    qdrant = req.app.state.qdrant
    schema_info = req.app.state.schema_info

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available. Configure LLM API keys in .env and restart.")

    # Input guardrail
    guard_result = _input_guard.check(message)
    if not guard_result.passed:
        def err_stream():
            yield f"data: {json.dumps({'type': 'final_text', 'content': 'I can only help with Titanic dataset analysis. Please rephrase your question.'})}\n\n"
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    # Embed the message
    embedding = None
    try:
        embedding = embed_text(message)
    except EmbeddingError as e:
        logger.error("chat_stream.embedding_failed", error=str(e))

    # Build context
    context = None
    if embedding:
        try:
            context = build_context(session_id, embedding, qdrant, schema_info)
        except Exception as e:
            logger.error("chat_stream.context_failed", error=str(e))

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

    def generate_events():
        yield f"data: {json.dumps({'type': 'start', 'content': 'Analyzing query...'})}\n\n"
        
        try:
            messages = []
            if cache_context != "No cached data available.":
                messages.append(SystemMessage(content=cache_context))

            if context and context.recent_messages:
                for msg in context.recent_messages:
                    if msg.role == "user":
                        messages.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        messages.append(AIMessage(content=msg.content))

            messages.append(HumanMessage(content=message))
            agent_input = {"messages": messages}
            config = {"recursion_limit": 10}

            final_ai_text = ""
            image_base64 = None
            tools_called = []
            trace_steps = []

            # Stream updates from the agent's internal thought loops
            for chunk in agent.stream(agent_input, config=config, stream_mode="updates"):
                # Agent node finished
                if "agent" in chunk:
                    msg = chunk["agent"]["messages"][0]
                    # Agent requested tools
                    if getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            name = tc.get("name", "unknown")
                            tools_called.append(name)
                            trace_steps.append({"type": "tool_call", "tool": name, "input": tc.get("args", {})})
                            yield f"data: {json.dumps({'type': 'tool_start', 'name': name})}\n\n"
                    # Agent sent final text
                    elif msg.content:
                        final_ai_text = msg.content

                # Tools node finished
                elif "tools" in chunk:
                    for msg in chunk["tools"]["messages"]:
                        name = getattr(msg, "name", "unknown")
                        content = msg.content or ""
                        trace_steps.append({"type": "tool_result", "tool": name, "output_preview": content[:200]})
                        
                        if content.startswith("BASE64:") and not image_base64:
                            image_base64 = content[7:]
                        
                        yield f"data: {json.dumps({'type': 'tool_end', 'name': name})}\n\n"

            # ---------------------------------------------------------
            # Post-processing (Guardrails & Persistence)
            # ---------------------------------------------------------
            if not final_ai_text or not final_ai_text.strip():
                final_ai_text = "I wasn't able to generate a response. Try rephrasing your question."
            if "Traceback (most recent call last)" in final_ai_text:
                final_ai_text = "An internal error occurred. Please try again."

            validated_text, guard_result = _output_guard.check(final_ai_text)

            # Strip markdown images from final text
            import re
            validated_text = re.sub(r"!\[[^\]]*\]\(\s*(?:data:image/[^\)]*|BASE64:[^\)]*)\)?", "", validated_text)
            validated_text = re.sub(r"(?:data:image/\S+|BASE64:\S+)", "", validated_text)
            validated_text = re.sub(r"!\[[^\]]*\]\s*\(\s*$", "", validated_text)
            validated_text = validated_text.strip()

            trace = {"steps": trace_steps, "tools_called": tools_called}

            try:
                save_message(session_id, "user", message)
                save_message(session_id, "assistant", validated_text, image_base64=image_base64, agent_trace=trace)
            except Exception as e:
                logger.error("chat_stream.persist_failed", error=str(e))

            if embedding and qdrant:
                try:
                    qdrant.upsert_cache("chat_history", embedding, {
                        "session_id": session_id,
                        "content": f"User: {message}\nAssistant: {validated_text}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    if context and context.cache_type_hit == "none":
                        if image_base64:
                            qdrant.upsert_cache("visualization_cache", embedding, {"image_base64": image_base64, "query": message})
                        elif tools_called:
                            qdrant.upsert_cache("execution_cache", embedding, {"result": validated_text, "query": message})
                except Exception as e:
                    logger.error("chat_stream.cache_upsert_failed", error=str(e))

            latency_ms = int((time.monotonic() - start) * 1000)
            cached = context.cache_type_hit != "none" if context else False
            logger.info("chat_stream.response", session_id=session_id, latency_ms=latency_ms, cached=cached, tools=tools_called)

            # Yield final text and optional image
            yield f"data: {json.dumps({'type': 'final_text', 'content': validated_text})}\n\n"
            if image_base64:
                yield f"data: {json.dumps({'type': 'image', 'content': image_base64})}\n\n"

        except Exception as e:
            logger.error("chat_stream.agent_failed", error=str(e))
            yield f"data: {json.dumps({'type': 'final_text', 'content': 'Service temporarily unavailable. Please try again in a moment.'})}\n\n"

    return StreamingResponse(generate_events(), media_type="text/event-stream")


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


@router.get("/")
@router.head("/")
def root_health():
    """Basic root health check for deployment platforms like Render."""
    return {"status": "ok", "service": "icberg-api"}


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

    # Strip inline base64/data image references - the image is rendered separately
    import re
    # Match ![...](data:image... or ![...](BASE64... even if unclosed
    text = re.sub(r"!\[[^\]]*\]\(\s*(?:data:image/[^\)]*|BASE64:[^\)]*)\)?", "", text)
    # Match any remaining raw base64 data strings
    text = re.sub(r"(?:data:image/\S+|BASE64:\S+)", "", text)
    # Strip any dangling/empty image markdown left over (e.g. `![Age Distribution](`)
    text = re.sub(r"!\[[^\]]*\]\s*\(\s*$", "", text)
    text = text.strip()

    trace = {"steps": trace_steps, "tools_called": tools_called}
    return text, image_base64, tools_called, trace
