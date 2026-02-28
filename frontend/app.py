"""IcBerg - Streamlit Chat Interface.

Thin client for the Titanic dataset conversational analysis agent.
All business logic lives in the FastAPI backend. This file handles:
  - Session UUID management (st.session_state)
  - Chat history restoration from backend on page load
  - POST /chat requests with loading states
  - Base64 chart image rendering inline in chat
"""

import base64
import json
import os
import time
from uuid import uuid4

import requests
import streamlit as st

# Config
API_URL = os.environ.get("API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_URL}/chat"
CHAT_STREAM_ENDPOINT = f"{API_URL}/chat/stream"
HISTORY_ENDPOINT = f"{API_URL}/history"
HEALTH_ENDPOINT = f"{API_URL}/health"

# Page setup
st.set_page_config(
    page_title="IcBerg - Titanic Analysis Agent",
    layout="centered",
)

# Custom styles
st.markdown("""
<style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    .stChatMessage {
        padding: 0.75rem 1rem;
    }
    div[data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-ok { background: #d4edda; color: #155724; }
    .status-err { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)



def init_session():
    """Initialize session state on first load."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history_loaded" not in st.session_state:
        st.session_state.history_loaded = False
    if "api_online" not in st.session_state:
        st.session_state.api_online = False
        st.session_state.retry_count = 0
        # Proactive ping on first visit to wake up Render cold-starts
        try:
            requests.get(HEALTH_ENDPOINT, timeout=1)
        except requests.RequestException:
            pass


def restore_history():
    """Fetch chat history from backend and populate session state."""
    if st.session_state.history_loaded:
        return

    try:
        resp = requests.get(
            f"{HISTORY_ENDPOINT}/{st.session_state.session_id}",
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            for msg in data.get("messages", []):
                st.session_state.messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "image": msg.get("image_base64"),
                })
    except requests.RequestException:
        pass  # Gracefully degrade - start with empty history

    st.session_state.history_loaded = True


def render_message(msg: dict):
    """Render a single chat message with optional chart image."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
            try:
                image_bytes = base64.b64decode(msg["image"])
                st.image(image_bytes, caption="Generated Chart", use_container_width=True)
            except Exception:
                st.warning("[WARN] Could not decode chart image.")


def send_message(user_input: str):
    """POST the user message to the streaming backend and handle the response."""
    # Append user message immediately
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "image": None,
    })

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send to backend with live SSE parsing
    with st.chat_message("assistant"):
        with st.status("[IcBerg] Analyzing...", expanded=True) as status:
            try:
                start_time = time.monotonic()
                resp = requests.post(
                    CHAT_STREAM_ENDPOINT,
                    json={
                        "session_id": st.session_state.session_id,
                        "message": user_input,
                    },
                    timeout=60,
                    stream=True
                )

                if resp.status_code == 200:
                    final_text = "No response received."
                    image_base64 = None
                    tools_used = []

                    for line in resp.iter_lines():
                        if not line:
                            continue

                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            raw_data = decoded_line[6:]
                            try:
                                data = json.loads(raw_data)
                            except json.JSONDecodeError:
                                continue

                            event_type = data.get("type")

                            if event_type == "start":
                                status.update(label=data.get("content", "Analyzing..."))
                            elif event_type == "tool_start":
                                name = data.get('name', 'unknown')
                                tools_used.append(name)
                                status.write(f"[SYS] Running tool: **{name}**")
                            elif event_type == "tool_end":
                                status.write(f"[OK] Finished: **{data.get('name')}**")
                            elif event_type == "final_text":
                                final_text = data.get("content", "")
                            elif event_type == "image":
                                image_base64 = data.get("content")
                            elif event_type == "error":
                                status.update(label="Server Error", state="error")
                                final_text = data.get("content", "Service temporarily unavailable.")

                    # Calculate latency
                    latency_ms = int((time.monotonic() - start_time) * 1000)

                    # Update status header text to act as metadata footer
                    status.update(
                        label=f"[TIME] {latency_ms}ms | [TOOL] {', '.join(tools_used) if tools_used else 'No tools'}",
                        state="complete",
                        expanded=False
                    )
                elif resp.status_code == 422:
                    status.update(label="Validation Error", state="error")
                    st.error("[ERROR] Invalid request. Please check your message.")
                else:
                    status.update(label="Server Error", state="error")
                    st.error(f"[ERROR] Server error ({resp.status_code}). Please try again.")

            except requests.Timeout:
                status.update(label="Timeout", state="error")
                st.error("[TIMEOUT] Request timed out. The server may be overloaded.")
                return
            except requests.ConnectionError:
                status.update(label="Connection Error", state="error")
                st.error("[DISCONNECT] Cannot connect to the backend. Is the API server running?")
                return
            except Exception as e:
                status.update(label="Request Failed", state="error")
                st.error(f"[ERROR] Connection error: {str(e)}")
                return

        # Render response outside the status box (Thinking component)
        if resp and resp.status_code == 200:
            st.markdown(final_text)

            if image_base64:
                try:
                    image_bytes = base64.b64decode(image_base64)
                    st.image(image_bytes, caption="Generated Chart", use_container_width=True)
                except Exception:
                    st.warning("[WARN] Could not decode chart image.")

            # Save to session
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_text,
                "image": image_base64,
            })
        elif resp and resp.status_code == 422:
            st.error("[ERROR] Invalid request. Please check your message.")
        elif resp: # For other non-200, non-422 status codes
            st.error(f"[ERROR] Server error ({resp.status_code}). Please try again.")



def main():
    """Run the Streamlit chat application."""
    init_session()
    restore_history()

    # 1. Perform Health Check immediately to set state for the rest of the page
    api_status = "unknown"
    try:
        # Cache-busting health check
        health_resp = requests.get(f"{HEALTH_ENDPOINT}?t={time.time()}", timeout=3)
        health = health_resp.json()
        api_status = health.get("status", "unknown")
        st.session_state.api_online = (api_status == "healthy")
    except requests.RequestException:
        st.session_state.api_online = False
        api_status = "offline"

    # Header
    st.title("IcBerg")
    st.caption("Conversational analysis of the Titanic passenger dataset")

    # 2. Show/Hide warning based on immediate health check
    if not st.session_state.api_online:
        st.warning("[WARN] The IcBerg API is currently offline or waking up. Please wait ~30s if you are seeing this for the first time.")

    # Sidebar
    with st.sidebar:
        st.markdown("### Session Info")
        st.code(st.session_state.session_id, language=None)

        # 3. Display Health Status in sidebar using the pre-calculated state
        with st.container():
            if st.session_state.api_online:
                st.markdown('<span class="status-badge status-ok">* API Healthy</span>',
                            unsafe_allow_html=True)
                st.session_state.retry_count = 0  # Reset on success
            elif api_status == "degraded":
                st.markdown('<span class="status-badge status-err">* API Degraded</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-err">* API Offline</span>',
                            unsafe_allow_html=True)
                st.info("[INFO] Backend may be sleeping (Render Free Tier).")

                # Sustainable Auto-ping logic: cap at 6 attempts (~1 min)
                if "retry_count" not in st.session_state:
                    st.session_state.retry_count = 0

                if st.session_state.retry_count < 6:
                    if "last_ping" not in st.session_state or time.time() - st.session_state.last_ping > 12:
                        st.session_state.last_ping = time.time()
                        st.session_state.retry_count += 1
                        with st.status(f"[SYS] Waking up backend (Attempt {st.session_state.retry_count}/6)...", expanded=False):
                            st.write("Sending wake-up ping...")
                            st.write("Waiting 10 seconds for cold-start...")
                            time.sleep(10)
                        st.rerun()
                else:
                    st.error("[ERROR] Backend failed to wake up after multiple attempts. Please check the logs or try manual refresh.")

        if st.button("Check Connection", use_container_width=True):
            st.rerun()

        st.divider()
        st.markdown("### About")
        st.markdown(
            "IcBerg uses a LangGraph ReAct agent to answer questions about "
            "the Titanic dataset. It can query data, compute statistics, "
            "and generate charts."
        )

        st.divider()
        if st.button("[DEL] New Session", use_container_width=True):
            st.session_state.session_id = str(uuid4())
            st.session_state.messages = []
            st.session_state.history_loaded = False
            st.rerun()

    # Render existing messages
    for msg in st.session_state.messages:
        render_message(msg)

    # Chat input
    if user_input := st.chat_input("Ask a question about the Titanic dataset..."):
        send_message(user_input)


if __name__ == "__main__":
    main()
