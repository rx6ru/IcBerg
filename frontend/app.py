"""IcBerg — Streamlit Chat Interface.

Thin client for the Titanic dataset conversational analysis agent.
All business logic lives in the FastAPI backend. This file handles:
  - Session UUID management (st.session_state)
  - Chat history restoration from backend on page load
  - POST /chat requests with loading states
  - Base64 chart image rendering inline in chat
"""

import base64
import os
from uuid import uuid4

import requests
import streamlit as st

# Config
API_URL = os.environ.get("API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_URL}/chat"
HISTORY_ENDPOINT = f"{API_URL}/history"
HEALTH_ENDPOINT = f"{API_URL}/health"

# Page setup
st.set_page_config(
    page_title="IcBerg — Titanic Analysis Agent",
    page_icon="🧊",
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
        pass  # Gracefully degrade — start with empty history

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
                st.warning("⚠️ Could not decode chart image.")


def send_message(user_input: str):
    """POST the user message to the backend and handle the response."""
    # Append user message immediately
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "image": None,
    })

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send to backend with loading indicator
    with st.chat_message("assistant"):
        with st.spinner("🧊 Analyzing..."):
            try:
                resp = requests.post(
                    CHAT_ENDPOINT,
                    json={
                        "session_id": st.session_state.session_id,
                        "message": user_input,
                    },
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("text", "No response received.")
                    image = data.get("image_base64")
                    cached = data.get("cached", False)
                    latency = data.get("latency_ms", 0)
                    tools = data.get("tools_called", [])

                    # Render response
                    st.markdown(text)

                    if image:
                        try:
                            image_bytes = base64.b64decode(image)
                            st.image(image_bytes, caption="Generated Chart",
                                     use_container_width=True)
                        except Exception:
                            st.warning("⚠️ Could not decode chart image.")

                    # Metadata footer
                    meta_parts = [f"⏱️ {latency}ms"]
                    if cached:
                        meta_parts.append("📦 cached")
                    if tools:
                        meta_parts.append(f"🔧 {', '.join(tools)}")
                    st.caption(" · ".join(meta_parts))

                    # Save to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": text,
                        "image": image,
                    })

                elif resp.status_code == 422:
                    st.error("❌ Invalid request. Please check your message.")
                else:
                    st.error(f"❌ Server error ({resp.status_code}). Please try again.")

            except requests.Timeout:
                st.error("⏰ Request timed out. The server may be overloaded.")
            except requests.ConnectionError:
                st.error("🔌 Cannot connect to the backend. Is the API server running?")
            except requests.RequestException as e:
                st.error(f"❌ Request failed: {e}")



def main():
    """Run the Streamlit chat application."""
    init_session()
    restore_history()

    # Header
    st.title("🧊 IcBerg")
    st.caption("Conversational analysis of the Titanic passenger dataset")

    # Sidebar
    with st.sidebar:
        st.markdown("### Session Info")
        st.code(st.session_state.session_id, language=None)

        # Health check
        try:
            health = requests.get(HEALTH_ENDPOINT, timeout=3).json()
            status = health.get("status", "unknown")
            if status == "healthy":
                st.markdown('<span class="status-badge status-ok">● API Healthy</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-err">● API Degraded</span>',
                            unsafe_allow_html=True)
        except requests.RequestException:
            st.markdown('<span class="status-badge status-err">● API Offline</span>',
                        unsafe_allow_html=True)

        st.divider()
        st.markdown("### About")
        st.markdown(
            "IcBerg uses a LangGraph ReAct agent to answer questions about "
            "the Titanic dataset. It can query data, compute statistics, "
            "and generate charts."
        )

        st.divider()
        if st.button("🗑️ New Session", use_container_width=True):
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
