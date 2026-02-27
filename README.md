# IcBerg

```text
  ___      ___                 
 |_ _|___ | _ ) ___ _ _ __ _  
  | |/ __|| _ \/ -_) '_/ _` | 
 |__|\___||___/\___|_| \__, | 
                       |___/  
```

IcBerg is a conversational analysis agent specifically designed to analyze the Titanic passenger dataset. It utilizes a ReAct-based LangChain agent, a sandboxed execution environment, and intelligent semantic caching to answer complex data questions while keeping system boundaries secure.

## Features

- **Conversational Interface**: Ask natural language questions about the Titanic dataset through a clean, real-time Streamlit interface with Server-Sent Events (SSE) streaming.
- **Agentic Execution**: An LLM-powered agent dynamically writes and executes pandas code to solve your queries, falling back across providers (Cerebras to Groq) for high availability.
- **Sandboxed Pandas**: All AI-authored python code executes in an isolated subprocess with strict memory limits, strict AST validation, and constrained library access, preventing abuse.
- **Semantic Caching**: Qdrant vector database caches previous executions and visualizations, vastly improving response times for similar queries via similarity search.
- **Robust Guardrails**: Prompt injection defenses and LLM output validators ensure the agent stays strictly on-topic (Titanic analysis) and refuses off-topic or inappropriate content.

## Architecture Highlights

1. **Frontend**: Streamlit application (`frontend/app.py`). Communicates with the API over HTTP and renders intermediate agent steps and charts dynamically.
2. **Backend**: FastAPI application (`backend/main.py` and `backend/api/routes.py`). Manages API routing, rate limiting, and dependencies.
3. **Agent Core**: LangGraph ReAct agent (`backend/agent/agent.py`). Orchestrates the main `query_data` and `visualize_data` tools.
4. **Isolated Sandbox**: Subprocess execution environment (`backend/core/sandbox.py`) paired with an AST validator (`backend/core/validator.py`).
5. **Caching & Embeddings**: Sentence-transformers for text embedding and Qdrant (`backend/core/qdrant_manager.py`) for similarity search caching.
6. **Persistence**: SQLite database (`backend/core/database.py`) tracks conversation history and session states.

## Prerequisites

- Python 3.11+
- [Cerebras AI API Key](https://cerebras.ai/)
- (Optional) [Groq API Key](https://groq.com/) for failover.

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd IcBerg
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Copy `.env.example` to `.env` and fill in your API keys:
   ```env
   CEREBRAS_API_KEY="your_cerebras_key_here"
   GROQ_API_KEY="your_groq_key_here"  # Optional failover
   ```

5. **Start the backend server:**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

6. **Start the frontend application:**
   In a new terminal window (with the virtual environment activated), run:
   ```bash
   streamlit run frontend/app.py
   ```

7. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`.

## Maintenance

To clear the SQLite database and Qdrant collections completely, run:
```bash
python scripts/reset_db.py
```

To clear local python caches (`__pycache__`, `.pytest_cache`, etc.):
```bash
bash scripts/clean_cache.sh
```

## Running Tests

The project includes a comprehensive Pytest suite. Ensure you have the required dependencies, then run:
```bash
pytest tests/ -v
```
