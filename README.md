# IcBerg

```text
  ___      ___                 
 |_ _|___ | _ ) ___ _ _ __ _  
  | |/ __|| _ \/ -_) '_/ _` | 
 |__|\___||___/\___|_| \__, | 
                       |___/  
```

**IcBerg** is a production-grade conversational analysis agent for the Titanic passenger dataset. It features a robust **LangGraph ReAct agent**, three tiers of **Qdrant semantic caching**, and a **process-isolated sandbox** for secure data analysis.

Built for the TailorTalk Internship Assessment (Backend AI Engineer).

## Key Features

- **Live Reasoning Stream**: Watch the agent's multi-step thought process in real-time via **Server-Sent Events (SSE)**.
- **Agentic Analysis**: Utilizes a LangGraph-powered ReAct loop to intelligently cycle between data querying, statistical computation, and chart visualization.
- **Three-Tier Semantic Cache (Qdrant)**:
  - **Execution Cache**: Skips redundant computation for similar analytical intents.
  - **Visualization Cache**: Instantly retrieves previously generated charts.
  - **History Retrieval**: Injects semantically relevant past exchanges for better follow-up context.
- **Security-First Execution**:
  - **AST Validation**: Statically analyzes AI-authored code to prevent dangerous imports or operations.
  - **Sandboxed Subprocess**: Executes pandas code in an isolated environment with strict resource limits and no network/file access.
- **Resilient LLM Core**: High-availability setup using **Cerebras** (primary, ~3000 tok/s) with **Groq** fallback.

## Technical Architecture

IcBerg follows a strict layered architecture with a downward-only dependency flow:

1.  **UI Layer (Streamlit)**: High-performance chat frontend with live status indicators.
2.  **Orchestration Layer (FastAPI)**: Manages SSE streaming, context assembly, and parallel metadata retrieval.
3.  **Agent Layer (LangGraph)**: Definitive ReAct state machine using tools (`query_data`, `visualize_data`, etc.).
4.  **Retrieval Layer (Qdrant + Gemini)**: High-dimensional vector search (3072-dim) using Google Gemini embeddings.
5.  **Execution Layer (validator.py + sandbox.py)**: The security foundation for AI code execution.
6.  **Data Layer (SQLite + DataFrame)**: Persistent chat history and thread-safe data access.

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Agent Framework** | LangGraph `create_react_agent` |
| **Primary LLM** | Cerebras (Model: `gpt-oss-120b`) |
| **Fallback LLM** | Groq (Model: `llama-3.3-70b-versatile`) |
| **Embeddings** | Google Gemini (`gemini-embedding-001`) |
| **Vector DB** | Qdrant Cloud |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Database** | SQLite + SQLAlchemy 2.0 |
| **DevOps** | Docker Compose v2 |

## Prerequisites

- Python 3.11 (exact)
- Docker & Docker Compose v2 (recommended for production)
- API Keys: Cerebras, Groq, Google AI (Gemini), and Qdrant Cloud.

## Getting Started

### Option A: Local Development (Terminal)

1. **Clone and Install**:
```bash
git clone <repo-url> && cd IcBerg
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

2. **Configure**:
Copy `.env.example` to `.env` and fill in your keys.

3. **Run Services**:
- **Backend**: `uvicorn backend.main:app --port 8000 --reload`
- **Frontend**: `streamlit run frontend/app.py`

### Option B: Docker Compose (Production-Ready)

```bash
docker compose up --build -d
```
The application will be available at `http://localhost:8501`.

## Maintenance & Testing

- **Full Suite**: `pytest tests/` (180 tests)
- **Reset Data**: `python scripts/reset_db.py` (Clears SQLite & Qdrant)
- **Clean Cache**: `./scripts/clean_cache.sh` (Clears Python bytecode)

## Environment Variables

| Variable | Description |
| :--- | :--- |
| `CEREBRAS_API_KEY` | Primary LLM Key |
| `GROQ_API_KEY` | Fallback LLM Key |
| `GEMINI_API_KEY` | Embedding Key |
| `QDRANT_URL`/`API_KEY` | Vector DB Credentials |
| `DATABASE_URL` | SQLite path (supports in-memory or file) |
