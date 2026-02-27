"""LangGraph agent factory.

Wires together the LLM adapter, tools, and system prompt
into a compiled ReAct agent graph.
"""

import pandas as pd
import structlog
from langgraph.prebuilt import create_react_agent

from backend.agent.prompts import build_system_prompt
from backend.agent.tools import (
    get_dataset_info,
    get_statistics,
    query_data,
    set_dataframe,
    visualize_data,
)
from backend.core.llm_adapter import LLMAdapter
from backend.core.validator import set_known_columns
from backend.data.loader import get_schema_metadata

logger = structlog.get_logger(__name__)

TOOLS = [get_dataset_info, query_data, visualize_data, get_statistics]


def create_agent(llm_adapter: LLMAdapter, df: pd.DataFrame):
    """Build the ReAct agent graph.

    Args:
        llm_adapter: Initialized LLM adapter with failover.
        df: Loaded Titanic DataFrame (singleton).

    Returns:
        Compiled LangGraph state graph, ready to invoke.
    """
    # Wire up the DataFrame for tools and column validation
    set_dataframe(df)
    set_known_columns(list(df.columns))

    # Build system prompt with schema
    schema = get_schema_metadata(df)
    system_prompt = build_system_prompt(schema)

    # Create the agent
    model = llm_adapter.get_chat_model()
    agent = create_react_agent(
        model=model,
        tools=TOOLS,
        prompt=system_prompt,
    )

    logger.info("agent.created", tools=[t.name for t in TOOLS])
    return agent
