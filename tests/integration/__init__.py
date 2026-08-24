"""Phase 3 integration-surface tests (`.devdocs/PHASE3_GATES.md`, P3.1-P3.6): the Python
SDK (`icberg`), policy YAML (`backend.core.policy`), the MCP server
(`backend.mcp_server`), and the LangChain/LangGraph adapter
(`backend.integrations.langgraph_tool`) — proving each governs a real proposed statement
end to end through the SAME `Gateway`/`GovernanceGate` the REST API uses, and that none
of them exposes a raw executor or database connection.
"""
