"""Framework adapters that plug IcBerg's governance gateway into third-party agent
frameworks. Every adapter in this package routes through `icberg.GovernedConnection` —
never a raw executor or DB connection — the same governance path the Python SDK
(`icberg`) and the MCP server (`backend.mcp_server`) use.
"""
