"""MCP server exposing IcBerg's governance gateway as governed MCP tools — the "native
agentic-integration path" from `.devdocs/FLAGSHIP_ROADMAP.md`'s "End product &
integration surfaces" section: any MCP-capable agent (Claude Desktop, Cursor, a custom
client) gets *governed* DB tools instead of raw credentials.

Tools registered on the server (agent-facing; per `.devdocs/PHASE3_GATES.md` P3.7):
    query(sql)                       - propose one SQL statement; governed end to end.
    list_pending_approvals()         - every pending, not-yet-expired write approval.
    audit_tail(n)                    - the last n entries of the tamper-evident audit trail.

Deliberately NOT a registered MCP tool: `approve`. Approval of a held write happens
out-of-band — an authenticated REST call or a human UI action, never a call an MCP
agent/client can make itself. The whole point of holding a write for approval is that
the proposing principal cannot also be the approving one; exposing `approve` as an
agent-callable MCP tool would let the very agent that proposed a write immediately
approve its own hold, collapsing the human-in-the-loop control to a no-op. See
`_ServerState.approve` below: the method still exists (for an out-of-band caller, e.g.
a REST layer, to invoke directly on `server.state`), it is simply never wrapped in a
`@server.tool(...)` registration, so no MCP protocol call can reach it.

`create_mcp_server(dsn, ...)` mirrors `backend.gateway_app.create_gateway_app(dsn)`'s
per-instance factory pattern exactly: each call builds one self-contained
`icberg.GovernedConnection` (its own `Gateway`/`AuditLog`/`ApprovalQueue`), so a test (or
an operator running several governed databases) can build multiple, fully isolated server
instances with no shared state — never a module-level singleton connection.

`sql` is untrusted input, exactly like the REST API's `POST /query` body — this module
never trusts it and never bypasses `icberg.GovernedConnection`, which itself never exposes
a raw executor or DB connection (see `icberg/__init__.py`'s module docstring and
`.devdocs/PHASE3_GATES.md` P3.5). There is no separate SQL execution path here at all: the
four tool functions below are thin wrappers over the SAME `GovernedConnection` methods the
Python SDK and the LangGraph tool call — one governance path, three integration surfaces.

Every tool handler is a plain method on `_ServerState` (attached to the returned server as
`server.state`), not a closure only reachable through the MCP protocol — so it is directly
unit-testable (`server.state.query("SELECT ...")`) with no live MCP client/transport
needed, per `.devdocs/PHASE3_GATES.md` P3.3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import structlog
from mcp.server.mcpserver import MCPServer

from icberg import GovernedConnection, Policy, governed_connection

logger = structlog.get_logger(__name__)

# Default actor recorded on every proposal this server makes when the MCP client/agent
# has no more specific identity to supply — `query`'s tool signature is fixed to `(sql)`
# only (per `.devdocs/PHASE3_GATES.md`'s tool contract), so the actor is a server-level
# identity, not a per-call argument. Override via `ICBERG_MCP_ACTOR`.
DEFAULT_ACTOR = "mcp-client"

_PolicyLike = str | dict[str, Any] | Policy | None


@dataclass
class _ServerState:
    """Holds the ONE `GovernedConnection` this MCP server instance routes every tool
    call through. `_connection` is intentionally named with a leading underscore (never
    a plain public attribute, even though `GovernedConnection` itself is already
    governance-only) — this object's own public surface exposes ONLY the methods below
    (three of which — `query`/`list_pending_approvals`/`audit_tail` — are registered as
    MCP tools; `approve` deliberately is not, see module docstring P3.7), mirroring
    `GovernedSQLTool`'s identical `_connection` convention.
    """

    _connection: GovernedConnection
    actor: str

    def query(self, sql: str) -> dict[str, Any]:
        """Propose `sql` (untrusted); governed end to end. Returns `action`
        (allow/block/hold), `reason`, `matched_rules`, `rows` (redacted PII, only on
        `allow`), `audit_seq`, and `approval_id` (only on `hold`).
        """
        return self._connection.query(sql, self.actor)

    def list_pending_approvals(self) -> list[dict[str, Any]]:
        """Every pending, not-yet-expired write approval, PII-value-redacted."""
        return self._connection.pending_approvals()

    def approve(self, id: str, approver: str) -> dict[str, Any]:  # noqa: A002 - matches the out-of-band caller's documented arg name
        """Approve a pending write by id: executes the EXACT SQL a human already
        reviewed, never re-derived. Raises `approvals.ApprovalError` for an
        unknown/expired/already-decided id or a self-approval attempt.

        NOT registered as an MCP tool (see module docstring, P3.7) — only reachable by
        an out-of-band caller invoking `server.state.approve(...)` directly (e.g. an
        authenticated REST endpoint or test code), never by an MCP client/agent over
        the protocol. This is what makes self-approval structurally impossible for an
        MCP-connected agent, not merely a policy `approve` happens to enforce.
        """
        return self._connection.approve(id, approver)

    def audit_tail(self, n: int = 20) -> list[dict[str, Any]]:
        """The last `n` entries of the tamper-evident audit trail, PII-value-redacted."""
        return self._connection.audit_tail(n)


def create_mcp_server(
    dsn: str,
    *,
    policy: _PolicyLike = None,
    actor: str = DEFAULT_ACTOR,
    name: str = "icberg",
) -> MCPServer:
    """Build one governed MCP server instance around the database named by `dsn`.

    Args:
        dsn: Passed straight to `icberg.governed_connection` (a SQLite path/URI,
            `postgres://...`, or `mysql://...`).
        policy: Optional policy YAML path, dict, or `Policy` — see
            `backend.core.policy.load_policy`. Applied to every `query` tool call.
        actor: Identity recorded on every proposal this server makes (the `query` tool's
            signature is fixed to `(sql)`, with no per-call actor — see module docstring).
        name: The MCP server's advertised name.

    Returns:
        An `MCPServer` with `query`/`list_pending_approvals`/`audit_tail` registered as
        agent-facing tools (see module docstring, P3.7 — deliberately no `approve` tool:
        approval is out-of-band, never an MCP-callable action), and `.state` set to the
        `_ServerState` those tools call into (for direct, client-free unit testing;
        `.state.approve(...)` remains callable directly by an out-of-band caller, just
        never via the MCP protocol).
    """
    state = _ServerState(_connection=governed_connection(dsn, policy=policy), actor=actor)

    server = MCPServer(
        name,
        description=(
            "IcBerg governance gateway exposed as governed MCP tools. Every `query` is "
            "decided (allow/block/hold), executed only if allowed (least-privilege, "
            "read-only), PII-redacted, and audited — the agent never gets a raw "
            "database connection or credentials. Writes are held for human approval "
            "granted out-of-band (REST/human UI), never by this agent-facing tool set — "
            "see `list_pending_approvals`; `audit_tail` shows recent decisions."
        ),
        version="0.3.0",
    )
    server.state = state

    @server.tool(
        name="query",
        description=(
            "Propose one SQL statement against the governed database. Untrusted input: "
            "evaluated by IcBerg's policy gate before anything executes. A destructive "
            "or out-of-scope statement (DROP/TRUNCATE/injection/RCE/DoS pattern/...) is "
            "blocked outright; a write is held for human approval, granted out-of-band "
            "by a human/REST caller — never by this agent (see list_pending_approvals); "
            "a safe, bounded read executes with PII redacted from every returned row."
        ),
    )
    def query(sql: str) -> dict[str, Any]:
        return state.query(sql)

    @server.tool(
        name="list_pending_approvals",
        description="List every pending, not-yet-expired write approval (PII-redacted).",
    )
    def list_pending_approvals() -> list[dict[str, Any]]:
        return state.list_pending_approvals()

    @server.tool(
        name="audit_tail",
        description="The last n entries of the tamper-evident, hash-chained audit trail (PII-redacted).",
    )
    def audit_tail(n: int = 20) -> list[dict[str, Any]]:
        return state.audit_tail(n)

    return server


def __getattr__(name: str) -> MCPServer:
    """PEP 562 module `__getattr__`: builds the module-level `server` object lazily, on
    first access, rather than as an import-time side effect — mirrors
    `backend.gateway_app`'s identical `__getattr__` pattern for its own lazy `app`, so
    `from backend.mcp_server import create_mcp_server` (what every test in this repo
    does) never touches the filesystem or opens a database nobody asked for.

    Configured entirely from environment variables, with a temp-directory default so an
    ad hoc stdio smoke-test run never writes into the repo itself:
        ICBERG_MCP_DSN     - database DSN (default: a temp-dir SQLite file).
        ICBERG_MCP_POLICY  - optional policy YAML path.
        ICBERG_MCP_ACTOR   - actor identity recorded on every proposal (default: "mcp-client").
    """
    if name != "server":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import tempfile

    default_db_path = os.path.join(tempfile.gettempdir(), "icberg_mcp.sqlite")
    dsn = os.environ.get("ICBERG_MCP_DSN", default_db_path)
    policy_path = os.environ.get("ICBERG_MCP_POLICY")
    actor = os.environ.get("ICBERG_MCP_ACTOR", DEFAULT_ACTOR)
    built = create_mcp_server(dsn, policy=policy_path, actor=actor)
    globals()["server"] = built
    return built


if __name__ == "__main__":  # pragma: no cover - manual stdio smoke-test entry point only
    # `server` is only materialized via module `__getattr__` above (PEP 562), which
    # intercepts genuine attribute access on the module object, not a bare global name
    # lookup inside the module's own body — call `__getattr__` directly rather than
    # referencing a bare `server` name that was never actually assigned.
    __getattr__("server").run(transport="stdio")
