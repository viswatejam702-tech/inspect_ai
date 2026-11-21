import contextlib
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, cast

import anyio
from pydantic import JsonValue

from inspect_ai._util._async import raise_unwrapped
from inspect_ai.agent._agent import AgentState
from inspect_ai.agent._bridge.sandbox.bridge import (
    _error_after_unwind,
    _monitor_failure,
    _register_bridged_tools,
)
from inspect_ai.agent._bridge.sandbox.service import (
    GenerateMethod,
    _forward_provider_errors,
    call_tool,
    generate_anthropic,
    generate_completions,
    generate_google,
    generate_responses,
    list_tools,
)
from inspect_ai.agent._bridge.sandbox.types import SandboxAgentBridge
from inspect_ai.agent._bridge.util import (
    resolve_bridge_code_execution,
    resolve_bridge_web_search,
)
from inspect_ai.model._compaction.types import CompactionStrategy
from inspect_ai.model._model import (
    GenerateFilter,
    Model,
    ModelEventSink,
    ModelResolver,
    ModelResponseFilter,
)
from inspect_ai.tool._mcp._tools_bridge import BridgedToolsSpec
from inspect_ai.tool._tools._code_execution import CodeExecutionProviders
from inspect_ai.tool._tools._web_search._web_search import WebSearchProviders
from inspect_ai.util._anyio import inner_exception
from inspect_ai.util._checkpoint.checkpointer import Checkpointer
from inspect_ai.util._sandbox.service import find_control_flow

if TYPE_CHECKING:
    from inspect_ai.approval._policy import ApprovalPolicy

logger = getLogger(__name__)


class AgentBridgeSession:
    """Reusable agent bridge session decoupled from transport mechanics.

    An `AgentBridgeSession` encapsulates the state, model resolution, prompt
    conversion, response filtering, tool approval enforcement, and error
    forwarding logic of the Inspect agent bridge without binding to a specific
    transport (such as the container file-based RPC service or a local HTTP proxy).

    Transports receive raw client requests, validate transport-level constraints
    (authentication, framing, size limits), and invoke the corresponding session
    handler (`chat_completions`, `responses`, `anthropic_messages`,
    `google_generate`, `list_tools`, `call_tool`).
    """

    def __init__(
        self,
        bridge: SandboxAgentBridge,
        web_search: WebSearchProviders | None = None,
        code_execution: CodeExecutionProviders | None = None,
    ) -> None:
        """Initialize an agent bridge session.

        Args:
            bridge: The configured `SandboxAgentBridge` governing model routing,
                approvals, filters, and state tracking.
            web_search: Optional resolved web search capability grant.
            code_execution: Optional resolved code execution capability grant.
        """
        self._bridge = bridge
        self._web_search = web_search
        self._code_execution = code_execution
        self._is_active = True
        self._request_count = 0
        self._latencies: list[float] = []

        # Instantiate error-forwarding generation handlers
        self._completions_handler: GenerateMethod = _forward_provider_errors(
            generate_completions(bridge), bridge
        )
        self._responses_handler: GenerateMethod = _forward_provider_errors(
            generate_responses(web_search, code_execution, bridge), bridge
        )
        self._anthropic_handler: GenerateMethod = _forward_provider_errors(
            generate_anthropic(web_search, code_execution, bridge), bridge
        )
        self._google_handler: GenerateMethod = _forward_provider_errors(
            generate_google(web_search, code_execution, bridge), bridge
        )
        self._list_tools_handler: Callable[[str], Any] = list_tools(bridge)
        self._call_tool_handler: Callable[[str, str, JsonValue], Any] = call_tool(
            bridge
        )

    @property
    def bridge(self) -> SandboxAgentBridge:
        """Return the underlying SandboxAgentBridge."""
        return self._bridge

    @property
    def state(self) -> AgentState:
        """Return the active agent state (message history and token counters)."""
        return self._bridge.state

    @property
    def is_active(self) -> bool:
        """Whether this session is currently active and accepting requests."""
        return self._is_active

    @property
    def request_count(self) -> int:
        """Total number of requests processed by this session."""
        return self._request_count

    @property
    def metrics(self) -> dict[str, Any]:
        """Telemetry and performance metrics for the session."""
        if not self._latencies:
            return {
                "total_requests": self._request_count,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "mean_latency_ms": 0.0,
            }
        sorted_lats = sorted(self._latencies)
        n = len(sorted_lats)
        p50 = sorted_lats[int(n * 0.50)] * 1000.0
        p95 = sorted_lats[min(int(n * 0.95), n - 1)] * 1000.0
        mean = (sum(sorted_lats) / n) * 1000.0
        return {
            "total_requests": self._request_count,
            "p50_latency_ms": round(p50, 3),
            "p95_latency_ms": round(p95, 3),
            "mean_latency_ms": round(mean, 3),
        }

    def _record_latency(self, duration: float) -> None:
        self._request_count += 1
        self._latencies.append(duration)

    async def chat_completions(
        self,
        json_data: dict[str, JsonValue],
        headers: dict[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        """Process an OpenAI Chat Completions API request.

        Args:
            json_data: Parsed JSON request body matching OpenAI chat completions schema.
            headers: Optional client HTTP request headers to filter and forward.

        Returns:
            Serialized JSON response dict (or provider error payload).
        """
        if not self._is_active:
            raise RuntimeError(
                "AgentBridgeSession is closed and no longer accepts requests."
            )
        start = time.perf_counter()
        try:
            return await self._completions_handler(json_data, headers)
        finally:
            self._record_latency(time.perf_counter() - start)

    async def responses(
        self,
        json_data: dict[str, JsonValue],
        headers: dict[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        """Process an OpenAI Responses API request.

        Args:
            json_data: Parsed JSON request body matching OpenAI responses schema.
            headers: Optional client HTTP request headers to filter and forward.

        Returns:
            Serialized JSON response dict (or provider error payload).
        """
        if not self._is_active:
            raise RuntimeError(
                "AgentBridgeSession is closed and no longer accepts requests."
            )
        start = time.perf_counter()
        try:
            return await self._responses_handler(json_data, headers)
        finally:
            self._record_latency(time.perf_counter() - start)

    async def anthropic_messages(
        self,
        json_data: dict[str, JsonValue],
        headers: dict[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        """Process an Anthropic Messages API request.

        Args:
            json_data: Parsed JSON request body matching Anthropic messages schema.
            headers: Optional client HTTP request headers to filter and forward.

        Returns:
            Serialized JSON response dict (or provider error payload).
        """
        if not self._is_active:
            raise RuntimeError(
                "AgentBridgeSession is closed and no longer accepts requests."
            )
        start = time.perf_counter()
        try:
            return await self._anthropic_handler(json_data, headers)
        finally:
            self._record_latency(time.perf_counter() - start)

    async def google_generate(
        self,
        json_data: dict[str, JsonValue],
        headers: dict[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        """Process a Google Gemini generateContent API request.

        Args:
            json_data: Parsed JSON request body matching Google generateContent schema.
            headers: Optional client HTTP request headers.

        Returns:
            Serialized JSON response dict (or provider error payload).
        """
        if not self._is_active:
            raise RuntimeError(
                "AgentBridgeSession is closed and no longer accepts requests."
            )
        start = time.perf_counter()
        try:
            return await self._google_handler(json_data, headers)
        finally:
            self._record_latency(time.perf_counter() - start)

    async def list_tools(self, server: str) -> JsonValue:
        """List tool definitions exposed by a bridged MCP server.

        Args:
            server: Name of the registered bridged MCP server.

        Returns:
            List of tool schema objects matching MCP protocol specifications.
        """
        return cast(JsonValue, await self._list_tools_handler(server))

    async def call_tool(
        self, server: str, tool: str, arguments: JsonValue
    ) -> JsonValue:
        """Execute a bridged tool on the host.

        Args:
            server: Name of the registered bridged MCP server.
            tool: Name of the tool function to call.
            arguments: Tool arguments matching the tool's JSON schema.

        Returns:
            Result of tool execution.
        """
        return cast(JsonValue, await self._call_tool_handler(server, tool, arguments))

    def close(self) -> None:
        """Mark the session as closed."""
        self._is_active = False


@contextlib.asynccontextmanager
async def agent_bridge_session(
    state: AgentState | None = None,
    *,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    model_resolver: ModelResolver | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    compaction: CompactionStrategy | None = None,
    port: int = 13131,
    web_search: WebSearchProviders | bool | None = None,
    code_execution: CodeExecutionProviders | bool | None = None,
    client_mcp_servers: bool | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    model_event_sink: ModelEventSink | None = None,
    forward_generation_config: bool = False,
    approval: list["ApprovalPolicy"] | None = None,
    checkpointer: Checkpointer | None = None,
    forward_client_headers: Mapping[str, Sequence[str]] | None = None,
    response_filter: ModelResponseFilter | None = None,
) -> AsyncIterator[AgentBridgeSession]:
    """Create a reusable agent bridge processing session without starting a transport.

    This context manager initializes all Inspect model resolution, response filtering,
    tool approval, generation configuration, and error forwarding infrastructure.
    It yields an `AgentBridgeSession` which exposes async handlers for OpenAI,
    Anthropic, and Google request dialects.

    Example:
        ```python
        async with agent_bridge_session(
            model="openai/gpt-4o",
            approval=approval_policies,
        ) as session:
            response = await session.chat_completions(request_json)
        ```

    Args:
        state: Initial agent state (messages and token counters).
        model: Fallback or pinned model identifier.
        model_aliases: Explicit alias mapping for requested model names.
        model_resolver: Dynamic model routing resolver function.
        filter: Generation filter.
        retry_refusals: Number of refusal retries.
        compaction: Conversation context compaction strategy.
        port: Virtual port identifier for bridged tools MCP URLs.
        web_search: Web search provider configuration or boolean grant.
        code_execution: Code execution provider configuration or boolean grant.
        client_mcp_servers: Whether to permit agent-declared MCP servers.
        bridged_tools: Host tools to expose to the agent via MCP protocol.
        model_event_sink: Custom transcript sink for ModelEvents.
        forward_generation_config: Whether client generation parameters take precedence.
        approval: Tool call approval policies.
        checkpointer: Checkpointer to tick after generations.
        forward_client_headers: Allowlist mapping of client headers to permitted values.
        response_filter: Post-generation response filter.

    Yields:
        AgentBridgeSession: The active session ready to process requests.
    """

    # Resolve capabilities
    web_search_grant = resolve_bridge_web_search(web_search, default_grant=False)
    code_execution_grant = resolve_bridge_code_execution(
        code_execution, default_grant=False
    )
    allow_remote_mcp = False if client_mcp_servers is None else client_mcp_servers

    state = state or AgentState(messages=[])
    session_completed = False
    bridge: SandboxAgentBridge | None = None

    try:
        async with anyio.create_task_group() as tg:
            # Create the underlying SandboxAgentBridge
            bridge = SandboxAgentBridge(
                state=state,
                filter=filter,
                retry_refusals=retry_refusals,
                compaction=compaction,
                port=port,
                model=model,
                model_aliases=model_aliases,
                model_resolver=model_resolver,
                model_event_sink=model_event_sink,
                forward_generation_config=forward_generation_config,
                approval=approval,
                checkpointer=checkpointer,
                allow_remote_mcp=allow_remote_mcp,
                forward_client_headers=forward_client_headers,
                response_filter=response_filter,
            )

            # Register bridged tools
            seen_names: set[str] = set()
            for spec in bridged_tools or []:
                if spec.name in seen_names:
                    raise ValueError(
                        f"Duplicate bridged_tools name: '{spec.name}'. "
                        "Each BridgedToolsSpec must have a unique name."
                    )
                seen_names.add(spec.name)
                config = _register_bridged_tools(bridge, spec, port)
                bridge.mcp_server_configs.append(config)
            bridge.warn_indistinct_tools()

            # Monitor for sample failures requested from service tasks
            tg.start_soon(_monitor_failure, bridge)

            # Instantiate session wrapper
            session = AgentBridgeSession(
                bridge=bridge,
                web_search=web_search_grant,
                code_execution=code_execution_grant,
            )

            try:
                yield session
                session_completed = True
            finally:
                session.close()
                tg.cancel_scope.cancel()

    except Exception as ex:
        termination = find_control_flow(ex)
        if termination is not None:
            raise termination from ex

        error = await _error_after_unwind(bridge, ex)
        if error is not None:
            raise_unwrapped(ex, error)

        if session_completed:
            logger.warning(
                f"Error during agent_bridge_session cleanup (session completed successfully): {inner_exception(ex)}"
            )
        else:
            raise_unwrapped(ex, inner_exception(ex))

    error = await _error_after_unwind(bridge, None)
    if error is not None:
        raise error
