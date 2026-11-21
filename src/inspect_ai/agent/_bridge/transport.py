import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Mapping

import anyio
from pydantic import JsonValue

from inspect_ai.agent._bridge._errors import PROVIDER_ERROR_KEY
from inspect_ai.agent._bridge.bridge import (
    _BLOCKED_CLIENT_HEADER_PREFIXES,
    _BLOCKED_CLIENT_HEADERS,
)
from inspect_ai.agent._bridge.session import AgentBridgeSession

logger = getLogger(__name__)

DEFAULT_MAX_PAYLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_REQUEST_TIMEOUT = 120.0  # seconds


@dataclass
class HttpResponse:
    """Standard HTTP response container for transport adapters."""

    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, JsonValue] | list[JsonValue] | str = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Whether the HTTP status indicates success (2xx)."""
        return 200 <= self.status_code < 300


class AgentBridgeTransport(ABC):
    """Abstract base class for agent bridge transports."""

    @abstractmethod
    async def serve(
        self, session: AgentBridgeSession, started: anyio.Event | None = None
    ) -> None:
        """Serve the transport layer until cancelled or session completes."""
        ...


class AsyncMemoryTransport(AgentBridgeTransport):
    """Zero-overhead direct in-process asynchronous transport.

    Designed for fast unit tests, synthetic benchmarks, direct agent scaffolds,
    and fuzzing without socket overhead, process supervision, or IPC serialization.
    """

    def __init__(self) -> None:
        self._session: AgentBridgeSession | None = None
        self._running = False
        self._requests_handled = 0

    @property
    def is_running(self) -> bool:
        """Whether the transport is actively bound to a running session."""
        return self._running and self._session is not None and self._session.is_active

    @property
    def requests_handled(self) -> int:
        """Total requests handled by this transport."""
        return self._requests_handled

    async def serve(
        self, session: AgentBridgeSession, started: anyio.Event | None = None
    ) -> None:
        """Bind session to this memory transport."""
        self._session = session
        self._running = True
        if started is not None:
            started.set()
        try:
            # Keep transport open until cancelled or session closes
            while self._running and session.is_active:
                await anyio.sleep(0.05)
        finally:
            self._running = False
            self._session = None

    def _ensure_session(self) -> AgentBridgeSession:
        if self._session is None or not self._session.is_active:
            raise RuntimeError(
                "AsyncMemoryTransport is not currently bound to an active AgentBridgeSession."
            )
        return self._session

    async def chat_completions(
        self,
        payload: dict[str, JsonValue],
        headers: dict[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        """Dispatch OpenAI Chat Completions directly to session."""
        session = self._ensure_session()
        self._requests_handled += 1
        return await session.chat_completions(payload, headers)

    async def responses(
        self,
        payload: dict[str, JsonValue],
        headers: dict[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        """Dispatch OpenAI Responses directly to session."""
        session = self._ensure_session()
        self._requests_handled += 1
        return await session.responses(payload, headers)

    async def anthropic_messages(
        self,
        payload: dict[str, JsonValue],
        headers: dict[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        """Dispatch Anthropic Messages directly to session."""
        session = self._ensure_session()
        self._requests_handled += 1
        return await session.anthropic_messages(payload, headers)

    async def google_generate(
        self,
        payload: dict[str, JsonValue],
        headers: dict[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        """Dispatch Google generateContent directly to session."""
        session = self._ensure_session()
        self._requests_handled += 1
        return await session.google_generate(payload, headers)

    async def list_tools(self, server: str) -> JsonValue:
        """List bridged MCP tools."""
        session = self._ensure_session()
        return await session.list_tools(server)

    async def call_tool(
        self, server: str, tool: str, arguments: JsonValue
    ) -> JsonValue:
        """Execute a bridged host tool."""
        session = self._ensure_session()
        return await session.call_tool(server, tool, arguments)


class HttpTransportAdapter(AgentBridgeTransport):
    """Controlled HTTP transport adapter for out-of-sandbox or network agents.

    Enforces:
    - Payload size limits (default 10 MB, HTTP 413).
    - Format validation (valid JSON, HTTP 400).
    - Bearer authentication boundaries (optional token, HTTP 401).
    - Header restrictions (strips Authorization, x-api-key, cookie, x-amz-*, x-stainless-*).
    - Timeouts (configurable per-request timeout with graceful cancellation).
    - Rate/concurrency limiting (Semaphore-governed concurrency ceiling).
    - Dialect routing and error payload translation.
    """

    def __init__(
        self,
        *,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        auth_token: str | None = None,
        max_concurrency: int = 64,
    ) -> None:
        """Initialize the HTTP transport adapter.

        Args:
            max_payload_bytes: Maximum allowed request body size in bytes.
            request_timeout: Seconds before a request times out.
            auth_token: Optional shared bearer token for authentication.
            max_concurrency: Maximum number of concurrent requests allowed.
        """
        self.max_payload_bytes = max_payload_bytes
        self.request_timeout = request_timeout
        self.auth_token = auth_token
        self.semaphore = anyio.Semaphore(max_concurrency)
        self._session: AgentBridgeSession | None = None
        self._running = False

    async def serve(
        self, session: AgentBridgeSession, started: anyio.Event | None = None
    ) -> None:
        """Bind session to this HTTP transport adapter."""
        self._session = session
        self._running = True
        if started is not None:
            started.set()
        try:
            while self._running and session.is_active:
                await anyio.sleep(0.05)
        finally:
            self._running = False
            self._session = None

    def _sanitize_headers(self, headers: Mapping[str, str] | None) -> dict[str, str]:
        """Strip sensitive/credential headers that must never cross the boundary."""
        if not headers:
            return {}
        sanitized: dict[str, str] = {}
        for k, v in headers.items():
            lower = k.lower().strip()
            if lower in _BLOCKED_CLIENT_HEADERS or lower.startswith(
                _BLOCKED_CLIENT_HEADER_PREFIXES
            ):
                continue
            sanitized[k] = v
        return sanitized

    async def handle_request(
        self,
        method: str,
        path: str,
        body: bytes | str | dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        """Handle an incoming HTTP-style request.

        Args:
            method: HTTP method (e.g. "POST", "GET").
            path: Target URL path (e.g. "/v1/chat/completions", "/v1/messages").
            body: Request body (bytes, JSON string, or pre-parsed dict).
            headers: Client HTTP request headers.

        Returns:
            HttpResponse: Standard response with status_code and JSON payload.
        """
        if self._session is None or not self._session.is_active:
            return HttpResponse(
                status_code=503,
                body={
                    "error": {
                        "message": "Agent bridge session is unavailable or closed."
                    }
                },
            )

        # 1. Validate Authentication
        if self.auth_token is not None:
            header_map = {k.lower(): v for k, v in (headers or {}).items()}
            auth_header = header_map.get("authorization", "")
            expected = f"Bearer {self.auth_token}"
            if auth_header.strip() != expected:
                return HttpResponse(
                    status_code=401,
                    body={
                        "error": {
                            "message": "Unauthorized: invalid or missing bearer token."
                        }
                    },
                )

        # 2. Check Payload Size
        raw_bytes: bytes
        if isinstance(body, bytes):
            raw_bytes = body
        elif isinstance(body, str):
            raw_bytes = body.encode("utf-8")
        elif isinstance(body, dict):
            raw_bytes = json.dumps(body).encode("utf-8")
        elif body is None:
            raw_bytes = b""
        else:
            return HttpResponse(
                status_code=400,
                body={
                    "error": {
                        "message": f"Unsupported request body type: {type(body).__name__}"
                    }
                },
            )

        if len(raw_bytes) > self.max_payload_bytes:
            return HttpResponse(
                status_code=413,
                body={
                    "error": {
                        "message": (
                            f"Payload Too Large: {len(raw_bytes)} bytes exceeds limit of "
                            f"{self.max_payload_bytes} bytes."
                        )
                    }
                },
            )

        # 3. Parse JSON Body
        parsed_json: dict[str, JsonValue] = {}
        if method.upper() in ("POST", "PUT", "PATCH") and raw_bytes:
            try:
                parsed = json.loads(raw_bytes.decode("utf-8"))
                if not isinstance(parsed, dict):
                    return HttpResponse(
                        status_code=400,
                        body={
                            "error": {"message": "Request body must be a JSON object."}
                        },
                    )
                parsed_json = parsed
            except Exception as e:
                return HttpResponse(
                    status_code=400,
                    body={"error": {"message": f"Malformed JSON payload: {e}"}},
                )

        # 4. Sanitize Headers
        client_headers = self._sanitize_headers(headers)

        # 5. Route Request with Concurrency and Timeout Controls
        try:
            async with self.semaphore:
                with anyio.fail_after(self.request_timeout):
                    return await self._dispatch(
                        method.upper(), path, parsed_json, client_headers
                    )
        except TimeoutError:
            return HttpResponse(
                status_code=504,
                body={
                    "error": {
                        "message": f"Gateway Timeout: request exceeded {self.request_timeout}s limit."
                    }
                },
            )
        except Exception as e:
            logger.exception(f"Unhandled exception in HttpTransportAdapter: {e}")
            return HttpResponse(
                status_code=500,
                body={"error": {"message": f"Internal Bridge Error: {e}"}},
            )

    async def _dispatch(
        self,
        method: str,
        path: str,
        payload: dict[str, JsonValue],
        headers: dict[str, str],
    ) -> HttpResponse:
        """Route request to appropriate session handler."""
        assert self._session is not None
        clean_path = path.rstrip("/")

        # OpenAI Chat Completions
        if clean_path.endswith("/v1/chat/completions"):
            res = await self._session.chat_completions(payload, headers)
            return self._format_response(res)

        # OpenAI Responses API
        elif clean_path.endswith("/v1/responses"):
            res = await self._session.responses(payload, headers)
            return self._format_response(res)

        # Anthropic Messages API
        elif clean_path.endswith("/v1/messages"):
            res = await self._session.anthropic_messages(payload, headers)
            return self._format_response(res)

        # Google Gemini API
        elif "generateContent" in clean_path:
            res = await self._session.google_generate(payload, headers)
            return self._format_response(res)

        # MCP Tools Listing
        elif clean_path.startswith("/mcp/") and clean_path.endswith("/tools"):
            parts = clean_path.split("/")
            server_name = parts[2]
            res_val = await self._session.list_tools(server_name)
            return HttpResponse(status_code=200, body={"tools": res_val})

        # MCP Tool Calling
        elif clean_path.startswith("/mcp/") and clean_path.endswith("/tools/call"):
            parts = clean_path.split("/")
            server_name = parts[2]
            tool_name = str(payload.get("name", ""))
            arguments = payload.get("arguments", {})
            res_val = await self._session.call_tool(server_name, tool_name, arguments)
            return HttpResponse(status_code=200, body={"result": res_val})

        else:
            return HttpResponse(
                status_code=404,
                body={"error": {"message": f"Unknown endpoint path: '{path}'."}},
            )

    def _format_response(self, response_dict: dict[str, JsonValue]) -> HttpResponse:
        """Format session handler response into HttpResponse, handling provider errors."""
        if PROVIDER_ERROR_KEY in response_dict:
            error_payload = response_dict[PROVIDER_ERROR_KEY]
            status_code = 502
            if isinstance(error_payload, dict):
                # Check for explicit HTTP status code in provider error payload
                raw_status = error_payload.get("status_code") or error_payload.get(
                    "code"
                )
                if isinstance(raw_status, int) and 400 <= raw_status < 600:
                    status_code = raw_status
                elif "rate_limit" in str(error_payload).lower():
                    status_code = 429
            return HttpResponse(
                status_code=status_code,
                body={"error": error_payload},
            )
        return HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=response_dict,
        )
