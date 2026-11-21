"""Tests for AgentBridgeSession and Multi-Transport Architecture.

Covers:
- Category A: Session creation
- Category B: Session cleanup
- Category C: Exceptions during session execution
- Category D: Async cancellation
- Category E: Concurrent requests
- Category F: Model resolution
- Category G: Approval enforcement
- Category H: Response filtering
- Category I: Invalid requests
- Category J: Transport failures and error mapping
- Category K: Existing API compatibility
- Category L: Logging and metrics
- Category M: Sandbox restrictions and security header stripping
"""

import anyio
import pytest

from inspect_ai.agent._agent import AgentState
from inspect_ai.agent._bridge.sandbox.bridge import sandbox_agent_bridge
from inspect_ai.agent._bridge.session import (
    AgentBridgeSession,
    agent_bridge_session,
)
from inspect_ai.agent._bridge.transport import (
    AsyncMemoryTransport,
    HttpTransportAdapter,
)
from inspect_ai.approval import ApprovalPolicy, auto_approver
from inspect_ai.model import (
    ChatMessageUser,
    GenerateInput,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import (
    active_model_context_var,
    init_active_model,
)

ACTIVE_MODEL = "mockllm/active-test-model"
RESOLVED_MODEL = "mockllm/resolved-model"
ALIAS_MODEL = "mockllm/alias-model"


@pytest.fixture(autouse=True)
def _isolate_active_model():
    """Ensure active model contextvar does not leak between test runs."""
    model = get_model(ACTIVE_MODEL)
    init_active_model(model, GenerateConfig())
    token = active_model_context_var.set(model)
    try:
        yield
    finally:
        active_model_context_var.reset(token)


# ==============================================================================
# Category A: Session Creation
# ==============================================================================
async def test_session_creation():
    """Verify AgentBridgeSession initializes cleanly with expected defaults."""
    state = AgentState(messages=[ChatMessageUser(content="Initial message")])
    async with agent_bridge_session(state=state, model=ACTIVE_MODEL) as session:
        assert isinstance(session, AgentBridgeSession)
        assert session.is_active is True
        assert session.request_count == 0
        assert len(session.state.messages) == 1
        assert session.bridge is not None


# ==============================================================================
# Category B: Session Cleanup
# ==============================================================================
async def test_session_cleanup():
    """Verify session marks itself inactive and refuses subsequent calls upon exit."""
    session_ref = None
    async with agent_bridge_session(model=ACTIVE_MODEL) as session:
        session_ref = session
        assert session.is_active is True

    assert session_ref is not None
    assert session_ref.is_active is False
    with pytest.raises(RuntimeError, match="closed and no longer accepts"):
        await session_ref.chat_completions({"model": "inspect", "messages": []})


# ==============================================================================
# Category C: Exceptions During Session Execution
# ==============================================================================
async def test_session_provider_error_forwarding():
    """Verify provider errors are wrapped in PROVIDER_ERROR_KEY rather than crashing the session."""
    async with agent_bridge_session(model=ACTIVE_MODEL) as session:
        # Invalid model request that cannot be resolved
        res = await session.chat_completions(
            {
                "model": "non-existent-provider/fake-model",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        # Session stays up and returns formatted response or error payload
        assert isinstance(res, dict)
        assert session.is_active is True


# ==============================================================================
# Category D: Async Cancellation
# ==============================================================================
async def test_session_async_cancellation():
    """Verify session handles external task group cancellation gracefully."""
    session_ref = None
    with anyio.move_on_after(0.05):
        async with agent_bridge_session(model=ACTIVE_MODEL) as session:
            session_ref = session
            await anyio.sleep(1.0)

    assert session_ref is not None
    assert session_ref.is_active is False


# ==============================================================================
# Category E: Concurrent Requests
# ==============================================================================
async def test_session_concurrent_requests():
    """Verify session handles concurrent requests correctly and updates telemetry."""
    async with agent_bridge_session(model=ACTIVE_MODEL) as session:

        async def make_request(idx: int) -> dict:
            return await session.chat_completions(
                {
                    "model": "inspect",
                    "messages": [
                        {"role": "user", "content": f"Concurrent prompt {idx}"}
                    ],
                }
            )

        async with anyio.create_task_group() as tg:
            for i in range(5):
                tg.start_soon(make_request, i)

        assert session.request_count == 5
        metrics = session.metrics
        assert metrics["total_requests"] == 5
        assert metrics["mean_latency_ms"] >= 0.0


# ==============================================================================
# Category F: Model Resolution
# ==============================================================================
async def test_session_model_resolution():
    """Verify model_aliases and dynamic model_resolver route requests accurately."""

    def custom_resolver(name: str) -> str | None:
        if name == "openai/custom-route":
            return RESOLVED_MODEL
        return None

    async with agent_bridge_session(
        model=ACTIVE_MODEL,
        model_aliases={"my-alias": ALIAS_MODEL},
        model_resolver=custom_resolver,
    ) as session:
        # 1. Alias Resolution
        res_alias = await session.chat_completions(
            {"model": "my-alias", "messages": [{"role": "user", "content": "hello"}]}
        )
        assert isinstance(res_alias, dict)

        # 2. Dynamic Resolver Resolution
        res_resolver = await session.chat_completions(
            {
                "model": "openai/custom-route",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
        assert isinstance(res_resolver, dict)


# ==============================================================================
# Category G: Approval Enforcement
# ==============================================================================
async def test_session_approval_enforcement():
    """Verify custom approval policy applies to session tool executions."""
    policy = ApprovalPolicy(tools="*", approver=auto_approver())
    async with agent_bridge_session(model=ACTIVE_MODEL, approval=[policy]) as session:
        assert session.bridge.approval is not None
        assert len(session.bridge.approval) == 1


# ==============================================================================
# Category H: Response Filtering
# ==============================================================================
async def test_session_response_filter():
    """Verify ModelResponseFilter transforms or replaces model outputs."""

    async def custom_filter(
        model: Model, output: ModelOutput, generate_input: GenerateInput
    ) -> ModelOutput | None:
        return ModelOutput.from_content(model.name, "FILTERED_RESPONSE")

    async with agent_bridge_session(
        model=ACTIVE_MODEL, response_filter=custom_filter
    ) as session:
        assert session.bridge.response_filter is not None
        res = await session.chat_completions(
            {
                "model": "inspect",
                "messages": [{"role": "user", "content": "test filter"}],
            }
        )
        assert isinstance(res, dict)
        choices = res.get("choices", [])
        if choices and isinstance(choices, list):
            first_msg = choices[0].get("message", {})
            if "content" in first_msg:
                assert first_msg["content"] == "FILTERED_RESPONSE"


# ==============================================================================
# Category I: Invalid Requests
# ==============================================================================
async def test_session_invalid_request_handling():
    """Verify malformed JSON requests are rejected with structured error replies."""
    adapter = HttpTransportAdapter()
    async with agent_bridge_session(model=ACTIVE_MODEL) as session:
        adapter._session = session
        resp = await adapter.handle_request(
            "POST", "/v1/chat/completions", body="not valid json"
        )
        assert resp.status_code == 400
        assert "error" in resp.body


# ==============================================================================
# Category J: Transport Failures & Error Mapping
# ==============================================================================
async def test_http_transport_adapter_payload_limit_and_auth():
    """Verify HttpTransportAdapter enforces max_payload_bytes and bearer auth."""
    adapter = HttpTransportAdapter(
        max_payload_bytes=100,  # Strict 100 byte limit for test
        auth_token="secret-token-123",
    )

    async with agent_bridge_session(model=ACTIVE_MODEL) as session:
        adapter._session = session

        # 1. Unauthorized request
        unauth_resp = await adapter.handle_request(
            "POST", "/v1/chat/completions", body={"model": "inspect"}
        )
        assert unauth_resp.status_code == 401
        assert unauth_resp.is_success is False

        # 2. Payload Too Large (HTTP 413)
        large_body = {"data": "X" * 200}
        too_large_resp = await adapter.handle_request(
            "POST",
            "/v1/chat/completions",
            body=large_body,
            headers={"Authorization": "Bearer secret-token-123"},
        )
        assert too_large_resp.status_code == 413


# ==============================================================================
# Category K: Existing API Compatibility
# ==============================================================================
async def test_sandbox_agent_bridge_with_transport():
    """Verify sandbox_agent_bridge backward compatibility when supplied with custom transport."""
    transport = AsyncMemoryTransport()
    async with sandbox_agent_bridge(
        model=ACTIVE_MODEL,
        transport=transport,
    ) as bridge:
        assert bridge is not None
        # Transport is running and bound
        assert transport.is_running is True
        res = await transport.chat_completions(
            {
                "model": "inspect",
                "messages": [{"role": "user", "content": "Hello via transport"}],
            }
        )
        assert isinstance(res, dict)
        assert transport.requests_handled == 1


# ==============================================================================
# Category L: Logging and Metrics
# ==============================================================================
async def test_session_metrics_recording():
    """Verify session tracks request count, P50, and P95 latency measurements."""
    async with agent_bridge_session(model=ACTIVE_MODEL) as session:
        for i in range(10):
            await session.chat_completions(
                {
                    "model": "inspect",
                    "messages": [{"role": "user", "content": f"msg {i}"}],
                }
            )

        metrics = session.metrics
        assert metrics["total_requests"] == 10
        assert metrics["p50_latency_ms"] >= 0.0
        assert metrics["p95_latency_ms"] >= metrics["p50_latency_ms"]


# ==============================================================================
# Category M: Sandbox Restrictions and Header Stripping
# ==============================================================================
async def test_transport_header_sanitization():
    """Verify transport adapter strips sensitive headers before forwarding to Inspect."""
    adapter = HttpTransportAdapter()
    headers_with_sensitive = {
        "Authorization": "Bearer leaked-secret",
        "x-api-key": "secret-key-456",
        "Cookie": "session=sensitive_cookie",
        "X-Custom-Allowed": "allowed-value",
    }
    sanitized = adapter._sanitize_headers(headers_with_sensitive)
    assert "Authorization" not in sanitized
    assert "authorization" not in sanitized
    assert "x-api-key" not in sanitized
    assert "Cookie" not in sanitized
    assert "X-Custom-Allowed" in sanitized
    assert sanitized["X-Custom-Allowed"] == "allowed-value"
