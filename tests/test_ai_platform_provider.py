"""AI Platform native transport: iB2B token exchange + chat call + refresh."""
import io
import json

import pytest

from src.efp_runtime.llm import provider as provider_mod
from src.efp_runtime.llm.provider import (
    AIPlatformHTTPTransport,
    AIPlatformProvider,
    ProviderTransportError,
    RecordingTransport,
    ai_platform_endpoint_is_responses,
)
from src.efp_runtime.llm.request import ProviderRequest, RequestToolSchema
from src.efp_runtime.loop import RuntimeRequest


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def _urlopen_script(steps):
    """steps: list of dict-body (returned) or urllib HTTPError (raised)."""
    calls = []
    it = iter(steps)

    def _open(request, timeout=None):
        calls.append(request)
        step = next(it)
        if isinstance(step, BaseException):
            raise step
        return _FakeResp(json.dumps(step).encode("utf-8"))

    return _open, calls


def _http_error(code):
    from urllib import error as urllib_error

    return urllib_error.HTTPError("https://x", code, "denied", {}, io.BytesIO(b'{"error":"nope"}'))


def _transport(**kw):
    base = dict(
        chat_endpoint="https://chat.int/v1/api/v1/chat/completions",
        ib2b_endpoint="https://ib2b.int/dsp",
        username="u",
        password="pw",
        usercase="uc",
        trust_token_header="X-Trust",
        tracking_prefix="EFP",
    )
    base.update(kw)
    return AIPlatformHTTPTransport(**base)


def test_exchange_then_chat_carries_trust_token(monkeypatch):
    t = _transport()
    open_fn, calls = _urlopen_script([
        {"issued_token": "JWT-1"},
        {"choices": [{"message": {"content": "pong"}}]},
    ])
    monkeypatch.setattr(provider_mod.urllib_request, "urlopen", open_fn)

    out = t._send_sync({"model": "gpt-5.4", "messages": [{"role": "user", "content": "ping"}]})
    assert out["choices"][0]["message"]["content"] == "pong"
    assert len(calls) == 2
    # 1) iB2B exchange with the CREDENTIAL body
    body0 = json.loads(calls[0].data)
    assert body0["input_token_state"]["token_type"] == "CREDENTIAL"
    assert body0["input_token_state"]["username"] == "u"
    # 2) chat call carries the exchanged JWT in the trust-token header + tracking
    header_values = list(calls[1].headers.values())
    assert "JWT-1" in header_values
    assert any(str(v).startswith("EFP-") for v in header_values)


def test_token_is_reused_until_expiry(monkeypatch):
    t = _transport()
    open_fn, calls = _urlopen_script([
        {"issued_token": "JWT-1"},
        {"choices": []},
        {"choices": []},
    ])
    monkeypatch.setattr(provider_mod.urllib_request, "urlopen", open_fn)
    t._send_sync({"messages": []})
    t._send_sync({"messages": []})
    # 1 exchange + 2 chat calls (token reused for the second)
    assert len(calls) == 3


def test_reexchanges_once_on_401(monkeypatch):
    t = _transport()
    open_fn, calls = _urlopen_script([
        {"issued_token": "JWT-1"},   # initial exchange
        _http_error(401),            # chat -> 401
        {"issued_token": "JWT-2"},   # forced re-exchange
        {"choices": []},             # chat retry OK
    ])
    monkeypatch.setattr(provider_mod.urllib_request, "urlopen", open_fn)
    t._send_sync({"messages": []})
    assert len(calls) == 4
    assert "JWT-2" in list(calls[3].headers.values())


def test_missing_credentials_raises(monkeypatch):
    t = _transport(username="", password="", ib2b_endpoint="")
    with pytest.raises(ProviderTransportError):
        t._send_sync({"messages": []})


def test_resolve_ai_platform_model_coerces_non_ai_platform_model(monkeypatch):
    import src.gateway.runtime_chat as rc

    monkeypatch.setattr(rc.config, "_config", {"llm": {"provider": "ai_platform"}}, raising=False)
    assert rc._resolve_ai_platform_model("gpt-5.6-terra") == "gpt-5.4"  # copilot id coerced
    assert rc._resolve_ai_platform_model("gpt-5.4") == "gpt-5.4"
    assert rc._resolve_ai_platform_model(None) == "gpt-5.4"


def test_build_llm_provider_dispatches_to_ai_platform(monkeypatch):
    import src.gateway.runtime_chat as rc
    from src.efp_runtime.llm.provider import AIPlatformProvider

    monkeypatch.setattr(
        rc.config,
        "_config",
        {
            "llm": {
                "provider": "ai_platform",
                "model": "gpt-5.4",
                "ai_platform": {
                    "chat": {"host": "https://c", "uri": "/v1/api/v1/chat/completions"},
                    "auth": {"username": "u", "password": "p"},
                },
            }
        },
        raising=False,
    )
    provider, model = rc._build_llm_provider(None)
    assert isinstance(provider, AIPlatformProvider)
    assert model == "gpt-5.4"


def test_ai_platform_provider_removes_metadata_from_payload():
    provider = AIPlatformProvider(
        transport=RecordingTransport([]),
        model="gpt-5.4",
        metadata={"trace_id": "trace-1"},
        reasoning_effort="high",
        usercase="uc",
    )
    request = RuntimeRequest(
        session_id="session-1",
        messages=[],
        iteration=1,
        max_iterations=1,
        metadata={"request_id": "request-1"},
        provider_request=ProviderRequest(
            messages=[],
            metadata={"request_id": "request-1"},
        ),
    )

    payload = provider.build_payload(request)

    assert "metadata" not in payload
    assert payload["model"] == "gpt-5.4"
    assert payload["reasoning_effort"] == "high"
    assert payload["user"] == "uc"


def test_build_ai_platform_provider_requires_chat_host(monkeypatch):
    import src.gateway.runtime_chat as rc

    monkeypatch.setattr(
        rc.config,
        "_config",
        {"llm": {"provider": "ai_platform", "ai_platform": {"auth": {"username": "u"}}}},
        raising=False,
    )
    with pytest.raises(rc.RuntimeChatError):
        rc._build_llm_provider(None)


def test_direct_token_skips_exchange(monkeypatch):
    t = _transport(username="", password="", ib2b_endpoint="", token="JWT-DIRECT")
    open_fn, calls = _urlopen_script([{"choices": []}])
    monkeypatch.setattr(provider_mod.urllib_request, "urlopen", open_fn)
    t._send_sync({"messages": []})
    assert len(calls) == 1  # no exchange, straight to chat
    assert "JWT-DIRECT" in list(calls[0].headers.values())


# --- Responses endpoint ------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://gw.int/v1/uc/responses", True),
        ("https://gw.int/v1/uc/responses/", True),
        ("https://gw.int/v1/uc/RESPONSES", True),
        ("https://gw.int/v1/uc/responses/compact", True),
        ("https://gw.int/v1/uc/responses?stream=true", True),
        ("https://gw.int/v1/api/v1/chat/completions", False),
        ("https://gw.int/v1/responses-legacy/chat", False),
        ("", False),
        (None, False),
    ],
)
def test_ai_platform_endpoint_is_responses(url, expected):
    assert ai_platform_endpoint_is_responses(url) is expected


def test_transport_adds_bearer_only_for_responses_endpoint(monkeypatch):
    chat = _transport(token="JWT-1")
    responses = _transport(chat_endpoint="https://gw.int/v1/uc/responses", token="JWT-1")

    assert "Authorization" not in chat._headers()
    assert responses._headers()["Authorization"] == "Bearer JWT-1"
    # The trust-token header stays on both endpoints.
    assert responses._headers()["X-Trust"] == "JWT-1"
    assert responses._headers(stream=True)["Accept"] == "text/event-stream"


def _runtime_request(*, with_tools: bool = False) -> RuntimeRequest:
    tools = (
        [RequestToolSchema(id="read_file", name="read_file", description="Read a file", json_schema={"type": "object"})]
        if with_tools
        else []
    )
    return RuntimeRequest(
        session_id="session-1",
        messages=[],
        iteration=1,
        max_iterations=1,
        metadata={"request_id": "request-1"},
        provider_request=ProviderRequest(messages=[], tools=tools, metadata={"request_id": "request-1"}),
    )


def test_ai_platform_provider_defaults_to_chat_endpoint():
    provider = AIPlatformProvider(transport=RecordingTransport([]), model="gpt-5.4")
    assert provider.endpoint == "chat"


def test_ai_platform_responses_payload_uses_reasoning_object_not_reasoning_effort():
    provider = AIPlatformProvider(
        transport=RecordingTransport([]),
        model="gpt-5.6-terra",
        endpoint="responses",
        metadata={"trace_id": "trace-1"},
        reasoning_effort="high",
        usercase="uc",
        stream=True,
    )

    payload = provider.build_payload(_runtime_request(with_tools=True))

    # This is the shape the gateway accepts for gpt-5.6 with function tools:
    # Responses-style ``reasoning`` plus ``input``, and none of the
    # chat/completions-only fields it rejects.
    assert payload["model"] == "gpt-5.6-terra"
    assert payload["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in payload
    assert "input" in payload and "messages" not in payload
    assert payload["stream"] is True
    assert payload["tools"] and payload["tools"][0]["name"] == "read_file"
    assert "metadata" not in payload
    assert "user" not in payload  # the use case is routed via the URL


def test_ai_platform_responses_payload_omits_reasoning_when_unset():
    provider = AIPlatformProvider(transport=RecordingTransport([]), model="gpt-5.4", endpoint="responses")
    payload = provider.build_payload(_runtime_request())
    assert "reasoning" not in payload
    assert "reasoning_effort" not in payload


def test_ai_platform_chat_payload_is_unchanged_by_responses_support():
    provider = AIPlatformProvider(
        transport=RecordingTransport([]),
        model="gpt-5.4",
        reasoning_effort="high",
        usercase="uc",
    )
    payload = provider.build_payload(_runtime_request(with_tools=True))
    assert payload["reasoning_effort"] == "high"
    assert payload["user"] == "uc"
    assert "reasoning" not in payload
    assert "messages" in payload


def _ai_platform_runtime_config(ai_platform: dict) -> dict:
    return {"llm": {"provider": "ai_platform", "model": "gpt-5.4", "ai_platform": ai_platform}}


def test_build_ai_platform_provider_prefers_configured_responses_endpoint(monkeypatch):
    import src.gateway.runtime_chat as rc

    monkeypatch.setattr(
        rc.config,
        "_config",
        _ai_platform_runtime_config(
            {
                "chat": {"host": "https://gw.int", "uri": "/v1/api/v1/chat/completions"},
                "responses": {"uri": "/v1/uc/responses"},  # host falls back to chat.host
                "auth": {"username": "u", "password": "p", "usercase": "uc"},
            }
        ),
        raising=False,
    )

    provider, model = rc._build_llm_provider(None)

    assert model == "gpt-5.4"
    assert provider.endpoint == "responses"
    assert provider.transport.endpoint == "https://gw.int/v1/uc/responses"


def test_build_ai_platform_provider_uses_responses_host_when_given(monkeypatch):
    import src.gateway.runtime_chat as rc

    monkeypatch.setattr(
        rc.config,
        "_config",
        _ai_platform_runtime_config(
            {
                "chat": {"host": "https://chat.int"},
                "responses": {"host": "https://responses.int", "uri": "/v1/uc/responses"},
                "auth": {"username": "u", "password": "p", "usercase": "uc"},
            }
        ),
        raising=False,
    )

    provider, _model = rc._build_llm_provider(None)

    assert provider.transport.endpoint == "https://responses.int/v1/uc/responses"


def test_build_ai_platform_provider_stays_on_chat_without_responses_uri(monkeypatch):
    import src.gateway.runtime_chat as rc

    monkeypatch.setattr(
        rc.config,
        "_config",
        _ai_platform_runtime_config(
            {
                "chat": {"host": "https://gw.int", "uri": "/v1/api/v1/chat/completions"},
                "responses": {"host": "https://gw.int"},  # a host alone selects nothing
                "auth": {"username": "u", "password": "p", "usercase": "uc"},
            }
        ),
        raising=False,
    )

    provider, _model = rc._build_llm_provider(None)

    assert provider.endpoint == "chat"
    assert provider.transport.endpoint == "https://gw.int/v1/api/v1/chat/completions"


def test_build_ai_platform_provider_requires_some_endpoint_host(monkeypatch):
    import src.gateway.runtime_chat as rc

    monkeypatch.setattr(
        rc.config,
        "_config",
        _ai_platform_runtime_config(
            {
                "responses": {"uri": "/v1/uc/responses"},
                "auth": {"username": "u", "password": "p", "usercase": "uc"},
            }
        ),
        raising=False,
    )

    with pytest.raises(rc.RuntimeChatError) as excinfo:
        rc._build_llm_provider(None)
    assert excinfo.value.error_type == "invalid_config"
    assert "responses.host" in str(excinfo.value)


def test_portal_managed_field_tree_accepts_ai_platform_responses_block():
    from src.config import Config

    ai_platform = Config.PORTAL_MANAGED_FIELD_TREE["llm"]["ai_platform"]
    assert ai_platform["responses"] == {"host": True, "uri": True}
    assert ai_platform["chat"] == {"host": True, "uri": True}
