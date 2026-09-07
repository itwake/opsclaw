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


def test_resolve_ai_platform_model_keeps_every_catalog_model(monkeypatch):
    # Every model the Portal offers for ai_platform must survive untouched;
    # the previous one-entry list coerced gpt-5.6-* back to gpt-5.4, which made
    # both the runtime profile model and the chat model switch a no-op.
    import src.gateway.runtime_chat as rc

    monkeypatch.setattr(rc.config, "_config", {"llm": {"provider": "ai_platform"}}, raising=False)
    for model_id in ("gpt-5.4", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"):
        assert rc._resolve_ai_platform_model(model_id) == model_id


def test_resolve_ai_platform_model_falls_back_from_profile_config(monkeypatch):
    import src.gateway.runtime_chat as rc

    monkeypatch.setattr(
        rc.config, "_config", {"llm": {"provider": "ai_platform", "model": "gpt-5.6-sol"}}, raising=False
    )
    assert rc._resolve_ai_platform_model(None) == "gpt-5.6-sol"  # profile model, no request override
    assert rc._resolve_ai_platform_model("gpt-5.6-terra") == "gpt-5.6-terra"  # request override wins


def test_resolve_ai_platform_model_coerces_non_ai_platform_model(monkeypatch):
    import src.gateway.runtime_chat as rc

    monkeypatch.setattr(rc.config, "_config", {"llm": {"provider": "ai_platform"}}, raising=False)
    assert rc._resolve_ai_platform_model("gpt-5.5") == "gpt-5.4"  # copilot-only id coerced
    assert rc._resolve_ai_platform_model("not-a-model") == "gpt-5.4"
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


def _ai_platform_request(*, with_tools: bool) -> RuntimeRequest:
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


@pytest.mark.parametrize("model", ["gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-sol", "ai_platform/gpt-5.6-terra"])
def test_ai_platform_gpt56_with_tools_disables_reasoning_effort(model):
    # The gateway's /v1/chat/completions rejects function tools combined with a
    # reasoning effort for gpt-5.6 ("set reasoning_effort to 'none'").
    provider = AIPlatformProvider(transport=RecordingTransport([]), model=model, reasoning_effort="high")

    payload = provider.build_payload(_ai_platform_request(with_tools=True))

    assert payload["tools"]
    assert payload["reasoning_effort"] == "none"


def test_ai_platform_gpt56_without_tools_keeps_reasoning_effort():
    provider = AIPlatformProvider(transport=RecordingTransport([]), model="gpt-5.6-terra", reasoning_effort="high")

    payload = provider.build_payload(_ai_platform_request(with_tools=False))

    assert not payload.get("tools")
    assert payload["reasoning_effort"] == "high"


def test_ai_platform_gpt54_with_tools_keeps_reasoning_effort():
    provider = AIPlatformProvider(transport=RecordingTransport([]), model="gpt-5.4", reasoning_effort="high")

    payload = provider.build_payload(_ai_platform_request(with_tools=True))

    assert payload["tools"]
    assert payload["reasoning_effort"] == "high"


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
