"""Provider transport facade for EFP runtime OpenAI-compatible clients.

The facade classes in this module do not import an OpenAI SDK. A caller injects
the transport boundary, which receives the projected payload and returns raw
provider data.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import os
import re
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Union
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from .adapter import DefaultLLMEventAdapter, LLMEventAdapter
from .errors import (
    ProviderContextOverflowError,
    ProviderTransientError,
)
from .models import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROVIDER_ID,
    SUPPORTED_COPILOT_MODEL_IDS,
    canonicalize_copilot_model_id,
)
from .openai import (
    normalize_responses_call_id,
    provider_request_to_openai_chat,
    provider_request_to_openai_responses,
)

if TYPE_CHECKING:
    from ..loop.provider import ProviderOutput, RuntimeRequest


TransportOutput = Union[
    Mapping[str, Any],
    Iterable[Mapping[str, Any]],
    AsyncIterable[Mapping[str, Any]],
]


class ProviderTransport(Protocol):
    """Injectable boundary that sends a projected provider payload."""

    async def send(self, payload: dict[str, Any]) -> TransportOutput:
        ...


class ProviderTransportError(RuntimeError):
    """Raised by transports or helpers when provider transport fails."""


# Statuses where the request was fine and the far side was not: retrying the
# same request is the correct response. Anything else (400, 401 after a token
# refresh, 403, 404, 422) will fail again identically.
RETRYABLE_TRANSPORT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class ProviderTransportTransientError(ProviderTransportError, ProviderTransientError):
    """A transport failure worth retrying.

    Inherits from both so it lands in the loop's ``except ProviderTransientError``
    branch and still matches the ``except ProviderTransportError`` handlers at
    the gateway boundary once retries are exhausted.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        ProviderTransientError.__init__(
            self,
            message,
            retryable=True,
            code=code or "transport_transient",
            metadata={"status_code": status_code} if status_code is not None else {},
        )
        self.status_code = status_code


def _transport_error_for_http_status(message: str, status_code: int | None) -> ProviderTransportError:
    """Pick the retryable or fatal transport error for an HTTP status."""

    if status_code in RETRYABLE_TRANSPORT_STATUS_CODES:
        return ProviderTransportTransientError(message, status_code=status_code)
    return ProviderTransportError(message)


class ProviderModelUnavailableError(ProviderTransportError):
    """Raised when GitHub Copilot rejects the requested model."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        available_models_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.available_models_text = available_models_text


@dataclass(frozen=True)
class CopilotTokenExchange:
    """Result of exchanging a GitHub source token for a Copilot plugin token."""

    token: str
    expires_at: int
    api_base_url: str | None = None


DEFAULT_COPILOT_REASONING_EFFORT = "high"
DEFAULT_GITHUB_COPILOT_TIMEOUT_SECONDS = 300
GITHUB_COPILOT_TOKEN_REFRESH_MARGIN_SECONDS = 300
SUPPORTED_COPILOT_REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
GITHUB_SOURCE_TOKEN_PREFIXES = (
    "ghp_",
    "ghu_",
    "gho_",
    "ghs_",
    "ghr_",
    "github_pat_",
)
_PROXY_ENDPOINT_PATTERN = re.compile(r"(?:^|[;&,\s])proxy-ep=([^;&,\s]+)")
# Substrings that identify a provider 400/413 as "the prompt is too big". The
# message is the primary discriminator, never the generic error code: the
# Copilot /responses endpoint returns "invalid_request_body" for tool-schema and
# call-id rejections too (see _sanitize_copilot_responses_payload), so matching
# on that code would classify a malformed-tool 400 as a context overflow and send
# the runner into a pointless compacted retry. Only the unambiguous codes in
# _CONTEXT_OVERFLOW_ERROR_CODES below classify on their own.
_CONTEXT_OVERFLOW_MESSAGE_MARKERS = (
    "exceeds the context window",
    "context_length_exceeded",
    "context length exceeded",
    "maximum context length",
)
# Secondary allowlist of provider error codes that mean overflow on their own.
_CONTEXT_OVERFLOW_ERROR_CODES = ("context_length_exceeded",)


class GitHubCopilotHTTPTransport:
    """Standard-library HTTP JSON transport for GitHub Copilot Responses."""

    DEFAULT_BASE_URL = "https://api.githubcopilot.com"
    DEFAULT_GITHUB_API_BASE_URL = "https://api.github.com"
    RESPONSES_PATH = "/responses"

    def __init__(
        self,
        *,
        token: str,
        base_url: Optional[str] = None,
        timeout: float | None = DEFAULT_GITHUB_COPILOT_TIMEOUT_SECONDS,
        github_api_base_url: Optional[str] = None,
        user_agent: str = "GitHubCopilotChat/0.41.0",
        editor_version: str = "vscode/1.133.0",
        editor_plugin_version: str = "copilot-chat/0.41.0",
        integration_id: str = "vscode-chat",
        initiator: str = "agent",
        exchange_source_token: bool = True,
    ) -> None:
        credential = _required_non_empty_string(token, "token")
        self.timeout = timeout
        self.user_agent = _required_non_empty_string(user_agent, "user_agent")
        self.editor_version = _required_non_empty_string(
            editor_version,
            "editor_version",
        )
        self.editor_plugin_version = _required_non_empty_string(
            editor_plugin_version,
            "editor_plugin_version",
        )
        self.integration_id = _required_non_empty_string(
            integration_id,
            "integration_id",
        )
        self.initiator = _required_non_empty_string(initiator, "initiator")
        self.github_api_base_url = _normalize_base_url(
            github_api_base_url,
            default=self.DEFAULT_GITHUB_API_BASE_URL,
        )
        explicit_base_url = (
            _normalize_base_url(base_url) if base_url is not None else None
        )
        self.token_expires_at: int | None = None
        self.token_source = "copilot"
        self._explicit_base_url = explicit_base_url
        self._source_credential: str | None = None
        self._refresh_lock = threading.Lock()
        self._token = credential
        self.base_url = _normalize_base_url(self._explicit_base_url)
        self.endpoint = "{0}{1}".format(self.base_url, self.RESPONSES_PATH)

        if exchange_source_token and is_github_source_token(credential):
            self._source_credential = credential
            self.token_source = "github_exchange"
            self._refresh_source_token(force=True)

    async def send(self, payload: dict[str, Any]) -> TransportOutput:
        if payload.get("stream") is True:
            return self._send_stream(payload)
        return await asyncio.to_thread(self._send_sync, payload)

    async def _send_stream(self, payload: dict[str, Any]) -> AsyncIterator[Mapping[str, Any]]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        done = object()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for chunk in self._send_stream_sync(payload):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except BaseException as exc:  # noqa: BLE001 - propagated through async iterator.
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if not task.done():
            await task

    def _send_stream_sync(self, payload: dict[str, Any]) -> Iterable[Mapping[str, Any]]:
        try:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProviderTransportError(
                "GitHub Copilot HTTP transport received a non-JSON payload"
            ) from exc

        self._refresh_source_token_if_needed()
        request = urllib_request.Request(
            self.endpoint,
            data=body,
            headers=self._headers(stream=True),
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                yield from parse_sse_json_events(_iter_response_lines(response))
        except urllib_error.HTTPError as exc:
            response_text = _read_http_error_body(exc)
            if self._refresh_after_token_expired_http_error(exc, response_text):
                request = urllib_request.Request(
                    self.endpoint,
                    data=body,
                    headers=self._headers(stream=True),
                    method="POST",
                )
                try:
                    with urllib_request.urlopen(request, timeout=self.timeout) as response:
                        yield from parse_sse_json_events(_iter_response_lines(response))
                        return
                except urllib_error.HTTPError as retry_exc:
                    exc = retry_exc
                    response_text = _read_http_error_body(retry_exc)
                except urllib_error.URLError as retry_exc:
                    reason = _redact_secret(
                        str(getattr(retry_exc, "reason", retry_exc)),
                        self._token,
                    )
                    raise ProviderTransportTransientError(
                        "GitHub Copilot HTTP transport failed: {0}".format(reason)
                    ) from None
                except TimeoutError:
                    raise ProviderTransportTransientError(
                        _format_timeout_message(
                            "GitHub Copilot HTTP transport",
                            self.timeout,
                        )
                    ) from None
            model_unavailable_error = _model_unavailable_error_from_http_error(
                exc,
                response_text=response_text,
                token=self._token,
            )
            if model_unavailable_error is not None:
                raise model_unavailable_error from None
            overflow_error = _context_overflow_error_from_http_error(
                exc,
                response_text=response_text,
                token=self._token,
            )
            if overflow_error is not None:
                raise overflow_error from None
            message = _format_http_error(
                exc,
                self._token,
                response_text=response_text,
            )
            raise _transport_error_for_http_status(message, getattr(exc, "code", None)) from None
        except urllib_error.URLError as exc:
            reason = _redact_secret(str(getattr(exc, "reason", exc)), self._token)
            # The connection never completed, so nothing was consumed upstream.
            raise ProviderTransportTransientError(
                "GitHub Copilot HTTP transport failed: {0}".format(reason)
            ) from None
        except TimeoutError:
            raise ProviderTransportTransientError(
                _format_timeout_message("GitHub Copilot HTTP transport", self.timeout)
            ) from None

    def _send_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProviderTransportError(
                "GitHub Copilot HTTP transport received a non-JSON payload"
            ) from exc

        self._refresh_source_token_if_needed()
        request = urllib_request.Request(
            self.endpoint,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read()
        except urllib_error.HTTPError as exc:
            response_text = _read_http_error_body(exc)
            if self._refresh_after_token_expired_http_error(exc, response_text):
                request = urllib_request.Request(
                    self.endpoint,
                    data=body,
                    headers=self._headers(),
                    method="POST",
                )
                try:
                    with urllib_request.urlopen(request, timeout=self.timeout) as response:
                        raw_body = response.read()
                except urllib_error.HTTPError as retry_exc:
                    exc = retry_exc
                    response_text = _read_http_error_body(retry_exc)
                except urllib_error.URLError as retry_exc:
                    reason = _redact_secret(
                        str(getattr(retry_exc, "reason", retry_exc)),
                        self._token,
                    )
                    raise ProviderTransportError(
                        "GitHub Copilot HTTP transport failed: {0}".format(reason)
                    ) from None
                except TimeoutError:
                    raise ProviderTransportError(
                        _format_timeout_message(
                            "GitHub Copilot HTTP transport",
                            self.timeout,
                        )
                    ) from None
                else:
                    return self._decode_json_response(raw_body)
            model_unavailable_error = _model_unavailable_error_from_http_error(
                exc,
                response_text=response_text,
                token=self._token,
            )
            if model_unavailable_error is not None:
                raise model_unavailable_error from None
            overflow_error = _context_overflow_error_from_http_error(
                exc,
                response_text=response_text,
                token=self._token,
            )
            if overflow_error is not None:
                raise overflow_error from None
            message = _format_http_error(
                exc,
                self._token,
                response_text=response_text,
            )
            raise ProviderTransportError(message) from None
        except urllib_error.URLError as exc:
            reason = _redact_secret(str(getattr(exc, "reason", exc)), self._token)
            raise ProviderTransportError(
                "GitHub Copilot HTTP transport failed: {0}".format(reason)
            ) from None
        except TimeoutError:
            raise ProviderTransportError(
                _format_timeout_message("GitHub Copilot HTTP transport", self.timeout)
            ) from None

        return self._decode_json_response(raw_body)

    def _decode_json_response(self, raw_body: bytes) -> dict[str, Any]:
        try:
            text = raw_body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProviderTransportError(
                "GitHub Copilot HTTP transport returned non-UTF-8 response data"
            ) from exc

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderTransportError(
                "GitHub Copilot HTTP transport returned invalid JSON"
            ) from exc
        if not isinstance(data, dict):
            raise ProviderTransportError(
                "GitHub Copilot HTTP transport returned a non-object JSON response"
            )
        return data

    def _refresh_source_token_if_needed(self) -> None:
        self._refresh_source_token(force=False)

    def _refresh_source_token(self, *, force: bool) -> None:
        if self._source_credential is None:
            return
        if not force and not self._token_refresh_due():
            return
        with self._refresh_lock:
            if not force and not self._token_refresh_due():
                return
            exchange = exchange_github_token_for_copilot_token(
                self._source_credential,
                github_api_base_url=self.github_api_base_url,
                timeout=self.timeout,
                user_agent=self.user_agent,
                editor_version=self.editor_version,
                editor_plugin_version=self.editor_plugin_version,
                integration_id=self.integration_id,
            )
            self._apply_token_exchange(exchange)

    def _token_refresh_due(self) -> bool:
        if self._source_credential is None or self.token_expires_at is None:
            return False
        return (
            self.token_expires_at - int(time.time())
            < GITHUB_COPILOT_TOKEN_REFRESH_MARGIN_SECONDS
        )

    def _apply_token_exchange(self, exchange: CopilotTokenExchange) -> None:
        self._token = exchange.token
        self.token_expires_at = exchange.expires_at
        self.base_url = _normalize_base_url(
            self._explicit_base_url or exchange.api_base_url
        )
        self.endpoint = "{0}{1}".format(self.base_url, self.RESPONSES_PATH)

    def _refresh_after_token_expired_http_error(
        self,
        exc: urllib_error.HTTPError,
        response_text: str | None,
    ) -> bool:
        if self._source_credential is None:
            return False
        if getattr(exc, "code", None) != 401:
            return False
        reason = getattr(exc, "reason", None) or getattr(exc, "msg", "")
        combined = "{0} {1}".format(reason, response_text or "").lower()
        if "expired" not in combined:
            return False
        self._refresh_source_token(force=True)
        return True

    def _headers(self, *, stream: bool = False) -> dict[str, str]:
        return {
            "Authorization": "Bearer {0}".format(self._token),
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/vnd.github.copilot-chat-preview+json",
            "User-Agent": self.user_agent,
            "Editor-Version": self.editor_version,
            "Editor-Plugin-Version": self.editor_plugin_version,
            "Copilot-Integration-Id": self.integration_id,
            "Openai-Intent": "conversation-edits",
            "x-initiator": self.initiator,
        }


class OpenAICompatibleProvider:
    """LLMProvider implementation for OpenAI-compatible payload transports."""

    def __init__(
        self,
        *,
        model: str,
        transport: ProviderTransport,
        endpoint: str = "chat",
        instructions: Optional[str] = None,
        stream: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        adapter: Optional[LLMEventAdapter] = None,
    ) -> None:
        if endpoint not in {"chat", "responses"}:
            raise ValueError("endpoint must be 'chat' or 'responses'")
        self.model = model
        self.transport = transport
        self.endpoint = endpoint
        self.instructions = instructions
        self.stream = stream
        self.metadata = dict(metadata or {})
        self.reasoning_effort = reasoning_effort
        self.adapter = adapter or DefaultLLMEventAdapter()

    def build_payload(self, request: RuntimeRequest) -> dict[str, Any]:
        """Project a RuntimeRequest into the configured provider payload."""

        payload_model = _requested_model(request) or self.model
        if self.endpoint == "responses":
            return provider_request_to_openai_responses(
                request.provider_request,
                model=payload_model,
                instructions=self.instructions,
                stream=self.stream,
                metadata=self.metadata,
                reasoning_effort=self.reasoning_effort,
            )
        return provider_request_to_openai_chat(
            request.provider_request,
            model=payload_model,
            instructions=self.instructions,
            stream=self.stream,
            metadata=self.metadata,
        )

    async def invoke(self, request: RuntimeRequest) -> ProviderOutput:
        payload = self.build_payload(request)
        try:
            raw_output = self.transport.send(payload)
            if inspect.isawaitable(raw_output):
                raw_output = await raw_output
        except ProviderContextOverflowError:
            # A context overflow is recoverable by the runner's compacted retry,
            # which only sees it if it stays an exception. Absorbing it into an
            # error mapping ends the run. Deliberately narrow: every other
            # transport failure keeps mapping to a provider error response.
            raise
        except Exception as exc:
            return self._transport_error_response(exc)

        if self.stream:
            if isinstance(raw_output, Mapping):
                return self.adapter.normalize_response(raw_output)
            return self.adapter.normalize_stream(raw_output)

        if not isinstance(raw_output, Mapping):
            return self._transport_error_response(
                ProviderTransportError("non-stream transport returned a stream response")
            )
        return raw_output

    def _transport_error_response(self, exc: BaseException) -> dict[str, Any]:
        message = str(exc) or exc.__class__.__name__
        return {
            "error": {
                "message": "OpenAI-compatible transport failed: {0}".format(message),
                "type": "transport_error",
                "exception": exc.__class__.__name__,
            },
            "metadata": {
                "provider": "openai",
                "endpoint": self.endpoint,
                "model": self.model,
            },
        }


class GitHubCopilotProvider(OpenAICompatibleProvider):
    """Thin OpenAI-compatible facade for GitHub Copilot payload tests."""

    def __init__(
        self,
        *,
        transport: ProviderTransport,
        model: str = DEFAULT_MODEL_ID,
        endpoint: str = "responses",
        instructions: Optional[str] = None,
        stream: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        reasoning_effort: str = DEFAULT_COPILOT_REASONING_EFFORT,
        adapter: Optional[LLMEventAdapter] = None,
    ) -> None:
        if endpoint != "responses":
            raise ValueError("GitHub Copilot provider endpoint must be 'responses'")
        canonical_model = canonicalize_copilot_model_id(model)
        canonical_reasoning_effort = validate_copilot_reasoning_effort(reasoning_effort)
        provider_metadata = dict(metadata or {})
        provider_metadata.update(
            {
                "provider": DEFAULT_PROVIDER_ID,
                "provider_id": DEFAULT_PROVIDER_ID,
            }
        )
        super().__init__(
            model=canonical_model,
            transport=transport,
            endpoint=endpoint,
            instructions=instructions,
            stream=stream,
            metadata=provider_metadata,
            reasoning_effort=canonical_reasoning_effort,
            adapter=adapter,
        )

    def build_payload(self, request: RuntimeRequest) -> dict[str, Any]:
        """Project a request and apply GitHub Copilot request quirks."""

        payload = super().build_payload(request)
        payload["model"] = canonicalize_copilot_model_id(payload.get("model"))
        if self.endpoint == "responses":
            payload["reasoning"] = {"effort": self.reasoning_effort}
        _inject_copilot_noop_tool_fallback(payload, request)
        return _sanitize_copilot_responses_payload(payload)

    def _transport_error_response(self, exc: BaseException) -> dict[str, Any]:
        response = super()._transport_error_response(exc)
        metadata = response.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["provider"] = DEFAULT_PROVIDER_ID
            metadata["provider_id"] = DEFAULT_PROVIDER_ID
        return response


DEFAULT_AI_PLATFORM_TOKEN_TTL_SECONDS = 30
AI_PLATFORM_TOKEN_REFRESH_MARGIN_SECONDS = 5
DEFAULT_AI_PLATFORM_TRUST_TOKEN_HEADER = "X-XXXX-E2E-Trust-Token"
DEFAULT_AI_PLATFORM_TRACKING_PREFIX = "EFP"


class AIPlatformHTTPTransport:
    """HTTP JSON transport for the AI Platform OpenAI-compatible chat endpoint.

    Auth is two-legged: username/password/usercase are exchanged at an iB2B STS
    endpoint for a short-lived JWT ("trust token"), which is sent in a
    configurable trust-token header on each chat call (plus tracking ids). The
    JWT is re-exchanged when missing or within a refresh margin of expiry, and
    once reactively on a 401/403.
    """

    def __init__(
        self,
        *,
        chat_endpoint: str,
        ib2b_endpoint: str = "",
        username: str = "",
        password: str = "",
        usercase: str = "",
        token: str = "",
        trust_token_header: str = DEFAULT_AI_PLATFORM_TRUST_TOKEN_HEADER,
        tracking_prefix: str = DEFAULT_AI_PLATFORM_TRACKING_PREFIX,
        timeout: float | None = DEFAULT_GITHUB_COPILOT_TIMEOUT_SECONDS,
        token_ttl_seconds: int = DEFAULT_AI_PLATFORM_TOKEN_TTL_SECONDS,
    ) -> None:
        self.endpoint = _required_non_empty_string(chat_endpoint, "chat_endpoint")
        self.ib2b_endpoint = (ib2b_endpoint or "").strip()
        self.timeout = timeout
        self.token_source = "ai_platform"
        self._username = (username or "").strip()
        self._password = (password or "").strip()
        self._usercase = (usercase or "").strip()
        self._trust_token_header = (trust_token_header or DEFAULT_AI_PLATFORM_TRUST_TOKEN_HEADER).strip()
        self._tracking_prefix = (tracking_prefix or DEFAULT_AI_PLATFORM_TRACKING_PREFIX).strip()
        self._token_ttl_seconds = max(1, int(token_ttl_seconds or DEFAULT_AI_PLATFORM_TOKEN_TTL_SECONDS))
        self._token = (token or "").strip()
        self._token_expires_at: float | None = (
            time.time() + self._token_ttl_seconds if self._token else None
        )
        self._refresh_lock = threading.Lock()

    # -- token management --------------------------------------------------

    def _token_is_fresh(self) -> bool:
        if not self._token:
            return False
        if self._token_expires_at is None:
            return True
        return self._token_expires_at > time.time() + AI_PLATFORM_TOKEN_REFRESH_MARGIN_SECONDS

    def _ensure_token(self, *, force: bool = False) -> None:
        if not force and self._token_is_fresh():
            return
        with self._refresh_lock:
            if not force and self._token_is_fresh():
                return
            self._exchange_token()

    def _exchange_token(self) -> None:
        if not (self._username and self._password and self.ib2b_endpoint):
            if self._token:
                return
            raise ProviderTransportError(
                "AI Platform requires a token, or username/password plus an iB2B endpoint."
            )
        body = json.dumps(
            {
                "input_token_state": {
                    "token_type": "CREDENTIAL",
                    "username": self._username,
                    "password": self._password,
                },
                "output_token_state": {"token_type": "JWT"},
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib_request.Request(
            self.ib2b_endpoint,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib_error.HTTPError as exc:
            text = _read_http_error_body(exc)
            raise ProviderTransportError(
                "AI Platform token exchange failed ({0}): {1}".format(
                    exc.code, _redact_secret(text, self._password)
                )
            ) from None
        except urllib_error.URLError as exc:
            reason = _redact_secret(str(getattr(exc, "reason", exc)), self._password)
            raise ProviderTransportError(
                "AI Platform token exchange failed: {0}".format(reason)
            ) from None
        except TimeoutError:
            raise ProviderTransportError(
                _format_timeout_message("AI Platform token exchange", self.timeout)
            ) from None
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderTransportError(
                "AI Platform token exchange returned invalid JSON"
            ) from exc
        token = str(data.get("issued_token") or "").strip() if isinstance(data, dict) else ""
        if not token:
            raise ProviderTransportError(
                "AI Platform token exchange did not return issued_token."
            )
        self._token = token
        self._token_expires_at = time.time() + self._token_ttl_seconds

    def _tracking_id(self) -> str:
        prefix = self._tracking_prefix or DEFAULT_AI_PLATFORM_TRACKING_PREFIX
        return "{0}-{1}".format(prefix, time.strftime("%Y%m%d%H%M%S", time.gmtime()))

    def _headers(self, *, stream: bool = False) -> dict[str, str]:
        tracking = self._tracking_id()
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            self._trust_token_header: self._token,
            "x-correlation-id": tracking,
            "x-usersession-id": tracking,
        }

    @staticmethod
    def _should_reexchange(exc: urllib_error.HTTPError) -> bool:
        return getattr(exc, "code", None) in (401, 403)

    # -- send --------------------------------------------------------------

    async def send(self, payload: dict[str, Any]) -> TransportOutput:
        if payload.get("stream") is True:
            return self._send_stream(payload)
        return await asyncio.to_thread(self._send_sync, payload)

    async def _send_stream(self, payload: dict[str, Any]) -> AsyncIterator[Mapping[str, Any]]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        done = object()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for chunk in self._send_stream_sync(payload):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except BaseException as exc:  # noqa: BLE001 - propagated through async iterator.
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    def _encode_payload(self, payload: dict[str, Any]) -> bytes:
        try:
            return json.dumps(payload, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProviderTransportError(
                "AI Platform HTTP transport received a non-JSON payload"
            ) from exc

    def _send_stream_sync(self, payload: dict[str, Any]) -> Iterable[Mapping[str, Any]]:
        body = self._encode_payload(payload)
        self._ensure_token()
        for attempt in range(2):
            request = urllib_request.Request(
                self.endpoint, data=body, headers=self._headers(stream=True), method="POST"
            )
            try:
                with urllib_request.urlopen(request, timeout=self.timeout) as response:
                    yield from parse_sse_json_events(_iter_response_lines(response))
                return
            except urllib_error.HTTPError as exc:
                if attempt == 0 and self._should_reexchange(exc):
                    self._ensure_token(force=True)
                    continue
                text = _read_http_error_body(exc)
                overflow_error = _context_overflow_error_from_http_error(
                    exc,
                    response_text=text,
                    token=self._token,
                    provider_label="AI Platform",
                )
                if overflow_error is not None:
                    raise overflow_error from None
                raise _transport_error_for_http_status(
                    "AI Platform HTTP transport failed ({0}): {1}".format(
                        exc.code, _redact_secret(text, self._token)
                    ),
                    getattr(exc, "code", None),
                ) from None
            except urllib_error.URLError as exc:
                reason = _redact_secret(str(getattr(exc, "reason", exc)), self._token)
                raise ProviderTransportTransientError(
                    "AI Platform HTTP transport failed: {0}".format(reason)
                ) from None
            except TimeoutError:
                raise ProviderTransportTransientError(
                    _format_timeout_message("AI Platform HTTP transport", self.timeout)
                ) from None

    def _send_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = self._encode_payload(payload)
        self._ensure_token()
        for attempt in range(2):
            request = urllib_request.Request(
                self.endpoint, data=body, headers=self._headers(), method="POST"
            )
            try:
                with urllib_request.urlopen(request, timeout=self.timeout) as response:
                    return self._decode_json_response(response.read())
            except urllib_error.HTTPError as exc:
                if attempt == 0 and self._should_reexchange(exc):
                    self._ensure_token(force=True)
                    continue
                text = _read_http_error_body(exc)
                overflow_error = _context_overflow_error_from_http_error(
                    exc,
                    response_text=text,
                    token=self._token,
                    provider_label="AI Platform",
                )
                if overflow_error is not None:
                    raise overflow_error from None
                raise _transport_error_for_http_status(
                    "AI Platform HTTP transport failed ({0}): {1}".format(
                        exc.code, _redact_secret(text, self._token)
                    ),
                    getattr(exc, "code", None),
                ) from None
            except urllib_error.URLError as exc:
                reason = _redact_secret(str(getattr(exc, "reason", exc)), self._token)
                raise ProviderTransportTransientError(
                    "AI Platform HTTP transport failed: {0}".format(reason)
                ) from None
            except TimeoutError:
                raise ProviderTransportTransientError(
                    _format_timeout_message("AI Platform HTTP transport", self.timeout)
                ) from None
        raise ProviderTransportError("AI Platform HTTP transport exhausted retries")

    def _decode_json_response(self, raw_body: bytes) -> dict[str, Any]:
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderTransportError(
                "AI Platform HTTP transport returned invalid JSON"
            ) from exc
        if not isinstance(data, dict):
            raise ProviderTransportError(
                "AI Platform HTTP transport returned a non-object JSON response"
            )
        return data


class AIPlatformProvider(OpenAICompatibleProvider):
    """OpenAI-compatible facade for the AI Platform chat/completions endpoint."""

    def __init__(
        self,
        *,
        transport: ProviderTransport,
        model: str,
        instructions: Optional[str] = None,
        stream: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        usercase: Optional[str] = None,
        adapter: Optional[LLMEventAdapter] = None,
    ) -> None:
        provider_metadata = dict(metadata or {})
        provider_metadata.update({"provider": "ai_platform", "provider_id": "ai_platform"})
        self._usercase = (usercase or "").strip()
        super().__init__(
            model=model,
            transport=transport,
            endpoint="chat",
            instructions=instructions,
            stream=stream,
            metadata=provider_metadata,
            reasoning_effort=reasoning_effort,
            adapter=adapter,
        )

    def build_payload(self, request: RuntimeRequest) -> dict[str, Any]:
        payload = super().build_payload(request)
        # The AI Platform chat endpoint rejects the EFP/OpenAI-compatible
        # extension field even though it is useful for other providers.
        payload.pop("metadata", None)
        if self.reasoning_effort:
            payload.setdefault("reasoning_effort", self.reasoning_effort)
        if self._usercase:
            payload.setdefault("user", self._usercase)
        return payload

    def _transport_error_response(self, exc: BaseException) -> dict[str, Any]:
        response = super()._transport_error_response(exc)
        metadata = response.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["provider"] = "ai_platform"
            metadata["provider_id"] = "ai_platform"
        return response


class RecordingTransport:
    """Small deterministic transport for tests and local prototypes."""

    def __init__(self, responses: Iterable[Union[TransportOutput, BaseException]]) -> None:
        self._responses = list(responses)
        self.payloads: List[dict[str, Any]] = []

    async def send(self, payload: dict[str, Any]) -> TransportOutput:
        self.payloads.append(deepcopy(payload))
        if not self._responses:
            raise AssertionError("RecordingTransport has no response left")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    @property
    def requests(self) -> List[dict[str, Any]]:
        return self.payloads

    @property
    def remaining(self) -> int:
        return len(self._responses)


def github_copilot_provider_from_env(
    *,
    model: str = DEFAULT_MODEL_ID,
    endpoint: str = "responses",
    instructions: Optional[str] = None,
    stream: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    adapter: Optional[LLMEventAdapter] = None,
    timeout: float | None = DEFAULT_GITHUB_COPILOT_TIMEOUT_SECONDS,
    user_agent: str = "GitHubCopilotChat/0.41.0",
    initiator: str = "agent",
    env: Optional[Mapping[str, str]] = None,
) -> GitHubCopilotProvider:
    """Create a GitHub Copilot provider using caller-supplied environment auth."""

    environ = os.environ if env is None else env
    token = _env_string(environ, "EFP_GITHUB_COPILOT_TOKEN") or _env_string(
        environ,
        "GITHUB_COPILOT_TOKEN",
    )
    if token is None:
        raise ProviderTransportError(
            "GitHub Copilot token is required; set EFP_GITHUB_COPILOT_TOKEN "
            "or GITHUB_COPILOT_TOKEN"
        )
    configured_reasoning_effort = (
        reasoning_effort
        or _env_string(environ, "EFP_GITHUB_COPILOT_REASONING_EFFORT")
        or _env_string(environ, "EFP_LLM_REASONING_EFFORT")
        or DEFAULT_COPILOT_REASONING_EFFORT
    )
    transport = GitHubCopilotHTTPTransport(
        token=token,
        base_url=_env_string(environ, "EFP_GITHUB_COPILOT_BASE_URL"),
        timeout=timeout,
        user_agent=user_agent,
        initiator=initiator,
    )
    return GitHubCopilotProvider(
        transport=transport,
        model=model,
        endpoint=endpoint,
        instructions=instructions,
        stream=stream,
        metadata=metadata,
        reasoning_effort=configured_reasoning_effort,
        adapter=adapter,
    )


def parse_sse_json_events(lines: Iterable[bytes | str]) -> Iterable[Mapping[str, Any]]:
    """Parse text/event-stream data blocks into JSON object chunks."""

    data_lines: list[str] = []

    def flush() -> Iterable[Mapping[str, Any]]:
        if not data_lines:
            return []
        raw_data = "\n".join(data_lines).strip()
        data_lines.clear()
        if not raw_data or raw_data == "[DONE]":
            return []
        try:
            parsed = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise ProviderTransportError(
                "GitHub Copilot HTTP transport returned invalid SSE JSON"
            ) from exc
        if isinstance(parsed, Mapping):
            return [dict(parsed)]
        return [{"data": parsed}]

    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = str(raw_line)
        line = line.rstrip("\r\n")
        if not line:
            yield from flush()
            continue
        if line.startswith("#"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    yield from flush()


def validate_copilot_reasoning_effort(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(_unsupported_reasoning_message(value))
    effort = value.strip().lower()
    if effort not in SUPPORTED_COPILOT_REASONING_EFFORTS:
        raise ValueError(_unsupported_reasoning_message(value))
    return effort


def _requested_model(request: RuntimeRequest) -> Optional[str]:
    requested_model = request.metadata.get("requested_model")
    if not isinstance(requested_model, str):
        return None
    requested_model = requested_model.strip()
    if not requested_model:
        return None
    return requested_model


def _inject_copilot_noop_tool_fallback(
    payload: dict[str, Any],
    request: RuntimeRequest,
) -> None:
    if request.provider_request.tools:
        return
    tools = payload.get("tools")
    if tools:
        return
    if not _provider_request_has_tool_call(request):
        return
    payload["tools"] = [_copilot_noop_tool_payload(payload)]


def _sanitize_copilot_responses_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    if "model" in payload:
        sanitized["model"] = payload["model"]
    if "input" in payload:
        sanitized["input"] = _sanitize_copilot_responses_input(payload.get("input"))
    tools = _sanitize_copilot_tools(payload.get("tools"))
    if tools:
        sanitized["tools"] = tools
    if "stream" in payload:
        sanitized["stream"] = payload["stream"]
    if payload.get("instructions") is not None:
        sanitized["instructions"] = payload["instructions"]
    if payload.get("reasoning") is not None:
        sanitized["reasoning"] = deepcopy(payload["reasoning"])
    return sanitized


def _sanitize_copilot_responses_input(input_value: Any) -> list[dict[str, Any]]:
    if not isinstance(input_value, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for item in input_value:
        sanitized.extend(_sanitize_copilot_responses_input_item(item))
    return sanitized


def _sanitize_copilot_responses_input_item(item: Any) -> list[dict[str, Any]]:
    if not isinstance(item, Mapping):
        return []
    item_type = item.get("type")
    if item_type == "function_call":
        return [_sanitize_copilot_function_call_item(item)]
    if item_type == "function_call_output":
        return [_sanitize_copilot_function_call_output_item(item)]
    if "content" not in item:
        return []

    role = item.get("role")
    if not isinstance(role, str) or not role:
        role = "user"
    content = item.get("content")
    if isinstance(content, str):
        return [
            {
                "role": role,
                "content": [
                    {"type": _copilot_text_content_type(role), "text": content}
                ],
            }
        ]
    if not isinstance(content, list):
        return []

    sanitized: list[dict[str, Any]] = []
    buffered_content: list[dict[str, Any]] = []

    def flush_message() -> None:
        if not buffered_content:
            return
        sanitized.append({"role": role, "content": list(buffered_content)})
        buffered_content.clear()

    for content_item in content:
        if isinstance(content_item, str):
            buffered_content.append(
                {
                    "type": _copilot_text_content_type(role),
                    "text": content_item,
                }
            )
            continue
        if not isinstance(content_item, Mapping):
            continue
        content_type = content_item.get("type")
        if content_type == "function_call":
            flush_message()
            sanitized.append(_sanitize_copilot_function_call_item(content_item))
            continue
        if content_type == "function_call_output":
            flush_message()
            sanitized.append(_sanitize_copilot_function_call_output_item(content_item))
            continue
        projected = _sanitize_copilot_message_content_item(content_item, role=role)
        if projected is not None:
            buffered_content.append(projected)

    flush_message()
    return sanitized


def _sanitize_copilot_message_content_item(
    item: Mapping[str, Any],
    *,
    role: str,
) -> Optional[dict[str, Any]]:
    item_type = item.get("type")
    if item_type in {"input_text", "output_text", "text"}:
        return {
            "type": _copilot_text_content_type(role),
            "text": _copilot_string(item.get("text", "")),
        }
    if role == "assistant":
        if item_type == "refusal":
            return {
                "type": "refusal",
                "refusal": _copilot_string(
                    item.get("refusal", item.get("text", ""))
                ),
            }
        return None
    if item_type == "input_image" and "image_url" in item:
        return {"type": "input_image", "image_url": deepcopy(item["image_url"])}
    if item_type == "input_file":
        sanitized: dict[str, Any] = {"type": "input_file"}
        if item.get("file_id") is not None:
            sanitized["file_id"] = item["file_id"]
        else:
            if item.get("filename") is not None:
                sanitized["filename"] = item["filename"]
            if item.get("file_data") is not None:
                sanitized["file_data"] = item["file_data"]
        if len(sanitized) > 1:
            return sanitized
    return None


def _copilot_text_content_type(role: str) -> str:
    return "output_text" if role == "assistant" else "input_text"


def _sanitize_copilot_function_call_item(item: Mapping[str, Any]) -> dict[str, Any]:
    call_id = normalize_responses_call_id(_copilot_string(item.get("call_id", "")))
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": _copilot_string(item.get("name") or item.get("tool_name") or ""),
        "arguments": _copilot_arguments_text(item),
    }


def _sanitize_copilot_function_call_output_item(
    item: Mapping[str, Any],
) -> dict[str, Any]:
    output = item.get("output")
    if output is None:
        output = item.get("content")
    if output is None:
        output = item.get("error", "")
    call_id = normalize_responses_call_id(_copilot_string(item.get("call_id", "")))
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": _copilot_value_text(output),
    }


def _sanitize_copilot_tools(tools_value: Any) -> list[dict[str, Any]]:
    if not isinstance(tools_value, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for tool in tools_value:
        clean_tool = _sanitize_copilot_tool(tool)
        if clean_tool is not None:
            sanitized.append(clean_tool)
    return sanitized


def _sanitize_copilot_tool(tool: Any) -> Optional[dict[str, Any]]:
    if not isinstance(tool, Mapping):
        return None
    source: Mapping[str, Any] = tool
    function = tool.get("function")
    if isinstance(function, Mapping) and tool.get("name") is None:
        source = function

    name = source.get("name")
    if not isinstance(name, str) or not name:
        return None
    sanitized: dict[str, Any] = {
        "type": _copilot_string(tool.get("type") or "function"),
        "name": name,
    }
    if source.get("description") is not None:
        sanitized["description"] = source["description"]
    if source.get("parameters") is not None:
        sanitized["parameters"] = deepcopy(source["parameters"])
    return sanitized


def _copilot_arguments_text(item: Mapping[str, Any]) -> str:
    arguments = item.get("arguments")
    if arguments is not None:
        return _copilot_value_text(arguments)
    arguments_text = item.get("arguments_text")
    if arguments_text is not None:
        return _copilot_value_text(arguments_text)
    arguments_json = item.get("arguments_json")
    if arguments_json is not None:
        return _copilot_value_text(arguments_json)
    return ""


def _copilot_value_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (Mapping, list)):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
    return str(value)


def _copilot_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _provider_request_has_tool_call(request: RuntimeRequest) -> bool:
    for message in request.provider_request.messages:
        for part in message.parts:
            if part.tool_call is not None:
                return True
    return False


def _copilot_noop_tool_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "input" in payload:
        return {
            "type": "function",
            "name": "_noop",
            "description": "No-op fallback for GitHub Copilot tool-call history.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    return {
        "type": "function",
        "function": {
            "name": "_noop",
            "description": "No-op fallback for GitHub Copilot tool-call history.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def _env_string(environ: Mapping[str, str], name: str) -> Optional[str]:
    value = environ.get(name)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def is_github_source_token(token: Any) -> bool:
    if not isinstance(token, str):
        return False
    text = token.strip().lower()
    return text.startswith(GITHUB_SOURCE_TOKEN_PREFIXES)


def exchange_github_token_for_copilot_token(
    source_credential: str,
    *,
    github_api_base_url: Optional[str] = None,
    timeout: float | None = DEFAULT_GITHUB_COPILOT_TIMEOUT_SECONDS,
    user_agent: str = "GitHubCopilotChat/0.41.0",
    editor_version: str = "vscode/1.133.0",
    editor_plugin_version: str = "copilot-chat/0.41.0",
    integration_id: str = "vscode-chat",
) -> CopilotTokenExchange:
    """Exchange a GitHub source token for a Copilot plugin token."""

    source_token = _required_non_empty_string(source_credential, "source_credential")
    endpoint = "{0}/copilot_internal/v2/token".format(
        _normalize_base_url(
            github_api_base_url,
            default=GitHubCopilotHTTPTransport.DEFAULT_GITHUB_API_BASE_URL,
        )
    )
    request = urllib_request.Request(
        endpoint,
        headers={
            "Authorization": "Bearer {0}".format(source_token),
            "Accept": "application/json",
            **_copilot_plugin_headers(
                user_agent=user_agent,
                editor_version=editor_version,
                editor_plugin_version=editor_plugin_version,
                integration_id=integration_id,
            ),
        },
        method="GET",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
    except urllib_error.HTTPError as exc:
        response_text = _read_http_error_body(exc)
        raise ProviderTransportError(
            _format_token_exchange_http_error(
                exc,
                source_token,
                response_text=response_text,
            )
        ) from None
    except urllib_error.URLError as exc:
        reason = _redact_secret(str(getattr(exc, "reason", exc)), source_token)
        raise ProviderTransportError(
            "GitHub Copilot token exchange failed: {0}".format(reason)
        ) from None
    except TimeoutError:
        raise ProviderTransportError(
            _format_timeout_message("GitHub Copilot token exchange", timeout)
        ) from None

    try:
        text = raw_body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProviderTransportError(
            "GitHub Copilot token exchange returned non-UTF-8 response data"
        ) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderTransportError(
            "GitHub Copilot token exchange returned invalid JSON"
        ) from exc
    if not isinstance(data, Mapping):
        raise ProviderTransportError(
            "GitHub Copilot token exchange returned a non-object JSON response"
        )

    token = data.get("token")
    expires_at = data.get("expires_at")
    if not isinstance(token, str) or not token.strip():
        raise ProviderTransportError(
            "GitHub Copilot token exchange returned an invalid token"
        )
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        raise ProviderTransportError(
            "GitHub Copilot token exchange returned an invalid expires_at"
        )
    expires_at_int = int(expires_at)
    if expires_at_int <= 0:
        raise ProviderTransportError(
            "GitHub Copilot token exchange returned an invalid expires_at"
        )
    return CopilotTokenExchange(
        token=token.strip(),
        expires_at=expires_at_int,
        api_base_url=parse_copilot_api_base_url(token),
    )


def parse_copilot_api_base_url(token: str) -> str | None:
    match = _PROXY_ENDPOINT_PATTERN.search(token)
    if match is None:
        return None
    raw = urllib_parse.unquote(match.group(1).strip())
    parsed = urllib_parse.urlparse(raw)
    host = parsed.netloc
    if not host:
        host = parsed.path.split("/", 1)[0]
    host = host.strip()
    if not host:
        return None
    if host.startswith("proxy."):
        host = "api." + host.removeprefix("proxy.")
    return "https://{0}".format(host.rstrip("/"))


def _required_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError("{0} must be a non-empty string".format(field_name))
    value = value.strip()
    if not value:
        raise ValueError("{0} must be a non-empty string".format(field_name))
    return value


def _normalize_base_url(
    base_url: Optional[str],
    *,
    default: str | None = None,
) -> str:
    if base_url is None:
        return default or GitHubCopilotHTTPTransport.DEFAULT_BASE_URL
    base_url = _required_non_empty_string(base_url, "base_url")
    return base_url.rstrip("/")


def _format_http_error(
    exc: urllib_error.HTTPError,
    token: str,
    *,
    response_text: str | None = None,
) -> str:
    status = getattr(exc, "code", None)
    reason = getattr(exc, "reason", None) or getattr(exc, "msg", "")
    parts = ["GitHub Copilot HTTP transport failed"]
    if status is not None:
        parts.append("with status {0}".format(status))
    if reason:
        parts.append("({0})".format(_redact_secret(str(reason), token)))
    if response_text is None:
        response_text = _read_http_error_body(exc)
    if response_text:
        parts.append("response: {0}".format(_redact_secret(response_text, token)))
    return " ".join(parts)


def _format_token_exchange_http_error(
    exc: urllib_error.HTTPError,
    token: str,
    *,
    response_text: str | None = None,
) -> str:
    status = getattr(exc, "code", None)
    reason = getattr(exc, "reason", None) or getattr(exc, "msg", "")
    parts = ["GitHub Copilot token exchange failed"]
    if status is not None:
        parts.append("with status {0}".format(status))
    if reason:
        parts.append("({0})".format(_redact_secret(str(reason), token)))
    if response_text is None:
        response_text = _read_http_error_body(exc)
    if response_text:
        parts.append("response: {0}".format(_redact_secret(response_text, token)))
    return " ".join(parts)


def _model_unavailable_error_from_http_error(
    exc: urllib_error.HTTPError,
    *,
    response_text: str,
    token: str,
) -> ProviderModelUnavailableError | None:
    status = getattr(exc, "code", None)
    if status != 400 or not response_text:
        return None
    error_message = _json_error_message(response_text)
    if error_message is None or not _is_model_unavailable_message(error_message):
        return None
    safe_message = _redact_secret(error_message, token)
    available_models_text = _available_models_text(safe_message)
    return ProviderModelUnavailableError(
        "GitHub Copilot model is not available: {0}".format(safe_message),
        status_code=status,
        available_models_text=available_models_text,
    )


def _context_overflow_error_from_http_error(
    exc: urllib_error.HTTPError,
    *,
    response_text: str,
    token: str,
    provider_label: str = "GitHub Copilot",
) -> ProviderContextOverflowError | None:
    status = getattr(exc, "code", None)
    if status not in (400, 413):
        return None
    error_message = _json_error_message(response_text)
    error_code = _json_error_code(response_text)
    if error_message is None and _parse_json_object(response_text) is None:
        # Non-JSON envelope (a proxy's HTML or bare text): the whole body is the
        # only message there is. Deliberately NOT done for a JSON object that
        # simply lacks a message key - such a body often echoes the rejected
        # request back, and scanning it would classify a validation 400 as an
        # overflow whenever the user's own prompt mentions a context window.
        error_message = response_text.strip()
    if status == 413:
        # 413 Payload Too Large on a JSON prompt POST has exactly one meaning,
        # and the ingress or proxy that emits it usually sends an empty or HTML
        # body with no marker to match. Requiring a marker here would make 413
        # unreachable: every 413 that carries one would already match as a 400.
        error_message = error_message or "HTTP 413 Payload Too Large"
    elif not _is_context_overflow_message(error_message or "") and (
        error_code not in _CONTEXT_OVERFLOW_ERROR_CODES
    ):
        return None
    safe_message = _redact_secret(
        error_message or error_code or "HTTP {0}".format(status), token
    )
    return ProviderContextOverflowError(
        "{0} rejected the request as larger than the model context window: {1}".format(
            provider_label,
            safe_message,
        ),
        metadata={"status_code": status, "provider_error_code": error_code},
    )


def _parse_json_object(response_text: str) -> Mapping[str, Any] | None:
    """Return ``response_text`` decoded as a JSON object, or ``None``."""

    try:
        data = json.loads(response_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return data if isinstance(data, Mapping) else None


def _json_error_message(response_text: str) -> str | None:
    data = _parse_json_object(response_text)
    if data is None:
        return None

    error_value = data.get("error")
    if isinstance(error_value, Mapping):
        for key in ("message", "detail", "error"):
            value = error_value.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(error_value, str) and error_value.strip():
        return error_value.strip()
    for key in ("message", "detail"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _json_error_code(response_text: str) -> str | None:
    data = _parse_json_object(response_text)
    if data is None:
        return None

    error_value = data.get("error")
    if isinstance(error_value, Mapping):
        for key in ("code", "type"):
            value = error_value.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("code", "type"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_context_overflow_message(message: str) -> bool:
    text = message.lower()
    return any(marker in text for marker in _CONTEXT_OVERFLOW_MESSAGE_MARKERS)


def _is_model_unavailable_message(message: str) -> bool:
    text = message.lower()
    return (
        "requested model is not available" in text
        or "model is not available" in text
    )


def _available_models_text(message: str) -> str | None:
    match = re.search(r"available models:\s*(.+)$", message, flags=re.IGNORECASE)
    if match is None:
        return None
    available_models = match.group(1).strip()
    return available_models or None


def _read_http_error_body(exc: urllib_error.HTTPError) -> str:
    try:
        raw_body = exc.read(2048)
    except Exception:
        return ""
    if not raw_body:
        return ""
    try:
        return raw_body.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _iter_response_lines(response: Any) -> Iterable[bytes]:
    while True:
        line = response.readline()
        if not line:
            break
        yield line


def _redact_secret(text: str, secret: str) -> str:
    if not text or not secret:
        return text
    return text.replace(secret, "[redacted]")


def _copilot_plugin_headers(
    *,
    user_agent: str,
    editor_version: str,
    editor_plugin_version: str,
    integration_id: str,
) -> dict[str, str]:
    return {
        "User-Agent": _required_non_empty_string(user_agent, "user_agent"),
        "Editor-Version": _required_non_empty_string(
            editor_version,
            "editor_version",
        ),
        "Editor-Plugin-Version": _required_non_empty_string(
            editor_plugin_version,
            "editor_plugin_version",
        ),
        "Copilot-Integration-Id": _required_non_empty_string(
            integration_id,
            "integration_id",
        ),
    }


def _unsupported_reasoning_message(value: Any) -> str:
    supported = ", ".join(SUPPORTED_COPILOT_REASONING_EFFORTS)
    return "unsupported GitHub Copilot reasoning effort {0!r}; supported values: {1}".format(
        value,
        supported,
    )


def _format_timeout_message(prefix: str, timeout: Any) -> str:
    if timeout is None:
        return "{0} timed out".format(prefix)
    return "{0} timed out after {1} seconds".format(
        prefix,
        _format_timeout_seconds(timeout),
    )


def _format_timeout_seconds(timeout: Any) -> str:
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        return str(timeout)
    return "{0:g}".format(value)


__all__ = [
    "CopilotTokenExchange",
    "DEFAULT_COPILOT_REASONING_EFFORT",
    "DEFAULT_GITHUB_COPILOT_TIMEOUT_SECONDS",
    "GitHubCopilotHTTPTransport",
    "GitHubCopilotProvider",
    "OpenAICompatibleProvider",
    "ProviderModelUnavailableError",
    "ProviderTransport",
    "ProviderTransportError",
    "RecordingTransport",
    "SUPPORTED_COPILOT_MODEL_IDS",
    "SUPPORTED_COPILOT_REASONING_EFFORTS",
    "TransportOutput",
    "exchange_github_token_for_copilot_token",
    "github_copilot_provider_from_env",
    "is_github_source_token",
    "parse_sse_json_events",
    "parse_copilot_api_base_url",
    "validate_copilot_reasoning_effort",
]
