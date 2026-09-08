"""Copilot-first model context profiles for EFP runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


DEFAULT_PROVIDER_ID = "github-copilot"
DEFAULT_MODEL_ID = "gpt-5.6-terra"
DEFAULT_CHARS_PER_TOKEN = 4
MIN_PRESERVE_RECENT_TOKENS = 2_000
MAX_PRESERVE_RECENT_TOKENS = 8_000
DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS = 8_000
CONTEXT_SAFETY_MARGIN_WINDOW_DIVISOR = 20
SUPPORTED_COPILOT_MODEL_IDS = (
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
)

# AI Platform (enterprise gateway) models. Must stay aligned with the Portal
# catalog (app/contracts/llm_catalog.py AI_PLATFORM_MODELS): the runtime coerces
# any model outside this tuple back to DEFAULT_AI_PLATFORM_MODEL, so a model the
# Portal offers but this list lacks is silently replaced by gpt-5.4.
DEFAULT_AI_PLATFORM_MODEL = "gpt-5.4"
AI_PLATFORM_MODEL_IDS = (
    "gpt-5.4",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
)


@dataclass(frozen=True)
class ModelContextProfile:
    """Deterministic context sizing metadata for a Copilot-hosted model."""

    provider_id: str
    model_id: str
    context_window_tokens: int
    default_reserve_tokens: int
    default_preserve_recent_tokens: int
    chars_per_token: int = DEFAULT_CHARS_PER_TOKEN

    def tokens_to_chars(self, tokens: int) -> int:
        """Convert token counts to deterministic approximate character counts."""

        return tokens_to_chars(tokens, chars_per_token=self.chars_per_token)


def resolve_model_context_profile(
    model: Any = None,
    *,
    provider_id: Any = DEFAULT_PROVIDER_ID,
) -> ModelContextProfile:
    """Resolve a model context profile with a conservative fallback.

    The catalog is keyed by model id alone: a context window is a property of
    the model, not of the gateway serving it. ``provider_id`` is accepted (and
    a ``provider/model`` prefix is stripped from ``model``) so callers can pass
    a qualified id, but it does not gate the lookup - ai_platform serves the
    same GPT-5.x line (see ``AI_PLATFORM_MODEL_IDS``) with the same 1M window
    as Copilot, and this profile is the default source of the native runtime's
    context budget, so falling back to the 64k profile purely because of a
    provider id would compact those sessions roughly seven times too early.
    """

    del provider_id  # accepted for call-site compatibility; see docstring
    _, requested_model = _split_model_id(model)
    model_id = requested_model or DEFAULT_MODEL_ID
    return _COPILOT_PROFILES.get(
        model_id,
        replace(_CONSERVATIVE_FALLBACK_PROFILE, model_id=model_id),
    )


def canonicalize_copilot_model_id(value: Any) -> str:
    """Return a supported canonical GitHub Copilot model id or raise."""

    text = _model_identifier(value)
    if text is None:
        raise ValueError(_unsupported_model_message(value))
    if "/" in text:
        provider, model = text.split("/", 1)
        if _normalize_identifier(provider) not in {
            DEFAULT_PROVIDER_ID,
            "github_copilot",
            "copilot",
        }:
            raise ValueError(_unsupported_model_message(value))
        text = model
    model_id = _normalize_model_identifier(text)
    if model_id not in SUPPORTED_COPILOT_MODEL_IDS:
        raise ValueError(_unsupported_model_message(value))
    return model_id


def tokens_to_chars(
    tokens: int,
    *,
    chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Deterministically approximate a token budget as a character budget."""

    _validate_non_negative_int(tokens, "tokens")
    _validate_positive_int(chars_per_token, "chars_per_token")
    return int(tokens) * int(chars_per_token)


def context_safety_margin_tokens(profile: ModelContextProfile) -> int:
    """Return the token margin held back from a model's declared window.

    Mirrors the legacy safety-margin rule in ``src/runtime/progressive_context.py``
    (``min(configured_safety, int(context_window * 0.05))`` with a configured
    safety of 8000 tokens). Combined with ``default_reserve_tokens`` this
    reproduces the legacy prompt budget ``context_window - reserved - safety``;
    the legacy path additionally clamps by a configured ``max_prompt_tokens``,
    which has no native-runtime equivalent, and has no table entry for the
    ``gpt-5.6-*`` models at all.
    """

    return min(
        DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS,
        profile.context_window_tokens // CONTEXT_SAFETY_MARGIN_WINDOW_DIVISOR,
    )


def default_max_context_tokens(profile: ModelContextProfile) -> int:
    """Return the catalog-derived prompt budget for a model, in tokens.

    The model's response headroom (``default_reserve_tokens``) is NOT subtracted
    here; callers subtract it separately via ``ContextBudget.reserve_chars``.
    """

    return max(1, profile.context_window_tokens - context_safety_margin_tokens(profile))


def is_catalog_model_context_profile(profile: ModelContextProfile) -> bool:
    """Report whether ``profile`` came from the catalog rather than the fallback.

    ``resolve_model_context_profile`` returns
    ``replace(_CONSERVATIVE_FALLBACK_PROFILE, model_id=model_id)`` on a miss, so
    ``profile.model_id`` alone cannot tell a real catalog entry from a guess -
    an uncatalogued model reports its own id next to a 64k window it never
    declared.

    Callers must not narrow an operator's explicit context budget on a profile
    this returns ``False`` for: a newly released or gateway-only model needs an
    override precisely because the conservative fallback is wrong about it, and
    clamping to 64k would defeat the only purpose the override has.
    """

    return _COPILOT_PROFILES.get(profile.model_id) == profile


def _profile(
    model_id: str,
    *,
    context_window_tokens: int,
    default_reserve_tokens: int,
    chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
) -> ModelContextProfile:
    return ModelContextProfile(
        provider_id=DEFAULT_PROVIDER_ID,
        model_id=model_id,
        context_window_tokens=context_window_tokens,
        default_reserve_tokens=default_reserve_tokens,
        default_preserve_recent_tokens=_default_preserve_recent_tokens(
            context_window_tokens=context_window_tokens,
            reserve_tokens=default_reserve_tokens,
        ),
        chars_per_token=chars_per_token,
    )


def _default_preserve_recent_tokens(
    *,
    context_window_tokens: int,
    reserve_tokens: int,
) -> int:
    usable_tokens = max(0, context_window_tokens - reserve_tokens)
    return min(
        MAX_PRESERVE_RECENT_TOKENS,
        max(MIN_PRESERVE_RECENT_TOKENS, usable_tokens // 4),
    )


_COPILOT_PROFILES = {
    "gpt-5.4": _profile(
        "gpt-5.4",
        context_window_tokens=1_000_000,
        default_reserve_tokens=128_000,
    ),
    "gpt-5.5": _profile(
        "gpt-5.5",
        context_window_tokens=1_000_000,
        default_reserve_tokens=128_000,
    ),
    "gpt-5.6-luna": _profile(
        "gpt-5.6-luna",
        context_window_tokens=1_000_000,
        default_reserve_tokens=128_000,
    ),
    "gpt-5.6-sol": _profile(
        "gpt-5.6-sol",
        context_window_tokens=1_000_000,
        default_reserve_tokens=128_000,
    ),
    "gpt-5.6-terra": _profile(
        "gpt-5.6-terra",
        context_window_tokens=1_000_000,
        default_reserve_tokens=128_000,
    ),
}

_CONSERVATIVE_FALLBACK_PROFILE = _profile(
    "unknown",
    context_window_tokens=64_000,
    default_reserve_tokens=4_000,
)


def _split_model_id(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, None
    text = value.strip()
    if not text:
        return None, None
    if "/" not in text:
        return None, _normalize_model_identifier(text)
    provider, model = text.split("/", 1)
    return _normalize_identifier(provider), _normalize_model_identifier(model)


def _normalize_identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _model_identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _normalize_model_identifier(value: Any) -> str | None:
    text = _model_identifier(value)
    if text is None:
        return None
    return "-".join(text.split())


def _unsupported_model_message(value: Any) -> str:
    supported = ", ".join(SUPPORTED_COPILOT_MODEL_IDS)
    return "unsupported GitHub Copilot model {0!r}; supported models: {1}".format(
        value,
        supported,
    )


def _validate_non_negative_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


__all__ = [
    "CONTEXT_SAFETY_MARGIN_WINDOW_DIVISOR",
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_PROVIDER_ID",
    "MAX_PRESERVE_RECENT_TOKENS",
    "MIN_PRESERVE_RECENT_TOKENS",
    "ModelContextProfile",
    "SUPPORTED_COPILOT_MODEL_IDS",
    "canonicalize_copilot_model_id",
    "context_safety_margin_tokens",
    "default_max_context_tokens",
    "is_catalog_model_context_profile",
    "resolve_model_context_profile",
    "tokens_to_chars",
]
