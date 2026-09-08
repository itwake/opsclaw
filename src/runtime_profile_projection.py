"""Per-runtime projection of the canonical runtime-profile config.

The runtime-profile Secret now stores a single, runtime-agnostic canonical
config (``config.json``). Each runtime must apply its OWN projection to that
parsed canonical config at boot. This module is a self-contained (stdlib-only)
verbatim port of the portal's projection so that, for any canonical config::

    project_canonical_for_runtime(canonical, "<runtime>")

produces byte-identical output to what the portal's old
``build_runtime_profile_context_config(profile_config, runtime_type="<runtime>")``
baked into the per-runtime Secret payload.

For the native runtime this re-normalizes the LLM to itself (github_copilot +
bare model, a no-op), leaves opencode-restricted fields untouched (profile
configs never carry them), and re-adds the native CLI tool instructions. For
opencode it would instead normalize the LLM to ``github-copilot`` +
``provider/model`` and drop the opencode-restricted fields.

Ported verbatim from:
  - engineering-flow-platform-portal app/contracts/provider_projection.py
  - engineering-flow-platform-portal app/services/runtime_profile_llm_projection.py
  - engineering-flow-platform-portal app/services/runtime_profile_context_projection.py
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# provider_projection.py (verbatim)
# ---------------------------------------------------------------------------


_AI_PLATFORM_ALIASES = {"ai_platform", "ai-platform", "ai platform"}


def normalize_provider_for_portal(value: str | None) -> str:
    # Two supported providers: github_copilot (default) and ai_platform.
    # Anything blank/unknown (incl. copilot aliases and legacy openai/anthropic)
    # falls back to github_copilot.
    raw = (value or "").strip().lower()
    if raw in _AI_PLATFORM_ALIASES:
        return "ai_platform"
    return "github_copilot"


def normalize_provider_for_runtime(runtime_type: str, provider: str | None) -> str:
    portal_provider = normalize_provider_for_portal(provider)
    if (runtime_type or "").strip().lower() == "opencode":
        if portal_provider == "github_copilot":
            return "github-copilot"
        if portal_provider == "ai_platform":
            return "ai-platform"
    return portal_provider


def normalize_model_for_runtime(runtime_type: str, provider: str | None, model: str | None) -> str | None:
    if not model:
        return None
    model = str(model).strip()
    runtime_provider = normalize_provider_for_runtime(runtime_type, provider)
    if "/" in model:
        prefix, rest = model.split("/", 1)
        normalized_prefix = normalize_provider_for_runtime(runtime_type, prefix)
        return f"{normalized_prefix}/{rest}"
    if (runtime_type or "").strip().lower() == "opencode" and runtime_provider:
        return f"{runtime_provider}/{model}"
    return model


# ---------------------------------------------------------------------------
# runtime_profile_llm_projection.py (verbatim)
# ---------------------------------------------------------------------------


def _is_copilot_provider(provider: str | None) -> bool:
    return normalize_provider_for_portal(provider) == "github_copilot"


def _provider_hint_from_llm(llm: dict) -> str | None:
    provider = llm.get("provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    model = llm.get("model")
    if isinstance(model, str) and "/" in model:
        prefix = model.split("/", 1)[0].strip()
        if prefix:
            return prefix
    return None


def project_llm_for_runtime(llm: dict, runtime_type: str) -> dict:
    projected = deepcopy(llm)
    provider_hint = _provider_hint_from_llm(projected)
    runtime_type = "opencode" if str(runtime_type or "").strip().lower() == "opencode" else "native"
    if provider_hint:
        projected["provider"] = normalize_provider_for_runtime(runtime_type, provider_hint)
    if projected.get("model"):
        normalized_model = normalize_model_for_runtime(runtime_type, provider_hint, projected.get("model"))
        if normalized_model:
            projected["model"] = normalized_model

    # Portal-only oauth fields never reach a runtime.
    projected.pop("oauth", None)
    projected.pop("oauth_by_runtime", None)

    if _is_copilot_provider(provider_hint):
        token = str(projected.get("api_key") or "").strip()
        if token:
            projected["api_key"] = token
        else:
            projected.pop("api_key", None)
    # For ai_platform the rich block (llm.ai_platform: chat/responses/ib2b endpoints +
    # auth credentials) is preserved as-is for the runtime to consume.
    return projected


# ---------------------------------------------------------------------------
# runtime_profile_context_projection.py (verbatim subset)
# ---------------------------------------------------------------------------


RUNTIME_PROFILE_CLI_TOOL_INSTRUCTIONS = (
    "Use bash for runtime profile CLI tools: jira/confluence for Atlassian, "
    "gh for GitHub issues, PRs, and api calls, aws for AWS operations, "
    "jenkins for Jenkins controller operations, mobile-auto for BrowserStack/Appium device automation, "
    "and git for clone, fetch, push, and status. "
    "For every jira, confluence, jenkins, and mobile-auto command add --json. For complex jira/confluence/jenkins/mobile-auto calls, "
    "inspect commands, schema, or help llm first, for example `jira commands --json`, "
    "`jira schema <command> --json`, `jira help llm --json`, and the matching confluence/jenkins/mobile-auto commands. "
    "For mobile work, start with `mobile-auto doctor --json` and `mobile-auto auth test --json`; use BrowserStackLocal through "
    "`private-external` with a supplied local identifier or `private-managed` only when the runtime image has BrowserStackLocal installed. "
    "Jenkins runtime profile credentials are available as EFP_JENKINS_USERNAME and EFP_JENKINS_PASSWORD; "
    "when the user provides a Jenkins controller URL or pipeline/job, configure or log in to that controller at that time and pass the password through stdin, never by echoing it. "
    "For AWS, prefer `aws --output json` for inspection and avoid changing cloud resources unless the user asks. "
    "Run write operations with --dry-run before executing them. Use --yes only for destructive "
    "operations after the user explicitly confirms. Runtime profile credentials are applied in "
    "the runtime container through CLIs or environment variables; if a CLI returns auth_failed, report a runtime profile "
    "configuration problem instead of guessing or inventing tokens."
)

OPENCODE_RUNTIME_RESTRICTION_FIELDS = frozenset(
    {
        "enabled_tools",
        "disabled_tools",
        "tool_permissions",
        "allowed_external_systems",
        "allowed_actions",
        "allowed_adapter_actions",
        "allowed_capability_ids",
        "allowed_capability_types",
        "resolved_action_mappings",
        "unresolved_tools",
        "unresolved_skills",
        "unresolved_channels",
        "unresolved_actions",
        "skill_details",
        "allowed_skills",
        "denied_skills",
        "denied_actions",
        "denied_capability_types",
        "skill_set",
        "policy_context",
        "derived_runtime_rules",
    }
)


def is_opencode_runtime_type(runtime_type: str | None) -> bool:
    return str(runtime_type or "").strip().lower() == "opencode"


def strip_opencode_runtime_restrictions(
    config: dict[str, Any] | None,
    runtime_type: str | None,
) -> dict[str, Any]:
    projected = deepcopy(config) if isinstance(config, dict) else {}
    if not is_opencode_runtime_type(runtime_type):
        return projected
    for key in OPENCODE_RUNTIME_RESTRICTION_FIELDS:
        projected.pop(key, None)
    return projected


def _has_enabled_instance_section(config: dict[str, Any], section: str) -> bool:
    section_config = config.get(section)
    if not isinstance(section_config, dict) or section_config.get("enabled") is not True:
        return False
    instances = section_config.get("instances")
    if not isinstance(instances, list):
        return False
    for item in instances:
        if not isinstance(item, dict) or item.get("enabled") is False:
            continue
        endpoint = ""
        for key in ("url", "base_url", "baseUrl", "uri"):
            endpoint = str(item.get(key) or "").strip()
            if endpoint:
                break
        if endpoint:
            return True
    return False


def _has_enabled_github_config(config: dict[str, Any]) -> bool:
    github = config.get("github")
    if not isinstance(github, dict) or github.get("enabled") is not True:
        return False
    return bool(str(github.get("api_token") or github.get("base_url") or "").strip())


def _has_git_config(config: dict[str, Any]) -> bool:
    git = config.get("git")
    if not isinstance(git, dict):
        return False
    user = git.get("user")
    if not isinstance(user, dict):
        return False
    return bool(str(user.get("name") or user.get("email") or "").strip())


def _has_enabled_aws_config(config: dict[str, Any]) -> bool:
    aws = config.get("aws")
    if not isinstance(aws, dict) or aws.get("enabled") is not True:
        return False
    domain = str(aws.get("domain") or "").strip()
    username = str(aws.get("username") or "").strip()
    password = str(aws.get("password") or "").strip()
    return bool(domain and username and password)


def _has_enabled_jenkins_config(config: dict[str, Any]) -> bool:
    jenkins = config.get("jenkins")
    if not isinstance(jenkins, dict) or jenkins.get("enabled") is not True:
        return False
    if isinstance(jenkins.get("instances"), list):
        # Multi-instance Jenkins profiles are shaped like jira/confluence.
        return _has_enabled_instance_section(config, "jenkins")
    # Legacy flat section (pre multi-instance Jenkins UI): judged by the same
    # rule, as a single instance. Requiring username+password here instead
    # would get it wrong in both directions -- it claimed Jenkins was usable
    # with no endpoint at all (the projection drops such a section, so the
    # agent was told about a CLI it could not use), and it denied a valid
    # url+token profile that the CLI does support.
    return _has_enabled_instance_section(
        {"jenkins": {"enabled": True, "instances": [jenkins]}}, "jenkins"
    )


def _has_enabled_mobile_config(config: dict[str, Any]) -> bool:
    mobile = config.get("mobile-auto")
    if not isinstance(mobile, dict) or mobile.get("enabled") is not True:
        return False
    browserstack = mobile.get("browserstack")
    if not isinstance(browserstack, dict):
        return False
    return bool(
        str(browserstack.get("username") or browserstack.get("username_env") or "").strip()
        or str(browserstack.get("access_key") or browserstack.get("access_key_env") or "").strip()
        or str(browserstack.get("api_base_url") or browserstack.get("appium_base_url") or "").strip()
    )


def _has_enabled_external_cli_config(config: dict[str, Any]) -> bool:
    return (
        _has_enabled_instance_section(config, "jira")
        or _has_enabled_instance_section(config, "confluence")
        or _has_enabled_jenkins_config(config)
        or _has_enabled_mobile_config(config)
        or _has_enabled_github_config(config)
        or _has_enabled_aws_config(config)
        or _has_git_config(config)
    )


def _with_native_cli_tool_instructions(
    config: dict[str, Any],
    runtime_type: str | None,
) -> dict[str, Any]:
    if is_opencode_runtime_type(runtime_type) or not _has_enabled_external_cli_config(config):
        return config
    projected = deepcopy(config)
    instruction_texts = projected.get("instruction_texts")
    if not isinstance(instruction_texts, list):
        instruction_texts = []
    if RUNTIME_PROFILE_CLI_TOOL_INSTRUCTIONS not in instruction_texts:
        instruction_texts.append(RUNTIME_PROFILE_CLI_TOOL_INSTRUCTIONS)
    projected["instruction_texts"] = instruction_texts
    return projected


def project_canonical_for_runtime(
    canonical: dict[str, Any] | None,
    runtime_type: str | None,
) -> dict[str, Any]:
    """Apply the per-runtime projection to a canonical config.

    The runtimes call this at boot on the config parsed from the Secret. Native
    re-normalizes the LLM to itself (a no-op) and gains the CLI tool
    instructions; opencode re-normalizes the LLM to its ``provider/model`` form
    and drops the opencode-restricted fields.
    """
    projected = deepcopy(canonical) if isinstance(canonical, dict) else {}
    llm = projected.get("llm")
    if isinstance(llm, dict):
        projected_llm = project_llm_for_runtime(llm, runtime_type)
        max_context_tokens = projected_llm.pop("max_context_tokens", None)
        projected["llm"] = projected_llm
        if not is_opencode_runtime_type(runtime_type) and isinstance(max_context_tokens, int):
            projected["max_context_tokens"] = max_context_tokens
    projected = strip_opencode_runtime_restrictions(projected, runtime_type)
    return _with_native_cli_tool_instructions(projected, runtime_type)
