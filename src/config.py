"""Configuration loader for Engineering Flow Platform."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from src.runtime_profile_encryption import decrypt_sensitive_fields
from src.runtime_profile_projection import project_canonical_for_runtime


logger = logging.getLogger(__name__)

# Module-level YAML instance for reuse
_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.indent(mapping=2, sequence=4, offset=2)


def _home_path() -> Path:
    return Path(os.environ.get("HOME") or Path.home())


DEFAULT_LLM_MODEL = "gpt-5.4"
DEFAULT_LLM_TEMPERATURE = 0.7

PORTAL_MANAGED_RUNTIME_FIELDS = frozenset(
    {
        "enabled_tools",
        "disabled_tools",
        "tool_permissions",
        "max_iterations",
        "doom_loop_threshold",
        "max_context_parts",
        "max_context_chars",
        "max_context_tokens",
        "context_reserve_chars",
        "context_reserve_tokens",
        "compaction_auto",
        "compaction_rewrite_stored_history",
        "compaction_prune",
        "compaction_tail_turns",
        "compaction_preserve_recent_chars",
        "compaction_preserve_recent_tokens",
        "compaction_reserved_chars",
        "compaction_tool_output_max_chars",
        "compaction_prune_min_chars",
        "compaction_prune_protect_chars",
        "enable_compaction_summarizer",
        "enable_context_overflow_retry",
        "enable_session_revert_snapshots",
        "skill_directories",
        "active_skills",
        "command_directories",
        "enable_command_expansion",
        "system_prompt_texts",
        "system_prompt_paths",
        "include_default_system_prompt",
        "include_environment_context",
        "max_system_prompt_chars",
        "include_runtime_reminders",
        "instruction_texts",
        "instruction_paths",
        "include_default_instructions",
        "attach_read_instructions",
        "max_instruction_chars",
        "include_skill_sidecar_content",
        "max_skill_sidecar_chars",
        "max_command_chars",
        "resolve_prompt_references",
        "max_prompt_reference_chars",
        "max_prompt_directory_entries",
        "runtime_mode",
        "enable_plan_tool",
        "plan_mode_read_only",
        "enable_question_tool",
        "enable_lsp_tool",
        "inject_background_task_results",
        "model_aware_tool_selection",
        "structured_output_schema",
        "tool_output_max_lines",
        "tool_output_max_bytes",
        "tool_output_truncation_direction",
        "archive_truncated_tool_outputs",
        "tool_output_dir",
        "emit_llm_stream_events",
        "track_usage",
    }
)

_ATLASSIAN_INSTANCE_URL_FIELDS = ("url", "base_url", "baseUrl", "uri")


def _first_atlassian_instance_url(value: Dict[str, Any]) -> str:
    if not isinstance(value, dict):
        return ""
    for field in _ATLASSIAN_INSTANCE_URL_FIELDS:
        text = str(value.get(field) or "").strip()
        if text:
            return text.rstrip("/")
    return ""


class Config:
    """Configuration management.
    
    Searches for config.yaml in the following order:
    1. Project directory (same directory as this file)
    2. ~/.efp/config.yaml
    """
    
    DEFAULT_PATHS = [
        Path.home() / ".efp" / "config.yaml",  # User config directory
        Path(__file__).parent / "config.yaml",  # Project directory
    ]
    PROXY_ENV_VARS = (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "no_proxy",
        "NO_PROXY",
    )
    JENKINS_ENV_VARS = (
        "EFP_JENKINS_USERNAME",
        "EFP_JENKINS_PASSWORD",
        "JENKINS_USERNAME",
        "JENKINS_PASSWORD",
    )
    MOBILE_ENV_VARS = (
        "BROWSERSTACK_USERNAME",
        "BROWSERSTACK_ACCESS_KEY",
    )
    # Where the runtime image installs the BrowserStack Local binary (matches
    # the opencode runtime's bundled path). Used as the default so mobile-auto
    # resolves BROWSERSTACK_LOCAL_BINARY directly instead of relying on a bare
    # PATH lookup of "BrowserStackLocal".
    DEFAULT_BROWSERSTACK_LOCAL_BINARY_PATH = "/usr/local/bin/BrowserStackLocal"

    PROJECT_EXAMPLE = Path(__file__).parent.parent / 'config.yaml.example'
    MANAGED_OVERLAY_SECTIONS = {
        "llm",
        "proxy",
        "jira",
        "confluence",
        "github",
        "aws",
        "jenkins",
        "mobile-auto",
        "git",
        "debug",
        *PORTAL_MANAGED_RUNTIME_FIELDS,
    }
    PORTAL_MANAGED_FIELD_TREE = {
        **{field: True for field in sorted(PORTAL_MANAGED_RUNTIME_FIELDS)},
        # Keep hidden/deprecated Portal LLM fields in this field tree.
        # Portal may stop rendering temperature/tools/response_flow controls, but
        # the EFP_PROFILE_CONFIG overlay filter must still accept older
        # Portal-managed values so they merge into the effective config.
        "llm": {
            "provider": True,
            "model": True,
            "api_key": True,
            "reasoning_effort": True,
            "timeout_ms": True,
            "timeout_seconds": True,
            "timeout": True,
            "temperature": True,
            "reasoning_replay": True,
            "max_tokens": True,
            "tools": True,
            "context_budget": True,
            "context_projection": True,
            "response_flow": True,
            "tool_loop": True,
            # AI Platform rich config (chat/responses/ib2b endpoints + credentials).
            "ai_platform": {
                "chat": {"host": True, "uri": True},
                "responses": {"host": True, "uri": True},
                "ib2b": {"host": True, "uri": True},
                "auth": {
                    "username": True,
                    "password": True,
                    "usercase": True,
                    "trust_token_header": True,
                    "tracking_prefix": True,
                    "token": True,
                },
            },
        },
        "proxy": {
            "enabled": True,
            "url": True,
            "username": True,
            "password": True,
            "no_proxy": True,
            "noProxy": True,
        },
        "jira": {
            "enabled": True,
            "instances": True,
            "default_instance": True,
        },
        "confluence": {
            "enabled": True,
            "instances": True,
            "default_instance": True,
        },
        "github": {
            "enabled": True,
            "api_token": True,
            "token": True,
            "access_token": True,
            "base_url": True,
            "api_base_url": True,
        },
        "aws": {
            "enabled": True,
            "domain": True,
            "username": True,
            "password": True,
        },
        "jenkins": {
            "enabled": True,
            "instances": True,
            "default_instance": True,
            # Legacy flat single-instance Jenkins fields: profiles saved before
            # the Portal grew a multi-instance Jenkins UI still carry these and
            # must keep merging into the effective config.
            "url": True,
            "username": True,
            "password": True,
        },
        "mobile-auto": {
            "enabled": True,
            "default_provider": True,
            "state_dir": True,
            "artifacts_dir": True,
            "retention_hours": True,
            "defaults": {
                "platform": True,
                "network_mode": True,
                "idle_timeout_seconds": True,
                "new_command_timeout_seconds": True,
                "interactive_debugging": True,
                "video": True,
            },
            "browserstack": {
                "api_base_url": True,
                "appium_base_url": True,
                "username_env": True,
                "access_key_env": True,
                "username": True,
                "access_key": True,
                "verify_ssl": True,
                "ca_cert": True,
                "http_proxy": True,
                "local": True,
            },
        },
        "git": {
            "user": {
                "name": True,
                "email": True,
            },
        },
        "debug": {
            "enabled": True,
            "log_level": True,
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Find first existing config file
            self.config_path = self._find_config()
        else:
            self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._base_config: Dict[str, Any] = {}
        self._env_overlay: Dict[str, Any] = {}
        self._profile_env_present: bool = False
        self._profile_load_error: Optional[str] = None
        self._managed_overlay_meta: Dict[str, Any] = {
            "runtime_profile_id": None,
            "revision": None,
        }
        self._managed_sections: List[str] = []
        self._mobile_env_vars: set[str] = set()
        self._external_config_status: Dict[str, Any] = {
            "success": True,
            "error": None,
            "operation": None,
        }
        self._yaml = _yaml  # Use module-level instance
        self.load()

    def _find_config(self) -> Path:
        """Find the first existing config file from default paths."""
        default_paths = [
            _home_path() / ".efp" / "config.yaml",
            Path(__file__).parent / "config.yaml",
        ]
        for path in default_paths:
            if path.exists():
                return path
        # Return the primary path even if it doesn't exist
        import shutil
        target = default_paths[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.PROJECT_EXAMPLE, target)
        return target

    def load(self) -> None:
        """Load the read-only base config.yaml and overlay EFP_PROFILE_CONFIG.

        The base config ships with the image and is never rewritten. Portal-managed
        runtime-profile fields arrive as a JSON payload in the EFP_PROFILE_CONFIG
        environment variable (rendered into the pod from a per-profile Secret) and
        are merged in memory only. Absent env var means dev mode (base config only).
        """
        self._base_config = self._load_yaml_document(self.config_path)
        self._warn_on_encrypted_values(self._base_config)
        self._parse_profile_env()
        self._rebuild_effective_config()

    def _warn_on_encrypted_values(self, obj: Any, path: str = "") -> None:
        """Warn about legacy ENC: values; encrypted config is no longer supported."""
        if self._is_mapping(obj):
            for key, value in obj.items():
                child = f"{path}.{key}" if path else str(key)
                if isinstance(value, str) and value.startswith("ENC:"):
                    logger.warning(
                        "Config value %s starts with 'ENC:'. Encrypted config values are no "
                        "longer supported; the value will be used as-is. Portal-managed "
                        "credentials now arrive via the EFP_PROFILE_CONFIG environment payload.",
                        child,
                    )
                else:
                    self._warn_on_encrypted_values(value, child)
        elif self._is_sequence(obj):
            for item in obj:
                self._warn_on_encrypted_values(item, path)

    def _parse_profile_env(self) -> None:
        """Parse the EFP_PROFILE_CONFIG apply-payload into the managed overlay."""
        self._env_overlay = {}
        self._profile_env_present = False
        self._profile_load_error = None
        self._managed_overlay_meta = {"runtime_profile_id": None, "revision": None}
        self._managed_sections = []

        raw = os.environ.get("EFP_PROFILE_CONFIG")
        if raw is None:
            return
        self._profile_env_present = True

        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
        except Exception as exc:
            self._profile_load_error = f"Invalid EFP_PROFILE_CONFIG payload: {exc}"
            logger.error(self._profile_load_error)
            return

        overlay_config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        # The portal encrypts the sensitive VALUES (api keys, tokens, passwords)
        # of the canonical config as ENC:<fernet-token> so broad Secret readers
        # see ciphertext. Decrypt them here, immediately after parsing the payload
        # and BEFORE the per-runtime projection, so the projection and every
        # downstream consumer see plaintext. Raises if an ENC: value is present
        # but EFP_CONFIG_KEY is unset; surface that like a parse error below.
        try:
            overlay_config = decrypt_sensitive_fields(overlay_config)
        except Exception as exc:
            self._profile_load_error = f"Invalid EFP_PROFILE_CONFIG payload: {exc}"
            logger.error(self._profile_load_error)
            return
        # The Secret stores a single runtime-agnostic canonical config; this
        # native runtime applies its own projection at boot (the portal used to
        # bake this into the per-runtime Secret payload). For native this
        # re-adds the CLI tool instructions and keeps the LLM in canonical
        # github_copilot/bare-model form. The payload no longer carries a
        # runtime_type field; native is always the native runtime.
        overlay_config = project_canonical_for_runtime(overlay_config, "native")
        overlay = self._filter_managed_overlay_sections(overlay_config)
        self._env_overlay = overlay
        self._managed_overlay_meta = {
            "runtime_profile_id": payload.get("runtime_profile_id"),
            "revision": payload.get("revision"),
        }
        self._managed_sections = sorted(overlay.keys())

    def _load_yaml_document(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return CommentedMap()
        with open(path, "r", encoding="utf-8") as f:
            document = self._yaml.load(f) or CommentedMap()
        return document if isinstance(document, dict) else CommentedMap()

    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(base)
        self._deep_merge_into(result, overlay)
        return result

    def _deep_merge_into(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in (overlay or {}).items():
            if self._is_mapping(base.get(key)) and self._is_mapping(value):
                self._deep_merge_into(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
        return base

    def _filter_by_field_tree(self, source: Dict[str, Any], field_tree: Dict[str, Any]) -> Dict[str, Any]:
        filtered: Dict[str, Any] = {}
        if not self._is_mapping(source) or not self._is_mapping(field_tree):
            return filtered
        for key, subtree in field_tree.items():
            if key not in source:
                continue
            value = source.get(key)
            if subtree is True:
                filtered[key] = copy.deepcopy(value)
                continue
            if self._is_mapping(value):
                nested = self._filter_by_field_tree(value, subtree)
                if nested:
                    filtered[key] = nested
        return filtered

    def _filter_managed_overlay_sections(self, overlay_config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(overlay_config, dict):
            return {}
        filtered: Dict[str, Any] = {}
        for section, value in overlay_config.items():
            if section in self.MANAGED_OVERLAY_SECTIONS:
                filtered[section] = copy.deepcopy(value)
        return self._filter_by_field_tree(filtered, self.PORTAL_MANAGED_FIELD_TREE)

    def _rebuild_effective_config(self) -> None:
        self._config = self._deep_merge(self._base_config, self._env_overlay)
        llm_cfg = self._config.get("llm")
        if isinstance(llm_cfg, dict):
            llm_cfg.setdefault("reasoning_effort", "high")
            llm_cfg.setdefault("reasoning_replay", False)

    def _set_external_config_status(
        self,
        operation: Optional[str],
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        self._external_config_status = {
            "success": bool(success),
            "error": error if error else None,
            "operation": operation if operation in {"apply", "clear"} else None,
        }

    def get_effective_config(self) -> Dict[str, Any]:
        return copy.deepcopy(self._config)

    def get_managed_overlay_meta(self) -> Dict[str, Any]:
        return {
            "runtime_profile_id": self._managed_overlay_meta.get("runtime_profile_id"),
            "revision": self._managed_overlay_meta.get("revision"),
            "managed_sections": sorted(self._managed_sections),
        }

    def get_external_config_status(self) -> Dict[str, Any]:
        return {
            "success": bool(self._external_config_status.get("success")),
            "error": self._external_config_status.get("error"),
            "operation": self._external_config_status.get("operation"),
        }

    def _is_mapping(self, obj: Any) -> bool:
        """Check if obj is a mapping (dict or CommentedMap)."""
        from collections.abc import Mapping
        return isinstance(obj, Mapping)
    
    def _is_sequence(self, obj: Any) -> bool:
        """Check if obj is a sequence (list or CommentedSeq)."""
        from collections.abc import Sequence
        return isinstance(obj, Sequence) and not isinstance(
            obj, (str, bytes, bytearray, memoryview)
        )
    
    @property
    def config_source(self) -> str:
        """Return the path to the loaded config file."""
        return str(self.config_path)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (supports dot notation)."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def llm(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self._config.get("llm", {})

    @property
    def session(self) -> Dict[str, Any]:
        """Get session configuration."""
        return self._config.get("session", {})

    @property
    def server(self) -> Dict[str, Any]:
        """Get server configuration."""
        return self._config.get("server", {})

    @property
    def jira(self) -> Dict[str, Any]:
        """Get Jira configuration."""
        return self._config.get("jira", {})
    
    def get_jira_instances(self) -> List[Dict[str, Any]]:
        """Get Jira instances as a list (supports both old and new format)."""
        jira_config = self.jira
        instances = jira_config.get("instances", [])
        
        # Backward compatibility: if instances is empty but url exists, convert old format
        if not instances and _first_atlassian_instance_url(jira_config):
            instances = [{
                "name": "Default",
                "url": _first_atlassian_instance_url(jira_config),
                "project": jira_config.get("project", ""),
                "username": jira_config.get("username", ""),
                "password": jira_config.get("password", ""),
                "token": jira_config.get("token", ""),
                "api_version": jira_config.get("api_version", "3"),
                "timeout": jira_config.get("timeout", 30.0),
            }]
        
        return self._normalize_atlassian_instances(instances)
    
    def find_jira_instance(self, url: str = None, name: str = None) -> Optional[Dict[str, Any]]:
        """Find Jira instance by URL or name."""
        instances = self.get_jira_instances()
        
        if not instances:
            return None
        
        # Match by name first
        if name:
            for inst in instances:
                if inst.get("name", "").lower() == name.lower():
                    return inst
        
        # Match by URL
        if url:
            for inst in instances:
                inst_url = inst.get("url", "")
                if inst_url and url.startswith(inst_url):
                    return inst
        
        # Return first instance as default
        return instances[0] if instances else None

    @property
    def confluence(self) -> Dict[str, Any]:
        """Get Confluence configuration."""
        return self._config.get("confluence", {})
    
    def get_confluence_instances(self) -> List[Dict[str, Any]]:
        """Get Confluence instances as a list (supports both old and new format)."""
        confluence_config = self.confluence
        instances = confluence_config.get("instances", [])
        
        # Backward compatibility: if instances is empty but url exists, convert old format
        if not instances and _first_atlassian_instance_url(confluence_config):
            instances = [{
                "name": "Default",
                "url": _first_atlassian_instance_url(confluence_config),
                "username": confluence_config.get("username", ""),
                "password": confluence_config.get("password", ""),
                "token": confluence_config.get("token", ""),
                "space": confluence_config.get("space", ""),
            }]
        
        return self._normalize_atlassian_instances(instances)

    def _normalize_atlassian_instances(self, instances: Any) -> List[Dict[str, Any]]:
        if not isinstance(instances, list):
            return []
        normalized: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for raw in instances:
            if not isinstance(raw, dict):
                continue
            url = _first_atlassian_instance_url(raw)
            if not url:
                continue
            item = copy.deepcopy(raw)
            item["url"] = url
            key = (str(item.get("name") or "").strip().lower(), url.lower())
            if key in seen:
                continue
            seen.add(key)
            normalized.append(item)
        return normalized
    
    def find_confluence_instance(
        self,
        url: str = None,
        name: str = None,
        strict: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Find Confluence instance by URL or name."""
        instances = self.get_confluence_instances()
        
        if not instances:
            return None
        
        # Match by name first
        if name:
            for inst in instances:
                if inst.get("name", "").lower() == name.lower():
                    return inst
        
        # Match by URL
        if url:
            for inst in instances:
                inst_url = inst.get("url", "")
                if inst_url and url.startswith(inst_url):
                    return inst
        
        if strict:
            return None

        # Return first instance as default
        return instances[0] if instances else None
    
    @property
    def debug(self) -> Dict[str, Any]:
        """Get debug configuration."""
        return self._config.get("debug", {})
    
    @property
    def proxy(self) -> Dict[str, Any]:
        """Get proxy configuration."""
        return self._config.get("proxy", {})

    def _clear_proxy_env(self) -> None:
        for var in self.PROXY_ENV_VARS:
            os.environ.pop(var, None)
    
    def apply_proxy(self) -> None:
        """Apply proxy settings to os.environ."""
        from src.utils.proxy import no_proxy_value, proxy_url_with_credentials

        proxy_config = self.proxy
        if proxy_config.get("enabled") and proxy_config.get("url"):
            url = proxy_url_with_credentials(
                proxy_config.get("url", ""),
                proxy_config.get("username"),
                proxy_config.get("password"),
            )
            
            os.environ["http_proxy"] = url
            os.environ["https_proxy"] = url
            os.environ["HTTP_PROXY"] = url
            os.environ["HTTPS_PROXY"] = url
            os.environ["all_proxy"] = url
            os.environ["ALL_PROXY"] = url
            # Handle no_proxy for internal addresses
            no_proxy = no_proxy_value(proxy_config)
            os.environ["no_proxy"] = no_proxy
            os.environ["NO_PROXY"] = no_proxy
        elif "proxy" in self._config:
            # Only clear if proxy section exists but is disabled
            # Don't clear inherited env vars when proxy section is absent
            self._clear_proxy_env()

    @property
    def jenkins(self) -> Dict[str, Any]:
        return self._config.get("jenkins", {})

    def _clear_jenkins_env(self) -> None:
        for var in self.JENKINS_ENV_VARS:
            os.environ.pop(var, None)

    @staticmethod
    def _jenkins_default_instance(jenkins_config: Dict[str, Any]) -> Dict[str, Any]:
        """Return the Jenkins instance the CLI uses when no --instance is passed.

        Multi-instance profiles carry ``instances`` plus an optional
        ``default_instance`` name; the legacy flat profile shape is its own
        single instance.
        """
        instances = jenkins_config.get("instances")
        if not isinstance(instances, list):
            return jenkins_config
        # Ask the projection rather than re-deriving the choice. It filters out
        # instances with no base URL and de-duplicates repeated/blank names
        # (a second "ci" is projected as "ci-2"), and ``default_instance`` is
        # matched against those *projected* names. Any second implementation
        # drifts from it, and the drift is dangerous in one specific way: the
        # exported EFP_JENKINS_USERNAME/PASSWORD would belong to a different
        # controller than the exported EFP_JENKINS_DEFAULT_INSTANCE, so an
        # agent following them authenticates against the wrong host.
        try:
            from src.external_cli.profile_config import (
                _build_product_instances,
                _default_instance_name,
            )
        except Exception:  # pragma: no cover - defensive, keeps boot working
            return {}
        projected = _build_product_instances(jenkins_config, product="jenkins")
        if not projected:
            return {}
        chosen_name = _default_instance_name(jenkins_config, projected)
        chosen = next(
            (item for item in projected if item.get("name") == chosen_name),
            projected[0],
        )
        auth = chosen.get("auth") if isinstance(chosen.get("auth"), dict) else {}
        # Only basic password auth maps onto the flat username/password env
        # pair, which is what this has always exported.
        if auth.get("type") != "basic_password":
            return {}
        return {
            "name": chosen.get("name", ""),
            "url": chosen.get("base_url", ""),
            "username": auth.get("username", ""),
            "password": auth.get("secret", ""),
        }

    def apply_jenkins_env(self) -> None:
        jenkins_config = self.jenkins
        instance = self._jenkins_default_instance(jenkins_config) if isinstance(jenkins_config, dict) else {}
        username = str(instance.get("username") or "").strip() if isinstance(instance, dict) else ""
        password = str(instance.get("password") or "").strip() if isinstance(instance, dict) else ""
        if isinstance(jenkins_config, dict) and jenkins_config.get("enabled") and username and password:
            os.environ["EFP_JENKINS_USERNAME"] = username
            os.environ["EFP_JENKINS_PASSWORD"] = password
            os.environ["JENKINS_USERNAME"] = username
            os.environ["JENKINS_PASSWORD"] = password
        else:
            self._clear_jenkins_env()

    @property
    def mobile(self) -> Dict[str, Any]:
        return self._config.get("mobile-auto", {})

    def _clear_mobile_env(self) -> None:
        for var in set(self.MOBILE_ENV_VARS) | set(self._mobile_env_vars):
            os.environ.pop(var, None)
        self._mobile_env_vars = set()

    def apply_mobile_env(self) -> None:
        mobile_config = self.mobile
        browserstack = (
            mobile_config.get("browserstack")
            if isinstance(mobile_config, dict) and isinstance(mobile_config.get("browserstack"), dict)
            else {}
        )
        username = str(browserstack.get("username") or "").strip()
        access_key = str(browserstack.get("access_key") or "").strip()
        username_env = str(browserstack.get("username_env") or "BROWSERSTACK_USERNAME").strip()
        access_key_env = str(browserstack.get("access_key_env") or "BROWSERSTACK_ACCESS_KEY").strip()

        self._clear_mobile_env()
        if not (isinstance(mobile_config, dict) and mobile_config.get("enabled") and isinstance(browserstack, dict)):
            return
        if username and username_env:
            os.environ[username_env] = username
            os.environ["BROWSERSTACK_USERNAME"] = username
            self._mobile_env_vars.update({username_env, "BROWSERSTACK_USERNAME"})
        if access_key and access_key_env:
            os.environ[access_key_env] = access_key
            os.environ["BROWSERSTACK_ACCESS_KEY"] = access_key
            self._mobile_env_vars.update({access_key_env, "BROWSERSTACK_ACCESS_KEY"})

        # Expose the BrowserStack Local binary path so the mobile-auto CLI
        # resolves it directly (parity with the opencode runtime, which sets
        # BROWSERSTACK_LOCAL_BINARY from the bundled path). Honor an explicit
        # profile-configured path unconditionally; otherwise fall back to the
        # bundled default only when it actually exists, so a missing binary
        # leaves mobile-auto's PATH lookup of "BrowserStackLocal" intact
        # instead of pinning BROWSERSTACK_LOCAL_BINARY at a phantom path.
        local = browserstack.get("local") if isinstance(browserstack.get("local"), dict) else {}
        binary_path = str(local.get("binary") or "").strip() if isinstance(local, dict) else ""
        if not binary_path and os.path.exists(self.DEFAULT_BROWSERSTACK_LOCAL_BINARY_PATH):
            binary_path = self.DEFAULT_BROWSERSTACK_LOCAL_BINARY_PATH
        if binary_path:
            os.environ["BROWSERSTACK_LOCAL_BINARY"] = binary_path
            self._mobile_env_vars.add("BROWSERSTACK_LOCAL_BINARY")
    
    @property
    def heartbeat(self) -> Dict[str, Any]:
        """Get heartbeat configuration."""
        return self._config.get("heartbeat", {})


DEFAULT_MODEL_LIMITS: Dict[str, Dict[str, int]] = {
    "gpt-5-mini": {
        "max_context_window_tokens": 264000,
        "max_prompt_tokens": 128000,
        "max_output_tokens": 64000,
    },
    "gpt-5.3-codex": {
        "max_context_window_tokens": 400000,
        "max_prompt_tokens": 272000,
        "max_output_tokens": 128000,
    },
    "gpt-5.4": {
        "max_context_window_tokens": 400000,
        "max_prompt_tokens": 272000,
        "max_output_tokens": 128000,
    },
    "gpt-5.4-mini": {
        "max_context_window_tokens": 400000,
        "max_prompt_tokens": 272000,
        "max_output_tokens": 128000,
    },
    "gpt-5.5": {
        "max_context_window_tokens": 400000,
        "max_prompt_tokens": 272000,
        "max_output_tokens": 128000,
    },
    "gpt-5.6-luna": {
        "max_context_window_tokens": 328000,
        "max_prompt_tokens": 200000,
        "max_output_tokens": 128000,
    },
    "gpt-5.6-sol": {
        "max_context_window_tokens": 400000,
        "max_prompt_tokens": 272000,
        "max_output_tokens": 128000,
    },
    "gpt-5.6-terra": {
        "max_context_window_tokens": 400000,
        "max_prompt_tokens": 272000,
        "max_output_tokens": 128000,
    },
    "gemini-2.5-pro": {
        "max_context_window_tokens": 128_000,
        "max_prompt_tokens": 128_000,
        "max_output_tokens": 64000,
    },
    "gemini-3.5-flash": {
        "max_context_window_tokens": 128_000,
        "max_prompt_tokens": 128_000,
        "max_output_tokens": 64000,
    },
}


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except Exception:
        return default


def resolve_llm_temperature(explicit: Optional[Any] = None) -> float:
    source = explicit if explicit is not None else config.llm.get("temperature", DEFAULT_LLM_TEMPERATURE)
    if isinstance(source, bool):
        return float(DEFAULT_LLM_TEMPERATURE)
    if source is None:
        return float(DEFAULT_LLM_TEMPERATURE)
    if isinstance(source, str):
        source = source.strip()
        if not source:
            return float(DEFAULT_LLM_TEMPERATURE)
    try:
        parsed = float(source)
    except (TypeError, ValueError):
        return float(DEFAULT_LLM_TEMPERATURE)
    if not math.isfinite(parsed) or parsed < 0 or parsed > 2:
        return float(DEFAULT_LLM_TEMPERATURE)
    return float(parsed)


def resolve_model_limits(model: Optional[str] = None) -> Dict[str, int]:
    llm_cfg = config.llm if isinstance(config.llm, dict) else {}
    configured_model = _canonical_model_limit_key(
        model or llm_cfg.get("model") or DEFAULT_LLM_MODEL
    )
    configured_limits = llm_cfg.get("model_limits") if isinstance(llm_cfg.get("model_limits"), dict) else {}
    candidates: Dict[str, Dict[str, int]] = dict(DEFAULT_MODEL_LIMITS)
    for key, raw in configured_limits.items():
        if not isinstance(raw, dict):
            continue
        candidates[_canonical_model_limit_key(key)] = {
            "max_context_window_tokens": _safe_positive_int(raw.get("max_context_window_tokens"), 264000),
            "max_prompt_tokens": _safe_positive_int(raw.get("max_prompt_tokens"), 128000),
            "max_output_tokens": _safe_positive_int(raw.get("max_output_tokens"), _safe_positive_int(llm_cfg.get("max_tokens"), 64000)),
        }

    selected = candidates.get(configured_model, {})
    if not selected and configured_model:
        for key in sorted(candidates.keys(), key=len, reverse=True):
            if key in configured_model:
                selected = candidates[key]
                break
    if not selected:
        selected = {
            "max_context_window_tokens": 200000,
            "max_prompt_tokens": 128000,
            "max_output_tokens": _safe_positive_int(llm_cfg.get("max_tokens"), 64000),
        }
    selected = dict(selected)
    selected["max_output_tokens"] = _safe_positive_int(selected.get("max_output_tokens"), _safe_positive_int(llm_cfg.get("max_tokens"), 64000))
    selected["max_prompt_tokens"] = _safe_positive_int(selected.get("max_prompt_tokens"), 128000)
    selected["max_context_window_tokens"] = _safe_positive_int(selected.get("max_context_window_tokens"), 264000)
    return selected


def _canonical_model_limit_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "/" in text:
        text = text.split("/", 1)[1]
    return "-".join(text.split())


def resolve_output_boundary(model: Optional[str] = None) -> Dict[str, int | str]:
    llm_cfg = config.llm if isinstance(config.llm, dict) else {}
    output_cfg = llm_cfg.get("output_controller") if isinstance(llm_cfg.get("output_controller"), dict) else {}
    limits = resolve_model_limits(model)
    max_output_tokens = _safe_positive_int(limits.get("max_output_tokens"), _safe_positive_int(llm_cfg.get("max_tokens"), 64000))
    configured_chat_tokens = output_cfg.get("max_chat_output_tokens")
    default_chat_tokens = max(1, int(max_output_tokens * 0.9375))
    max_chat_output_tokens = _safe_positive_int(configured_chat_tokens, default_chat_tokens)
    max_chat_output_tokens = min(max_chat_output_tokens, max_output_tokens)
    chars_per_token = _safe_positive_int(output_cfg.get("chars_per_token_estimate"), 4)
    configured_chars = output_cfg.get("max_chat_output_chars")
    derived_chars = max_chat_output_tokens * chars_per_token
    min_reasonable_chars = int(derived_chars * 0.25)
    legacy_ignored = False
    boundary_source = "model_limits_derived"
    if configured_chars in (None, "", "null"):
        max_chat_output_chars = derived_chars
    else:
        parsed_chars = _safe_positive_int(configured_chars, derived_chars)
        allow_low = bool(output_cfg.get("allow_low_max_chat_output_chars", False))
        if parsed_chars < min_reasonable_chars and not allow_low:
            max_chat_output_chars = derived_chars
            legacy_ignored = True
            boundary_source = "model_limits_legacy_override_ignored"
        else:
            max_chat_output_chars = parsed_chars
            boundary_source = "config_override"
    strategy = str(output_cfg.get("oversized_output_strategy") or "save_and_manifest")
    return {
        "max_context_window_tokens": int(limits.get("max_context_window_tokens") or 264000),
        "max_prompt_tokens": int(limits.get("max_prompt_tokens") or 128000),
        "max_output_tokens": max_output_tokens,
        "max_chat_output_tokens": max_chat_output_tokens,
        "chars_per_token_estimate": chars_per_token,
        "max_chat_output_chars": max_chat_output_chars,
        "allow_low_max_chat_output_chars": bool(output_cfg.get("allow_low_max_chat_output_chars", False)),
        "configured_max_chat_output_chars": str(configured_chars) if configured_chars is not None else None,
        "legacy_max_chat_output_chars_ignored": legacy_ignored,
        "output_boundary_source": boundary_source,
        "oversized_output_strategy": strategy,
    }


# Global config instance
config = Config()


# Boot-time profile projection state consumed by GET /ready. The gateway stays
# unready (503) until bootstrap_profile_boot() has completed successfully.
_profile_boot_state: Dict[str, Any] = {
    "completed": False,
    "ready": False,
    "error": None,
}


def get_profile_boot_state() -> Dict[str, Any]:
    return dict(_profile_boot_state)


def _set_profile_boot_state(*, completed: bool, ready: bool, error: Optional[str]) -> None:
    _profile_boot_state["completed"] = completed
    _profile_boot_state["ready"] = ready
    _profile_boot_state["error"] = error


def bootstrap_profile_boot() -> bool:
    """Project the EFP_PROFILE_CONFIG overlay exactly once at process boot.

    Runs from main.py after Config construction and before importing
    src.gateway.server (Gateway() executes at import). Steps:

    1. Project gh/aws/git external CLI config from the overlay via real CLIs.
       Jira/Confluence/Jenkins/mobile-auto/visual reach the Go CLIs through
       the EFP_-prefixed tools config env vars only.
    2. Export the tools config env vars (EFP_-prefixed indexed vars flattened
       from the tools RootConfig-shaped subset of the effective config, e.g.
       EFP_JIRA_INSTANCES_0_BASE_URL / EFP_AWS_DOMAIN) for every CLI child
       process.
    3. Apply proxy / jenkins / mobile env exactly once.
    4. Scrub EFP_PROFILE_CONFIG from os.environ so no child process can see the
       full profile blob.
    5. Record success/failure for GET /ready.

    A projection failure keeps the process alive (liveness /health stays ok) but
    leaves /ready at 503 so the pod never becomes ready with a broken profile.
    """
    error: Optional[str] = config._profile_load_error
    profile_env_present = config._profile_env_present

    if error is None and profile_env_present:
        from src.external_cli.profile_config import (
            apply_runtime_profile_external_config,
            redact_runtime_profile_external_config_error,
        )

        try:
            apply_runtime_profile_external_config(config._env_overlay, config_path=config.config_path)
        except Exception as exc:
            error = redact_runtime_profile_external_config_error(exc, config._env_overlay)
            logger.warning(
                "Runtime profile external CLI config apply failed: %s",
                error,
                exc_info=(RuntimeError, RuntimeError(error), exc.__traceback__),
            )

    if profile_env_present:
        try:
            from src.external_cli.profile_config import (
                build_tools_config_json,
                flatten_config_to_env,
            )

            root = build_tools_config_json(config.get_effective_config())
            for key, value in flatten_config_to_env(root).items():
                os.environ[key] = value
        except Exception as exc:
            if error is None:
                error = f"Failed to build tools config env vars: {exc}"
            logger.warning("Failed to build tools config env vars", exc_info=True)

    config.apply_proxy()
    config.apply_jenkins_env()
    config.apply_mobile_env()

    # Scrub the full profile blob AFTER the external CLI projection took its
    # os.environ.copy() snapshot; children must never inherit it.
    os.environ.pop("EFP_PROFILE_CONFIG", None)

    success = error is None
    if profile_env_present:
        config._set_external_config_status("apply", success, error)
    _set_profile_boot_state(completed=True, ready=success, error=error)
    if success:
        logger.info(
            "Runtime profile boot projection completed: profile_id=%s revision=%s sections=%s",
            config._managed_overlay_meta.get("runtime_profile_id"),
            config._managed_overlay_meta.get("revision"),
            config.get_managed_overlay_meta().get("managed_sections"),
        )
    return success
