from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only used before dependencies are installed.

    def load_dotenv() -> None:
        return None


load_dotenv()

DEFAULT_PUBLIC_MODELS = [
    "proxy-default",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]


def _env_first(names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value.strip()
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _json_object_from(value: str, env_name: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{env_name} must be a JSON object")
    return parsed


def _env_json_object(*names: str) -> dict[str, Any]:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return _json_object_from(value.strip(), name)
    return {}


def _env_mapping() -> dict[str, str]:
    """Read model mapping from JSON or comma-separated key=value pairs."""

    raw_json = os.getenv("MODEL_MAPPING_JSON")
    if raw_json and raw_json.strip():
        parsed = _json_object_from(raw_json.strip(), "MODEL_MAPPING_JSON")
        return {str(k): str(v) for k, v in parsed.items()}

    raw = _env_first(("MODEL_MAP", "MODEL_MAPPING"), "")
    if not raw:
        return {}

    if raw.startswith("{"):
        parsed = _json_object_from(raw, "MODEL_MAP")
        return {str(k): str(v) for k, v in parsed.items()}

    mapping: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        delimiter = "=" if "=" in item else ":"
        if delimiter not in item:
            raise ValueError("MODEL_MAP entries must look like claude-model=gpt-model")
        source, target = item.split(delimiter, 1)
        mapping[source.strip()] = target.strip()
    return mapping


def _env_public_models() -> list[str]:
    raw = os.getenv("PUBLIC_MODELS", "").strip()
    if not raw:
        return DEFAULT_PUBLIC_MODELS.copy()
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    openai_api_endpoint: str = field(
        default_factory=lambda: _env_first(
            ("OPENAI_API_ENDPOINT", "OPENAI_BASE_URL"), "https://api.openai.com/v1"
        )
    )
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip())
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.5").strip())
    openai_chat_completions_path: str = field(
        default_factory=lambda: os.getenv("OPENAI_CHAT_COMPLETIONS_PATH", "/chat/completions").strip()
    )
    openai_api_key_header: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY_HEADER", "Authorization").strip()
    )
    openai_api_key_prefix: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY_PREFIX", "Bearer").strip()
    )
    openai_organization: str = field(
        default_factory=lambda: _env_first(("OPENAI_ORGANIZATION", "OPENAI_ORG"), "")
    )
    openai_project: str = field(default_factory=lambda: os.getenv("OPENAI_PROJECT", "").strip())
    openai_max_tokens_field: str = field(
        default_factory=lambda: os.getenv("OPENAI_MAX_TOKENS_FIELD", "auto").strip()
    )
    openai_stream_include_usage: bool = field(
        default_factory=lambda: _env_bool("OPENAI_STREAM_INCLUDE_USAGE", False)
    )
    openai_extra_headers: dict[str, Any] = field(
        default_factory=lambda: _env_json_object("OPENAI_EXTRA_HEADERS_JSON", "OPENAI_EXTRA_HEADERS")
    )
    openai_extra_body: dict[str, Any] = field(
        default_factory=lambda: _env_json_object("OPENAI_EXTRA_BODY_JSON", "OPENAI_EXTRA_BODY")
    )

    model_mapping: dict[str, str] = field(default_factory=_env_mapping)
    model_passthrough: bool = field(default_factory=lambda: _env_bool("MODEL_PASSTHROUGH", False))
    public_models: list[str] = field(default_factory=_env_public_models)

    proxy_auth_token: str = field(
        default_factory=lambda: _env_first(("PROXY_AUTH_TOKEN", "PROXY_API_KEY"), "")
    )
    require_openai_api_key: bool = field(default_factory=lambda: _env_bool("REQUIRE_OPENAI_API_KEY", True))
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").strip())
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0").strip())
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8080")))
    approximate_chars_per_token: int = field(
        default_factory=lambda: int(os.getenv("APPROXIMATE_CHARS_PER_TOKEN", "4"))
    )

    def chat_completions_url(self) -> str:
        base = self.openai_api_endpoint.strip().rstrip("/")
        path = self.openai_chat_completions_path.strip()
        if not path:
            return base
        normalized_path = path if path.startswith("/") else f"/{path}"
        if base.endswith(normalized_path):
            return base
        return f"{base}{normalized_path}"

    def resolved_model(self, incoming_model: str | None) -> str:
        if incoming_model and incoming_model in self.model_mapping:
            return self.model_mapping[incoming_model]
        if self.model_passthrough and incoming_model:
            return incoming_model
        return self.openai_model or incoming_model or ""

    def resolved_max_tokens_field(self) -> str:
        if self.openai_max_tokens_field and self.openai_max_tokens_field != "auto":
            return self.openai_max_tokens_field
        host = urlparse(self.openai_api_endpoint).hostname or ""
        if host.endswith("api.openai.com"):
            return "max_completion_tokens"
        return "max_tokens"

    def upstream_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "claude-code-proxy/0.1.0",
        }
        if self.openai_api_key and self.openai_api_key_header:
            if self.openai_api_key_header.lower() == "authorization":
                prefix = self.openai_api_key_prefix.strip()
                headers["Authorization"] = (
                    f"{prefix} {self.openai_api_key}" if prefix else self.openai_api_key
                )
            else:
                headers[self.openai_api_key_header] = self.openai_api_key
        if self.openai_organization:
            headers["OpenAI-Organization"] = self.openai_organization
        if self.openai_project:
            headers["OpenAI-Project"] = self.openai_project
        headers.update({str(k): str(v) for k, v in self.openai_extra_headers.items()})
        return headers

    def is_ready(self) -> bool:
        if not self.openai_api_endpoint or not self.openai_model:
            return False
        if self.require_openai_api_key and not self.openai_api_key:
            return False
        return True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
