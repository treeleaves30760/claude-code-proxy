from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only used before dependencies are installed.

    def load_dotenv() -> None:
        return None


load_dotenv()

DEFAULT_PUBLIC_MODELS = [
    "proxy-default",
    "openai-gpt-5.5",
    "openai-gpt-5.5-pro",
    "openrouter-gpt-5.5",
    "openrouter-claude-opus-4.7",
    "openrouter-claude-sonnet-4.6",
    "ollama-gpt-oss-20b",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


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


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def interpolate_env(value: Any) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            fallback = match.group(2) or ""
            return os.getenv(name, fallback)

        return ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [interpolate_env(item) for item in value]
    if isinstance(value, dict):
        return {str(k): interpolate_env(v) for k, v in value.items()}
    return value


def compact_url(base_url: str, path: str) -> str:
    base = base_url.strip().rstrip("/")
    endpoint_path = path.strip()
    if not endpoint_path:
        return base
    normalized_path = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
    if base.endswith(normalized_path):
        return base
    return f"{base}{normalized_path}"


@dataclass
class ProviderConfig:
    name: str
    type: str = "openai-compatible"
    base_url: str = "https://api.openai.com/v1"
    chat_completions_path: str = "/chat/completions"
    api_key: str = ""
    api_key_env: str = ""
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    require_api_key: bool = True
    max_tokens_field: str = "auto"
    stream_include_usage: bool = False
    headers: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> ProviderConfig:
        provider_type = str(data.get("type") or data.get("kind") or "openai-compatible")
        defaults = provider_type_defaults(provider_type)
        merged = interpolate_env({**defaults, **data})
        api_key = str(merged.get("api_key") or "")
        api_key_env = str(merged.get("api_key_env") or "")
        if api_key_env and not api_key:
            api_key = os.getenv(api_key_env, "")
        return cls(
            name=name,
            type=provider_type,
            base_url=str(merged.get("base_url") or merged.get("endpoint") or defaults["base_url"]),
            chat_completions_path=str(
                merged.get("chat_completions_path")
                or merged.get("chat_path")
                or defaults["chat_completions_path"]
            ),
            api_key=api_key,
            api_key_env=api_key_env,
            api_key_header=str(merged.get("api_key_header") or defaults["api_key_header"]),
            api_key_prefix=str(merged.get("api_key_prefix") or defaults["api_key_prefix"]),
            require_api_key=as_bool(merged.get("require_api_key"), defaults["require_api_key"]),
            max_tokens_field=str(merged.get("max_tokens_field") or defaults["max_tokens_field"]),
            stream_include_usage=as_bool(
                merged.get("stream_include_usage"), defaults["stream_include_usage"]
            ),
            headers=dict(merged.get("headers") or {}),
            extra_body=dict(merged.get("extra_body") or {}),
        )

    @property
    def chat_completions_url(self) -> str:
        return compact_url(self.base_url, self.chat_completions_path)

    def resolved_max_tokens_field(self) -> str:
        if self.max_tokens_field and self.max_tokens_field != "auto":
            return self.max_tokens_field
        host = urlparse(self.base_url).hostname or ""
        if host.endswith("api.openai.com"):
            return "max_completion_tokens"
        return "max_tokens"

    def upstream_headers(self, extra_headers: dict[str, Any] | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "claude-code-proxy/0.1.0",
        }
        if self.api_key and self.api_key_header:
            if self.api_key_header.lower() == "authorization":
                prefix = self.api_key_prefix.strip()
                headers["Authorization"] = f"{prefix} {self.api_key}" if prefix else self.api_key
            else:
                headers[self.api_key_header] = self.api_key
        headers.update({str(k): str(v) for k, v in self.headers.items() if v not in (None, "")})
        if extra_headers:
            headers.update({str(k): str(v) for k, v in extra_headers.items() if v not in (None, "")})
        return headers

    def readiness_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.base_url:
            errors.append(f"{self.name}.base_url")
        if self.require_api_key and not self.api_key:
            suffix = f" env {self.api_key_env}" if self.api_key_env else ""
            errors.append(f"{self.name}.api_key{suffix}")
        return errors


@dataclass
class ModelRoute:
    public_model: str
    provider_name: str
    upstream_model: str
    aliases: list[str] = field(default_factory=list)
    extra_headers: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_tokens_field: str = ""


@dataclass
class ResolvedRoute:
    public_model: str
    provider: ProviderConfig
    upstream_model: str
    extra_headers: dict[str, Any] = field(default_factory=dict)
    route_extra_body: dict[str, Any] = field(default_factory=dict)
    route_max_tokens_field: str = ""

    def resolved_model(self, _: str | None = None) -> str:
        return self.upstream_model

    def resolved_max_tokens_field(self) -> str:
        if self.route_max_tokens_field:
            return self.route_max_tokens_field
        return self.provider.resolved_max_tokens_field()

    @property
    def openai_stream_include_usage(self) -> bool:
        return self.provider.stream_include_usage

    @property
    def openai_extra_body(self) -> dict[str, Any]:
        return {**self.provider.extra_body, **self.route_extra_body}

    def upstream_headers(self) -> dict[str, str]:
        return self.provider.upstream_headers(self.extra_headers)

    @property
    def chat_completions_url(self) -> str:
        return self.provider.chat_completions_url

    def readiness_errors(self) -> list[str]:
        errors = self.provider.readiness_errors()
        if not self.upstream_model:
            errors.append(f"{self.public_model}.model")
        return errors

    def is_ready(self) -> bool:
        return not self.readiness_errors()


@dataclass
class RoutingConfig:
    providers: dict[str, ProviderConfig]
    routes: dict[str, ModelRoute]
    default_model: str
    public_models: list[str]
    model_passthrough: bool = False

    def resolve(self, incoming_model: str | None) -> ResolvedRoute:
        model = str(incoming_model or "").strip()
        route = self.routes.get(model)
        if route is None:
            route = self._resolve_alias(model)
        if route is None and ":" in model:
            provider_name, upstream_model = model.split(":", 1)
            provider = self.providers.get(provider_name)
            if provider and upstream_model:
                return ResolvedRoute(model, provider, upstream_model)
        if route is None and self.model_passthrough and model:
            provider = self.providers[self.default_provider_name()]
            return ResolvedRoute(model, provider, model)
        if route is None:
            route = self.routes[self.default_model]

        provider = self.providers[route.provider_name]
        return ResolvedRoute(
            public_model=route.public_model,
            provider=provider,
            upstream_model=route.upstream_model,
            extra_headers=route.extra_headers,
            route_extra_body=route.extra_body,
            route_max_tokens_field=route.max_tokens_field,
        )

    def _resolve_alias(self, model: str) -> ModelRoute | None:
        if not model:
            return None
        for route in self.routes.values():
            if model in route.aliases:
                return route
        return None

    def default_provider_name(self) -> str:
        return self.routes[self.default_model].provider_name

    def readiness(self) -> dict[str, Any]:
        route_states = {}
        for model, route in self.routes.items():
            resolved = self.resolve(model)
            route_states[model] = {
                "provider": route.provider_name,
                "upstream_model": route.upstream_model,
                "ready": resolved.is_ready(),
                "errors": resolved.readiness_errors(),
            }
        return route_states


def provider_type_defaults(provider_type: str) -> dict[str, Any]:
    normalized = provider_type.lower()
    defaults = {
        "base_url": "https://api.openai.com/v1",
        "chat_completions_path": "/chat/completions",
        "api_key_header": "Authorization",
        "api_key_prefix": "Bearer",
        "require_api_key": True,
        "max_tokens_field": "auto",
        "stream_include_usage": False,
        "headers": {},
        "extra_body": {},
    }
    if normalized == "openai":
        return {**defaults, "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"}
    if normalized == "openrouter":
        return {
            **defaults,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "max_tokens_field": "max_tokens",
            "headers": {
                "HTTP-Referer": "${OPENROUTER_SITE_URL:-}",
                "X-Title": "${OPENROUTER_APP_NAME:-claude-code-proxy}",
            },
        }
    if normalized == "ollama":
        return {
            **defaults,
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "api_key_prefix": "Bearer",
            "require_api_key": False,
            "max_tokens_field": "max_tokens",
        }
    return defaults


def load_yaml_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML config root must be an object")
    return interpolate_env(loaded)


def routing_from_yaml(raw: dict[str, Any], legacy: Settings) -> RoutingConfig:
    provider_defs = raw.get("providers") or {}
    if not isinstance(provider_defs, dict) or not provider_defs:
        raise ValueError("YAML config must define providers")

    providers = {
        str(name): ProviderConfig.from_dict(str(name), dict(data or {}))
        for name, data in provider_defs.items()
    }

    raw_models = raw.get("models") or raw.get("routes") or {}
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("YAML config must define models")

    routes: dict[str, ModelRoute] = {}
    for public_model, route_data in raw_models.items():
        if not isinstance(route_data, dict):
            raise ValueError(f"Model route {public_model} must be an object")
        provider_name = str(route_data.get("provider") or "")
        if provider_name not in providers:
            raise ValueError(f"Model route {public_model} references unknown provider {provider_name}")
        upstream_model = str(route_data.get("model") or route_data.get("upstream_model") or public_model)
        aliases = route_data.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        routes[str(public_model)] = ModelRoute(
            public_model=str(public_model),
            provider_name=provider_name,
            upstream_model=upstream_model,
            aliases=[str(alias) for alias in aliases],
            extra_headers=dict(route_data.get("headers") or {}),
            extra_body=dict(route_data.get("extra_body") or {}),
            max_tokens_field=str(route_data.get("max_tokens_field") or ""),
        )

    default_model = str(raw.get("default_model") or next(iter(routes)))
    if default_model not in routes:
        raise ValueError(f"default_model {default_model} is not defined in models")

    public_models = raw.get("public_models")
    if isinstance(public_models, list):
        visible_models = [str(item) for item in public_models]
    else:
        visible_models = list(routes)

    return RoutingConfig(
        providers=providers,
        routes=routes,
        default_model=default_model,
        public_models=visible_models,
        model_passthrough=as_bool(raw.get("model_passthrough"), legacy.model_passthrough),
    )


def legacy_routing(legacy: Settings) -> RoutingConfig:
    provider = ProviderConfig(
        name="openai",
        type="openai-compatible",
        base_url=legacy.openai_api_endpoint,
        chat_completions_path=legacy.openai_chat_completions_path,
        api_key=legacy.openai_api_key,
        api_key_header=legacy.openai_api_key_header,
        api_key_prefix=legacy.openai_api_key_prefix,
        require_api_key=legacy.require_openai_api_key,
        max_tokens_field=legacy.openai_max_tokens_field,
        stream_include_usage=legacy.openai_stream_include_usage,
        headers={
            **({"OpenAI-Organization": legacy.openai_organization} if legacy.openai_organization else {}),
            **({"OpenAI-Project": legacy.openai_project} if legacy.openai_project else {}),
            **legacy.openai_extra_headers,
        },
        extra_body=legacy.openai_extra_body,
    )
    routes = {
        "proxy-default": ModelRoute("proxy-default", "openai", legacy.openai_model),
    }
    for public_model, upstream_model in legacy.model_mapping.items():
        routes[public_model] = ModelRoute(public_model, "openai", upstream_model)
    default_model = "proxy-default"
    public_models = legacy.public_models or list(routes)
    for public_model in public_models:
        if public_model not in routes:
            routes[public_model] = ModelRoute(public_model, "openai", legacy.openai_model)
    return RoutingConfig(
        providers={"openai": provider},
        routes=routes,
        default_model=default_model,
        public_models=public_models,
        model_passthrough=legacy.model_passthrough,
    )


@dataclass
class Settings:
    """Runtime configuration loaded from YAML first, then legacy environment variables."""

    config_path: str = field(
        default_factory=lambda: _env_first(("MODEL_CONFIG_PATH", "PROXY_CONFIG_PATH"), "config.yaml")
    )
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

    @cached_property
    def routing(self) -> RoutingConfig:
        raw = load_yaml_config(self.config_path)
        if raw:
            return routing_from_yaml(raw, self)
        return legacy_routing(self)

    def resolve_route(self, incoming_model: str | None) -> ResolvedRoute:
        return self.routing.resolve(incoming_model)

    def public_model_ids(self) -> list[str]:
        model_ids = list(self.routing.public_models)
        for route in self.routing.routes.values():
            model_ids.extend(route.aliases)
        return list(dict.fromkeys(model_ids))

    def chat_completions_url(self) -> str:
        return self.resolve_route(None).chat_completions_url

    def resolved_model(self, incoming_model: str | None) -> str:
        return self.resolve_route(incoming_model).upstream_model

    def resolved_max_tokens_field(self) -> str:
        return self.resolve_route(None).resolved_max_tokens_field()

    def upstream_headers(self) -> dict[str, str]:
        return self.resolve_route(None).upstream_headers()

    def is_ready(self) -> bool:
        return self.resolve_route(self.routing.default_model).is_ready()

    def readiness(self) -> dict[str, Any]:
        return deepcopy(self.routing.readiness())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
