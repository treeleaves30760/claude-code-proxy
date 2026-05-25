from __future__ import annotations

from copy import deepcopy
from typing import Any

from .config import ResolvedRoute
from .transform import remove_none_values


def prepare_openai_compatible_request(body: dict[str, Any], route: ResolvedRoute) -> dict[str, Any]:
    payload = deepcopy(body)
    payload["model"] = route.upstream_model

    desired_tokens_field = route.resolved_max_tokens_field()
    if desired_tokens_field == "max_completion_tokens":
        if "max_completion_tokens" not in payload and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
    elif desired_tokens_field == "max_tokens":
        if "max_tokens" not in payload and "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")

    if payload.get("stream") and route.openai_stream_include_usage:
        stream_options = payload.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        stream_options.setdefault("include_usage", True)
        payload["stream_options"] = stream_options

    if route.openai_extra_body:
        payload.update(route.openai_extra_body)

    return remove_none_values(payload)


def restore_public_model(response_body: dict[str, Any], public_model: str) -> dict[str, Any]:
    if response_body.get("object") == "chat.completion" and public_model:
        response_body = deepcopy(response_body)
        response_body["model"] = public_model
    return response_body
