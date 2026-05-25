from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from typing import Any

from .config import Settings


ANTHROPIC_STOP_REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


class ToolNameMapper:
    """Keep OpenAI function names valid while preserving Anthropic-facing names."""

    _valid_name_re = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
    _invalid_chars_re = re.compile(r"[^a-zA-Z0-9_-]")

    def __init__(self) -> None:
        self.original_to_openai: dict[str, str] = {}
        self.openai_to_original: dict[str, str] = {}

    def to_openai(self, original: Any) -> str:
        original_name = str(original or "tool")
        if original_name in self.original_to_openai:
            return self.original_to_openai[original_name]

        candidate = original_name
        if not self._valid_name_re.match(candidate):
            candidate = self._invalid_chars_re.sub("_", candidate)
        candidate = candidate.strip("_") or "tool"
        if len(candidate) > 64:
            suffix = uuid.uuid5(uuid.NAMESPACE_URL, original_name).hex[:8]
            candidate = f"{candidate[:55]}_{suffix}"

        base = candidate
        counter = 2
        while candidate in self.openai_to_original and self.openai_to_original[candidate] != original_name:
            suffix = f"_{counter}"
            candidate = f"{base[: 64 - len(suffix)]}{suffix}"
            counter += 1

        self.original_to_openai[original_name] = candidate
        self.openai_to_original[candidate] = original_name
        return candidate

    def to_anthropic(self, openai_name: Any) -> str:
        name = str(openai_name or "tool")
        return self.openai_to_original.get(name, name)


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {compact_json(data)}\n\n"


def anthropic_to_openai_request(
    body: dict[str, Any], settings: Settings, tool_name_mapper: ToolNameMapper | None = None
) -> dict[str, Any]:
    mapper = tool_name_mapper or ToolNameMapper()
    model = settings.resolved_model(as_str(body.get("model")))
    if not model:
        raise ValueError("No model configured. Set OPENAI_MODEL or pass an OpenAI model name.")

    payload: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages_to_openai(body, mapper),
    }

    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        payload[settings.resolved_max_tokens_field()] = max_tokens

    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stream", "stream"),
        ("user", "user"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("seed", "seed"),
    ):
        if source in body and body[source] is not None:
            payload[target] = body[source]

    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]
    elif body.get("stop"):
        payload["stop"] = body["stop"]

    tools = anthropic_tools_to_openai(body.get("tools"), mapper)
    if tools:
        payload["tools"] = tools
        tool_choice = anthropic_tool_choice_to_openai(body.get("tool_choice"), mapper)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    if payload.get("stream") and settings.openai_stream_include_usage:
        payload["stream_options"] = {"include_usage": True}

    if settings.openai_extra_body:
        payload.update(settings.openai_extra_body)
    request_extra_body = body.get("openai_extra_body")
    if isinstance(request_extra_body, dict):
        payload.update(request_extra_body)

    return remove_none_values(payload)


def anthropic_messages_to_openai(
    body: dict[str, Any], tool_name_mapper: ToolNameMapper | None = None
) -> list[dict[str, Any]]:
    mapper = tool_name_mapper or ToolNameMapper()
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": content_to_text(system)})

    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content", "")
        if role == "assistant":
            messages.append(assistant_message_to_openai(content, mapper))
        elif role == "user":
            messages.extend(user_message_to_openai(content))
        elif role == "tool":
            normalized = dict(message)
            normalized["content"] = tool_result_content_to_text(content)
            messages.append(normalized)
        else:
            messages.append({"role": role or "user", "content": content_to_text(content)})

    return messages


def assistant_message_to_openai(
    content: Any, tool_name_mapper: ToolNameMapper | None = None
) -> dict[str, Any]:
    mapper = tool_name_mapper or ToolNameMapper()
    blocks = normalize_content_blocks(content)
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if text:
                text_parts.append(str(text))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or f"call_{uuid.uuid4().hex}"),
                    "type": "function",
                    "function": {
                        "name": mapper.to_openai(block.get("name") or "tool"),
                        "arguments": compact_json(block.get("input") or {}),
                    },
                }
            )

    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def user_message_to_openai(content: Any) -> list[dict[str, Any]]:
    blocks = normalize_content_blocks(content)
    messages: list[dict[str, Any]] = []
    user_parts: list[Any] = []

    def flush_user_parts() -> None:
        nonlocal user_parts
        if not user_parts:
            return
        messages.append({"role": "user", "content": simplify_openai_user_parts(user_parts)})
        user_parts = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "tool_result":
            flush_user_parts()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or block.get("id") or ""),
                    "content": tool_result_content_to_text(block.get("content", "")),
                }
            )
        elif block_type == "text":
            user_parts.append({"type": "text", "text": str(block.get("text") or "")})
        elif block_type == "image":
            image_part = anthropic_image_block_to_openai(block)
            if image_part:
                user_parts.append(image_part)
            else:
                user_parts.append({"type": "text", "text": "[Unsupported image block]"})
        elif block_type in {"thinking", "redacted_thinking"}:
            continue
        else:
            user_parts.append({"type": "text", "text": content_to_text(block)})

    flush_user_parts()
    if not messages:
        messages.append({"role": "user", "content": ""})
    return messages


def anthropic_tools_to_openai(
    tools: Any, tool_name_mapper: ToolNameMapper | None = None
) -> list[dict[str, Any]]:
    mapper = tool_name_mapper or ToolNameMapper()
    if not isinstance(tools, list):
        return []

    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        parameters = tool.get("input_schema") or tool.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": mapper.to_openai(tool["name"]),
                    "description": str(tool.get("description") or ""),
                    "parameters": parameters,
                },
            }
        )
    return converted


def anthropic_tool_choice_to_openai(
    tool_choice: Any, tool_name_mapper: ToolNameMapper | None = None
) -> str | dict[str, Any] | None:
    mapper = tool_name_mapper or ToolNameMapper()
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        return {"type": "function", "function": {"name": mapper.to_openai(tool_choice)}}
    if not isinstance(tool_choice, dict):
        return None

    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": mapper.to_openai(tool_choice["name"])}}
    return None


def openai_to_anthropic_response(
    upstream: dict[str, Any],
    original_body: dict[str, Any],
    settings: Settings,
    tool_name_mapper: ToolNameMapper | None = None,
) -> dict[str, Any]:
    mapper = tool_name_mapper or ToolNameMapper()
    choice = first_choice(upstream)
    openai_message = choice.get("message") or {}
    content_blocks: list[dict[str, Any]] = []

    text = openai_message.get("content")
    if isinstance(text, list):
        text = content_to_text(text)
    if text:
        content_blocks.append({"type": "text", "text": str(text)})

    for tool_call in openai_message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": str(tool_call.get("id") or f"call_{uuid.uuid4().hex}"),
                "name": mapper.to_anthropic(function.get("name") or "tool"),
                "input": parse_tool_arguments(function.get("arguments")),
            }
        )

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": new_message_id(),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": as_str(original_body.get("model")) or settings.resolved_model(None),
        "stop_reason": map_finish_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": openai_usage_to_anthropic(upstream.get("usage")),
    }


def openai_usage_to_anthropic(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
    }


def map_finish_reason(finish_reason: Any) -> str:
    if not finish_reason:
        return "end_turn"
    return ANTHROPIC_STOP_REASONS.get(str(finish_reason), "end_turn")


def approximate_token_count(body: dict[str, Any], settings: Settings) -> int:
    text = content_to_text(body.get("system", ""))
    text += "\n" + content_to_text(body.get("messages", []))
    text += "\n" + content_to_text(body.get("tools", []))
    chars_per_token = max(settings.approximate_chars_per_token, 1)
    return max(1, (len(text) + chars_per_token - 1) // chars_per_token)


def normalize_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                blocks.append(item)
            else:
                blocks.append({"type": "text", "text": str(item)})
        return blocks
    if isinstance(content, dict):
        return [content]
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def simplify_openai_user_parts(parts: list[Any]) -> str | list[Any]:
    if all(isinstance(part, dict) and part.get("type") == "text" for part in parts):
        return "\n".join(str(part.get("text") or "") for part in parts)
    return parts


def anthropic_image_block_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source")
    if not isinstance(source, dict):
        return None

    if source.get("type") == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": str(source["url"])}}

    if source.get("type") == "base64" and source.get("data"):
        media_type = str(source.get("media_type") or "image/png")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{source['data']}"},
        }
    return None


def tool_result_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, dict):
                parts.append(compact_json(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    if isinstance(content, dict):
        return compact_json(content)
    return str(content)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        block_type = content.get("type")
        if block_type == "text":
            return str(content.get("text") or "")
        if block_type == "tool_result":
            return tool_result_content_to_text(content.get("content", ""))
        if block_type == "tool_use":
            return compact_json(
                {
                    "tool_use": {
                        "id": content.get("id"),
                        "name": content.get("name"),
                        "input": content.get("input"),
                    }
                }
            )
        if block_type in {"thinking", "redacted_thinking"}:
            return ""
        return compact_json(content)
    if isinstance(content, Iterable):
        return "\n".join(content_to_text(item) for item in content)
    return str(content)


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            return {"_raw": arguments}
    return {"value": arguments}


def first_choice(upstream: dict[str, Any]) -> dict[str, Any]:
    choices = upstream.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            return choice
    return {"message": {"content": ""}, "finish_reason": "stop"}


def as_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def remove_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        for key in list(value.keys()):
            if value[key] is None:
                del value[key]
            else:
                remove_none_values(value[key])
    elif isinstance(value, list):
        for item in value:
            remove_none_values(item)
    return value
