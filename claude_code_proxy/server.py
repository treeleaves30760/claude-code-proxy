from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import __version__
from .config import ResolvedRoute, Settings, get_settings
from .openai_compat import prepare_openai_compatible_request, restore_public_model
from .transform import (
    ToolNameMapper,
    anthropic_to_openai_request,
    approximate_token_count,
    compact_json,
    map_finish_reason,
    new_message_id,
    openai_to_anthropic_response,
    openai_usage_to_anthropic,
    sse_event,
)


logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Claude Code Multi-Provider Proxy", version=__version__)


async def verify_proxy_auth(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    """Optionally require a shared token from Claude Code callers.

    Claude Code sends ANTHROPIC_AUTH_TOKEN as an Authorization: Bearer header and
    ANTHROPIC_API_KEY as x-api-key. Accept both so users can choose either env var.
    """

    if not settings.proxy_auth_token:
        return

    expected = settings.proxy_auth_token
    candidates: list[str] = []

    authorization = request.headers.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        candidates.append(authorization[7:].strip())
    elif authorization:
        candidates.append(authorization)

    for header_name in ("x-api-key", "anthropic-api-key"):
        header_value = request.headers.get(header_name)
        if header_value:
            candidates.append(header_value.strip())

    if expected not in candidates:
        raise HTTPException(status_code=401, detail="Invalid proxy auth token")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return anthropic_error_response(
        exc.status_code,
        str(exc.detail),
        "authentication_error" if exc.status_code == 401 else "invalid_request_error",
    )


@app.get("/")
async def root(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "name": "claude-code-proxy",
        "status": "ok",
        "ready": settings.is_ready(),
        "version": __version__,
        "default_model": settings.routing.default_model,
        "providers": list(settings.routing.providers),
        "models": settings.public_model_ids(),
        "endpoints": [
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/chat/completions",
            "/v1/models",
            "/healthz",
            "/readyz",
        ],
    }


@app.head("/")
async def root_head() -> Response:
    return Response(status_code=200)


@app.get("/healthz")
async def healthz(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "default_model": settings.routing.default_model,
        "providers": list(settings.routing.providers),
    }


@app.get("/readyz")
async def readyz(settings: Settings = Depends(get_settings)) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_200_OK if settings.is_ready() else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if settings.is_ready() else "not_ready",
            "default_model": settings.routing.default_model,
            "routes": settings.readiness(),
        },
    )


@app.get("/v1/models", dependencies=[Depends(verify_proxy_auth)])
async def list_models(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    model_ids = settings.public_model_ids()
    data = [
        {
            "id": model_id,
            "object": "model",
            "created": 1704067200,
            "owned_by": "claude-code-proxy",
            "type": "model",
            "display_name": model_id,
            "created_at": "2024-01-01T00:00:00Z",
        }
        for model_id in dict.fromkeys(model_ids)
    ]
    return {
        "object": "list",
        "data": data,
        "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }


@app.get("/v1/models/{model_id}", dependencies=[Depends(verify_proxy_auth)])
async def retrieve_model(model_id: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 1704067200,
        "owned_by": "claude-code-proxy",
        "type": "model",
        "display_name": model_id,
        "created_at": "2024-01-01T00:00:00Z",
    }


@app.post("/v1/messages/count_tokens", dependencies=[Depends(verify_proxy_auth)])
async def count_tokens(
    request: Request, settings: Settings = Depends(get_settings)
) -> dict[str, int]:
    body = await read_json_object(request)
    return {"input_tokens": approximate_token_count(body, settings)}


@app.post("/v1/messages", dependencies=[Depends(verify_proxy_auth)])
async def create_message(request: Request, settings: Settings = Depends(get_settings)) -> Any:
    body = await read_json_object(request)
    if not body.get("messages"):
        raise HTTPException(status_code=400, detail="messages is required")

    route = settings.resolve_route(as_optional_str(body.get("model")))
    if not route.is_ready():
        return anthropic_error_response(
            503,
            "Proxy route is not ready. Missing: " + ", ".join(route.readiness_errors()),
            "api_error",
        )

    tool_name_mapper = ToolNameMapper()
    try:
        upstream_payload = anthropic_to_openai_request(body, route, tool_name_mapper)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "proxying anthropic message provider=%s upstream_model=%s public_model=%s stream=%s",
        route.provider.name,
        upstream_payload.get("model"),
        route.public_model,
        bool(upstream_payload.get("stream")),
    )

    if upstream_payload.get("stream"):
        return StreamingResponse(
            stream_openai_as_anthropic(body, upstream_payload, settings, route, tool_name_mapper),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            upstream_response = await client.post(
                route.chat_completions_url,
                headers=route.upstream_headers(),
                json=upstream_payload,
            )
    except httpx.TimeoutException:
        return anthropic_error_response(504, "OpenAI upstream request timed out", "api_error")
    except httpx.HTTPError as exc:
        return anthropic_error_response(502, f"OpenAI upstream request failed: {exc}", "api_error")

    if upstream_response.status_code >= 400:
        return openai_error_response(upstream_response)

    try:
        upstream_json = upstream_response.json()
    except json.JSONDecodeError:
        return anthropic_error_response(502, "OpenAI upstream returned non-JSON response", "api_error")
    return openai_to_anthropic_response(upstream_json, body, route, tool_name_mapper)


@app.post("/v1/chat/completions", dependencies=[Depends(verify_proxy_auth)])
async def create_chat_completion(request: Request, settings: Settings = Depends(get_settings)) -> Any:
    body = await read_json_object(request)
    if not body.get("messages"):
        return openai_proxy_error_response(400, "messages is required", "invalid_request_error")

    incoming_model = as_optional_str(body.get("model"))
    route = settings.resolve_route(incoming_model)
    if not route.is_ready():
        return openai_proxy_error_response(
            503,
            "Proxy route is not ready. Missing: " + ", ".join(route.readiness_errors()),
            "api_error",
        )

    upstream_payload = prepare_openai_compatible_request(body, route)
    logger.info(
        "proxying openai-compatible chat provider=%s upstream_model=%s public_model=%s stream=%s",
        route.provider.name,
        upstream_payload.get("model"),
        route.public_model,
        bool(upstream_payload.get("stream")),
    )

    if upstream_payload.get("stream"):
        return StreamingResponse(
            stream_openai_passthrough(upstream_payload, settings, route),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            upstream_response = await client.post(
                route.chat_completions_url,
                headers=route.upstream_headers(),
                json=upstream_payload,
            )
    except httpx.TimeoutException:
        return openai_proxy_error_response(504, "Upstream request timed out", "api_error")
    except httpx.HTTPError as exc:
        return openai_proxy_error_response(502, f"Upstream request failed: {exc}", "api_error")

    if upstream_response.status_code >= 400:
        return openai_error_response(upstream_response, anthropic=False)

    try:
        response_json = upstream_response.json()
    except json.JSONDecodeError:
        return openai_proxy_error_response(502, "Upstream returned non-JSON response", "api_error")
    return restore_public_model(response_json, incoming_model or route.public_model)


async def stream_openai_as_anthropic(
    original_body: dict[str, Any],
    upstream_payload: dict[str, Any],
    settings: Settings,
    route: ResolvedRoute,
    tool_name_mapper: ToolNameMapper,
) -> AsyncIterator[str]:
    message_id = new_message_id()
    yield sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": str(original_body.get("model") or route.public_model),
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    state = StreamState(tool_name_mapper)
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    finish_reason = "end_turn"

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            async with client.stream(
                "POST",
                route.chat_completions_url,
                headers=route.upstream_headers(),
                json=upstream_payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    yield sse_event(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "type": map_http_status_to_anthropic_error(response.status_code),
                                "message": decode_error_body(body),
                            },
                        },
                    )
                    return

                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("Skipping non-JSON upstream SSE data: %s", data)
                        continue

                    if chunk.get("usage"):
                        usage = openai_usage_to_anthropic(chunk["usage"])

                    async for event in state.consume_chunk(chunk):
                        yield event

                    choice = first_stream_choice(chunk)
                    if choice and choice.get("finish_reason"):
                        finish_reason = map_finish_reason(choice["finish_reason"])
    except asyncio.CancelledError:
        raise
    except httpx.TimeoutException:
        yield sse_event(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": "OpenAI upstream stream timed out"},
            },
        )
        return
    except httpx.HTTPError as exc:
        yield sse_event(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": f"OpenAI upstream stream failed: {exc}"},
            },
        )
        return

    async for event in state.finish():
        yield event

    yield sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": finish_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage.get("output_tokens", 0)},
        },
    )
    yield sse_event("message_stop", {"type": "message_stop"})


async def stream_openai_passthrough(
    upstream_payload: dict[str, Any], settings: Settings, route: ResolvedRoute
) -> AsyncIterator[str]:
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            async with client.stream(
                "POST",
                route.chat_completions_url,
                headers=route.upstream_headers(),
                json=upstream_payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    yield "data: " + compact_json(
                        {
                            "error": {
                                "type": map_http_status_to_anthropic_error(response.status_code),
                                "message": decode_error_body(body),
                            }
                        }
                    ) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk.decode("utf-8", errors="replace")
    except asyncio.CancelledError:
        raise
    except httpx.TimeoutException:
        yield "data: " + compact_json(
            {"error": {"type": "api_error", "message": "Upstream stream timed out"}}
        ) + "\n\n"
        yield "data: [DONE]\n\n"
    except httpx.HTTPError as exc:
        yield "data: " + compact_json(
            {"error": {"type": "api_error", "message": f"Upstream stream failed: {exc}"}}
        ) + "\n\n"
        yield "data: [DONE]\n\n"


class StreamState:
    def __init__(self, tool_name_mapper: ToolNameMapper) -> None:
        self.tool_name_mapper = tool_name_mapper
        self.next_index = 0
        self.text_index: int | None = None
        self.text_open = False
        self.tools: dict[int, ToolStreamState] = {}

    async def consume_chunk(self, chunk: dict[str, Any]) -> AsyncIterator[str]:
        choice = first_stream_choice(chunk)
        if not choice:
            return

        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content:
            async for event in self.consume_text(str(content)):
                yield event

        for tool_call in delta.get("tool_calls") or []:
            if self.text_open:
                yield sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self.text_index},
                )
                self.text_open = False
            async for event in self.consume_tool_call_delta(tool_call):
                yield event

    async def consume_text(self, text: str) -> AsyncIterator[str]:
        if not self.text_open:
            self.text_index = self.next_index
            self.next_index += 1
            self.text_open = True
            yield sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.text_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        yield sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self.text_index,
                "delta": {"type": "text_delta", "text": text},
            },
        )

    async def consume_tool_call_delta(self, tool_call: dict[str, Any]) -> AsyncIterator[str]:
        openai_index = int(tool_call.get("index") or 0)
        tool_state = self.tools.get(openai_index)
        if not tool_state:
            tool_state = ToolStreamState(anthropic_index=self.next_index)
            self.next_index += 1
            self.tools[openai_index] = tool_state

        if tool_call.get("id"):
            tool_state.id = str(tool_call["id"])

        function = tool_call.get("function") or {}
        if function.get("name"):
            tool_state.name = self.tool_name_mapper.to_anthropic(function["name"])

        arguments_delta = function.get("arguments")
        if arguments_delta:
            tool_state.pending_arguments.append(str(arguments_delta))

        if tool_state.can_start and not tool_state.started:
            tool_state.started = True
            yield sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": tool_state.anthropic_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_state.id,
                        "name": tool_state.name,
                        "input": {},
                    },
                },
            )

        if tool_state.started and tool_state.pending_arguments:
            for partial_json in tool_state.pending_arguments:
                yield sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": tool_state.anthropic_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": partial_json,
                        },
                    },
                )
            tool_state.pending_arguments = []

    async def finish(self) -> AsyncIterator[str]:
        if self.text_open:
            yield sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": self.text_index},
            )
            self.text_open = False

        for tool_state in sorted(self.tools.values(), key=lambda item: item.anthropic_index):
            if not tool_state.started:
                tool_state.id = tool_state.id or f"call_{uuid.uuid4().hex}"
                tool_state.name = tool_state.name or "tool"
                tool_state.started = True
                yield sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": tool_state.anthropic_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_state.id,
                            "name": tool_state.name,
                            "input": {},
                        },
                    },
                )
                for partial_json in tool_state.pending_arguments:
                    yield sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tool_state.anthropic_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": partial_json,
                            },
                        },
                    )
                tool_state.pending_arguments = []

            yield sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": tool_state.anthropic_index},
            )


class ToolStreamState:
    def __init__(self, anthropic_index: int) -> None:
        self.anthropic_index = anthropic_index
        self.id = ""
        self.name = ""
        self.started = False
        self.pending_arguments: list[str] = []

    @property
    def can_start(self) -> bool:
        return bool(self.id and self.name)


async def read_json_object(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return body


def first_stream_choice(chunk: dict[str, Any]) -> dict[str, Any] | None:
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return None


def openai_error_response(response: httpx.Response, anthropic: bool = True) -> JSONResponse:
    error_type = map_http_status_to_anthropic_error(response.status_code)
    message = decode_error_body(response.content)
    if anthropic:
        return anthropic_error_response(response.status_code, message, error_type)
    return openai_proxy_error_response(response.status_code, message, error_type)


def anthropic_error_response(status_code: int, message: str, error_type: str = "api_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


def openai_proxy_error_response(status_code: int, message: str, error_type: str = "api_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"type": error_type, "message": message}},
    )


def map_http_status_to_anthropic_error(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 429:
        return "rate_limit_error"
    return "api_error"


def decode_error_body(body: bytes) -> str:
    if not body:
        return "Upstream API request failed"
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if parsed.get("message"):
                return str(parsed["message"])
            return compact_json(parsed)
    except json.JSONDecodeError:
        pass
    return body.decode("utf-8", errors="replace")


def as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    uvicorn.run("claude_code_proxy.server:app", host=settings.host, port=settings.port)
