from __future__ import annotations

import unittest

from claude_code_proxy.config import Settings
from claude_code_proxy.transform import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
)


class TransformTests(unittest.TestCase):
    def test_request_maps_messages_tools_and_model(self) -> None:
        settings = Settings(
            openai_api_endpoint="https://api.openai.com/v1",
            openai_model="gpt-test",
            openai_api_key="test",
        )
        body = {
            "model": "claude-sonnet",
            "max_tokens": 100,
            "system": "You are terse.",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
            "tool_choice": {"type": "auto"},
        }

        payload = anthropic_to_openai_request(body, settings)

        self.assertEqual(payload["model"], "gpt-test")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "You are terse."})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "Hello"})
        self.assertEqual(payload["tools"][0]["function"]["name"], "Read")
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["max_completion_tokens"], 100)

    def test_request_maps_tool_use_and_tool_result(self) -> None:
        settings = Settings(
            openai_api_endpoint="https://api.openai.com/v1",
            openai_model="gpt-test",
            openai_api_key="test",
        )
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll inspect it."},
                        {
                            "type": "tool_use",
                            "id": "call_123",
                            "name": "Read",
                            "input": {"path": "README.md"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": [{"type": "text", "text": "contents"}],
                        }
                    ],
                },
            ]
        }

        payload = anthropic_to_openai_request(body, settings)

        assistant = payload["messages"][0]
        tool = payload["messages"][1]
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_123")
        self.assertEqual(assistant["tool_calls"][0]["function"]["arguments"], '{"path":"README.md"}')
        self.assertEqual(tool, {"role": "tool", "tool_call_id": "call_123", "content": "contents"})

    def test_response_maps_text_and_tool_calls(self) -> None:
        settings = Settings(
            openai_api_endpoint="https://api.openai.com/v1",
            openai_model="gpt-test",
            openai_api_key="test",
        )
        upstream = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Need a file.",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"path":"pyproject.toml"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        response = openai_to_anthropic_response(upstream, {"model": "claude-sonnet"}, settings)

        self.assertEqual(response["role"], "assistant")
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(response["usage"], {"input_tokens": 10, "output_tokens": 5})
        self.assertEqual(response["content"][0], {"type": "text", "text": "Need a file."})
        self.assertEqual(
            response["content"][1],
            {
                "type": "tool_use",
                "id": "call_abc",
                "name": "Read",
                "input": {"path": "pyproject.toml"},
            },
        )


if __name__ == "__main__":
    unittest.main()
