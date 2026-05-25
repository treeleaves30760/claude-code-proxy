from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from claude_code_proxy.config import Settings
from claude_code_proxy.openai_compat import prepare_openai_compatible_request


class ConfigTests(unittest.TestCase):
    def test_yaml_routes_models_to_multiple_providers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                """
default_model: proxy-default
providers:
  openai:
    type: openai
    api_key: test-openai
  openrouter:
    type: openrouter
    api_key: test-openrouter
  ollama:
    type: ollama
    base_url: http://localhost:11434/v1
    require_api_key: false
models:
  proxy-default:
    provider: openai
    model: gpt-test
    aliases: [claude-sonnet-4-6]
  router:
    provider: openrouter
    model: openai/gpt-test
  local:
    provider: ollama
    model: gpt-oss:20b
public_models: [proxy-default, router, local]
""",
                encoding="utf-8",
            )

            settings = Settings(config_path=str(path))

            self.assertEqual(settings.resolve_route("claude-sonnet-4-6").upstream_model, "gpt-test")
            self.assertEqual(settings.resolve_route("router").provider.name, "openrouter")
            self.assertEqual(
                settings.resolve_route("local").chat_completions_url,
                "http://localhost:11434/v1/chat/completions",
            )
            self.assertEqual(settings.public_model_ids(), ["proxy-default", "router", "local", "claude-sonnet-4-6"])

    def test_yaml_supports_env_interpolation(self) -> None:
        previous = os.environ.get("TEST_PROXY_MODEL")
        os.environ["TEST_PROXY_MODEL"] = "openai/gpt-env"
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.yaml"
                path.write_text(
                    """
default_model: routed
providers:
  openrouter:
    type: openrouter
    api_key: test
models:
  routed:
    provider: openrouter
    model: ${TEST_PROXY_MODEL}
""",
                    encoding="utf-8",
                )
                settings = Settings(config_path=str(path))
                self.assertEqual(settings.resolve_route("routed").upstream_model, "openai/gpt-env")
        finally:
            if previous is None:
                os.environ.pop("TEST_PROXY_MODEL", None)
            else:
                os.environ["TEST_PROXY_MODEL"] = previous

    def test_openai_compatible_request_uses_route_model_and_token_field(self) -> None:
        settings = Settings(
            openai_api_endpoint="https://api.openai.com/v1",
            openai_model="gpt-test",
            openai_api_key="test",
            config_path="/path/that/does/not/exist.yaml",
        )
        route = settings.resolve_route("proxy-default")
        payload = prepare_openai_compatible_request(
            {
                "model": "proxy-default",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
            route,
        )

        self.assertEqual(payload["model"], "gpt-test")
        self.assertEqual(payload["max_completion_tokens"], 100)
        self.assertNotIn("max_tokens", payload)


if __name__ == "__main__":
    unittest.main()
