from pathlib import Path
import os
import socket
import tempfile
import unittest
from unittest import mock

from cai.openrouter import (
    COMPILER_MODEL,
    COMPILER_REASONING,
    DEFAULT_CHAT_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    GUIDE_APPLICATION_MODEL,
    GUIDE_APPLICATION_REASONING,
    OpenRouterClient,
    OpenRouterError,
    OpenRouterSettings,
    _reasoning_from_args,
    build_parser,
    settings_from_env,
)


class FakeOpenRouterClient(OpenRouterClient):
    def __init__(self) -> None:
        super().__init__(OpenRouterSettings(api_key="test-key"))
        self.payloads = []

    def _post(self, path, payload):  # type: ignore[no-untyped-def]
        self.payloads.append((path, payload))
        return {"choices": [{"message": {"content": "ok"}}]}


class OpenRouterTests(unittest.TestCase):
    def test_loads_settings_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "OPENROUTER_API_KEY=test-key",
                        "APP_TITLE=cai-test",
                    ]
                ),
                encoding="utf-8",
            )

            env_keys = ["OPENROUTER_API_KEY", "APP_TITLE"]
            old_values = {key: os.environ.pop(key, None) for key in env_keys}
            try:
                settings = settings_from_env(path)
            finally:
                for key in env_keys:
                    os.environ.pop(key, None)
                    if old_values[key] is not None:
                        os.environ[key] = old_values[key] or ""

            self.assertEqual(settings.api_key, "test-key")
            self.assertEqual(settings.app_title, "cai-test")

    def test_local_base_url_does_not_require_openrouter_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_key = os.environ.pop("OPENROUTER_API_KEY", None)
            try:
                os.environ["OPENROUTER_API_KEY"] = "real-openrouter-key"
                settings = settings_from_env(Path(tmp) / ".env", base_url="http://127.0.0.1:8080/v1")
            finally:
                os.environ.pop("OPENROUTER_API_KEY", None)
                if old_key is not None:
                    os.environ["OPENROUTER_API_KEY"] = old_key

        self.assertEqual(settings.base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(settings.api_key, "local-token")

    def test_request_timeout_is_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")

            settings = settings_from_env(path, request_timeout=42)

        self.assertEqual(settings.request_timeout, 42)

    def test_model_roles_are_code_defaults_not_env_settings(self) -> None:
        self.assertEqual(COMPILER_MODEL, "deepseek/deepseek-v4-pro")
        self.assertEqual(GUIDE_APPLICATION_MODEL, "deepseek/deepseek-v3.2")
        self.assertEqual(DEFAULT_CHAT_MODEL, GUIDE_APPLICATION_MODEL)
        self.assertEqual(COMPILER_REASONING, {"effort": "high", "exclude": True})
        self.assertEqual(GUIDE_APPLICATION_REASONING, {"effort": "none", "exclude": True})

    def test_chat_includes_reasoning_when_provided(self) -> None:
        client = FakeOpenRouterClient()

        content = client.chat(
            [{"role": "user", "content": "hello"}],
            reasoning={"effort": "low", "exclude": True},
        )

        self.assertEqual(content, "ok")
        self.assertEqual(client.payloads[0][1]["model"], DEFAULT_CHAT_MODEL)
        self.assertEqual(client.payloads[0][1]["reasoning"], {"effort": "low", "exclude": True})

    def test_chat_uses_configured_base_url(self) -> None:
        client = FakeOpenRouterClient()
        client.settings = OpenRouterSettings(api_key="local-token", base_url="http://127.0.0.1:8080/v1")

        client.chat([{"role": "user", "content": "hello"}], model="local-model")

        self.assertEqual(client.payloads[0][1]["model"], "local-model")

    def test_post_wraps_socket_timeout(self) -> None:
        client = OpenRouterClient(OpenRouterSettings(api_key="test-key", request_timeout=0.01))

        with mock.patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            with self.assertRaisesRegex(OpenRouterError, "timed out"):
                client._post("/chat/completions", {"model": DEFAULT_CHAT_MODEL})

    def test_chat_parser_request_timeout_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["chat", "hello"])

        self.assertEqual(args.request_timeout, DEFAULT_REQUEST_TIMEOUT)

    def test_reasoning_effort_none_is_sent_to_openrouter_but_not_local(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["chat", "hello", "--reasoning-effort", "none"])

        self.assertEqual(_reasoning_from_args(args, base_url=None), {"effort": "none", "exclude": True})
        self.assertIsNone(_reasoning_from_args(args, base_url="http://127.0.0.1:8080/v1"))

    def test_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_key = os.environ.pop("OPENROUTER_API_KEY", None)
            try:
                with self.assertRaisesRegex(OpenRouterError, "OPENROUTER_API_KEY"):
                    settings_from_env(Path(tmp) / ".env")
            finally:
                if old_key is not None:
                    os.environ["OPENROUTER_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
