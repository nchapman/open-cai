from pathlib import Path
import os
import tempfile
import unittest

from cai.openrouter import (
    APPLY_RULES_MODEL,
    APPLY_RULES_REASONING,
    COMPILER_MODEL,
    COMPILER_REASONING,
    DEFAULT_CHAT_MODEL,
    OpenRouterClient,
    OpenRouterError,
    OpenRouterSettings,
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

    def test_model_roles_are_code_defaults_not_env_settings(self) -> None:
        self.assertEqual(COMPILER_MODEL, "deepseek/deepseek-v4-pro")
        self.assertEqual(APPLY_RULES_MODEL, "deepseek/deepseek-v4-flash")
        self.assertEqual(DEFAULT_CHAT_MODEL, APPLY_RULES_MODEL)
        self.assertEqual(COMPILER_REASONING, {"effort": "high", "exclude": True})
        self.assertEqual(APPLY_RULES_REASONING, {"effort": "medium", "exclude": True})

    def test_chat_includes_reasoning_when_provided(self) -> None:
        client = FakeOpenRouterClient()

        content = client.chat(
            [{"role": "user", "content": "hello"}],
            reasoning={"effort": "low", "exclude": True},
        )

        self.assertEqual(content, "ok")
        self.assertEqual(client.payloads[0][1]["model"], DEFAULT_CHAT_MODEL)
        self.assertEqual(client.payloads[0][1]["reasoning"], {"effort": "low", "exclude": True})

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
