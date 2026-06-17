from pathlib import Path
import os
import tempfile
import unittest

from cai.openrouter import OpenRouterError, settings_from_env


class OpenRouterTests(unittest.TestCase):
    def test_loads_settings_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "OPENROUTER_API_KEY=test-key",
                        "OPENROUTER_MODEL=mistralai/mistral-7b-instruct",
                        "APP_TITLE=cai-test",
                    ]
                ),
                encoding="utf-8",
            )

            old_values = {key: os.environ.pop(key, None) for key in ["OPENROUTER_API_KEY", "OPENROUTER_MODEL", "APP_TITLE"]}
            try:
                settings = settings_from_env(path)
            finally:
                for key in ["OPENROUTER_API_KEY", "OPENROUTER_MODEL", "APP_TITLE"]:
                    os.environ.pop(key, None)
                    if old_values[key] is not None:
                        os.environ[key] = old_values[key] or ""

            self.assertEqual(settings.api_key, "test-key")
            self.assertEqual(settings.model, "mistralai/mistral-7b-instruct")
            self.assertEqual(settings.app_title, "cai-test")

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
