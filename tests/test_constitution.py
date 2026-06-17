from pathlib import Path
import json
import tempfile
import unittest

import yaml

from cai.constitution import (
    ConstitutionError,
    compile_markdown,
    ruleset_from_json,
    ruleset_to_yaml,
    validate_markdown,
    validate_ruleset,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((messages, kwargs))
        return self.response


class ConstitutionTests(unittest.TestCase):
    def test_validates_source_markdown_without_parsing_principles(self) -> None:
        for path in [
            ROOT / "constitutions" / "core.md",
            ROOT / "constitutions" / "grok.md",
        ]:
            with self.subTest(path=path):
                warnings = validate_markdown(path)

                self.assertEqual(warnings, [])

    def test_compiles_markdown_through_model_json(self) -> None:
        response = json.dumps(
            {
                "rules": [
                    {
                        "id": "principle-01-privacy-consent",
                        "category": "privacy-consent",
                        "principle": "It should protect private data and avoid helping track people without consent.",
                        "critic": "Identify whether the response violates privacy or consent.",
                        "revision": "Rewrite the response to protect privacy and consent.",
                    }
                ]
            }
        )
        client = FakeClient(response)

        rules = compile_markdown("# Local\n\n- It should protect private data.", client)  # type: ignore[arg-type]

        self.assertEqual(rules[0]["category"], "privacy-consent")
        self.assertEqual(client.calls[0][1]["response_format"], {"type": "json_object"})
        self.assertEqual(client.calls[0][1]["temperature"], 0.0)

    def test_serializes_ruleset_to_reviewable_yaml(self) -> None:
        rules = [
            {
                "id": "principle-01-privacy-consent",
                "category": "privacy-consent",
                "principle": "It should protect private data.",
                "critic": "Identify whether the response violates privacy.",
                "revision": "Rewrite the response to protect privacy.",
            }
        ]

        text = ruleset_to_yaml(rules)
        parsed = yaml.safe_load(text)

        self.assertEqual(sorted(parsed[0]), ["category", "critic", "id", "principle", "revision"])
        self.assertIn("critic: |", text)

    def test_validates_ruleset_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.yaml"
            path.write_text(
                ruleset_to_yaml(
                    [
                        {
                            "id": "principle-01-privacy-consent",
                            "category": "privacy-consent",
                            "principle": "It should protect private data.",
                            "critic": "Identify whether the response violates privacy.",
                            "revision": "Rewrite the response to protect privacy.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            warnings = validate_ruleset(path)

        self.assertEqual(warnings, [])

    def test_rejects_invalid_model_json(self) -> None:
        with self.assertRaisesRegex(ConstitutionError, "missing fields"):
            ruleset_from_json('{"rules": [{"id": "principle-01"}]}')

    def test_rejects_unknown_category(self) -> None:
        payload = {
            "rules": [
                {
                    "id": "principle-01-custom",
                    "category": "custom",
                    "principle": "It should do something.",
                    "critic": "Identify issues.",
                    "revision": "Revise the answer.",
                }
            ]
        }

        with self.assertRaisesRegex(ConstitutionError, "unknown category"):
            ruleset_from_json(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
