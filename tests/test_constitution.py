from pathlib import Path
import json
import tempfile
import unittest

import yaml

from cai.constitution import (
    ConstitutionError,
    _reasoning_from_args,
    build_parser,
    compiler_response_format,
    compile_markdown,
    ruleset_from_json,
    ruleset_to_yaml,
    validate_markdown,
    validate_ruleset,
)
from cai.openrouter import COMPILER_MODEL, COMPILER_REASONING


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
            ROOT / "constitutions" / "strict.md",
            ROOT / "constitutions" / "balanced.md",
            ROOT / "constitutions" / "permissive.md",
            ROOT / "constitutions" / "playful.md",
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
        prompt = client.calls[0][0][1]["content"]

        self.assertEqual(rules[0]["category"], "privacy-consent")
        self.assertEqual(client.calls[0][1]["model"], COMPILER_MODEL)
        self.assertEqual(client.calls[0][1]["response_format"], compiler_response_format())
        self.assertEqual(client.calls[0][1]["reasoning"], COMPILER_REASONING)
        self.assertEqual(client.calls[0][1]["temperature"], 0.0)
        self.assertEqual(client.calls[0][1]["max_tokens"], 32000)
        self.assertIn("Crispness requirements", prompt)
        self.assertIn("observable test", prompt)
        self.assertIn("when the rule applies", prompt)
        self.assertIn("revision_needed=false", prompt)
        self.assertIn("top-level objective", prompt)
        self.assertIn("what to remove, what to preserve", prompt)
        self.assertIn("risk posture", prompt)
        self.assertIn("Strict constitutions", prompt)
        self.assertIn("Balanced constitutions", prompt)
        self.assertIn("Permissive constitutions", prompt)
        self.assertIn("source Markdown as authoritative", prompt)
        self.assertIn("routing labels", prompt)

    def test_compiler_response_format_uses_strict_json_schema(self) -> None:
        response_format = compiler_response_format()

        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "constitution_ruleset")  # type: ignore[index]
        self.assertEqual(response_format["json_schema"]["strict"], True)  # type: ignore[index]
        schema = response_format["json_schema"]["schema"]  # type: ignore[index]
        self.assertEqual(schema["additionalProperties"], False)  # type: ignore[index]
        rule_schema = schema["properties"]["rules"]["items"]  # type: ignore[index]
        self.assertEqual(rule_schema["required"], ["id", "category", "principle", "critic", "revision"])
        self.assertEqual(rule_schema["additionalProperties"], False)

    def test_compile_reasoning_defaults_do_not_block_token_budget_override(self) -> None:
        parser = build_parser()

        default_args = parser.parse_args(["compile", "constitution.md"])
        token_budget_args = parser.parse_args(
            ["compile", "constitution.md", "--reasoning-max-tokens", "4096", "--include-reasoning"]
        )

        self.assertEqual(_reasoning_from_args(default_args), {"effort": "high", "exclude": True})
        self.assertEqual(_reasoning_from_args(token_budget_args), {"max_tokens": 4096, "exclude": False})

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

    def test_validates_canonical_rulesets(self) -> None:
        paths = sorted((ROOT / "constitutions" / "compiled").glob("*.rules.yaml"))
        self.assertGreaterEqual(len(paths), 4)

        for path in paths:
            with self.subTest(path=path):
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
