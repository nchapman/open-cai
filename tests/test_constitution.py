from pathlib import Path
import json
import unittest

from cai.constitution import (
    ConstitutionError,
    _reasoning_from_args,
    build_parser,
    compiler_response_format,
    compile_markdown,
    guide_from_json,
    guide_to_markdown,
    validate_markdown,
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


def sample_guide_payload() -> dict[str, object]:
    return {
        "title": "Local Guide",
        "overview": "Protect private data while staying useful.",
        "response_posture": "Be direct, calm, and privacy-preserving.",
        "sections": [
            {
                "id": "privacy-consent",
                "title": "Privacy and consent",
                "when_to_apply": "The user asks for private information or tracking without consent.",
                "do": ["Refuse to provide private data.", "Offer public or consent-based alternatives."],
                "avoid": ["Do not reveal addresses, credentials, or tracking methods."],
                "examples": [
                    {
                        "user": "Where does this actor live?",
                        "good": "I cannot provide a private address, but I can suggest official contact channels.",
                        "bad": "Here is the actor's home address.",
                        "notes": "The good answer protects privacy while preserving a benign alternative.",
                    }
                ],
            }
        ],
    }


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

    def test_compiles_markdown_through_model_json_to_response_guide(self) -> None:
        client = FakeClient(json.dumps(sample_guide_payload()))

        guide = compile_markdown("# Local\n\n- It should protect private data.", client)  # type: ignore[arg-type]
        prompt = client.calls[0][0][1]["content"]

        self.assertEqual(guide["title"], "Local Guide")
        self.assertEqual(client.calls[0][1]["model"], COMPILER_MODEL)
        self.assertEqual(client.calls[0][1]["response_format"], compiler_response_format())
        self.assertEqual(client.calls[0][1]["reasoning"], COMPILER_REASONING)
        self.assertEqual(client.calls[0][1]["temperature"], 0.0)
        self.assertEqual(client.calls[0][1]["max_tokens"], 32000)
        self.assertIn("response guide", prompt)
        self.assertIn("complete operational guide", prompt)
        self.assertIn("not isolated random rules", prompt)
        self.assertIn("human reviewer", prompt)

    def test_compiler_response_format_uses_strict_json_schema(self) -> None:
        response_format = compiler_response_format()

        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "response_guide")  # type: ignore[index]
        self.assertEqual(response_format["json_schema"]["strict"], True)  # type: ignore[index]
        schema = response_format["json_schema"]["schema"]  # type: ignore[index]
        self.assertEqual(schema["additionalProperties"], False)  # type: ignore[index]
        section_schema = schema["properties"]["sections"]["items"]  # type: ignore[index]
        self.assertEqual(section_schema["required"], ["id", "title", "when_to_apply", "do", "avoid", "examples"])

    def test_serializes_guide_to_reviewable_markdown(self) -> None:
        text = guide_to_markdown(sample_guide_payload())

        self.assertIn("# Local Guide", text)
        self.assertIn("## Response Posture", text)
        self.assertIn("### privacy-consent: Privacy and consent", text)
        self.assertIn("**When to apply:**", text)
        self.assertIn("- Refuse to provide private data.", text)
        self.assertIn("- User: Where does this actor live?", text)

    def test_compile_reasoning_defaults_do_not_block_token_budget_override(self) -> None:
        parser = build_parser()

        default_args = parser.parse_args(["compile", "constitution.md"])
        token_budget_args = parser.parse_args(
            ["compile", "constitution.md", "--reasoning-max-tokens", "4096", "--include-reasoning"]
        )

        self.assertEqual(_reasoning_from_args(default_args), {"effort": "high", "exclude": True})
        self.assertEqual(_reasoning_from_args(token_budget_args), {"max_tokens": 4096, "exclude": False})

    def test_rejects_invalid_model_json(self) -> None:
        with self.assertRaisesRegex(ConstitutionError, "missing fields"):
            guide_from_json('{"sections": []}')

    def test_rejects_duplicate_section_ids(self) -> None:
        payload = sample_guide_payload()
        payload["sections"] = [payload["sections"][0], payload["sections"][0]]  # type: ignore[index]

        with self.assertRaisesRegex(ConstitutionError, "duplicate section id"):
            guide_from_json(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
