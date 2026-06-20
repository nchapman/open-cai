from pathlib import Path
import unittest

from cai.constitution import (
    ConstitutionError,
    _reasoning_from_args,
    build_parser,
    compile_markdown,
    validate_generated_markdown,
    validate_markdown,
)
from cai.openrouter import COMPILER_MODEL, COMPILER_REASONING


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((messages, kwargs))
        return self.responses.pop(0)


def sample_outline() -> str:
    return """# Local Guide

Protect private data while staying useful.

## Privacy and Consent

- Covers: requests for private information or tracking without consent.
- Distinguishes: public information and consent-based alternatives from privacy violations.
"""


def sample_guide() -> str:
    return """# Local Guide

Protect private data while staying useful. Be direct, calm, and privacy-preserving.

## Operating Guidance

### Privacy and Consent

The user asks for private information or tracking without consent.

**Practices:**

- Refuse to provide private data.
- Offer public or consent-based alternatives.

**Boundaries:**

- Do not reveal addresses, credentials, or tracking methods.

**Example:**

- User: Where does this actor live?
  - Good: I cannot provide a private address, but I can suggest official contact channels.
  - Bad: Here is the actor's home address.
"""


class ConstitutionTests(unittest.TestCase):
    def test_validates_source_markdown_without_parsing_principles(self) -> None:
        for path in [
            ROOT / "constitutions" / "protective.md",
            ROOT / "constitutions" / "balanced.md",
            ROOT / "constitutions" / "permissive.md",
        ]:
            with self.subTest(path=path):
                warnings = validate_markdown(path)

                self.assertEqual(warnings, [])

    def test_compiles_markdown_through_outline_to_response_guide(self) -> None:
        client = FakeClient(sample_outline(), sample_guide())

        guide = compile_markdown("# Local\n\n- It should protect private data.", client)  # type: ignore[arg-type]
        outline_prompt = client.calls[0][0][1]["content"]
        guide_prompt = client.calls[1][0][1]["content"]

        self.assertIn("# Local Guide", guide)
        self.assertIn("### Privacy and Consent", guide)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0][1]["model"], COMPILER_MODEL)
        self.assertNotIn("response_format", client.calls[0][1])
        self.assertNotIn("response_format", client.calls[1][1])
        self.assertEqual(client.calls[0][1]["reasoning"], COMPILER_REASONING)
        self.assertEqual(client.calls[0][1]["temperature"], 0.0)
        self.assertEqual(client.calls[0][1]["max_tokens"], 32000)
        self.assertIn("concise Markdown outline", outline_prompt)
        self.assertIn("Write the final response guide as concise Markdown", guide_prompt)
        self.assertIn("Preserve the source posture and boundaries exactly", guide_prompt)
        self.assertIn("Stay faithful to the source", client.calls[0][0][0]["content"])
        self.assertIn("Target 600 to 800 words", guide_prompt)
        self.assertIn("Every section must include a trigger sentence", guide_prompt)
        self.assertIn("same trigger, practice, and boundary", guide_prompt)
        self.assertIn("use 6 to 8 sections", guide_prompt)
        self.assertIn("mentioning it only in the introduction is not enough", guide_prompt)
        self.assertIn("self-contained enough to apply without guessing", guide_prompt)
        self.assertIn('Avoid vague bullets like "Use extra care"', guide_prompt)
        self.assertIn("Do not use placeholders", outline_prompt)
        self.assertIn("Do not generate examples", guide_prompt)
        self.assertIn("preserve only the behavioral lesson", guide_prompt)
        self.assertIn("Preserve permissions as permissions", guide_prompt)
        self.assertIn("Do not make a permission sound like a mandate", guide_prompt)
        self.assertIn("more legalistic, or more refusal-forward", guide_prompt)
        self.assertIn("Do not add named edge cases", guide_prompt)
        self.assertIn("Preserve the source's strength of language", guide_prompt)
        self.assertIn('Do not add "even if..." clauses', guide_prompt)
        self.assertIn("Preserve key source terms", guide_prompt)
        self.assertIn("Do not introduce substitute threshold terms", guide_prompt)
        self.assertIn('Use "Never" only when the source uses comparable force', guide_prompt)
        self.assertIn("mixed safe/unsafe requests", guide_prompt)
        self.assertIn("Do not wrap the response in a fenced code block", guide_prompt)
        self.assertIn('must be "#"', guide_prompt)

    def test_compile_reasoning_defaults_do_not_block_token_budget_override(self) -> None:
        parser = build_parser()

        default_args = parser.parse_args(["compile", "constitution.md"])
        token_budget_args = parser.parse_args(
            ["compile", "constitution.md", "--reasoning-max-tokens", "4096", "--include-reasoning"]
        )

        self.assertEqual(_reasoning_from_args(default_args), {"effort": "high", "exclude": True})
        self.assertEqual(_reasoning_from_args(token_budget_args), {"max_tokens": 4096, "exclude": False})

    def test_rejects_generated_markdown_without_title(self) -> None:
        with self.assertRaisesRegex(ConstitutionError, "no top-level title"):
            validate_generated_markdown("#\nNo title here.", "guide")

    def test_rejects_generated_markdown_that_does_not_start_with_title(self) -> None:
        with self.assertRaisesRegex(ConstitutionError, 'must start with "#"'):
            validate_generated_markdown(f"```markdown\n{sample_guide()}```", "guide")

    def test_generated_markdown_validation_does_not_police_content(self) -> None:
        text = sample_guide().replace("I cannot provide a private address", "The term [Taboo Word] means")

        self.assertIn("[Taboo Word]", validate_generated_markdown(text, "guide"))

if __name__ == "__main__":
    unittest.main()
