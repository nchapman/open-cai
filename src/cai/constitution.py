"""Compile freeform constitution Markdown into response guides."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Mapping

from cai.openrouter import (
    COMPILER_MODEL,
    COMPILER_REASONING,
    OpenRouterClient,
    OpenRouterError,
    settings_from_env,
)


HEADING = re.compile(r"^#[ \t]+(?P<title>\S.*?)\s*$", re.MULTILINE)
COMPILER_PROMPT_VERSION = "v36-no-examples-no-substitute-terms"


class ConstitutionError(ValueError):
    """Raised when a constitution or response guide cannot be compiled or validated."""


def compile_markdown(
    markdown: str,
    client: OpenRouterClient,
    *,
    model: str | None = COMPILER_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 32000,
    reasoning: Mapping[str, object] | None = COMPILER_REASONING,
) -> str:
    """Compile a complete Markdown constitution into a response-guide Markdown document."""

    if not markdown.strip():
        raise ConstitutionError("constitution Markdown is empty")

    outline = client.chat(
        _outline_messages(markdown),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )
    outline = validate_generated_markdown(outline, "outline")

    guide = client.chat(
        _guide_messages(markdown, outline),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )
    return validate_generated_markdown(guide, "guide")


def validate_markdown(path: str | Path) -> list[str]:
    """Validate source or guide Markdown."""

    content = Path(path).read_text(encoding="utf-8")
    if not content.strip():
        raise ConstitutionError("Markdown is empty")
    warnings: list[str] = []
    if not _find_title(content):
        warnings.append("Markdown has no top-level title")
    return warnings


def validate_generated_markdown(markdown: str, label: str = "generated Markdown") -> str:
    """Catch unusable compiler output without imposing content constraints."""

    text = _non_empty_string(markdown, label)
    if not text.startswith("#"):
        raise ConstitutionError(f'{label} must start with "#"')
    if not _find_title(text):
        raise ConstitutionError(f"{label} has no top-level title")
    return text.rstrip() + "\n"


def _compiler_system_message() -> str:
    return (
        "You compile constitution Markdown into concise human-readable response guides. "
        "Stay faithful to the source. Do not add your own policy preferences. "
        "Preserve permissions as permissions and boundaries as boundaries. "
        "Write clear, concrete Markdown only."
    )


def _outline_messages(markdown: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": _compiler_system_message(),
        },
        {
            "role": "user",
            "content": f"""Compiler prompt version: {COMPILER_PROMPT_VERSION}

Create a concise Markdown outline for a response guide based on the constitution below.

The outline is a planning artifact for the final guide. It should decide the document structure, not write the full guide.

Rules:
- Preserve the source posture and boundaries exactly.
- Treat the opening paragraph as source material, not background. Preserve its default posture and response style.
- Every source bullet or materially distinct sentence must have an obvious home in the outline.
- Do not add safety, political, moral, or cultural preferences that are not in the source.
- Do not add named edge cases, exceptions, or interpretations that are not in the source.
- Preserve permissions as permissions. If the source allows something, do not turn that into a requirement to answer with a specific depth, format, or style.
- Do not make the guide stricter, looser, more moralizing, more legalistic, or more refusal-forward than the constitution.
- Preserve the source's strength of language. Avoid intensifiers like "fully", "strict", "non-negotiable", "always", or "never" unless the source uses comparable force.
- Restate source boundaries only. Do not add "even if..." clauses, special cases, or non-exceptions that are not in the source.
- Preserve key source terms for conditions and thresholds. Do not replace the source's standard with a broader, narrower, or more legalistic standard.
- Do not introduce substitute threshold terms such as public, anonymized, legal, ethical, authorized, or professional unless those terms appear in the source.
- Merge only when source items produce the same trigger, practice, and boundary. Keep separate items that govern different domains, different user intent, different refusal style, or different allowed alternatives.
- Use 5 to 8 sections as needed to preserve distinct behavior.
- Name the critical behavioral decisions each section must cover.
- For each section, name the source permissions, boundaries, and failure modes the final guide must preserve.
- Do not combine source items merely because they share a broad theme; combine only when the resulting response behavior would be the same.
- Include omissions to avoid when they would help prevent source drift.
- Do not include examples.
- Do not use placeholders, bracketed stand-ins, ellipses, "TBD", "etc.", or template variables.
- Return only the Markdown document itself. The first character of your response must be "#".
- Do not wrap the response in a fenced code block.

Markdown shape:
# Short guide title

One compact paragraph describing the guide's central posture.

## Section Title

- Covers: concrete behavior this section must govern.
- Distinguishes: boundary or permission that keeps this section from collapsing into generic safety language.
- Must preserve: source-specific permission, boundary, or failure mode.

Markdown constitution:
```markdown
{markdown}
```""",
        },
    ]


def _guide_messages(markdown: str, outline: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": _compiler_system_message(),
        },
        {
            "role": "user",
            "content": f"""Compiler prompt version: {COMPILER_PROMPT_VERSION}

Write the final response guide as concise Markdown.

Use the outline to preserve structure and differentiation. Write the guide as a polished operating document for a human alignment team.

Rules:
- Preserve the source posture and boundaries exactly.
- Treat the constitution's opening paragraph as source material. Preserve its default posture and refusal style in the guide.
- Every source bullet or materially distinct sentence must be represented visibly in the guide.
- Do not add safety, political, moral, or cultural preferences that are not in the constitution.
- Do not add named edge cases, exceptions, or interpretations that are not in the constitution.
- Preserve permissions as permissions. If the source allows something, do not turn that into a requirement to answer with a specific depth, format, or style.
- Phrase permissions, obligations, and boundaries according to the constitution. Do not make a permission sound like a mandate, and do not make a boundary stricter or looser than the source.
- Do not make the guide stricter, looser, more moralizing, more legalistic, or more refusal-forward than the constitution.
- Preserve the source's strength of language. Avoid intensifiers like "fully", "strict", "non-negotiable", "always", or "never" unless the source uses comparable force.
- Restate source boundaries only. Do not add "even if..." clauses, special cases, or non-exceptions that are not in the constitution.
- Preserve key source terms for conditions and thresholds. Do not replace the source's standard with a broader, narrower, or more legalistic standard.
- Do not introduce substitute threshold terms such as public, anonymized, legal, ethical, authorized, or professional unless those terms appear in the constitution.
- Use compact, behavior-first prose.
- Target 600 to 800 words. Shorter is acceptable only when the source is genuinely simple.
- Keep materially different rules separate. Collapse sections from the outline only when they have the same trigger, practice, and boundary.
- Do not combine source items merely because they share a broad theme; combine only when the resulting response behavior would be the same.
- If the constitution has many distinct bullets, use 6 to 8 sections. Do not move a distinct source bullet only into the introduction.
- Every materially distinct source bullet must map to at least one body section under "Operating Guidance"; mentioning it only in the introduction is not enough.
- If the source discusses mixed safe/unsafe requests, blanket refusals, safe alternatives, or refusal style, represent that behavior in its own body section.
- Every section must include a trigger sentence, a Practices list, and a Boundaries list.
- Use 1 or 2 practice bullets and 1 or 2 boundary bullets per section.
- Make every practice and boundary bullet self-contained enough to apply without guessing.
- Avoid vague bullets like "Use extra care", "Be helpful", or "Preserve safety".
- Boundary bullets should usually start with "Do not" or "Avoid" when they describe a prohibition. Use "Never" only when the source uses comparable force.
- Do not generate examples. If the constitution contains explicit examples, preserve only the behavioral lesson they teach; do not copy or invent examples in the guide.
- Before returning, silently check that each outline section has actionable practices and concrete boundaries.
- Return only the Markdown document itself. The first character of your response must be "#".
- Do not wrap the response in a fenced code block.

Markdown shape:
# Short guide title

One or two compact paragraphs stating the guide's purpose, scope, default response posture, and boundaries. Do not label this paragraph "Objective".

## Operating Guidance

### Human-readable section title

When this applies, in one compact sentence.

**Practices:**

- Concrete response practice.

**Boundaries:**

- Do not cross this concrete boundary.

Markdown outline:
```markdown
{outline}
```

Markdown constitution:
```markdown
{markdown}
```""",
        },
    ]


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConstitutionError(f"{label} must be a non-empty string")
    return value.strip()


def _find_title(markdown: str) -> str | None:
    match = HEADING.search(markdown)
    if not match:
        return None
    return match.group("title").strip()


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        warnings = validate_markdown(args.path)
    except ConstitutionError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1

    print(f"valid: {args.path}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    try:
        markdown = args.path.read_text(encoding="utf-8")
        client = OpenRouterClient(settings_from_env(args.env))
        guide = compile_markdown(
            markdown,
            client,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning=_reasoning_from_args(args),
        )
    except (ConstitutionError, OpenRouterError) as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(guide, encoding="utf-8")
    else:
        print(guide, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Constitution helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate source or response-guide Markdown")
    validate.add_argument("path", type=Path)
    validate.set_defaults(func=_cmd_validate)

    compile_parser = subparsers.add_parser("compile", help="compile constitution Markdown to response-guide Markdown")
    compile_parser.add_argument("path", type=Path)
    compile_parser.add_argument("-o", "--output", type=Path)
    compile_parser.add_argument("--env", type=Path, default=Path(".env"))
    compile_parser.add_argument("--model", default=COMPILER_MODEL)
    compile_parser.add_argument("--temperature", type=float, default=0.0)
    compile_parser.add_argument("--max-tokens", type=int, default=32000)
    reasoning = compile_parser.add_mutually_exclusive_group()
    reasoning.add_argument("--reasoning-effort", choices=["xhigh", "high", "medium", "low", "minimal", "none"])
    reasoning.add_argument("--reasoning-max-tokens", type=int)
    compile_parser.add_argument("--include-reasoning", action="store_true")
    compile_parser.set_defaults(func=_cmd_compile)
    return parser


def _reasoning_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    if args.reasoning_effort and args.reasoning_max_tokens is not None:
        raise ConstitutionError("use either --reasoning-effort or --reasoning-max-tokens, not both")

    reasoning: dict[str, object] = {"exclude": not args.include_reasoning}
    if args.reasoning_max_tokens is not None:
        reasoning["max_tokens"] = args.reasoning_max_tokens
    else:
        reasoning["effort"] = args.reasoning_effort or COMPILER_REASONING["effort"]
    return reasoning


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
