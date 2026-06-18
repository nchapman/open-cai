"""Compile freeform constitution Markdown into response guides."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping

from cai.openrouter import (
    COMPILER_MODEL,
    COMPILER_REASONING,
    OpenRouterClient,
    OpenRouterError,
    settings_from_env,
)


GUIDE_SECTION_FIELDS = ("id", "title", "when_to_apply", "do", "avoid", "examples")
GUIDE_EXAMPLE_FIELDS = ("user", "good", "bad", "notes")
GUIDE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
HEADING = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
COMPILER_PROMPT_VERSION = "v8-response-guide"

GUIDE_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "overview": {"type": "string", "minLength": 1},
        "response_posture": {"type": "string", "minLength": 1},
        "sections": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]*$"},
                    "title": {"type": "string", "minLength": 1},
                    "when_to_apply": {"type": "string", "minLength": 1},
                    "do": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                    "avoid": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                    "examples": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "user": {"type": "string", "minLength": 1},
                                "good": {"type": "string", "minLength": 1},
                                "bad": {"type": "string", "minLength": 1},
                                "notes": {"type": "string", "minLength": 1},
                            },
                            "required": list(GUIDE_EXAMPLE_FIELDS),
                            "additionalProperties": False,
                        },
                    },
                },
                "required": list(GUIDE_SECTION_FIELDS),
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "overview", "response_posture", "sections"],
    "additionalProperties": False,
}


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
) -> dict[str, object]:
    """Compile a complete Markdown constitution into structured response-guide data."""

    if not markdown.strip():
        raise ConstitutionError("constitution Markdown is empty")

    response = client.chat(
        _compiler_messages(markdown),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=compiler_response_format(),
        reasoning=reasoning,
    )
    return guide_from_json(response)


def guide_from_json(text: str) -> dict[str, object]:
    """Parse and validate compiler JSON output."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConstitutionError(f"compiler returned invalid JSON: {exc}") from exc

    return validate_guide_data(payload)


def validate_guide_data(payload: object) -> dict[str, object]:
    """Validate structured response-guide data."""

    if not isinstance(payload, Mapping):
        raise ConstitutionError("guide JSON must be an object")

    keys = set(payload)
    expected = {"title", "overview", "response_posture", "sections"}
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise ConstitutionError(f"guide JSON missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ConstitutionError(f"guide JSON unexpected fields: {', '.join(sorted(extra))}")

    guide: dict[str, object] = {
        "title": _non_empty_string(payload["title"], "title"),
        "overview": _non_empty_string(payload["overview"], "overview"),
        "response_posture": _non_empty_string(payload["response_posture"], "response_posture"),
    }

    raw_sections = payload["sections"]
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ConstitutionError("guide sections must be a non-empty array")
    sections = [_validate_section(section, index) for index, section in enumerate(raw_sections, start=1)]
    _ensure_unique("section id", (str(section["id"]) for section in sections))
    guide["sections"] = sections
    return guide


def guide_to_markdown(guide: Mapping[str, object]) -> str:
    """Serialize guide data as reviewable Markdown."""

    data = validate_guide_data(guide)
    lines = [
        f"# {data['title']}",
        "",
        "<!-- Generated response guide. Edit the source constitution Markdown, then recompile. -->",
        "",
        "## Overview",
        "",
        str(data["overview"]),
        "",
        "## Response Posture",
        "",
        str(data["response_posture"]),
        "",
        "## Guide Sections",
        "",
    ]

    for section in data["sections"]:  # type: ignore[index]
        section_map = section if isinstance(section, Mapping) else {}
        lines.extend(
            [
                f"### {section_map['id']}: {section_map['title']}",
                "",
                "**When to apply:**",
                "",
                str(section_map["when_to_apply"]),
                "",
                "**Do:**",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in section_map["do"])  # type: ignore[index]
        lines.extend(["", "**Avoid:**", ""])
        lines.extend(f"- {item}" for item in section_map["avoid"])  # type: ignore[index]
        lines.extend(["", "**Examples:**", ""])
        for example in section_map["examples"]:  # type: ignore[index]
            example_map = example if isinstance(example, Mapping) else {}
            lines.extend(
                [
                    f"- User: {example_map['user']}",
                    f"  - Good: {example_map['good']}",
                    f"  - Bad: {example_map['bad']}",
                    f"  - Notes: {example_map['notes']}",
                ]
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def validate_markdown(path: str | Path) -> list[str]:
    """Validate source or guide Markdown."""

    content = Path(path).read_text(encoding="utf-8")
    if not content.strip():
        raise ConstitutionError("Markdown is empty")
    warnings: list[str] = []
    if not _find_title(content):
        warnings.append("Markdown has no top-level title")
    return warnings


def compiler_response_format() -> dict[str, object]:
    """Return OpenRouter's strict JSON Schema response format for compiler calls."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "response_guide",
            "strict": True,
            "schema": GUIDE_RESPONSE_SCHEMA,
        },
    }


def _compiler_messages(markdown: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a rigorous source-faithful response-guide compiler. "
                "You turn freeform constitution documents into practical Markdown guides a human reviewer could use to judge and rewrite assistant responses. "
                "Return only valid JSON matching the requested schema."
            ),
        },
        {
            "role": "user",
            "content": f"""Compiler prompt version: {COMPILER_PROMPT_VERSION}

Transform the full Markdown constitution below into a response guide.

Objective:
- Produce a complete operational guide for data generation and review.
- The guide must cover the constitution's critical alignment areas as a whole, not isolated random rules.
- Preserve the source constitution's actual posture. Do not make a strict constitution permissive, a permissive constitution strict, or a playful constitution generic.
- Make the guide useful to a human reviewer: crisp applicability criteria, concrete do/avoid instructions, and examples.

Return a JSON object with this shape:

{{
  "title": "Short guide title",
  "overview": "The constitution's purpose and boundaries.",
  "response_posture": "How the assistant should generally behave under this constitution.",
  "sections": [
    {{
      "id": "stable-lower-kebab-id",
      "title": "Section title",
      "when_to_apply": "Observable conditions for when this section matters.",
      "do": ["Concrete response behavior to follow."],
      "avoid": ["Concrete response behavior to avoid."],
      "examples": [
        {{
          "user": "A representative user request.",
          "good": "A concise response that follows the guide.",
          "bad": "A concise response that violates the guide.",
          "notes": "Why the good response is better."
        }}
      ]
    }}
  ]
}}

Requirements:
- Use 4 to 10 guide sections unless the source clearly requires fewer or more.
- Every section id must be stable lower-kebab-case.
- Each section must express a distinct, reusable alignment concern.
- Sections must be broad enough to apply across many prompts but specific enough to support concrete judgments.
- Include examples that test the constitution's differentiating behavior, especially strict vs balanced vs permissive vs playful posture when relevant.
- Do not add generic safety obligations unless they are grounded in the source.
- Do not produce critique/revision prompt snippets. Produce a guide, not a random-rule set.
- Return JSON only.

Markdown constitution:
```markdown
{markdown}
```""",
        },
    ]


def _validate_section(raw_section: object, index: int) -> dict[str, object]:
    if not isinstance(raw_section, Mapping):
        raise ConstitutionError(f"section {index}: must be an object")

    keys = set(raw_section)
    expected = set(GUIDE_SECTION_FIELDS)
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise ConstitutionError(f"section {index}: missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ConstitutionError(f"section {index}: unexpected fields: {', '.join(sorted(extra))}")

    section = {
        "id": _non_empty_string(raw_section["id"], f"section {index} id"),
        "title": _non_empty_string(raw_section["title"], f"section {index} title"),
        "when_to_apply": _non_empty_string(raw_section["when_to_apply"], f"section {index} when_to_apply"),
        "do": _string_list(raw_section["do"], f"section {index} do"),
        "avoid": _string_list(raw_section["avoid"], f"section {index} avoid"),
    }
    if not GUIDE_ID.match(str(section["id"])):
        raise ConstitutionError(f"section {index}: id must be lower-kebab-case")

    raw_examples = raw_section["examples"]
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ConstitutionError(f"section {index}: examples must be a non-empty array")
    section["examples"] = [_validate_example(example, index, example_index) for example_index, example in enumerate(raw_examples, start=1)]
    return section


def _validate_example(raw_example: object, section_index: int, example_index: int) -> dict[str, str]:
    if not isinstance(raw_example, Mapping):
        raise ConstitutionError(f"section {section_index} example {example_index}: must be an object")

    keys = set(raw_example)
    expected = set(GUIDE_EXAMPLE_FIELDS)
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise ConstitutionError(f"section {section_index} example {example_index}: missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ConstitutionError(f"section {section_index} example {example_index}: unexpected fields: {', '.join(sorted(extra))}")

    return {
        field: _non_empty_string(raw_example[field], f"section {section_index} example {example_index} {field}")
        for field in GUIDE_EXAMPLE_FIELDS
    }


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConstitutionError(f"{label} must be a non-empty string")
    return value.strip()


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConstitutionError(f"{label} must be a non-empty array")
    return [_non_empty_string(item, f"{label} item {index}") for index, item in enumerate(value, start=1)]


def _find_title(markdown: str) -> str | None:
    match = HEADING.search(markdown)
    if not match:
        return None
    return match.group("title").strip()


def _ensure_unique(label: str, values: Iterable[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ConstitutionError(f"duplicate {label}: {', '.join(sorted(duplicates))}")


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

    text = guide_to_markdown(guide)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
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
