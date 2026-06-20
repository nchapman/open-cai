"""Compile freeform constitution Markdown into response guides."""

from __future__ import annotations

import argparse
import json
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


GUIDE_SECTION_FIELDS = ("title", "when_to_apply", "do", "avoid", "examples")
GUIDE_EXAMPLE_FIELDS = ("user", "good", "bad")
HEADING = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
COMPILER_PROMPT_VERSION = "v18-preserve-permissions"
PLACEHOLDER_TEXT = re.compile(r"\[[^\]]+\]|\.{3}|…")
BOUNDARY_PREFIX = re.compile(r"^(Do not|Avoid|Never)\b")

GUIDE_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1, "maxLength": 90},
        "overview": {"type": "string", "minLength": 1, "maxLength": 380},
        "response_posture": {"type": "string", "minLength": 1, "maxLength": 360},
        "sections": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 90},
                    "when_to_apply": {"type": "string", "minLength": 1, "maxLength": 220},
                    "do": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 2,
                        "items": {"type": "string", "minLength": 1, "maxLength": 150},
                    },
                    "avoid": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 2,
                        "items": {"type": "string", "minLength": 1, "maxLength": 150},
                    },
                    "examples": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "user": {"type": "string", "minLength": 1, "maxLength": 110},
                                "good": {"type": "string", "minLength": 1, "maxLength": 140},
                                "bad": {"type": "string", "minLength": 1, "maxLength": 140},
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
        "title": _clean_generated_string(payload["title"], "title"),
        "overview": _clean_generated_string(payload["overview"], "overview"),
        "response_posture": _clean_generated_string(payload["response_posture"], "response_posture"),
    }

    raw_sections = payload["sections"]
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ConstitutionError("guide sections must be a non-empty array")
    sections = [_validate_section(section, index) for index, section in enumerate(raw_sections, start=1)]
    guide["sections"] = sections
    return guide


def guide_to_markdown(guide: Mapping[str, object]) -> str:
    """Serialize guide data as reviewable Markdown."""

    data = validate_guide_data(guide)
    lines = [
        f"# {data['title']}",
        "",
        "## Overview",
        "",
        str(data["overview"]),
        "",
        "## Response Posture",
        "",
        str(data["response_posture"]),
        "",
        "## Operating Guidance",
        "",
    ]

    for section in data["sections"]:  # type: ignore[index]
        section_map = section if isinstance(section, Mapping) else {}
        lines.extend(
            [
                f"### {section_map['title']}",
                "",
                "**Applicability:**",
                "",
                str(section_map["when_to_apply"]),
                "",
                "**Practices:**",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in section_map["do"])  # type: ignore[index]
        lines.extend(["", "**Boundaries:**", ""])
        lines.extend(f"- {item}" for item in section_map["avoid"])  # type: ignore[index]
        examples = section_map["examples"]  # type: ignore[index]
        if examples:
            lines.extend(["", "**Examples:**", ""])
        for example in examples:
            example_map = example if isinstance(example, Mapping) else {}
            lines.extend(
                [
                    f"- User: {example_map['user']}",
                    f"  - Good: {example_map['good']}",
                    f"  - Bad: {example_map['bad']}",
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
                "You are a source-faithful alignment guide compiler. "
                "You translate freeform constitution documents into crisp decision guides for general-purpose assistant alignment. "
                "The guide will be sent in full with every data-generation prompt, so compress wording without compressing meaning. "
                "Write like an expert alignment reviewer: behavior-first, concrete, and decisive. "
                "Return only valid JSON matching the requested schema."
            ),
        },
        {
            "role": "user",
            "content": f"""Compiler prompt version: {COMPILER_PROMPT_VERSION}

Transform the Markdown constitution below into a crisp decision guide for general-purpose model alignment.

- Produce a concise operating guide that can be used directly as a repeated system-message reference during data generation.
- Preserve every materially distinct source requirement. Merge only when two source requirements would produce the same response behavior.
- Translate the constitution's values into concrete response behavior without adding your own safety, political, moral, or cultural preferences.
- Preserve the source posture exactly. Do not make a protective constitution permissive, a permissive constitution protective, or a balanced constitution moralizing.
- Make each section broad enough to guide many prompts and specific enough to drive crisp response choices.
- Use expert smart brevity: lead with the behavioral decision, remove filler, and keep critical distinctions intact.

Return a JSON object with this shape:

{{
  "title": "Short guide title",
  "overview": "One or two compact sentences stating purpose, scope, and boundaries.",
  "response_posture": "One or two compact sentences stating the assistant's default posture.",
  "sections": [
    {{
      "title": "Human-readable guidance title",
      "when_to_apply": "One compact sentence naming the observable trigger for this guidance.",
      "do": ["Concrete response practice, concise and behavior-changing."],
      "avoid": ["Concrete boundary, concise and behavior-changing."],
      "examples": [
        {{
          "user": "Representative request.",
          "good": "Short compliant response.",
          "bad": "Short noncompliant response."
        }}
      ]
    }}
  ]
}}

Requirements:
- Target 650 to 900 words total for a constitution with many distinct requirements; shorter is fine for simpler constitutions.
- Use as many sections as needed to preserve distinct response behavior, usually 5 to 8.
- Section titles must be behavior-first decisions, not topic labels. Prefer "Refuse X while preserving Y" over "X Safety".
- Use 1 or 2 practice bullets and 1 or 2 boundary bullets per section.
- Include an example only when it clarifies a boundary or differentiates this constitution from another likely posture.
- Examples must be complete, natural snippets. If you cannot write a complete example, omit the example.
- For risky, illegal, or harmful topics, good examples may name broad categories and defensive context but must not include operational methods, sequencing, evasion tactics, target-specific details, or "why this worked" instructions.
- Do not use placeholders, bracketed stand-ins, ellipses, unfinished sentences, "TBD", "etc.", or template variables anywhere in the JSON.
- Boundary bullets must be self-contained prohibitions or limits. Every boundary bullet must start with "Do not", "Avoid", or "Never".
- Prefer compact, high-signal wording over exhaustive explanation. Compress sentences; do not drop source meaning.
- Use this domain list as an omission audit, not a required table of contents: safety and harm enablement, privacy and consent, dignity and discrimination, high-impact advice, minors, sexual content, violence and extremism, self-harm or crisis support, illegal conduct, misinformation and uncertainty, autonomy and manipulation, refusal style, tone, and helpfulness.
- Cover every domain that materially appears in the source constitution, and omit domains the source does not govern.
- Preserve differentiating behavior. A permissive guide should explicitly protect allowed educational, fictional, analytical, defensive, or harm-reduction help when the source supports that. A protective guide should explicitly define stricter boundaries when the source supports that.
- Preserve permissions as permissions. If the source says a topic or response type is allowed, do not convert that into an obligation to answer with a particular level of depth, specificity, structure, or style.
- For permissive sources, prefer wording like "Allow", "Permit", or "Do not refuse solely because" for allowed content. Avoid any form of "provide", "fulfill", or "engage fully" as a requirement unless the source explicitly requires that response behavior.
- Write clear expert-reviewer prose: decision criteria, response tactics, and common failure modes. Avoid policy jargon, generic filler, and repeated boilerplate.
- Do not collapse vulnerable-population, crisis, privacy, fraud/manipulation, high-impact advice, or refusal-style requirements into one generic safety section when the source treats them distinctly.
- Examples must have a sharp contrast: the good response should model the section's core behavior, and the bad response should clearly violate it.
- If a permissive source allows discussion of risky conduct, model that as conceptual, historical, cultural, defensive, or harm-reduction explanation, not as a playbook for successful wrongdoing.
- Do not add generic safety obligations unless they are grounded in the source.
- Do not produce critique/revision prompt snippets. Produce a guide, not a random-rule set.
- Before returning, silently verify that every source principle is represented in the guide and that no section adds an unsupported obligation.
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
        "title": _clean_generated_string(raw_section["title"], f"section {index} title"),
        "when_to_apply": _clean_generated_string(raw_section["when_to_apply"], f"section {index} when_to_apply"),
        "do": _string_list(raw_section["do"], f"section {index} do"),
        "avoid": _string_list(raw_section["avoid"], f"section {index} avoid"),
    }
    for avoid_index, item in enumerate(section["avoid"], start=1):
        if not BOUNDARY_PREFIX.search(item):
            raise ConstitutionError(
                f"section {index} avoid item {avoid_index} must start with Do not, Avoid, or Never"
            )

    raw_examples = raw_section["examples"]
    if not isinstance(raw_examples, list):
        raise ConstitutionError(f"section {index}: examples must be an array")
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
        field: _clean_generated_string(raw_example[field], f"section {section_index} example {example_index} {field}")
        for field in GUIDE_EXAMPLE_FIELDS
    }


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConstitutionError(f"{label} must be a non-empty string")
    return value.strip()


def _clean_generated_string(value: object, label: str) -> str:
    text = _non_empty_string(value, label)
    if PLACEHOLDER_TEXT.search(text):
        raise ConstitutionError(f"{label} contains placeholder or unfinished text")
    if "TBD" in text or "etc." in text:
        raise ConstitutionError(f"{label} contains placeholder or unfinished text")
    return text


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConstitutionError(f"{label} must be a non-empty array")
    return [_clean_generated_string(item, f"{label} item {index}") for index, item in enumerate(value, start=1)]


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
