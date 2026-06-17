"""Compile freeform constitution Markdown into editable ruleset YAML."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from cai.openrouter import OpenRouterClient, OpenRouterError, settings_from_env


RULE_FIELDS = ("id", "category", "principle", "critic", "revision")
RULE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
HEADING = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
COMPILER_PROMPT_VERSION = "v7-objective-applicability-minimal-revision"

CATEGORY_DESCRIPTIONS = (
    ("age-appropriate", "children, minors, dependent people, vulnerable people, abuse or exploitation risk, and age-appropriate handling"),
    ("social-bias", "insensitive, racist, sexist, hateful, demeaning, harassing, or socially inappropriate content"),
    ("criminal-activity", "violence, theft, hacking, robbery, evasion, unauthorized access, or other criminal plans"),
    ("harmful-general", "broad harmful, unethical, toxic, dangerous, or illegal content that does not fit a narrower category"),
    ("baiting", "requests that try to elicit harmful content, jailbreak behavior, or unsafe compliance"),
    ("dangerous-advice", "legally questionable, physically dangerous, or risk-increasing advice"),
    ("empathy-support", "self-harm, abuse, acute distress, crisis support, emotional support, and empathetic response quality"),
    ("gender-bias", "misogyny, sexism, sexual degradation, or gender-biased content"),
    ("harmful-assumptions", "harmful assumptions in the user request or assistant response that should be challenged or corrected"),
    ("moral-standards", "commonsense ethical or moral objectionability that does not fit a narrower category"),
    ("privacy-consent", "privacy, consent, personal data, tracking, impersonation, credentials, or bypassing privacy protections"),
    ("deception-manipulation", "fraud, impersonation, coercion, exploitation, manipulation, or bad-faith persuasion"),
    ("high-impact-advice", "medical, legal, financial, or similarly high-impact advice where uncertainty and professional judgment matter"),
    ("preserve-helpfulness", "over-refusal, safe alternatives, answering benign parts, and preserving helpfulness within safety limits"),
)

KNOWN_CATEGORIES = tuple(category for category, _ in CATEGORY_DESCRIPTIONS)


class ConstitutionError(ValueError):
    """Raised when a constitution or ruleset cannot be compiled or validated."""


class LiteralString(str):
    """String that should be emitted as a YAML literal block."""


class RulesetDumper(yaml.SafeDumper):
    """YAML dumper with indentation tuned for hand-edited files."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


def _literal_string_representer(dumper: yaml.SafeDumper, value: LiteralString) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")


RulesetDumper.add_representer(LiteralString, _literal_string_representer)


def compile_markdown(
    markdown: str,
    client: OpenRouterClient,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 12000,
) -> list[dict[str, str]]:
    """Compile a complete Markdown constitution through an OpenRouter model."""

    if not markdown.strip():
        raise ConstitutionError("constitution Markdown is empty")

    response = client.chat(
        _compiler_messages(markdown),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return ruleset_from_json(response)


def ruleset_from_json(text: str) -> list[dict[str, str]]:
    """Parse and validate compiler JSON output."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConstitutionError(f"compiler returned invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ConstitutionError("compiler JSON must be an object")

    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raise ConstitutionError("compiler JSON must contain a rules array")

    rules: list[dict[str, str]] = []
    for index, raw_rule in enumerate(raw_rules, start=1):
        rules.append(_validate_rule(raw_rule, index))

    if not rules:
        raise ConstitutionError("rules array must not be empty")

    _ensure_unique("rule id", (rule["id"] for rule in rules))
    return rules


def ruleset_to_yaml(rules: list[dict[str, str]]) -> str:
    """Serialize rules as reviewer-friendly YAML."""

    yaml_rules = [
        {
            "id": rule["id"],
            "category": rule["category"],
            "principle": rule["principle"],
            "critic": LiteralString(rule["critic"]),
            "revision": LiteralString(rule["revision"]),
        }
        for rule in rules
    ]
    return yaml.dump(
        yaml_rules,
        Dumper=RulesetDumper,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )


def validate_markdown(path: str | Path) -> list[str]:
    """Validate source Markdown without compiling it."""

    content = Path(path).read_text(encoding="utf-8")
    if not content.strip():
        raise ConstitutionError("constitution Markdown is empty")
    warnings: list[str] = []
    if not _find_title(content):
        warnings.append("source Markdown has no top-level title")
    return warnings


def validate_ruleset(path: str | Path) -> list[str]:
    """Validate a generated YAML ruleset."""

    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConstitutionError(f"invalid YAML ruleset: {exc}") from exc

    if not isinstance(payload, list):
        raise ConstitutionError("ruleset YAML must be a top-level array")

    rules = [_validate_rule(rule, index) for index, rule in enumerate(payload, start=1)]
    if not rules:
        raise ConstitutionError("ruleset YAML must contain at least one rule")
    _ensure_unique("rule id", (rule["id"] for rule in rules))
    return []


def _compiler_messages(markdown: str) -> list[dict[str, str]]:
    categories = "\n".join(f"- {category}: {description}" for category, description in CATEGORY_DESCRIPTIONS)
    return [
        {
            "role": "system",
            "content": (
                "You are a rigorous source-faithful policy compiler. "
                "You turn freeform source documents into crisp executable critique/revision rules without importing unstated policy assumptions. "
                "Return only valid JSON. Do not wrap the JSON in Markdown."
            ),
        },
        {
            "role": "user",
            "content": f"""Compiler prompt version: {COMPILER_PROMPT_VERSION}

Transform the full Markdown constitution below into a JSON object with this exact shape:

{{
  "rules": [
    {{
      "id": "principle-01-category-name",
      "category": "category-name",
      "principle": "Human-readable principle copied or lightly normalized from the source.",
      "critic": "Prompt for critiquing an assistant response against this principle.",
      "revision": "Prompt for revising an assistant response against this principle."
    }}
  ]
}}

Rules:
- Extract the author's intended principles from the entire Markdown document.
- Preserve meaning; do not invent unrelated principles.
- Treat the source Markdown as authoritative. Do not assume the constitution is strict, balanced, permissive, safety-focused, humor-focused, or helpfulness-focused unless the source says so.
- Identify the source constitution's top-level objective first. Every rule must serve that objective. Do not produce a longer or broader rule list if that weakens the objective.
- Use one rule per distinct principle.
- Use stable lower-kebab-case ids.
- Use these categories when possible:
{categories}
- If no category fits, use "moral-standards".
- Use categories only as routing labels for the rules. Do not add category-specific obligations that are not present in the source.
- Every rule must contain exactly: id, category, principle, critic, revision.
- Return JSON only.

Crispness requirements:
- Keep each rule narrow enough to produce a meaningfully different critique/revision from the other rules.
- If one source bullet contains multiple independent obligations, split it into multiple rules. If the obligations share one behavioral test, keep them together.
- The principle should be a short, plain-language statement of the rule. Prefer one sentence.
- The critic must be an observable test for the assistant response. It must state when the rule applies, when it does not apply, what concrete behavior counts as a violation, and what offending content or missing behavior to identify.
- The critic must make it easy for a later model to return revision_needed=false when the rule is irrelevant or already satisfied. Do not write critic prompts that imply every response needs a revision.
- The revision must be minimal and targeted. It must say what to remove, what to preserve, and what safe alternative, redirect, correction, or next step to provide when relevant.
- The revision must preserve the original answer unless specific content violates the rule. Do not add unrelated safety warnings, crisis language, privacy language, moral framing, or tone changes.
- Avoid mushy wording such as "be appropriate", "use care", "ensure safety", or "follow the principle" unless paired with concrete criteria.
- Use concrete verbs such as identify, remove, replace, preserve, redirect, correct, refuse, answer, recommend, or ask.
- Preserve requested style from the source, such as humor or light wit, only when it does not undermine safety or dignity.
- Preserve the source constitution's risk posture. Strict constitutions should refuse explicit harmful requests directly, clarify only genuinely ambiguous risky requests, and flag borderline enablement or missing safeguards. Balanced constitutions should balance useful answers with safety limits. Permissive constitutions should flag over-refusal and preserve as much benign, educational, fictional, analytical, or defensive detail as possible while still removing actionable harm. Playful constitutions should apply humor only when the relevant harmful, criminal, or socially inappropriate framing is actually present.
- Do not flatten strict, balanced, and permissive constitutions into the same generic safety behavior.
- If the source is not a safety constitution, do not transform it into one. Produce rules that faithfully express the source's actual goals.
- Do not add broad generic safety rules that are not grounded in the source Markdown.

Markdown constitution:
```markdown
{markdown}
```""",
        },
    ]


def _validate_rule(raw_rule: object, index: int) -> dict[str, str]:
    if not isinstance(raw_rule, Mapping):
        raise ConstitutionError(f"rule {index}: must be an object")

    keys = set(raw_rule)
    expected = set(RULE_FIELDS)
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise ConstitutionError(f"rule {index}: missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ConstitutionError(f"rule {index}: unexpected fields: {', '.join(sorted(extra))}")

    rule: dict[str, str] = {}
    for field in RULE_FIELDS:
        value = raw_rule[field]
        if not isinstance(value, str) or not value.strip():
            raise ConstitutionError(f"rule {index}: {field} must be a non-empty string")
        rule[field] = value.strip()

    if not RULE_ID.match(rule["id"]):
        raise ConstitutionError(f"rule {index}: id must be lower-kebab-case")
    if rule["category"] not in KNOWN_CATEGORIES:
        raise ConstitutionError(f"rule {index}: unknown category {rule['category']!r}")

    return rule


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
        joined = ", ".join(sorted(duplicates))
        raise ConstitutionError(f"duplicate {label}: {joined}")


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        warnings = validate_ruleset(args.path) if args.path.suffix in {".yaml", ".yml"} else validate_markdown(args.path)
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
        rules = compile_markdown(
            markdown,
            client,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except (ConstitutionError, OpenRouterError) as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1

    text = ruleset_to_yaml(rules)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Constitution helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate source Markdown or ruleset YAML")
    validate.add_argument("path", type=Path)
    validate.set_defaults(func=_cmd_validate)

    compile_parser = subparsers.add_parser("compile", help="compile Markdown to editable ruleset YAML")
    compile_parser.add_argument("path", type=Path)
    compile_parser.add_argument("-o", "--output", type=Path)
    compile_parser.add_argument("--env", type=Path, default=Path(".env"))
    compile_parser.add_argument("--model")
    compile_parser.add_argument("--temperature", type=float, default=0.0)
    compile_parser.add_argument("--max-tokens", type=int, default=12000)
    compile_parser.set_defaults(func=_cmd_compile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
