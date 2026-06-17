"""Load and validate human-readable constitution files."""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
import tomllib
from pathlib import Path
from typing import Iterable


FRONT_MATTER = "+++"
PRINCIPLE_HEADING = re.compile(r"^##\s+(?P<id>[a-z0-9][a-z0-9-]*):\s+(?P<title>.+?)\s*$")
SUBHEADING = re.compile(r"^###\s+(?P<title>.+?)\s*$")


@dataclasses.dataclass(frozen=True)
class Principle:
    """One critique/revision rule from a constitution."""

    id: str
    title: str
    tags: tuple[str, ...]
    weight: float
    body: str
    critique: str
    revision: str


@dataclasses.dataclass(frozen=True)
class Constitution:
    """A parsed constitution document."""

    id: str
    title: str
    version: str
    description: str
    tags: tuple[str, ...]
    principles: tuple[Principle, ...]


class ConstitutionError(ValueError):
    """Raised when a constitution file cannot be parsed or validated."""


def load_constitution(path: str | Path) -> Constitution:
    """Load a constitution Markdown file."""

    source = Path(path).read_text(encoding="utf-8")
    metadata, content = _split_front_matter(source)
    principles = tuple(_parse_principles(content))

    required = ("id", "title", "version", "description")
    missing = [key for key in required if not str(metadata.get(key, "")).strip()]
    if missing:
        raise ConstitutionError(f"missing front matter fields: {', '.join(missing)}")

    if not principles:
        raise ConstitutionError("constitution must define at least one principle")

    _ensure_unique("principle id", (principle.id for principle in principles))

    return Constitution(
        id=str(metadata["id"]),
        title=str(metadata["title"]),
        version=str(metadata["version"]),
        description=str(metadata["description"]),
        tags=tuple(str(tag) for tag in metadata.get("tags", ())),
        principles=principles,
    )


def validate_constitution(path: str | Path) -> list[str]:
    """Return validation warnings. Parsing errors are raised."""

    constitution = load_constitution(path)
    warnings: list[str] = []

    if not constitution.id:
        warnings.append("constitution id is empty")

    for principle in constitution.principles:
        if principle.weight <= 0:
            warnings.append(f"{principle.id}: weight should be positive")
        if not principle.tags:
            warnings.append(f"{principle.id}: no tags set")
        if len(principle.critique.split()) < 8:
            warnings.append(f"{principle.id}: critique prompt is very short")
        if len(principle.revision.split()) < 8:
            warnings.append(f"{principle.id}: revision prompt is very short")

    return warnings


def _split_front_matter(source: str) -> tuple[dict[str, object], str]:
    if not source.startswith(FRONT_MATTER):
        raise ConstitutionError("constitution must start with TOML front matter delimited by +++")

    try:
        _, raw_metadata, content = source.split(FRONT_MATTER, 2)
    except ValueError as exc:
        raise ConstitutionError("front matter must be closed with +++") from exc

    try:
        metadata = tomllib.loads(raw_metadata)
    except tomllib.TOMLDecodeError as exc:
        raise ConstitutionError(f"invalid TOML front matter: {exc}") from exc

    return metadata, content.strip()


def _parse_principles(content: str) -> Iterable[Principle]:
    lines = content.splitlines()
    starts = [(index, PRINCIPLE_HEADING.match(line)) for index, line in enumerate(lines)]
    starts = [(index, match) for index, match in starts if match]

    for offset, (start, match) in enumerate(starts):
        assert match is not None
        end = starts[offset + 1][0] if offset + 1 < len(starts) else len(lines)
        section_lines = lines[start + 1 : end]
        yield _parse_principle(
            principle_id=match.group("id"),
            title=match.group("title"),
            lines=section_lines,
        )


def _parse_principle(principle_id: str, title: str, lines: list[str]) -> Principle:
    fields: dict[str, str] = {}
    sections: dict[str, list[str]] = {"body": []}
    current = "body"

    for line in lines:
        subheading = SUBHEADING.match(line)
        if subheading:
            current = subheading.group("title").strip().lower()
            sections.setdefault(current, [])
            continue

        if current == "body" and ":" in line:
            key, value = line.split(":", 1)
            normalized_key = key.strip().lower()
            if normalized_key in {"tags", "weight"}:
                fields[normalized_key] = value.strip()
                continue

        sections.setdefault(current, []).append(line)

    tags = tuple(tag.strip() for tag in fields.get("tags", "").split(",") if tag.strip())
    try:
        weight = float(fields.get("weight", "1.0"))
    except ValueError as exc:
        raise ConstitutionError(f"{principle_id}: Weight must be a number") from exc

    body = _clean_section(sections.get("body", []))
    critique = _clean_section(sections.get("critique", []))
    revision = _clean_section(sections.get("revision", []))

    missing = [
        name
        for name, value in (
            ("Tags", tags),
            ("Critique", critique),
            ("Revision", revision),
        )
        if not value
    ]
    if missing:
        raise ConstitutionError(f"{principle_id}: missing required sections: {', '.join(missing)}")

    return Principle(
        id=principle_id,
        title=title,
        tags=tags,
        weight=weight,
        body=body,
        critique=critique,
        revision=revision,
    )


def _clean_section(lines: Iterable[str]) -> str:
    return "\n".join(lines).strip()


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
        constitution = load_constitution(args.path)
        warnings = validate_constitution(args.path)
    except ConstitutionError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1

    print(
        f"valid: {constitution.id} v{constitution.version} "
        f"({len(constitution.principles)} principles)"
    )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Constitution helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a constitution Markdown file")
    validate.add_argument("path", type=Path)
    validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
