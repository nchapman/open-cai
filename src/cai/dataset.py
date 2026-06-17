"""Generate Constitutional AI training data with OpenRouter."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from cai.constitution import validate_ruleset
from cai.openrouter import (
    APPLY_RULES_MODEL,
    APPLY_RULES_REASONING,
    OpenRouterClient,
    OpenRouterError,
    settings_from_env,
)


DEFAULT_DATASET = "Anthropic/hh-rlhf"
DEFAULT_DATA_DIR = "harmless-base"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_SAMPLES = 128
DEFAULT_CONCURRENCY = 16
DEFAULT_MAX_TOKENS = 6000
DEFAULT_TEMPERATURE = 1.0
STOP_SEQUENCES = ("User:", "###", "<|endoftext|>")
DEFAULT_ASSISTANT_SYSTEM_PROMPT = (
    "You are Cai, a helpful and unbiased AI assistant. "
    "Follow the active constitution faithfully. Apply its rules evenhandedly across people, groups, viewpoints, and topics."
)
HUMAN_SEGMENT = re.compile(r"(?:^|\n\n)Human:\s*(?P<prompt>.*?)(?=\n\nAssistant:|\n\nHuman:|\Z)", re.DOTALL)
CRITIC_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "revision_needed": {
            "type": "boolean",
            "description": "Whether the assistant response violates the sampled rule and should be rewritten.",
        },
        "critique": {
            "type": "string",
            "description": "A concise critique explaining the decision, with concrete offending content when revision is needed.",
        },
    },
    "required": ["revision_needed", "critique"],
    "additionalProperties": False,
}


class DatasetError(RuntimeError):
    """Raised when dataset generation cannot continue."""


@dataclasses.dataclass(frozen=True)
class Rule:
    id: str
    category: str
    principle: str
    critic: str
    revision: str


@dataclasses.dataclass(frozen=True)
class SourcePrompt:
    source_dataset: str
    source_split: str
    source_index: int
    prompt: str


def load_rules(path: str | Path) -> list[Rule]:
    """Load and validate an editable YAML ruleset."""

    rules_path = Path(path)
    validate_ruleset(rules_path)
    payload = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    return [Rule(**rule) for rule in payload]


def extract_human_prompt(conversation: str) -> str:
    """Extract the first HH-RLHF Human turn from a chosen/rejected conversation."""

    match = HUMAN_SEGMENT.search(conversation)
    if not match:
        raise DatasetError("conversation does not contain a Human turn")

    prompt = match.group("prompt").strip()
    if not prompt:
        raise DatasetError("Human turn is empty")
    return prompt


def select_rule(rules: Sequence[Rule], *, seed: int, source_index: int) -> Rule:
    """Select one rule deterministically for a dataset row."""

    if not rules:
        raise DatasetError("ruleset is empty")
    rng = random.Random(f"{seed}:{source_index}")
    return rng.choice(list(rules))


def completed_indices(path: str | Path, *, split: str) -> set[int]:
    """Return source indices already present in a JSONL output file."""

    output_path = Path(path)
    if not output_path.exists():
        return set()

    completed: set[int] = set()
    for line_number, line in enumerate(output_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{output_path}:{line_number}: invalid JSONL: {exc}") from exc
        if row.get("source_split") == split and isinstance(row.get("source_index"), int):
            completed.add(row["source_index"])
    return completed


def load_source_prompts(
    *,
    dataset_name: str = DEFAULT_DATASET,
    data_dir: str = DEFAULT_DATA_DIR,
    split: str = DEFAULT_SPLIT,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> list[SourcePrompt]:
    """Load HH-RLHF prompts using the same harmless-base source as the HF example."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only with a broken install
        raise DatasetError("the datasets package is required; run `uv sync`") from exc

    try:
        dataset = load_dataset(dataset_name, data_dir=data_dir, split=split)
    except Exception as exc:  # noqa: BLE001 - surface dataset/cache/auth failures as CLI errors.
        raise DatasetError(f"failed to load dataset {dataset_name}/{data_dir}:{split}: {exc}") from exc
    if max_samples >= 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts: list[SourcePrompt] = []
    for index, row in enumerate(dataset):
        chosen = row.get("chosen") if isinstance(row, Mapping) else None
        if not isinstance(chosen, str):
            raise DatasetError(f"dataset row {index} has no chosen conversation")
        prompts.append(
            SourcePrompt(
                source_dataset=dataset_name,
                source_split=split,
                source_index=index,
                prompt=extract_human_prompt(chosen),
            )
        )
    return prompts


def build_critic_prompt(rule: Rule) -> str:
    schema = json.dumps(CRITIC_RESPONSE_SCHEMA, sort_keys=True)
    return (
        f"{rule.critic}\n\n"
        f"Principle: {rule.principle}\n\n"
        "Evaluate only this principle. Return only JSON matching this schema:\n"
        f"{schema}\n\n"
        "Set revision_needed to true only when the assistant response violates this principle and should be rewritten. "
        "Set revision_needed to false when the response already satisfies this principle or the principle is not applicable. "
        "The critique must briefly explain the decision and identify concrete offending content when revision is needed."
    )


def build_revision_prompt(rule: Rule) -> str:
    return (
        f"{rule.revision}\n\n"
        f"Principle: {rule.principle}\n\n"
        "Rewrite the assistant's original response as the final answer to the human. "
        "Use the critique above, but return only the revised assistant response. "
        "Do not explain that you are revising, do not mention the critique, and do not refer to the previous response."
    )


def trim_stop_sequences(text: str, stop_sequences: Iterable[str] = STOP_SEQUENCES) -> str:
    stripped = text.strip()
    for stop_sequence in stop_sequences:
        if stripped.endswith(stop_sequence):
            return stripped[: -len(stop_sequence)].rstrip()
    return stripped


def critic_result_from_json(text: str) -> dict[str, object]:
    """Parse and validate the critic's structured JSON response."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"critic returned invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise DatasetError("critic JSON must be an object")

    keys = set(payload)
    expected = {"revision_needed", "critique"}
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise DatasetError(f"critic JSON missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise DatasetError(f"critic JSON unexpected fields: {', '.join(sorted(extra))}")

    revision_needed = payload["revision_needed"]
    if not isinstance(revision_needed, bool):
        raise DatasetError("critic JSON field revision_needed must be a boolean")

    critique = payload["critique"]
    if not isinstance(critique, str) or not critique.strip():
        raise DatasetError("critic JSON field critique must be a non-empty string")

    return {"revision_needed": revision_needed, "critique": critique.strip()}


def critic_response_format() -> dict[str, object]:
    """Return OpenRouter's strict JSON Schema response format for critic calls."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "critic_result",
            "strict": True,
            "schema": CRITIC_RESPONSE_SCHEMA,
        },
    }


def make_record(
    *,
    source: SourcePrompt,
    rule: Rule,
    init_response: str,
    critic_prompt: str,
    critic_response: str,
    revision_prompt: str,
    revision_response: str,
    revision_needed: bool = True,
) -> dict[str, object]:
    """Create the JSONL row, including later SFT/preference-friendly fields."""

    init_prompt = source.prompt.strip()
    revised = revision_response.strip()
    initial = init_response.strip()
    chosen = [
        {"role": "user", "content": init_prompt},
        {"role": "assistant", "content": revised},
    ]
    rejected = [
        {"role": "user", "content": init_prompt},
        {"role": "assistant", "content": initial},
    ]

    return {
        "source_dataset": source.source_dataset,
        "source_split": source.source_split,
        "source_index": source.source_index,
        "rule_id": rule.id,
        "rule_category": rule.category,
        "principle": rule.principle,
        "init_prompt": init_prompt,
        "init_response": initial,
        "critic_prompt": critic_prompt.strip(),
        "critic_response": critic_response.strip(),
        "revision_prompt": revision_prompt.strip(),
        "revision_response": revised,
        "revision_needed": revision_needed,
        "prompt": init_prompt,
        "messages": chosen,
        "chosen": chosen,
        "rejected": rejected,
    }


def generate_record(
    *,
    client: OpenRouterClient,
    source: SourcePrompt,
    rule: Rule,
    model: str | None = APPLY_RULES_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = 2,
    reasoning: Mapping[str, object] | None = APPLY_RULES_REASONING,
) -> dict[str, object]:
    """Run initial-response, critique, and revision calls for one source prompt."""

    critic_prompt = build_critic_prompt(rule)
    revision_prompt = build_revision_prompt(rule)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": DEFAULT_ASSISTANT_SYSTEM_PROMPT},
        {"role": "user", "content": source.prompt},
    ]

    init_response = _chat_with_retries(
        client,
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        reasoning=reasoning,
    )
    messages.append({"role": "assistant", "content": init_response})
    messages.append({"role": "user", "content": critic_prompt})

    critic_raw_response = _chat_with_retries(
        client,
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        response_format=critic_response_format(),
        reasoning=reasoning,
    )
    critic_result = critic_result_from_json(critic_raw_response)
    critic_response = str(critic_result["critique"])
    revision_needed = bool(critic_result["revision_needed"])
    if not revision_needed:
        return make_record(
            source=source,
            rule=rule,
            init_response=init_response,
            critic_prompt=critic_prompt,
            critic_response=critic_response,
            revision_prompt="Revision not needed: the critic found no violation of the sampled rule.",
            revision_response=init_response,
            revision_needed=False,
        )

    messages.append({"role": "assistant", "content": critic_response})
    messages.append({"role": "user", "content": revision_prompt})

    revision_response = _chat_with_retries(
        client,
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        reasoning=reasoning,
    )

    return make_record(
        source=source,
        rule=rule,
        init_response=init_response,
        critic_prompt=critic_prompt,
        critic_response=critic_response,
        revision_prompt=revision_prompt,
        revision_response=revision_response,
        revision_needed=True,
    )


def _chat_with_retries(
    client: OpenRouterClient,
    messages: Sequence[Mapping[str, str]],
    *,
    model: str | None,
    temperature: float,
    max_tokens: int,
    retries: int,
    response_format: Mapping[str, object] | None = None,
    reasoning: Mapping[str, object] | None = None,
) -> str:
    for attempt in range(retries + 1):
        try:
            return trim_stop_sequences(
                client.chat(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    reasoning=reasoning,
                )
            )
        except OpenRouterError:
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def write_jsonl_record(path: str | Path, record: Mapping[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_dataset(args: argparse.Namespace) -> int:
    try:
        rules = load_rules(args.rules)
        completed = completed_indices(args.output, split=args.split)
        sources = load_source_prompts(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            split=args.split,
            max_samples=args.max_samples,
        )
        pending = [source for source in sources if source.source_index not in completed]
        client = OpenRouterClient(settings_from_env(args.env))
        reasoning = _reasoning_from_args(args)
    except (DatasetError, OpenRouterError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"generating {len(pending)} records "
        f"({len(completed)} already complete) from {args.dataset}/{args.data_dir}:{args.split}",
        file=sys.stderr,
    )
    if not pending:
        return 0

    failures: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                generate_record,
                client=client,
                source=source,
                rule=select_rule(rules, seed=args.seed, source_index=source.source_index),
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retries=args.retries,
                reasoning=reasoning,
            ): source
            for source in pending
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - keep long generation runs moving.
                failures.append((source.source_index, str(exc)))
                print(f"failed source_index={source.source_index}: {exc}", file=sys.stderr)
                continue
            write_jsonl_record(args.output, record)
            print(f"wrote source_index={source.source_index}", file=sys.stderr)

    if failures:
        print(f"completed with {len(failures)} failed records", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Constitutional AI datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate critique/revision JSONL data")
    generate.add_argument("--rules", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--dataset", default=DEFAULT_DATASET)
    generate.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    generate.add_argument("--split", default=DEFAULT_SPLIT)
    generate.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    generate.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--env", type=Path, default=Path(".env"))
    generate.add_argument("--model", default=APPLY_RULES_MODEL)
    generate.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    generate.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    reasoning = generate.add_mutually_exclusive_group()
    reasoning.add_argument("--reasoning-effort", choices=["xhigh", "high", "medium", "low", "minimal", "none"])
    reasoning.add_argument("--reasoning-max-tokens", type=int)
    generate.add_argument("--include-reasoning", action="store_true")
    generate.add_argument("--retries", type=int, default=2)
    generate.set_defaults(func=generate_dataset)

    return parser


def _reasoning_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    if args.reasoning_effort and args.reasoning_max_tokens is not None:
        raise DatasetError("use either --reasoning-effort or --reasoning-max-tokens, not both")

    reasoning: dict[str, object] = {"exclude": not args.include_reasoning}
    if args.reasoning_max_tokens is not None:
        reasoning["max_tokens"] = args.reasoning_max_tokens
    else:
        reasoning["effort"] = args.reasoning_effort or APPLY_RULES_REASONING["effort"]
    return reasoning


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
