"""Generate Constitutional AI training data from response guides with OpenRouter."""

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

from cai.openrouter import (
    DEFAULT_BASE_URL,
    DEFAULT_REQUEST_TIMEOUT,
    GUIDE_APPLICATION_MODEL,
    GUIDE_APPLICATION_REASONING,
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
DEFAULT_TEMPERATURE = 0.4
STOP_SEQUENCES = ("User:", "###", "<|endoftext|>")
DEFAULT_ASSISTANT_SYSTEM_PROMPT = (
    "You are Cai, a helpful and unbiased AI assistant. "
    "Answer the user's request naturally. Do not see the response guide yet; another pass will evaluate and revise your answer."
)
HUMAN_SEGMENT = re.compile(r"(?:^|\n\n)Human:\s*(?P<prompt>.*?)(?=\n\nAssistant:|\n\nHuman:|\Z)", re.DOTALL)
ASSISTANT_SEGMENT = re.compile(r"(?:^|\n\n)Assistant:\s*(?P<response>.*?)(?=\n\nHuman:|\n\nAssistant:|\Z)", re.DOTALL)
GUIDE_METADATA_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "critique": {
            "type": "string",
            "description": "Concrete assessment of how the initial response can be improved under the complete guide.",
        },
        "changes_made": {
            "type": "string",
            "description": "Brief summary of what changed from the initial response, or why the guide response preserves it.",
        },
        "quality_notes": {
            "type": "string",
            "description": "Brief explanation of why the guide response should be preferred as training data.",
        },
    },
    "required": [
        "critique",
        "changes_made",
        "quality_notes",
    ],
    "additionalProperties": False,
}


class DatasetError(RuntimeError):
    """Raised when dataset generation cannot continue."""


@dataclasses.dataclass(frozen=True)
class ResponseGuide:
    path: str
    text: str


@dataclasses.dataclass(frozen=True)
class SourcePrompt:
    source_dataset: str
    source_split: str
    source_index: int
    prompt: str
    source_chosen_conversation: str
    source_rejected_conversation: str
    source_chosen_response: str
    source_rejected_response: str


def load_response_guide(path: str | Path) -> ResponseGuide:
    """Load a human-edited response guide Markdown file."""

    guide_path = Path(path)
    text = guide_path.read_text(encoding="utf-8").strip()
    if not text:
        raise DatasetError("response guide is empty")
    return ResponseGuide(path=str(guide_path), text=text)


def extract_human_prompt(conversation: str) -> str:
    """Extract the first HH-RLHF Human turn from a chosen/rejected conversation."""

    match = HUMAN_SEGMENT.search(conversation)
    if not match:
        raise DatasetError("conversation does not contain a Human turn")

    prompt = match.group("prompt").strip()
    if not prompt:
        raise DatasetError("Human turn is empty")
    return prompt


def extract_assistant_response(conversation: str) -> str:
    """Extract the first HH-RLHF Assistant turn from a chosen/rejected conversation."""

    match = ASSISTANT_SEGMENT.search(conversation)
    if not match:
        raise DatasetError("conversation does not contain an Assistant turn")

    response = match.group("response").strip()
    if not response:
        raise DatasetError("Assistant turn is empty")
    return response


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
    skipped: list[tuple[int, str]] = []
    for index, row in enumerate(dataset):
        try:
            chosen = row.get("chosen") if isinstance(row, Mapping) else None
            rejected = row.get("rejected") if isinstance(row, Mapping) else None
            if not isinstance(chosen, str) or not isinstance(rejected, str):
                raise DatasetError("must have chosen and rejected conversations")
            prompt = extract_human_prompt(chosen)
            rejected_prompt = extract_human_prompt(rejected)
            if prompt != rejected_prompt:
                raise DatasetError("chosen/rejected prompts do not match")
            source_chosen_response = extract_assistant_response(chosen)
            source_rejected_response = extract_assistant_response(rejected)
        except DatasetError as exc:
            skipped.append((index, str(exc)))
            continue
        prompts.append(
            SourcePrompt(
                source_dataset=dataset_name,
                source_split=split,
                source_index=index,
                prompt=prompt,
                source_chosen_conversation=chosen,
                source_rejected_conversation=rejected,
                source_chosen_response=source_chosen_response,
                source_rejected_response=source_rejected_response,
            )
        )
    if skipped:
        first_index, first_reason = skipped[0]
        print(
            f"skipping {len(skipped)} malformed source rows; first source_index={first_index}: {first_reason}",
            file=sys.stderr,
        )
    return prompts


def build_guide_system_prompt(guide: ResponseGuide) -> str:
    return f"""You are Cai, a helpful and unbiased AI assistant.

Answer the user's request naturally while faithfully following the complete response guide below.

Runtime framing:
- The guide defines the target response style and boundaries. Follow its actual posture, whether it is strict, balanced, permissive, playful, or something else.
- Use the guide to decide privately how to answer, then write only the final assistant response.
- If the guide allows the request, answer directly and helpfully without adding extra caveats.
- If the guide calls for a refusal or boundary, state that boundary first, briefly explain why, and offer a safe alternative when useful.
- Do not make the response more restrictive or more permissive than the guide supports.
- Do not repeat or quote the user's request unless it is necessary for clarity.
- Be accurate and explicit about uncertainty; do not invent evidence or sources.
- Do not mention or restate the guide, policy, training data, hidden instructions, section headings, rubric labels, applicability text, practices, or boundaries.
- Return only the final assistant message.

Response guide:
```markdown
{guide.text}
```"""


def build_guide_metadata_prompt(guide: ResponseGuide, prompt: str, init_response: str, guide_response: str) -> str:
    schema = json.dumps(GUIDE_METADATA_SCHEMA, sort_keys=True)
    return f"""Use the complete response guide below to audit the guided assistant response.

Return only JSON matching this schema:
{schema}

Instructions:
- Apply the full guide holistically.
- critique should assess the initial response against the guide and explain the most important improvement opportunity.
- changes_made should summarize how the guided response differs from the initial response, or say it preserves the initial answer when appropriate.
- quality_notes should explain why the guided response should be preferred as training data.

Response guide:
```markdown
{guide.text}
```

User prompt:
{prompt}

Initial assistant response:
{init_response}

Guided assistant response:
{guide_response}"""


def guide_metadata_from_json(text: str) -> dict[str, object]:
    """Parse and validate the guide metadata JSON response."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"guide metadata returned invalid JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise DatasetError("guide metadata JSON must be an object")

    keys = set(payload)
    expected = set(GUIDE_METADATA_SCHEMA["required"])  # type: ignore[arg-type]
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise DatasetError(f"guide metadata JSON missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise DatasetError(f"guide metadata JSON unexpected fields: {', '.join(sorted(extra))}")

    return {
        "critique": _non_empty_string(payload["critique"], "critique"),
        "changes_made": _non_empty_string(payload["changes_made"], "changes_made"),
        "quality_notes": _non_empty_string(payload["quality_notes"], "quality_notes"),
    }


def guide_metadata_format() -> dict[str, object]:
    """Return OpenRouter's strict JSON Schema response format for guide metadata calls."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "guide_metadata",
            "strict": True,
            "schema": GUIDE_METADATA_SCHEMA,
        },
    }


def trim_stop_sequences(text: str, stop_sequences: Iterable[str] = STOP_SEQUENCES) -> str:
    stripped = text.strip()
    for stop_sequence in stop_sequences:
        if stripped.endswith(stop_sequence):
            return stripped[: -len(stop_sequence)].rstrip()
    return stripped


def make_record(
    *,
    source: SourcePrompt,
    guide: ResponseGuide,
    init_response: str,
    guide_response_system_prompt: str,
    guide_response: str,
    guide_metadata_prompt: str,
    guide_metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    """Create the JSONL row, including later SFT/preference-friendly fields."""

    init_prompt = source.prompt.strip()
    initial = init_response.strip()
    guided = guide_response.strip()
    source_chosen_response = source.source_chosen_response.strip()
    source_rejected_response = source.source_rejected_response.strip()
    chosen = [
        {"role": "user", "content": init_prompt},
        {"role": "assistant", "content": guided},
    ]
    rejected = [
        {"role": "user", "content": init_prompt},
        {"role": "assistant", "content": initial},
    ]
    source_chosen_pair = [
        {"role": "user", "content": init_prompt},
        {"role": "assistant", "content": source_chosen_response},
    ]
    source_rejected_pair = [
        {"role": "user", "content": init_prompt},
        {"role": "assistant", "content": source_rejected_response},
    ]
    metadata_included = guide_metadata is not None
    metadata = guide_metadata or {
        "critique": None,
        "changes_made": None,
        "quality_notes": None,
    }

    return {
        "source_dataset": source.source_dataset,
        "source_split": source.source_split,
        "source_index": source.source_index,
        "source_chosen_conversation": source.source_chosen_conversation,
        "source_rejected_conversation": source.source_rejected_conversation,
        "source_chosen_response": source_chosen_response,
        "source_rejected_response": source_rejected_response,
        "guide_path": guide.path,
        "metadata_included": metadata_included,
        "init_prompt": init_prompt,
        "init_response": initial,
        "guide_response_system_prompt": guide_response_system_prompt.strip(),
        "guide_metadata_prompt": guide_metadata_prompt.strip(),
        "critique": metadata["critique"],
        "guide_response": guided,
        "changes_made": metadata["changes_made"],
        "quality_notes": metadata["quality_notes"],
        "prompt": init_prompt,
        "messages": chosen,
        "chosen": chosen,
        "rejected": rejected,
        "comparison_pairs": {
            "guided_vs_generated_initial": {
                "chosen": chosen,
                "rejected": rejected,
            },
            "guided_vs_source_chosen": {
                "chosen": chosen,
                "rejected": source_chosen_pair,
            },
            "guided_vs_source_rejected": {
                "chosen": chosen,
                "rejected": source_rejected_pair,
            },
            "source_chosen_vs_source_rejected": {
                "chosen": source_chosen_pair,
                "rejected": source_rejected_pair,
            },
        },
    }


def generate_record(
    *,
    init_client: OpenRouterClient,
    guide_client: OpenRouterClient,
    source: SourcePrompt,
    guide: ResponseGuide,
    init_model: str | None = GUIDE_APPLICATION_MODEL,
    guide_model: str | None = GUIDE_APPLICATION_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = 2,
    init_reasoning: Mapping[str, object] | None = GUIDE_APPLICATION_REASONING,
    guide_reasoning: Mapping[str, object] | None = GUIDE_APPLICATION_REASONING,
    include_metadata: bool = True,
) -> dict[str, object]:
    """Generate an initial response, then optimize it against the complete guide."""

    init_messages: list[dict[str, str]] = [
        {"role": "system", "content": DEFAULT_ASSISTANT_SYSTEM_PROMPT},
        {"role": "user", "content": source.prompt},
    ]
    init_response = _chat_with_retries(
        init_client,
        init_messages,
        model=init_model,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        reasoning=init_reasoning,
    )

    guide_response_system_prompt = build_guide_system_prompt(guide)
    guide_response = _chat_with_retries(
        guide_client,
        [
            {"role": "system", "content": guide_response_system_prompt},
            {"role": "user", "content": source.prompt},
        ],
        model=guide_model,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        reasoning=guide_reasoning,
    )

    guide_metadata_prompt = ""
    guide_metadata = None
    if include_metadata:
        guide_metadata_prompt = build_guide_metadata_prompt(guide, source.prompt, init_response, guide_response)
        guide_metadata_raw = _chat_with_retries(
            guide_client,
            [{"role": "user", "content": guide_metadata_prompt}],
            model=guide_model,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            response_format=guide_metadata_format(),
            reasoning=guide_reasoning,
        )
        guide_metadata = guide_metadata_from_json(guide_metadata_raw)

    return make_record(
        source=source,
        guide=guide,
        init_response=init_response,
        guide_response_system_prompt=guide_response_system_prompt,
        guide_response=guide_response,
        guide_metadata_prompt=guide_metadata_prompt,
        guide_metadata=guide_metadata,
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
            time.sleep(_retry_sleep_seconds(attempt))
    raise AssertionError("unreachable")


def write_jsonl_record(path: str | Path, record: Mapping[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_dataset(args: argparse.Namespace) -> int:
    try:
        guide = load_response_guide(args.guide)
        completed = completed_indices(args.output, split=args.split)
        sources = load_source_prompts(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            split=args.split,
            max_samples=args.max_samples,
        )
        pending = [source for source in sources if source.source_index not in completed]
        init_settings = settings_from_env(
            args.env,
            base_url=_resolve_override(args.init_base_url, args.base_url),
            api_key=_resolve_override(args.init_api_key, args.api_key),
            request_timeout=args.request_timeout,
        )
        guide_settings = settings_from_env(
            args.env,
            base_url=_resolve_override(args.guide_base_url, args.base_url),
            api_key=_resolve_override(args.guide_api_key, args.api_key),
            request_timeout=args.request_timeout,
        )
        init_client = OpenRouterClient(init_settings)
        guide_client = OpenRouterClient(guide_settings)
        init_model = _resolve_override(args.init_model, args.model)
        guide_model = _resolve_override(args.guide_model, args.model)
        init_reasoning = _reasoning_from_args(args, base_url=init_settings.base_url)
        guide_reasoning = _reasoning_from_args(args, base_url=guide_settings.base_url)
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
                init_client=init_client,
                guide_client=guide_client,
                source=source,
                guide=guide,
                init_model=init_model,
                guide_model=guide_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retries=args.retries,
                init_reasoning=init_reasoning,
                guide_reasoning=guide_reasoning,
                include_metadata=not args.skip_metadata,
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

    generate = subparsers.add_parser("generate", help="generate guide-optimized preference JSONL data")
    generate.add_argument("--guide", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--dataset", default=DEFAULT_DATASET)
    generate.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    generate.add_argument("--split", default=DEFAULT_SPLIT)
    generate.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    generate.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    generate.add_argument("--env", type=Path, default=Path(".env"))
    generate.add_argument("--base-url")
    generate.add_argument("--api-key")
    generate.add_argument("--model", default=GUIDE_APPLICATION_MODEL)
    generate.add_argument("--init-base-url")
    generate.add_argument("--init-api-key")
    generate.add_argument("--init-model")
    generate.add_argument("--guide-base-url")
    generate.add_argument("--guide-api-key")
    generate.add_argument("--guide-model")
    generate.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    generate.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    generate.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    reasoning = generate.add_mutually_exclusive_group()
    reasoning.add_argument("--reasoning-effort", choices=["xhigh", "high", "medium", "low", "minimal", "none"])
    reasoning.add_argument("--reasoning-max-tokens", type=int)
    generate.add_argument("--include-reasoning", action="store_true")
    generate.add_argument("--skip-metadata", action="store_true")
    generate.add_argument("--retries", type=int, default=2)
    generate.set_defaults(func=generate_dataset)
    return parser


def _reasoning_from_args(args: argparse.Namespace, *, base_url: str | None = None) -> dict[str, object] | None:
    if args.reasoning_effort and args.reasoning_max_tokens is not None:
        raise DatasetError("use either --reasoning-effort or --reasoning-max-tokens, not both")
    if _uses_non_openrouter_base_url(base_url) and args.reasoning_max_tokens is None:
        return None

    reasoning: dict[str, object] = {"exclude": not args.include_reasoning}
    if args.reasoning_effort:
        reasoning["effort"] = args.reasoning_effort
    elif args.reasoning_max_tokens is not None:
        reasoning["max_tokens"] = args.reasoning_max_tokens
    else:
        reasoning["effort"] = GUIDE_APPLICATION_REASONING["effort"]
    return reasoning


def _uses_non_openrouter_base_url(base_url: str | None) -> bool:
    resolved_base_url = (base_url or "").strip().rstrip("/")
    return bool(resolved_base_url and resolved_base_url != DEFAULT_BASE_URL)


def _resolve_override(value: str | None, fallback: str | None) -> str | None:
    return value if value is not None else fallback


def _retry_sleep_seconds(attempt: int) -> float:
    base_delay = min(2**attempt, 8)
    return base_delay + random.uniform(0, 0.5)


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"guide metadata JSON field {label} must be a non-empty string")
    return value.strip()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be greater than 0")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
