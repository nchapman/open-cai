"""Prepare SFT and DPO training datasets from local CAI data and HF anchors."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


DEFAULT_SEED = 42
DEFAULT_TEST_RATIO = 0.02
DEFAULT_OUTPUT_DIR = Path("data/training")
VALID_VIEWS = {"messages", "preference"}


class TrainingError(RuntimeError):
    """Raised when training data preparation cannot continue."""


def load_yaml_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrainingError(f"failed to read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TrainingError(f"failed to parse config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingError("training config must be a YAML object")
    return payload


def prepare_from_config(config: Mapping[str, object], *, output_dir: str | Path | None = None) -> dict[str, dict[str, int]]:
    seed = _int_config(config.get("seed", DEFAULT_SEED), "seed")
    test_ratio = _float_config(config.get("test_ratio", DEFAULT_TEST_RATIO), "test_ratio")
    if not 0 <= test_ratio < 1:
        raise TrainingError("test_ratio must be >= 0 and < 1")

    resolved_output_dir = Path(output_dir or config.get("output_dir") or DEFAULT_OUTPUT_DIR)
    summary: dict[str, dict[str, int]] = {}
    for section_name, expected_view in [("sft", "messages"), ("dpo", "preference")]:
        section = config.get(section_name)
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise TrainingError(f"{section_name} config must be an object")
        rows = prepare_section(section_name, section, expected_view=expected_view, seed=seed)
        train_rows, test_rows = split_rows(rows, test_ratio=test_ratio, seed=seed)
        section_dir = resolved_output_dir / section_name
        write_jsonl(section_dir / "train.jsonl", train_rows)
        write_jsonl(section_dir / "test.jsonl", test_rows)
        summary[section_name] = {"train": len(train_rows), "test": len(test_rows), "total": len(rows)}
    if not summary:
        raise TrainingError("config must define at least one of sft or dpo")
    return summary


def prepare_section(
    section_name: str,
    section: Mapping[str, object],
    *,
    expected_view: str,
    seed: int,
) -> list[dict[str, object]]:
    dataset_specs = section.get("datasets")
    if not isinstance(dataset_specs, list) or not dataset_specs:
        raise TrainingError(f"{section_name}.datasets must be a non-empty list")
    if not all(isinstance(item, Mapping) for item in dataset_specs):
        raise TrainingError(f"{section_name}.datasets entries must be objects")

    sample_counts = resolve_sample_counts(dataset_specs, section.get("total_samples"))
    prepared: list[dict[str, object]] = []
    skipped = 0
    for index, raw_spec in enumerate(dataset_specs):
        spec = dict(raw_spec)
        view = _string_config(spec.get("view", expected_view), f"{section_name}.datasets[{index}].view")
        if view != expected_view:
            raise TrainingError(f"{section_name}.datasets[{index}].view must be {expected_view!r}")
        rows = load_dataset_rows(spec, sample_count=sample_counts[index], seed=seed + index)
        selected = sample_rows(rows, sample_counts[index], seed=seed + index)
        dataset_prepared = 0
        for row_number, row in enumerate(selected):
            try:
                if expected_view == "messages":
                    prepared.append(normalize_messages_row(row, spec))
                else:
                    prepared.append(normalize_preference_row(row, spec))
            except TrainingError as exc:
                label = dataset_label(spec)
                skipped += 1
                print(f"skipping {section_name} row from {label} at sampled row {row_number}: {exc}", file=sys.stderr)
                continue
            dataset_prepared += 1
        if dataset_prepared == 0:
            raise TrainingError(f"{dataset_label(spec)} produced no valid {section_name} rows")

    random.Random(seed).shuffle(prepared)
    if skipped:
        print(f"skipped {skipped} malformed {section_name} rows", file=sys.stderr)
    return prepared


def resolve_sample_counts(dataset_specs: Sequence[Mapping[str, object]], total_samples: object) -> list[int | None]:
    explicit = [_optional_int_config(spec.get("samples"), "samples") for spec in dataset_specs]
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise TrainingError("either set samples on every dataset in a section or set none of them")
        return explicit

    if total_samples is None:
        return [None for _ in dataset_specs]

    total = _int_config(total_samples, "total_samples")
    if total < 1:
        raise TrainingError("total_samples must be at least 1")
    weights = [_float_config(spec.get("weight", 1.0), "weight") for spec in dataset_specs]
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise TrainingError("dataset weights must be non-negative and sum to more than 0")

    raw_counts = [total * weight / sum(weights) for weight in weights]
    counts = [int(value) for value in raw_counts]
    remainder = total - sum(counts)
    order = sorted(range(len(raw_counts)), key=lambda idx: raw_counts[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def load_dataset_rows(
    spec: Mapping[str, object],
    *,
    sample_count: int | None = None,
    seed: int = DEFAULT_SEED,
) -> list[Mapping[str, object]]:
    source = _string_config(spec.get("source"), "source")
    if source == "local":
        path = Path(_string_config(spec.get("path"), "path"))
        if sample_count is not None:
            return sample_jsonl(path, sample_count, seed=seed)
        return list(read_jsonl(path))
    if source == "hf":
        return load_hf_rows(spec, sample_count=sample_count, seed=seed)
    raise TrainingError("dataset source must be 'local' or 'hf'")


def load_hf_rows(
    spec: Mapping[str, object],
    *,
    sample_count: int | None = None,
    seed: int = DEFAULT_SEED,
) -> list[Mapping[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only with a broken install
        raise TrainingError("the datasets package is required; run `uv sync`") from exc

    dataset_id = _string_config(spec.get("id"), "id")
    config = spec.get("config")
    split = _string_config(spec.get("split", "train"), "split")
    kwargs: dict[str, object] = {"path": dataset_id, "split": split}
    if config is not None:
        kwargs["name"] = _string_config(config, "config")
    try:
        if sample_count is not None:
            dataset = load_dataset(**kwargs, streaming=True)
            if hasattr(dataset, "shuffle"):
                dataset = dataset.shuffle(seed=seed, buffer_size=max(1000, sample_count * 10))
            rows: list[Mapping[str, object]] = []
            for row in dataset:
                if isinstance(row, Mapping):
                    rows.append(row)
                if len(rows) >= sample_count:
                    break
            return rows
        dataset = load_dataset(**kwargs)
    except Exception as exc:  # noqa: BLE001 - surface dataset/cache/auth failures as CLI errors.
        raise TrainingError(f"failed to load HF dataset {dataset_id}:{split}: {exc}") from exc
    return [row for row in dataset if isinstance(row, Mapping)]


def read_jsonl(path: str | Path) -> Iterable[Mapping[str, object]]:
    jsonl_path = Path(path)
    try:
        with jsonl_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrainingError(f"{jsonl_path}:{line_number}: invalid JSONL: {exc}") from exc
                if not isinstance(row, Mapping):
                    raise TrainingError(f"{jsonl_path}:{line_number}: row must be a JSON object")
                yield row
    except OSError as exc:
        raise TrainingError(f"failed to read local dataset {jsonl_path}: {exc}") from exc


def sample_jsonl(path: str | Path, samples: int, *, seed: int) -> list[Mapping[str, object]]:
    """Reservoir sample a JSONL file without materializing the full file."""

    if samples < 1:
        raise TrainingError("samples must be at least 1")
    rng = random.Random(seed)
    reservoir: list[Mapping[str, object]] = []
    for index, row in enumerate(read_jsonl(path)):
        if len(reservoir) < samples:
            reservoir.append(row)
            continue
        replacement = rng.randint(0, index)
        if replacement < samples:
            reservoir[replacement] = row
    return reservoir


def sample_rows(rows: Sequence[Mapping[str, object]], samples: int | None, *, seed: int) -> list[Mapping[str, object]]:
    if samples is None or samples >= len(rows):
        return list(rows)
    if samples < 1:
        raise TrainingError("samples must be at least 1")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected_indices = sorted(indices[:samples])
    return [rows[index] for index in selected_indices]


def normalize_messages_row(row: Mapping[str, object], spec: Mapping[str, object]) -> dict[str, object]:
    messages = normalize_messages(row.get("messages"))
    return {
        "messages": messages,
        "source": dataset_label(spec),
    }


def normalize_preference_row(row: Mapping[str, object], spec: Mapping[str, object]) -> dict[str, object]:
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if isinstance(chosen, str) and isinstance(rejected, str):
        prompt = prompt_from_fields(row)
        chosen_messages = [assistant_message(chosen)]
        rejected_messages = [assistant_message(rejected)]
    else:
        prompt, chosen_messages, rejected_messages = split_preference_conversations(chosen, rejected, row)
    return {
        "prompt": prompt,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
        "source": dataset_label(spec),
    }


def prompt_from_fields(row: Mapping[str, object]) -> list[dict[str, str]]:
    input_text = row.get("input", row.get("prompt"))
    if not isinstance(input_text, str) or not input_text.strip():
        raise TrainingError("preference row with string chosen/rejected must include non-empty input or prompt")
    prompt: list[dict[str, str]] = []
    system = row.get("system")
    if isinstance(system, str) and system.strip():
        prompt.append({"role": "system", "content": system.strip()})
    prompt.append({"role": "user", "content": input_text.strip()})
    return prompt


def split_preference_conversations(
    chosen: object,
    rejected: object,
    row: Mapping[str, object],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    chosen_messages = normalize_messages(chosen)
    rejected_messages = normalize_messages(rejected)
    if not chosen_messages or not rejected_messages:
        raise TrainingError("chosen and rejected must be non-empty conversations")
    if chosen_messages[-1]["role"] != "assistant" or rejected_messages[-1]["role"] != "assistant":
        raise TrainingError("chosen and rejected conversations must end with assistant messages")

    chosen_prompt = chosen_messages[:-1]
    rejected_prompt = rejected_messages[:-1]
    if chosen_prompt == rejected_prompt:
        if not chosen_prompt:
            explicit_prompt = prompt_from_value(row.get("prompt"))
            if explicit_prompt:
                return explicit_prompt, [chosen_messages[-1]], [rejected_messages[-1]]
        return chosen_prompt, [chosen_messages[-1]], [rejected_messages[-1]]

    prompt_value = row.get("prompt")
    prompt = prompt_from_value(prompt_value)
    if not prompt:
        raise TrainingError("chosen/rejected prompts do not match and no usable prompt field exists")
    return prompt, [chosen_messages[-1]], [rejected_messages[-1]]


def prompt_from_value(value: object) -> list[dict[str, str]]:
    if isinstance(value, list):
        return normalize_messages(value)
    if isinstance(value, str) and value.strip():
        return [{"role": "user", "content": value.strip()}]
    return []


def normalize_messages(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TrainingError("messages must be a list")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise TrainingError(f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise TrainingError(f"message {index} has invalid role")
        if not isinstance(content, str) or not content.strip():
            raise TrainingError(f"message {index} has empty content")
        messages.append({"role": role, "content": content.strip()})
    return messages


def assistant_message(content: str) -> dict[str, str]:
    if not content.strip():
        raise TrainingError("assistant response is empty")
    return {"role": "assistant", "content": content.strip()}


def split_rows(rows: Sequence[dict[str, object]], *, test_ratio: float, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    test_count = int(round(len(shuffled) * test_ratio))
    if test_ratio > 0 and len(shuffled) > 1:
        test_count = max(1, min(test_count, len(shuffled) - 1))
    test_rows = shuffled[:test_count]
    train_rows = shuffled[test_count:]
    return train_rows, test_rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_label(spec: Mapping[str, object]) -> str:
    source = spec.get("source")
    if source == "hf":
        label = _string_config(spec.get("id"), "id")
        config = spec.get("config")
        if config is not None:
            label = f"{label}/{_string_config(config, 'config')}"
        return label
    if source == "local":
        return str(Path(_string_config(spec.get("path"), "path")))
    return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Open CAI training datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare mixed SFT and DPO JSONL files")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path)
    prepare.set_defaults(func=prepare_command)
    return parser


def prepare_command(args: argparse.Namespace) -> int:
    try:
        config = load_yaml_config(args.config)
        summary = prepare_from_config(config, output_dir=args.output_dir)
    except TrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for section, counts in summary.items():
        print(
            f"{section}: wrote {counts['train']} train, {counts['test']} test ({counts['total']} total)",
            file=sys.stderr,
        )
    return 0


def _string_config(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f"{label} must be a non-empty string")
    return value.strip()


def _int_config(value: object, label: str) -> int:
    if not isinstance(value, int):
        raise TrainingError(f"{label} must be an integer")
    return value


def _optional_int_config(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _int_config(value, label)


def _float_config(value: object, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise TrainingError(f"{label} must be a number")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
