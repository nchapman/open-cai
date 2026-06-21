"""Prepare SFT and DPO training datasets from local CAI data and HF anchors."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


DEFAULT_SEED = 42
DEFAULT_TEST_RATIO = 0.02
DEFAULT_OUTPUT_DIR = Path("data/training")
DEFAULT_SFT_OUTPUT_DIR = Path("outputs/sft")
DEFAULT_DPO_OUTPUT_DIR = Path("outputs/dpo")
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


def run_sft_from_config(
    config: Mapping[str, object],
    *,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    resume_from_checkpoint: str | bool | None = None,
) -> dict[str, object]:
    """Run or validate an SFT training config."""

    model_config = _mapping_config(config.get("model"), "model")
    data_config = _mapping_config(config.get("data"), "data")
    training_config = dict(_mapping_config(config.get("training", {}), "training"))
    sft_config = dict(_mapping_config(config.get("sft", {}), "sft"))

    train_path = Path(_string_config(data_config.get("train_path"), "data.train_path"))
    eval_path_value = data_config.get("eval_path")
    eval_path = Path(_string_config(eval_path_value, "data.eval_path")) if eval_path_value is not None and eval_is_enabled(training_config) else None
    train_rows = load_sft_messages(train_path)
    eval_rows = load_sft_messages(eval_path) if eval_path is not None else []
    dropped_train_rows = 0
    dropped_eval_rows = 0

    model_name = _string_config(model_config.get("name"), "model.name")
    resolved_output_dir = Path(output_dir or config.get("output_dir") or DEFAULT_SFT_OUTPUT_DIR / _path_safe_model_name(model_name))
    if not train_rows:
        raise TrainingError("SFT train dataset is empty")

    args_kwargs = build_sft_config_kwargs(
        training_config=training_config,
        sft_config=sft_config,
        model_config=model_config,
        output_dir=resolved_output_dir,
    )
    if _bool_config(data_config.get("drop_overlength", False), "data.drop_overlength"):
        tokenizer = load_length_tokenizer(model_config, model_name, chat_template_path=sft_config.get("chat_template_path"))
        max_length = _int_config(sft_config.get("max_length", 1024), "sft.max_length")
        num_proc = dataset_num_proc(training_config)
        train_rows, dropped_train_rows = drop_overlength_sft_rows(train_rows, tokenizer, max_length=max_length, num_proc=num_proc)
        eval_rows, dropped_eval_rows = drop_overlength_sft_rows(eval_rows, tokenizer, max_length=max_length, num_proc=num_proc)
        if not train_rows:
            raise TrainingError("SFT train dataset is empty after dropping overlength rows")
    peft_config = dict(_mapping_config(config.get("peft", {}), "peft"))
    peft_enabled = bool(peft_config.pop("enabled", False))
    summary: dict[str, object] = {
        "model": model_name,
        "output_dir": str(resolved_output_dir),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "dropped_train_rows": dropped_train_rows,
        "dropped_eval_rows": dropped_eval_rows,
        "peft": peft_enabled,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    try:
        from datasets import Dataset
        import torch
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise TrainingError("SFT training requires optional deps; run `uv sync --extra train`") from exc

    args_kwargs["model_init_kwargs"] = normalize_model_init_kwargs(args_kwargs.get("model_init_kwargs"), torch)
    checked_args_kwargs = validate_kwargs(SFTConfig, args_kwargs, "SFTConfig")
    training_args = SFTConfig(**checked_args_kwargs)
    tokenizer_name = _string_config(model_config.get("tokenizer_name") or model_name, "model.tokenizer_name")
    tokenizer_kwargs = dict(_mapping_config(model_config.get("tokenizer_kwargs", {}), "model.tokenizer_kwargs"))
    processing_class = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if processing_class.pad_token is None and processing_class.eos_token is not None:
        processing_class.pad_token = processing_class.eos_token
    trainer_kwargs: dict[str, object] = {
        "model": model_name,
        "args": training_args,
        "processing_class": processing_class,
        "train_dataset": Dataset.from_list(train_rows),
    }
    if eval_rows:
        trainer_kwargs["eval_dataset"] = Dataset.from_list(eval_rows)
    if peft_enabled:
        trainer_kwargs["peft_config"] = build_peft_config(peft_config)

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(resolved_output_dir))
    return summary


def run_dpo_from_config(
    config: Mapping[str, object],
    *,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    resume_from_checkpoint: str | bool | None = None,
) -> dict[str, object]:
    """Run or validate a DPO training config."""

    model_config = _mapping_config(config.get("model"), "model")
    data_config = _mapping_config(config.get("data"), "data")
    training_config = dict(_mapping_config(config.get("training", {}), "training"))
    dpo_config = dict(_mapping_config(config.get("dpo", {}), "dpo"))

    train_path = Path(_string_config(data_config.get("train_path"), "data.train_path"))
    eval_path_value = data_config.get("eval_path")
    eval_path = Path(_string_config(eval_path_value, "data.eval_path")) if eval_path_value is not None and eval_is_enabled(training_config) else None
    train_rows = load_dpo_preferences(train_path)
    eval_rows = load_dpo_preferences(eval_path) if eval_path is not None else []
    dropped_train_rows = 0
    dropped_eval_rows = 0

    model_name = _string_config(model_config.get("name"), "model.name")
    adapter_path = model_config.get("adapter_path")
    resolved_output_dir = Path(output_dir or config.get("output_dir") or DEFAULT_DPO_OUTPUT_DIR / _path_safe_model_name(model_name))
    if not train_rows:
        raise TrainingError("DPO train dataset is empty")

    args_kwargs = build_dpo_config_kwargs(
        training_config=training_config,
        dpo_config=dpo_config,
        model_config=model_config,
        output_dir=resolved_output_dir,
    )
    if _bool_config(data_config.get("drop_overlength", False), "data.drop_overlength"):
        tokenizer = load_length_tokenizer(model_config, model_name, chat_template_path=model_config.get("chat_template_path"))
        max_length = _int_config(dpo_config.get("max_length", 1024), "dpo.max_length")
        num_proc = dataset_num_proc(training_config)
        train_rows, dropped_train_rows = drop_overlength_dpo_rows(train_rows, tokenizer, max_length=max_length, num_proc=num_proc)
        eval_rows, dropped_eval_rows = drop_overlength_dpo_rows(eval_rows, tokenizer, max_length=max_length, num_proc=num_proc)
        if not train_rows:
            raise TrainingError("DPO train dataset is empty after dropping overlength rows")
    peft_config = dict(_mapping_config(config.get("peft", {}), "peft"))
    peft_enabled = bool(peft_config.pop("enabled", False))
    if adapter_path is not None and peft_enabled:
        raise TrainingError("use either model.adapter_path or peft.enabled, not both")

    summary: dict[str, object] = {
        "model": model_name,
        "adapter_path": str(adapter_path) if adapter_path is not None else "",
        "output_dir": str(resolved_output_dir),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "dropped_train_rows": dropped_train_rows,
        "dropped_eval_rows": dropped_eval_rows,
        "peft": peft_enabled or adapter_path is not None,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary

    try:
        from datasets import Dataset
        import torch
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise TrainingError("DPO training requires optional deps; run `uv sync --extra train`") from exc

    args_kwargs["model_init_kwargs"] = normalize_model_init_kwargs(args_kwargs.get("model_init_kwargs"), torch)
    if adapter_path is not None:
        model_init_kwargs = args_kwargs.pop("model_init_kwargs")
    else:
        model_init_kwargs = args_kwargs["model_init_kwargs"]
    checked_args_kwargs = validate_kwargs(DPOConfig, args_kwargs, "DPOConfig")
    training_args = DPOConfig(**checked_args_kwargs)
    processing_class = load_tokenizer(model_config, model_name, padding_side="left")
    model = load_train_model(model_config, model_name, model_init_kwargs)

    trainer_kwargs: dict[str, object] = {
        "model": model,
        "args": training_args,
        "processing_class": processing_class,
        "train_dataset": Dataset.from_list(train_rows),
    }
    if eval_rows:
        trainer_kwargs["eval_dataset"] = Dataset.from_list(eval_rows)
    if peft_enabled:
        trainer_kwargs["peft_config"] = build_peft_config(peft_config)

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(resolved_output_dir))
    return summary


def load_sft_messages(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_number, row in enumerate(read_jsonl(path), start=1):
        try:
            rows.append({"messages": normalize_messages(row.get("messages"))})
        except TrainingError as exc:
            raise TrainingError(f"{path}:{row_number}: invalid SFT row: {exc}") from exc
    return rows


def load_dpo_preferences(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_number, row in enumerate(read_jsonl(path), start=1):
        try:
            prompt = normalize_messages(row.get("prompt"))
            chosen = normalize_messages(row.get("chosen"))
            rejected = normalize_messages(row.get("rejected"))
            if not prompt:
                raise TrainingError("prompt is empty")
            if not chosen or chosen[-1]["role"] != "assistant":
                raise TrainingError("chosen must end with an assistant message")
            if not rejected or rejected[-1]["role"] != "assistant":
                raise TrainingError("rejected must end with an assistant message")
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        except TrainingError as exc:
            raise TrainingError(f"{path}:{row_number}: invalid DPO row: {exc}") from exc
    return rows


def analyze_sft_lengths_from_config(config: Mapping[str, object]) -> dict[str, object]:
    model_config = _mapping_config(config.get("model"), "model")
    data_config = _mapping_config(config.get("data"), "data")
    sft_config = _mapping_config(config.get("sft", {}), "sft")
    train_path = Path(_string_config(data_config.get("train_path"), "data.train_path"))
    model_name = _string_config(model_config.get("name"), "model.name")
    tokenizer = load_length_tokenizer(model_config, model_name, chat_template_path=sft_config.get("chat_template_path"))
    max_length = _int_config(sft_config.get("max_length", 1024), "sft.max_length")

    lengths = []
    for row in load_sft_messages(train_path):
        lengths.append(len(tokenizer.apply_chat_template(row["messages"], tokenize=True, return_dict=False)))
    return {
        "kind": "sft",
        "train_path": str(train_path),
        "max_length": max_length,
        "drop_overlength": _bool_config(data_config.get("drop_overlength", False), "data.drop_overlength"),
        "lengths": summarize_lengths(lengths, caps=[max_length]),
    }


def analyze_dpo_lengths_from_config(config: Mapping[str, object]) -> dict[str, object]:
    model_config = _mapping_config(config.get("model"), "model")
    data_config = _mapping_config(config.get("data"), "data")
    dpo_config = _mapping_config(config.get("dpo", {}), "dpo")
    train_path = Path(_string_config(data_config.get("train_path"), "data.train_path"))
    model_name = _string_config(model_config.get("name"), "model.name")
    tokenizer = load_length_tokenizer(model_config, model_name, chat_template_path=model_config.get("chat_template_path"))
    max_length = _int_config(dpo_config.get("max_length", 1024), "dpo.max_length")

    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    max_pair_lengths: list[int] = []
    prefix_mismatches = 0
    for row in load_dpo_preferences(train_path):
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        chosen_ids = tokenizer.apply_chat_template(row["prompt"] + row["chosen"], tokenize=True, return_dict=False)
        rejected_ids = tokenizer.apply_chat_template(row["prompt"] + row["rejected"], tokenize=True, return_dict=False)
        if chosen_ids[: len(prompt_ids)] != prompt_ids or rejected_ids[: len(prompt_ids)] != prompt_ids:
            prefix_mismatches += 1
        prompt_lengths.append(len(prompt_ids))
        chosen_lengths.append(len(chosen_ids))
        rejected_lengths.append(len(rejected_ids))
        max_pair_lengths.append(max(len(chosen_ids), len(rejected_ids)))
    return {
        "kind": "dpo",
        "train_path": str(train_path),
        "max_length": max_length,
        "drop_overlength": _bool_config(data_config.get("drop_overlength", False), "data.drop_overlength"),
        "prefix_mismatches": prefix_mismatches,
        "prompt_lengths": summarize_lengths(prompt_lengths, caps=[max_length]),
        "chosen_lengths": summarize_lengths(chosen_lengths, caps=[max_length]),
        "rejected_lengths": summarize_lengths(rejected_lengths, caps=[max_length]),
        "max_pair_lengths": summarize_lengths(max_pair_lengths, caps=[max_length]),
    }


def load_length_tokenizer(model_config: Mapping[str, object], model_name: str, *, chat_template_path: object) -> object:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TrainingError("length analysis requires optional deps; run `uv sync --extra train`") from exc

    tokenizer_name = _string_config(model_config.get("tokenizer_name", model_name), "model.tokenizer_name")
    tokenizer_kwargs = dict(_mapping_config(model_config.get("tokenizer_kwargs", {}), "model.tokenizer_kwargs"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if chat_template_path is not None:
        tokenizer.chat_template = Path(_string_config(chat_template_path, "chat_template_path")).read_text(encoding="utf-8")
    return tokenizer


def drop_overlength_sft_rows(
    rows: Sequence[dict[str, object]],
    tokenizer: object,
    *,
    max_length: int,
    num_proc: int = 1,
) -> tuple[list[dict[str, object]], int]:
    if num_proc > 1 and rows:
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise TrainingError("parallel length filtering requires optional deps; run `uv sync --extra train`") from exc

        dataset = Dataset.from_list(list(rows))

        def add_keep_flag(batch: Mapping[str, Sequence[object]]) -> dict[str, list[bool]]:
            return {
                "_keep": [
                    len(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)) <= max_length
                    for messages in batch["messages"]
                ]
            }

        filtered = dataset.map(add_keep_flag, batched=True, num_proc=num_proc, desc="Filtering overlength SFT rows")
        filtered = filtered.filter(lambda keep: keep, input_columns="_keep", num_proc=num_proc, desc="Keeping SFT rows")
        kept_rows = [{"messages": row["messages"]} for row in filtered.remove_columns("_keep").to_list()]
        return kept_rows, len(rows) - len(kept_rows)

    kept = []
    dropped = 0
    for row in rows:
        if len(tokenizer.apply_chat_template(row["messages"], tokenize=True, return_dict=False)) > max_length:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def drop_overlength_dpo_rows(
    rows: Sequence[dict[str, object]],
    tokenizer: object,
    *,
    max_length: int,
    num_proc: int = 1,
) -> tuple[list[dict[str, object]], int]:
    if num_proc > 1 and rows:
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise TrainingError("parallel length filtering requires optional deps; run `uv sync --extra train`") from exc

        dataset = Dataset.from_list(list(rows))

        def add_keep_flag(batch: Mapping[str, Sequence[object]]) -> dict[str, list[bool]]:
            keep = []
            for prompt, chosen, rejected in zip(batch["prompt"], batch["chosen"], batch["rejected"], strict=True):
                chosen_length = len(tokenizer.apply_chat_template(prompt + chosen, tokenize=True, return_dict=False))
                rejected_length = len(tokenizer.apply_chat_template(prompt + rejected, tokenize=True, return_dict=False))
                keep.append(max(chosen_length, rejected_length) <= max_length)
            return {"_keep": keep}

        filtered = dataset.map(add_keep_flag, batched=True, num_proc=num_proc, desc="Filtering overlength DPO rows")
        filtered = filtered.filter(lambda keep: keep, input_columns="_keep", num_proc=num_proc, desc="Keeping DPO rows")
        kept_rows = [
            {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
            for row in filtered.remove_columns("_keep").to_list()
        ]
        return kept_rows, len(rows) - len(kept_rows)

    kept = []
    dropped = 0
    for row in rows:
        chosen_length = len(tokenizer.apply_chat_template(row["prompt"] + row["chosen"], tokenize=True, return_dict=False))
        rejected_length = len(
            tokenizer.apply_chat_template(row["prompt"] + row["rejected"], tokenize=True, return_dict=False)
        )
        if max(chosen_length, rejected_length) > max_length:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def eval_is_enabled(training_config: Mapping[str, object]) -> bool:
    if "eval_strategy" not in training_config and "evaluation_strategy" not in training_config:
        return True
    strategy = training_config.get("eval_strategy", training_config.get("evaluation_strategy"))
    return str(strategy).lower() not in {"no", "none", "false"}


def dataset_num_proc(training_config: Mapping[str, object]) -> int:
    value = training_config.get("dataset_num_proc", 1)
    if value is None:
        return 1
    return max(1, _int_config(value, "training.dataset_num_proc"))


def summarize_lengths(values: Sequence[int], *, caps: Sequence[int]) -> dict[str, object]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    summary: dict[str, object] = {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p50": percentile(sorted_values, 0.50),
        "p75": percentile(sorted_values, 0.75),
        "p90": percentile(sorted_values, 0.90),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "over_cap": {},
    }
    over_cap: dict[int, dict[str, object]] = {}
    for cap in caps:
        count = sum(1 for value in sorted_values if value > cap)
        over_cap[cap] = {"count": count, "percent": count * 100 / len(sorted_values)}
    summary["over_cap"] = over_cap
    return summary


def percentile(sorted_values: Sequence[int], q: float) -> int:
    index = max(0, min(len(sorted_values) - 1, int(q * len(sorted_values) + 0.999999) - 1))
    return sorted_values[index]


def build_sft_config_kwargs(
    *,
    training_config: Mapping[str, object],
    sft_config: Mapping[str, object],
    model_config: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    args: dict[str, object] = {
        "output_dir": str(output_dir),
        "report_to": "none",
        "save_strategy": "steps",
        "eval_strategy": "no",
    }
    args.update(training_config)
    args.update(sft_config)

    model_init_kwargs = dict(_mapping_config(model_config.get("init_kwargs", {}), "model.init_kwargs"))
    if model_init_kwargs:
        args["model_init_kwargs"] = model_init_kwargs
    return args


def build_dpo_config_kwargs(
    *,
    training_config: Mapping[str, object],
    dpo_config: Mapping[str, object],
    model_config: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    args: dict[str, object] = {
        "output_dir": str(output_dir),
        "report_to": "none",
        "save_strategy": "steps",
        "eval_strategy": "no",
    }
    args.update(training_config)
    args.update(dpo_config)

    model_init_kwargs = dict(_mapping_config(model_config.get("init_kwargs", {}), "model.init_kwargs"))
    if model_init_kwargs:
        args["model_init_kwargs"] = model_init_kwargs
    return args


def load_tokenizer(model_config: Mapping[str, object], model_name: str, *, padding_side: str) -> object:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TrainingError("tokenizer loading requires optional deps; run `uv sync --extra train`") from exc

    tokenizer_name = _string_config(model_config.get("tokenizer_name", model_name), "model.tokenizer_name")
    tokenizer_kwargs = dict(_mapping_config(model_config.get("tokenizer_kwargs", {}), "model.tokenizer_kwargs"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    chat_template_path = model_config.get("chat_template_path")
    if chat_template_path is not None:
        tokenizer.chat_template = Path(_string_config(chat_template_path, "model.chat_template_path")).read_text(encoding="utf-8")
    return tokenizer


def load_train_model(model_config: Mapping[str, object], model_name: str, model_init_kwargs: object) -> object:
    adapter_path = model_config.get("adapter_path")
    if adapter_path is None:
        return model_name
    try:
        from peft import PeftModel
        from trl.trainer.dpo_trainer import create_model_from_path
    except ImportError as exc:
        raise TrainingError("adapter training requires optional deps; run `uv sync --extra train`") from exc

    kwargs = dict(_mapping_config(model_init_kwargs or {}, "model_init_kwargs"))
    if is_distributed_training():
        kwargs["device_map"] = None
    base_model = create_model_from_path(model_name, **kwargs)
    return PeftModel.from_pretrained(
        base_model,
        _string_config(adapter_path, "model.adapter_path"),
        is_trainable=True,
    )


def is_distributed_training() -> bool:
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except ValueError:
        return False


def build_peft_config(config: Mapping[str, object]) -> object:
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise TrainingError("LoRA training requires optional deps; run `uv sync --extra train`") from exc

    kwargs = {
        "r": _int_config(config.get("r", 16), "peft.r"),
        "lora_alpha": _int_config(config.get("lora_alpha", 32), "peft.lora_alpha"),
        "lora_dropout": _float_config(config.get("lora_dropout", 0.05), "peft.lora_dropout"),
        "bias": _string_config(config.get("bias", "none"), "peft.bias"),
        "task_type": _string_config(config.get("task_type", "CAUSAL_LM"), "peft.task_type"),
    }
    target_modules = config.get("target_modules")
    if target_modules is not None:
        kwargs["target_modules"] = _string_list_config(target_modules, "peft.target_modules")
    return LoraConfig(**kwargs)


def normalize_model_init_kwargs(value: object, torch_module: object) -> object:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TrainingError("model_init_kwargs must be an object")
    normalized = dict(value)
    dtype_key = "dtype" if "dtype" in normalized else "torch_dtype" if "torch_dtype" in normalized else None
    if dtype_key is not None and isinstance(normalized[dtype_key], str):
        dtype = normalized[dtype_key].strip()
        if dtype != "auto":
            dtype_map = {
                "bfloat16": getattr(torch_module, "bfloat16"),
                "float16": getattr(torch_module, "float16"),
                "float32": getattr(torch_module, "float32"),
            }
            if dtype not in dtype_map:
                raise TrainingError(f"unsupported model dtype {dtype!r}")
            normalized[dtype_key] = dtype_map[dtype]
    return normalized


def validate_kwargs(callable_object: object, kwargs: Mapping[str, object], label: str) -> dict[str, object]:
    signature = inspect.signature(callable_object)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    allowed = {name for name, parameter in parameters.items() if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TrainingError(f"{label} does not support config keys: {', '.join(unknown)}")
    return dict(kwargs)


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
    skipped_unusable_preferences = 0
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
                if expected_view == "preference" and str(exc) == "preference row is marked unusable":
                    skipped_unusable_preferences += 1
                    continue
                print(f"skipping {section_name} row from {label} at sampled row {row_number}: {exc}", file=sys.stderr)
                continue
            dataset_prepared += 1
        if dataset_prepared == 0:
            raise TrainingError(f"{dataset_label(spec)} produced no valid {section_name} rows")

    random.Random(seed).shuffle(prepared)
    malformed = skipped - skipped_unusable_preferences
    if skipped_unusable_preferences:
        print(f"skipped {skipped_unusable_preferences} unusable {section_name} rows", file=sys.stderr)
    if malformed:
        print(f"skipped {malformed} malformed {section_name} rows", file=sys.stderr)
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
    if row.get("preference_usable") is False:
        raise TrainingError("preference row is marked unusable")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if isinstance(chosen, str) and isinstance(rejected, str):
        if chosen.strip() == rejected.strip():
            raise TrainingError("chosen and rejected responses are identical")
        prompt = prompt_from_fields(row)
        chosen_messages = [assistant_message(chosen)]
        rejected_messages = [assistant_message(rejected)]
    else:
        prompt, chosen_messages, rejected_messages = split_preference_conversations(chosen, rejected, row)
    if chosen_messages[-1]["content"] == rejected_messages[-1]["content"]:
        raise TrainingError("chosen and rejected responses are identical")
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

    sft = subparsers.add_parser("sft", help="run supervised fine-tuning from prepared messages JSONL")
    sft.add_argument("--config", type=Path, required=True)
    sft.add_argument("--output-dir", type=Path)
    sft.add_argument("--dry-run", action="store_true", help="validate config and data without loading the model")
    sft.add_argument("--resume-from-checkpoint", nargs="?", const=True)
    sft.set_defaults(func=sft_command)

    dpo = subparsers.add_parser("dpo", help="run direct preference optimization from prepared preference JSONL")
    dpo.add_argument("--config", type=Path, required=True)
    dpo.add_argument("--output-dir", type=Path)
    dpo.add_argument("--dry-run", action="store_true", help="validate config and data without loading the model")
    dpo.add_argument("--resume-from-checkpoint", nargs="?", const=True)
    dpo.set_defaults(func=dpo_command)

    lengths = subparsers.add_parser("lengths", help="analyze configured training token lengths")
    lengths.add_argument("--config", type=Path, required=True)
    lengths.add_argument("--fail-on-truncation", action="store_true", help="exit non-zero if any row exceeds max_length")
    lengths.set_defaults(func=lengths_command)
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


def sft_command(args: argparse.Namespace) -> int:
    try:
        config = load_yaml_config(args.config)
        summary = run_sft_from_config(
            config,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except TrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    mode = "validated" if summary["dry_run"] else "trained"
    dropped = dropped_rows_label(summary)
    print(
        f"sft: {mode} {summary['train_rows']} train, {summary['eval_rows']} eval rows{dropped} "
        f"for {summary['model']} -> {summary['output_dir']}",
        file=sys.stderr,
    )
    return 0


def dpo_command(args: argparse.Namespace) -> int:
    try:
        config = load_yaml_config(args.config)
        summary = run_dpo_from_config(
            config,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except TrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    mode = "validated" if summary["dry_run"] else "trained"
    adapter = f" from {summary['adapter_path']}" if summary["adapter_path"] else ""
    dropped = dropped_rows_label(summary)
    print(
        f"dpo: {mode} {summary['train_rows']} train, {summary['eval_rows']} eval rows{dropped} "
        f"for {summary['model']}{adapter} -> {summary['output_dir']}",
        file=sys.stderr,
    )
    return 0


def dropped_rows_label(summary: Mapping[str, object]) -> str:
    dropped_train = _int_config(summary.get("dropped_train_rows", 0), "dropped_train_rows")
    dropped_eval = _int_config(summary.get("dropped_eval_rows", 0), "dropped_eval_rows")
    if not dropped_train and not dropped_eval:
        return ""
    return f" (dropped {dropped_train} train, {dropped_eval} eval overlength rows)"


def lengths_command(args: argparse.Namespace) -> int:
    try:
        config = load_yaml_config(args.config)
        if "sft" in config:
            report = analyze_sft_lengths_from_config(config)
        elif "dpo" in config:
            report = analyze_dpo_lengths_from_config(config)
        else:
            raise TrainingError("length analysis config must define sft or dpo")
    except TrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_length_report(report)
    if args.fail_on_truncation and length_report_has_truncation(report):
        print("error: configured max_length would truncate training rows", file=sys.stderr)
        return 1
    return 0


def print_length_report(report: Mapping[str, object]) -> None:
    kind = _string_config(report.get("kind"), "kind")
    print(f"{kind}: {report['train_path']}")
    print(f"max_length: {report['max_length']}")
    print(f"drop_overlength: {report['drop_overlength']}")
    if kind == "sft":
        print_length_summary("messages", _mapping_config(report.get("lengths"), "lengths"))
        return

    print(f"prefix_mismatches: {report['prefix_mismatches']}")
    print_length_summary("prompt", _mapping_config(report.get("prompt_lengths"), "prompt_lengths"))
    print_length_summary("chosen_total", _mapping_config(report.get("chosen_lengths"), "chosen_lengths"))
    print_length_summary("rejected_total", _mapping_config(report.get("rejected_lengths"), "rejected_lengths"))
    print_length_summary("max_pair_total", _mapping_config(report.get("max_pair_lengths"), "max_pair_lengths"))


def print_length_summary(label: str, summary: Mapping[str, object]) -> None:
    count = _int_config(summary.get("count"), f"{label}.count")
    if count == 0:
        print(f"{label}: empty")
        return
    print(
        f"{label}: n={count} min={summary['min']} p50={summary['p50']} p75={summary['p75']} "
        f"p90={summary['p90']} p95={summary['p95']} p99={summary['p99']} max={summary['max']}"
    )
    over_cap = _mapping_config(summary.get("over_cap"), f"{label}.over_cap")
    for cap, raw in over_cap.items():
        cap_summary = _mapping_config(raw, f"{label}.over_cap.{cap}")
        over_count = _int_config(cap_summary.get("count"), f"{label}.over_cap.{cap}.count")
        over_percent = _float_config(cap_summary.get("percent"), f"{label}.over_cap.{cap}.percent")
        print(f"{label}: >{cap} {over_count} ({over_percent:.2f}%)")


def length_report_has_truncation(report: Mapping[str, object]) -> bool:
    if _bool_config(report.get("drop_overlength", False), "drop_overlength"):
        return False
    summaries = ["lengths"] if report.get("kind") == "sft" else ["max_pair_lengths"]
    for key in summaries:
        summary = _mapping_config(report.get(key), key)
        over_cap = _mapping_config(summary.get("over_cap"), f"{key}.over_cap")
        for raw in over_cap.values():
            cap_summary = _mapping_config(raw, f"{key}.over_cap")
            if _int_config(cap_summary.get("count"), f"{key}.count") > 0:
                return True
    return False


def _string_config(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingError(f"{label} must be a non-empty string")
    return value.strip()


def _string_list_config(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise TrainingError(f"{label} must be a non-empty list")
    strings = [_string_config(item, label) for item in value]
    return strings


def _mapping_config(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrainingError(f"{label} must be an object")
    return value


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


def _bool_config(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingError(f"{label} must be a boolean")
    return value


def _path_safe_model_name(value: str) -> str:
    return value.strip().replace("/", "-")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
