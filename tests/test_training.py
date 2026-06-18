from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from cai.training import (
    TrainingError,
    normalize_messages_row,
    normalize_preference_row,
    normalize_model_init_kwargs,
    prepare_from_config,
    resolve_sample_counts,
    run_dpo_from_config,
    run_sft_from_config,
    sample_jsonl,
)


class TrainingTests(unittest.TestCase):
    def test_resolves_total_samples_by_weight(self) -> None:
        counts = resolve_sample_counts(
            [
                {"source": "hf", "id": "a", "weight": 0.8},
                {"source": "local", "path": "b", "weight": 0.2},
            ],
            10,
        )

        self.assertEqual(counts, [8, 2])

    def test_requires_all_or_no_explicit_samples(self) -> None:
        with self.assertRaisesRegex(TrainingError, "samples on every dataset"):
            resolve_sample_counts(
                [
                    {"source": "hf", "id": "a", "samples": 1},
                    {"source": "local", "path": "b"},
                ],
                None,
            )

    def test_normalizes_cai_messages_row(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": " Hello "},
                {"role": "assistant", "content": " Hi "},
            ]
        }

        normalized = normalize_messages_row(row, {"source": "local", "path": "data/cai.jsonl"})

        self.assertEqual(normalized["messages"][0]["content"], "Hello")  # type: ignore[index]
        self.assertEqual(normalized["source"], "data/cai.jsonl")

    def test_normalizes_full_conversation_preference_row_to_explicit_prompt(self) -> None:
        row = {
            "chosen": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Good"},
            ],
            "rejected": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Bad"},
            ],
        }

        normalized = normalize_preference_row(row, {"source": "hf", "id": "prefs"})

        self.assertEqual(normalized["prompt"], [{"role": "user", "content": "Question"}])
        self.assertEqual(normalized["chosen"], [{"role": "assistant", "content": "Good"}])
        self.assertEqual(normalized["rejected"], [{"role": "assistant", "content": "Bad"}])

    def test_normalizes_explicit_prompt_preference_row(self) -> None:
        row = {
            "prompt": [{"role": "user", "content": "Question"}],
            "chosen": [{"role": "assistant", "content": "Good"}],
            "rejected": [{"role": "assistant", "content": "Bad"}],
        }

        normalized = normalize_preference_row(row, {"source": "hf", "id": "prefs"})

        self.assertEqual(normalized["prompt"], [{"role": "user", "content": "Question"}])
        self.assertEqual(normalized["chosen"], [{"role": "assistant", "content": "Good"}])
        self.assertEqual(normalized["rejected"], [{"role": "assistant", "content": "Bad"}])

    def test_normalizes_string_preference_row_with_system_prompt(self) -> None:
        row = {
            "system": "You are helpful.",
            "input": "Question",
            "chosen": "Good",
            "rejected": "Bad",
        }

        normalized = normalize_preference_row(row, {"source": "hf", "id": "orca"})

        self.assertEqual(
            normalized["prompt"],
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Question"},
            ],
        )
        self.assertEqual(normalized["chosen"], [{"role": "assistant", "content": "Good"}])

    def test_prepare_from_config_writes_splits_from_local_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sft_path = root / "sft.jsonl"
            dpo_path = root / "dpo.jsonl"
            output_dir = root / "prepared"
            sft_rows = [
                {"messages": [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]}
                for i in range(4)
            ]
            dpo_rows = [
                {
                    "chosen": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"good{i}"}],
                    "rejected": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"bad{i}"}],
                }
                for i in range(4)
            ]
            sft_path.write_text("\n".join(json.dumps(row) for row in sft_rows), encoding="utf-8")
            dpo_path.write_text("\n".join(json.dumps(row) for row in dpo_rows), encoding="utf-8")

            summary = prepare_from_config(
                {
                    "seed": 1,
                    "test_ratio": 0.25,
                    "output_dir": str(output_dir),
                    "sft": {"datasets": [{"source": "local", "path": str(sft_path), "view": "messages"}]},
                    "dpo": {"datasets": [{"source": "local", "path": str(dpo_path), "view": "preference"}]},
                }
            )

            self.assertEqual(summary["sft"], {"train": 3, "test": 1, "total": 4})
            self.assertEqual(summary["dpo"], {"train": 3, "test": 1, "total": 4})
            self.assertEqual(len((output_dir / "sft" / "train.jsonl").read_text(encoding="utf-8").splitlines()), 3)
            self.assertEqual(len((output_dir / "dpo" / "test.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    def test_prepare_from_config_loads_hf_sources(self) -> None:
        fake_dataset = [
            {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]},
        ]
        with tempfile.TemporaryDirectory() as tmp, mock.patch("datasets.load_dataset", return_value=fake_dataset) as load:
            summary = prepare_from_config(
                {
                    "output_dir": tmp,
                    "test_ratio": 0,
                    "sft": {
                        "datasets": [
                            {
                                "source": "hf",
                                "id": "HuggingFaceTB/smoltalk",
                                "config": "smol-magpie-ultra",
                                "split": "train",
                                "view": "messages",
                            }
                        ]
                    },
                }
            )

        self.assertEqual(summary["sft"], {"train": 1, "test": 0, "total": 1})
        load.assert_called_once_with(path="HuggingFaceTB/smoltalk", name="smol-magpie-ultra", split="train")

    def test_prepare_from_config_skips_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("sys.stderr"):
            path = Path(tmp) / "sft.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"messages": [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "yes"}]}),
                        json.dumps({"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "bad"}]}),
                    ]
                ),
                encoding="utf-8",
            )

            summary = prepare_from_config(
                {
                    "output_dir": tmp,
                    "test_ratio": 0,
                    "sft": {"datasets": [{"source": "local", "path": str(path), "view": "messages"}]},
                }
            )

        self.assertEqual(summary["sft"], {"train": 1, "test": 0, "total": 1})

    def test_sample_jsonl_does_not_require_full_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": str(i)}]}) for i in range(20)), encoding="utf-8")

            rows = sample_jsonl(path, 5, seed=123)

        self.assertEqual(len(rows), 5)
        self.assertEqual(len({row["messages"][0]["content"] for row in rows}), 5)  # type: ignore[index]

    def test_sft_dry_run_validates_prepared_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.jsonl"
            eval_path = root / "test.jsonl"
            train_path.write_text(
                json.dumps({"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}),
                encoding="utf-8",
            )
            eval_path.write_text(
                json.dumps({"messages": [{"role": "user", "content": "eu"}, {"role": "assistant", "content": "ea"}]}),
                encoding="utf-8",
            )

            summary = run_sft_from_config(
                {
                    "output_dir": str(root / "out"),
                    "model": {"name": "Qwen/Qwen3-0.6B"},
                    "data": {"train_path": str(train_path), "eval_path": str(eval_path)},
                    "training": {"max_steps": 1},
                    "sft": {"max_length": 512, "assistant_only_loss": True},
                    "peft": {"enabled": True, "r": 8},
                },
                dry_run=True,
            )

        self.assertEqual(summary["train_rows"], 1)
        self.assertEqual(summary["eval_rows"], 1)
        self.assertEqual(summary["model"], "Qwen/Qwen3-0.6B")
        self.assertTrue(summary["peft"])
        self.assertTrue(summary["dry_run"])

    def test_sft_rejects_invalid_messages_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.jsonl"
            train_path.write_text(json.dumps({"messages": [{"role": "user", "content": ""}]}), encoding="utf-8")

            with self.assertRaisesRegex(TrainingError, "invalid SFT row"):
                run_sft_from_config(
                    {
                        "model": {"name": "Qwen/Qwen3-0.6B"},
                        "data": {"train_path": str(train_path)},
                    },
                    dry_run=True,
                )

    def test_normalizes_yaml_dtype_for_model_init_kwargs(self) -> None:
        class FakeTorch:
            bfloat16 = object()
            float16 = object()
            float32 = object()

        normalized = normalize_model_init_kwargs({"dtype": "bfloat16"}, FakeTorch)

        self.assertEqual(normalized, {"dtype": FakeTorch.bfloat16})

    def test_dpo_dry_run_validates_prepared_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.jsonl"
            eval_path = root / "test.jsonl"
            row = {
                "prompt": [{"role": "user", "content": "Question"}],
                "chosen": [{"role": "assistant", "content": "Good"}],
                "rejected": [{"role": "assistant", "content": "Bad"}],
            }
            train_path.write_text(json.dumps(row), encoding="utf-8")
            eval_path.write_text(json.dumps(row), encoding="utf-8")

            summary = run_dpo_from_config(
                {
                    "output_dir": str(root / "out"),
                    "model": {"name": "tvall43/Qwen3.5-4B-heretic", "adapter_path": str(root / "sft")},
                    "data": {"train_path": str(train_path), "eval_path": str(eval_path)},
                    "training": {"max_steps": 1},
                    "dpo": {"beta": 0.1, "max_length": 512},
                },
                dry_run=True,
            )

        self.assertEqual(summary["train_rows"], 1)
        self.assertEqual(summary["eval_rows"], 1)
        self.assertEqual(summary["model"], "tvall43/Qwen3.5-4B-heretic")
        self.assertEqual(summary["adapter_path"], str(root / "sft"))
        self.assertTrue(summary["peft"])

    def test_dpo_rejects_invalid_preference_rows_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.jsonl"
            train_path.write_text(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": "Question"}],
                        "chosen": [{"role": "user", "content": "Wrong role"}],
                        "rejected": [{"role": "assistant", "content": "Bad"}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TrainingError, "invalid DPO row"):
                run_dpo_from_config(
                    {
                        "model": {"name": "tvall43/Qwen3.5-4B-heretic"},
                        "data": {"train_path": str(train_path)},
                    },
                    dry_run=True,
                )

    def test_dpo_rejects_adapter_path_with_new_peft_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.jsonl"
            train_path.write_text(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": "Question"}],
                        "chosen": [{"role": "assistant", "content": "Good"}],
                        "rejected": [{"role": "assistant", "content": "Bad"}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TrainingError, "adapter_path or peft.enabled"):
                run_dpo_from_config(
                    {
                        "model": {"name": "tvall43/Qwen3.5-4B-heretic", "adapter_path": str(root / "sft")},
                        "data": {"train_path": str(train_path)},
                        "peft": {"enabled": True},
                    },
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
