import tempfile
import unittest
from pathlib import Path

from cai.evaluation import (
    EvaluationError,
    group_eval_models,
    is_refusal,
    parse_capability_config,
    parse_generation_config,
    parse_model_specs,
    parse_prompt_specs,
    parse_refusal_config,
    render_markdown_report,
    render_suite_summary,
    run_eval_from_config,
    run_suite_from_config,
    strip_thinking_blocks,
)


class EvaluationTests(unittest.TestCase):
    def test_eval_dry_run_validates_config_shape(self) -> None:
        summary = run_eval_from_config(
            {
                "output_dir": "outputs/eval/test",
                "models": [{"label": "base", "name": "model"}],
                "prompts": [{"id": "helpful", "prompt": "Hello"}],
                "generation": {"max_new_tokens": 12, "temperature": 0.0},
            },
            dry_run=True,
        )

        self.assertEqual(summary["models"], 1)
        self.assertEqual(summary["prompts"], 1)
        self.assertTrue(summary["dry_run"])

    def test_suite_dry_run_validates_enabled_sections(self) -> None:
        summary = run_suite_from_config(
            {
                "output_dir": "outputs/eval/fast",
                "models": [{"label": "base", "name": "model"}],
                "refusals": {
                    "dataset": "mlabonne/harmful_behaviors",
                    "split": "test[:2]",
                    "column": "text",
                },
                "drift": {
                    "dataset": "mlabonne/harmless_alpaca",
                    "split": "test[:2]",
                    "column": "text",
                },
            },
            dry_run=True,
        )

        self.assertEqual(summary["models"], 1)
        self.assertEqual(summary["sections"], ["refusals", "drift"])
        self.assertTrue(summary["dry_run"])

    def test_parse_prompt_specs_accepts_messages_or_prompt(self) -> None:
        prompts = parse_prompt_specs(
            [
                {"id": "plain", "prompt": "Hello"},
                {"id": "messages", "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]},
            ]
        )

        self.assertEqual(prompts[0]["messages"], [{"role": "user", "content": "Hello"}])
        self.assertEqual(prompts[1]["messages"][0]["role"], "system")  # type: ignore[index]

    def test_parse_model_specs_rejects_duplicate_labels(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "duplicate model label"):
            parse_model_specs([{"label": "base", "name": "a"}, {"label": "base", "name": "b"}])

    def test_group_eval_models_reuses_shared_base_model(self) -> None:
        models = parse_model_specs(
            [
                {"label": "base", "name": "model", "init_kwargs": {"dtype": "bfloat16"}},
                {"label": "sft", "name": "model", "adapter_path": "outputs/sft", "init_kwargs": {"dtype": "bfloat16"}},
                {"label": "other", "name": "other-model", "init_kwargs": {"dtype": "bfloat16"}},
            ]
        )

        groups = group_eval_models(models)

        self.assertEqual([[model["label"] for model in group] for group in groups], [["base", "sft"], ["other"]])

    def test_parse_generation_config_rejects_invalid_max_new_tokens(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "max_new_tokens"):
            parse_generation_config({"max_new_tokens": 0})

    def test_parse_capability_config_accepts_task_limits(self) -> None:
        parsed = parse_capability_config({"tasks": [{"task": "hellaswag", "limit": 10}, "ifeval"]})

        self.assertEqual(parsed["tasks"][0], {"task": "hellaswag", "limit": 10})  # type: ignore[index]
        self.assertEqual(parsed["tasks"][1], {"task": "ifeval"})  # type: ignore[index]

    def test_parse_refusal_config_uses_default_markers(self) -> None:
        parsed = parse_refusal_config({"dataset": "bad", "split": "test[:2]", "column": "text"})

        self.assertEqual(parsed["dataset"], "bad")
        self.assertIn("i cannot", parsed["markers"])

    def test_is_refusal_matches_heretic_style_markers(self) -> None:
        self.assertTrue(is_refusal("I'm sorry, I cannot help with that.", ["sorry", "i cannot"]))
        self.assertTrue(is_refusal("", ["sorry"]))
        self.assertFalse(is_refusal("Here is a normal answer.", ["sorry"]))

    def test_strip_thinking_blocks_removes_visible_reasoning(self) -> None:
        self.assertEqual(strip_thinking_blocks("<think>\nnotes\n</think>\n\nFinal"), "Final")
        self.assertEqual(strip_thinking_blocks("Before <think>notes</think> after"), "Before  after")
        self.assertEqual(strip_thinking_blocks("<think>unfinished"), "<think>unfinished")

    def test_render_markdown_report_groups_by_prompt_then_model(self) -> None:
        models = [{"label": "base"}, {"label": "sft"}]
        prompts = [{"id": "p1", "messages": [{"role": "user", "content": "Question"}]}]
        records = [
            {"prompt_id": "p1", "model_label": "base", "response": "Base answer"},
            {"prompt_id": "p1", "model_label": "sft", "response": "SFT answer"},
        ]

        report = render_markdown_report(records, models=models, prompts=prompts)

        self.assertIn("## p1", report)
        self.assertIn("### base", report)
        self.assertIn("Base answer", report)
        self.assertIn("### sft", report)
        self.assertIn("SFT answer", report)

    def test_render_suite_summary_includes_key_tables(self) -> None:
        report = render_suite_summary(
            {
                "capability": {
                    "path": "capability.jsonl",
                    "metrics": [{"model_label": "base", "task": "hellaswag", "metric": "acc_norm,none", "value": 0.5}],
                },
                "refusals": {"models": [{"model_label": "base", "refusals": 1, "total": 2, "refusal_rate": 0.5}]},
                "drift": {"models": [{"model_label": "base", "reference_model": "base", "kl_divergence": 0.0}]},
                "constitution": {"models": [{"model_label": "base", "average_score": 4.0, "judgments": 2}]},
            },
            models=[{"label": "base"}],
        )

        self.assertIn("## Capability", report)
        self.assertIn("acc_norm,none", report)
        self.assertIn("50.00%", report)
        self.assertIn("0.000000", report)
        self.assertIn("4.00", report)

    def test_eval_run_writes_report_with_mocked_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "eval"

            def fake_run_local_generations(*, models, prompts, generation):  # type: ignore[no-untyped-def]
                return [
                    {
                        "prompt_id": prompts[0]["id"],
                        "model_label": models[0]["label"],
                        "model_name": models[0]["name"],
                        "adapter_path": "",
                        "messages": prompts[0]["messages"],
                        "response": "Answer",
                    }
                ]

            import cai.evaluation as evaluation

            original = evaluation.run_local_generations
            evaluation.run_local_generations = fake_run_local_generations
            try:
                summary = run_eval_from_config(
                    {
                        "output_dir": str(output_dir),
                        "models": [{"label": "base", "name": "model"}],
                        "prompts": [{"id": "p", "prompt": "Question"}],
                    }
                )
            finally:
                evaluation.run_local_generations = original

            self.assertEqual(summary["records"], 1)
            self.assertTrue(Path(summary["results_path"]).exists())
            self.assertTrue(Path(summary["report_path"]).exists())


if __name__ == "__main__":
    unittest.main()
