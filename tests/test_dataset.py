from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from cai.dataset import (
    DatasetError,
    Rule,
    SourcePrompt,
    completed_indices,
    critic_response_format,
    critic_result_from_json,
    extract_human_prompt,
    generate_record,
    build_critic_prompt,
    build_revision_prompt,
    load_source_prompts,
    load_rules,
    make_record,
    select_rule,
)
from cai.constitution import ruleset_to_yaml


class FakeClient:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.calls = []
        self.responses = responses or ["initial answer", "critique text", "revised answer"]

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((list(messages), kwargs))
        return self.responses.pop(0)


class DatasetTests(unittest.TestCase):
    def test_extracts_first_human_prompt(self) -> None:
        conversation = "\n\nHuman: Can you help?\n\nAssistant: Sure.\n\nHuman: Follow up?\n\nAssistant: Done."

        self.assertEqual(extract_human_prompt(conversation), "Can you help?")

    def test_rejects_conversation_without_human_turn(self) -> None:
        with self.assertRaisesRegex(DatasetError, "Human"):
            extract_human_prompt("Assistant: Hello")

    def test_loads_rules_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.yaml"
            path.write_text(
                ruleset_to_yaml(
                    [
                        {
                            "id": "principle-01-harmful-general",
                            "category": "harmful-general",
                            "principle": "It should avoid harm.",
                            "critic": "Identify harm.",
                            "revision": "Remove harm.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            rules = load_rules(path)

        self.assertEqual(rules[0].id, "principle-01-harmful-general")

    def test_selects_rules_deterministically(self) -> None:
        rules = [
            Rule("a", "harmful-general", "A", "Critic A", "Revision A"),
            Rule("b", "harmful-general", "B", "Critic B", "Revision B"),
        ]

        first = select_rule(rules, seed=42, source_index=7)
        second = select_rule(rules, seed=42, source_index=7)

        self.assertEqual(first, second)

    def test_generation_prompts_handle_irrelevant_rules(self) -> None:
        rule = Rule("r1", "harmful-general", "It should avoid harm.", "Critic", "Revision")

        critic = build_critic_prompt(rule)
        revision = build_revision_prompt(rule)

        self.assertIn("Evaluate only this principle", critic)
        self.assertIn('"revision_needed"', critic)
        self.assertIn('"critique"', critic)
        self.assertIn('"additionalProperties": false', critic)
        self.assertIn("Set revision_needed to false", critic)
        self.assertIn("return only the revised assistant response", revision)

    def test_reads_completed_indices_for_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"source_split": "train", "source_index": 1}),
                        json.dumps({"source_split": "test", "source_index": 2}),
                    ]
                ),
                encoding="utf-8",
            )

            completed = completed_indices(path, split="train")

        self.assertEqual(completed, {1})

    def test_wraps_dataset_load_failures(self) -> None:
        with mock.patch("datasets.load.load_dataset", side_effect=PermissionError("cache denied")):
            with self.assertRaisesRegex(DatasetError, "failed to load dataset"):
                load_source_prompts(max_samples=1)

    def test_make_record_includes_preference_fields(self) -> None:
        source = SourcePrompt("Anthropic/hh-rlhf", "train", 3, "Tell me something")
        rule = Rule("r1", "harmful-general", "It should avoid harm.", "Critic", "Revision")

        record = make_record(
            source=source,
            rule=rule,
            init_response="bad answer",
            critic_prompt="critic prompt",
            critic_response="critique",
            revision_prompt="revision prompt",
            revision_response="better answer",
        )

        self.assertEqual(record["chosen"][1]["content"], "better answer")  # type: ignore[index]
        self.assertEqual(record["rejected"][1]["content"], "bad answer")  # type: ignore[index]
        self.assertEqual(record["rule_id"], "r1")
        self.assertEqual(record["revision_needed"], True)

    def test_parses_structured_critic_result(self) -> None:
        result = critic_result_from_json('{"revision_needed": false, "critique": "No issue."}')

        self.assertEqual(result, {"revision_needed": False, "critique": "No issue."})

    def test_critic_response_format_uses_strict_json_schema(self) -> None:
        response_format = critic_response_format()

        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["strict"], True)  # type: ignore[index]
        self.assertEqual(response_format["json_schema"]["schema"]["additionalProperties"], False)  # type: ignore[index]

    def test_rejects_invalid_structured_critic_result(self) -> None:
        with self.assertRaisesRegex(DatasetError, "revision_needed"):
            critic_result_from_json('{"revision_needed": "false", "critique": "No issue."}')

    def test_generate_record_runs_initial_critique_revision_sequence(self) -> None:
        client = FakeClient(["initial answer", '{"revision_needed": true, "critique": "critique text"}', "revised answer"])
        source = SourcePrompt("Anthropic/hh-rlhf", "train", 0, "Unsafe request")
        rule = Rule("r1", "harmful-general", "It should avoid harm.", "Critic", "Revision")

        record = generate_record(client=client, source=source, rule=rule)

        self.assertEqual(record["init_response"], "initial answer")
        self.assertEqual(record["critic_response"], "critique text")
        self.assertEqual(record["revision_response"], "revised answer")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(client.calls[0][0], [{"role": "user", "content": "Unsafe request"}])
        self.assertIn("Critic", client.calls[1][0][2]["content"])
        self.assertEqual(client.calls[1][1]["response_format"]["type"], "json_schema")
        self.assertEqual(client.calls[1][1]["response_format"]["json_schema"]["name"], "critic_result")
        self.assertIn("Revision", client.calls[2][0][4]["content"])

    def test_generate_record_skips_revision_when_critic_says_revision_not_needed(self) -> None:
        client = FakeClient(["initial answer", '{"revision_needed": false, "critique": "No issue."}'])
        source = SourcePrompt("Anthropic/hh-rlhf", "train", 0, "Benign request")
        rule = Rule("r1", "harmful-general", "It should avoid harm.", "Critic", "Revision")

        record = generate_record(client=client, source=source, rule=rule)

        self.assertEqual(record["init_response"], "initial answer")
        self.assertEqual(record["revision_response"], "initial answer")
        self.assertEqual(record["revision_needed"], False)
        self.assertIn("Revision not needed", record["revision_prompt"])
        self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
