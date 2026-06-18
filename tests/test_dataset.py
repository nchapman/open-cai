from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from cai.dataset import (
    DEFAULT_ASSISTANT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DatasetError,
    GUIDE_APPLICATION_MODEL,
    GUIDE_APPLICATION_REASONING,
    ResponseGuide,
    SourcePrompt,
    _reasoning_from_args,
    build_guide_metadata_prompt,
    build_guide_system_prompt,
    build_parser,
    completed_indices,
    extract_assistant_response,
    extract_human_prompt,
    generate_record,
    guide_metadata_format,
    guide_metadata_from_json,
    load_response_guide,
    load_source_prompts,
    make_record,
)


class FakeClient:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.calls = []
        self.responses = responses or ["initial answer", "guided answer", json.dumps(sample_guide_metadata_payload())]

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((list(messages), kwargs))
        return self.responses.pop(0)


def sample_guide() -> ResponseGuide:
    return ResponseGuide(
        path="guide.md",
        text=(
            "# Test Guide\n\n"
            "## Guide Sections\n\n"
            "### privacy-consent: Privacy and consent\n\n"
            "**When to apply:** private data requests\n\n"
            "**Do:**\n\n- Refuse private data.\n\n"
            "**Avoid:**\n\n- Do not reveal addresses.\n"
        ),
        section_ids=("privacy-consent",),
    )


def sample_guide_metadata_payload() -> dict[str, object]:
    return {
        "applicable_section_ids": ["privacy-consent"],
        "primary_section_id": "privacy-consent",
        "critique": "The response reveals private data.",
        "changes_made": "Removed private data and preserved a safe alternative.",
        "quality_notes": "The guide response better follows the privacy section.",
    }


def sample_source(prompt: str = "Tell me something") -> SourcePrompt:
    return SourcePrompt(
        source_dataset="Anthropic/hh-rlhf",
        source_split="train",
        source_index=3,
        prompt=prompt,
        source_chosen_conversation=f"\n\nHuman: {prompt}\n\nAssistant: source chosen answer",
        source_rejected_conversation=f"\n\nHuman: {prompt}\n\nAssistant: source rejected answer",
        source_chosen_response="source chosen answer",
        source_rejected_response="source rejected answer",
    )


class DatasetTests(unittest.TestCase):
    def test_extracts_first_human_prompt(self) -> None:
        conversation = "\n\nHuman: Can you help?\n\nAssistant: Sure.\n\nHuman: Follow up?\n\nAssistant: Done."

        self.assertEqual(extract_human_prompt(conversation), "Can you help?")

    def test_rejects_conversation_without_human_turn(self) -> None:
        with self.assertRaisesRegex(DatasetError, "Human"):
            extract_human_prompt("Assistant: Hello")

    def test_extracts_first_assistant_response(self) -> None:
        conversation = "\n\nHuman: Can you help?\n\nAssistant: Sure.\n\nHuman: Follow up?\n\nAssistant: Done."

        self.assertEqual(extract_assistant_response(conversation), "Sure.")

    def test_rejects_conversation_without_assistant_turn(self) -> None:
        with self.assertRaisesRegex(DatasetError, "Assistant"):
            extract_assistant_response("Human: Hello")

    def test_loads_response_guide_section_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guide.md"
            path.write_text(sample_guide().text, encoding="utf-8")

            guide = load_response_guide(path)

        self.assertEqual(guide.section_ids, ("privacy-consent",))
        self.assertIn("Test Guide", guide.text)

    def test_rejects_response_guide_without_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guide.md"
            path.write_text("# Empty\n", encoding="utf-8")

            with self.assertRaisesRegex(DatasetError, "section"):
                load_response_guide(path)

    def test_guide_system_prompt_uses_complete_guide(self) -> None:
        prompt = build_guide_system_prompt(sample_guide())

        self.assertIn("Runtime framing", prompt)
        self.assertIn("whether it is strict, balanced, permissive", prompt)
        self.assertIn("more restrictive or more permissive", prompt)
        self.assertIn("privacy-consent", prompt)
        self.assertIn("Do not repeat or quote", prompt)
        self.assertIn("explicit about uncertainty", prompt)
        self.assertIn("rubric labels", prompt)
        self.assertIn("Return only the final assistant message", prompt)

    def test_guide_metadata_prompt_uses_complete_guide_and_responses(self) -> None:
        prompt = build_guide_metadata_prompt(
            sample_guide(),
            "Where does this actor live?",
            "Here is the address.",
            "I cannot provide private data.",
        )

        self.assertIn("audit the guided assistant response", prompt)
        self.assertIn("privacy-consent", prompt)
        self.assertIn("Here is the address.", prompt)
        self.assertIn("I cannot provide private data.", prompt)
        self.assertIn('"quality_notes"', prompt)
        self.assertNotIn('"guide_response"', prompt)

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

    def test_guide_metadata_format_uses_strict_json_schema(self) -> None:
        response_format = guide_metadata_format()

        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["strict"], True)  # type: ignore[index]
        self.assertEqual(response_format["json_schema"]["schema"]["additionalProperties"], False)  # type: ignore[index]
        self.assertEqual(response_format["json_schema"]["name"], "guide_metadata")  # type: ignore[index]

    def test_parses_structured_guide_metadata_result(self) -> None:
        result = guide_metadata_from_json(json.dumps(sample_guide_metadata_payload()), sample_guide())

        self.assertEqual(result["primary_section_id"], "privacy-consent")
        self.assertEqual(result["changes_made"], "Removed private data and preserved a safe alternative.")

    def test_rejects_unknown_guide_metadata_section(self) -> None:
        payload = sample_guide_metadata_payload()
        payload["primary_section_id"] = "unknown"
        payload["applicable_section_ids"] = ["unknown"]

        with self.assertRaisesRegex(DatasetError, "unknown"):
            guide_metadata_from_json(json.dumps(payload), sample_guide())

    def test_generate_reasoning_defaults_do_not_block_token_budget_override(self) -> None:
        parser = build_parser()

        default_args = parser.parse_args(["generate", "--guide", "guide.md", "--output", "out.jsonl"])
        token_budget_args = parser.parse_args(
            [
                "generate",
                "--guide",
                "guide.md",
                "--output",
                "out.jsonl",
                "--reasoning-max-tokens",
                "2048",
            ]
        )

        self.assertEqual(_reasoning_from_args(default_args, base_url=None), {"effort": "none", "exclude": True})
        self.assertEqual(_reasoning_from_args(token_budget_args, base_url=None), {"max_tokens": 2048, "exclude": True})

    def test_make_record_includes_preference_fields_and_guide_traceability(self) -> None:
        record = make_record(
            source=sample_source(),
            guide=sample_guide(),
            init_response="bad answer",
            guide_response_system_prompt="guide response system prompt",
            guide_response="I cannot provide private data, but official channels may help.",
            guide_metadata_prompt="guide metadata prompt",
            guide_metadata=sample_guide_metadata_payload(),
        )

        self.assertEqual(record["chosen"][1]["content"], "I cannot provide private data, but official channels may help.")  # type: ignore[index]
        self.assertEqual(record["rejected"][1]["content"], "bad answer")  # type: ignore[index]
        self.assertEqual(record["primary_section_id"], "privacy-consent")
        self.assertEqual(record["guide_section_ids"], ["privacy-consent"])
        self.assertEqual(record["guide_response"], "I cannot provide private data, but official channels may help.")
        self.assertEqual(record["changes_made"], "Removed private data and preserved a safe alternative.")
        self.assertEqual(record["metadata_included"], True)
        self.assertEqual(record["guide_response_system_prompt"], "guide response system prompt")
        self.assertEqual(record["guide_metadata_prompt"], "guide metadata prompt")
        self.assertEqual(record["source_chosen_response"], "source chosen answer")
        self.assertEqual(record["source_rejected_response"], "source rejected answer")
        self.assertEqual(
            record["comparison_pairs"]["guided_vs_source_chosen"]["rejected"][1]["content"],  # type: ignore[index]
            "source chosen answer",
        )
        self.assertEqual(
            record["comparison_pairs"]["source_chosen_vs_source_rejected"]["chosen"][1]["content"],  # type: ignore[index]
            "source chosen answer",
        )

    def test_make_record_always_prefers_guide_response(self) -> None:
        record = make_record(
            source=sample_source(),
            guide=sample_guide(),
            init_response="initial answer",
            guide_response_system_prompt="guide response system prompt",
            guide_response="guided answer",
            guide_metadata_prompt="guide metadata prompt",
            guide_metadata=sample_guide_metadata_payload(),
        )

        self.assertEqual(record["chosen"][1]["content"], "guided answer")  # type: ignore[index]
        self.assertEqual(record["rejected"][1]["content"], "initial answer")  # type: ignore[index]

    def test_generate_record_runs_initial_guide_and_metadata_sequence(self) -> None:
        client = FakeClient(["initial answer", "guided answer", json.dumps(sample_guide_metadata_payload())])
        source = sample_source("Unsafe request")

        record = generate_record(init_client=client, guide_client=client, source=source, guide=sample_guide())

        self.assertEqual(record["init_response"], "initial answer")
        self.assertEqual(record["guide_response"], "guided answer")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(
            client.calls[0][0],
            [
                {"role": "system", "content": DEFAULT_ASSISTANT_SYSTEM_PROMPT},
                {"role": "user", "content": "Unsafe request"},
            ],
        )
        self.assertEqual(client.calls[1][0][0]["role"], "system")
        self.assertIn("Response guide", client.calls[1][0][0]["content"])
        self.assertEqual(client.calls[1][0][1], {"role": "user", "content": "Unsafe request"})
        self.assertIsNone(client.calls[1][1]["response_format"])
        self.assertIn("audit the guided assistant response", client.calls[2][0][0]["content"])
        self.assertIn("guided answer", client.calls[2][0][0]["content"])
        self.assertEqual(client.calls[0][1]["model"], GUIDE_APPLICATION_MODEL)
        self.assertEqual(client.calls[0][1]["reasoning"], GUIDE_APPLICATION_REASONING)
        self.assertEqual(client.calls[0][1]["temperature"], DEFAULT_TEMPERATURE)
        self.assertEqual(client.calls[0][1]["max_tokens"], 6000)
        self.assertEqual(client.calls[2][1]["response_format"]["type"], "json_schema")
        self.assertEqual(client.calls[2][1]["response_format"]["json_schema"]["name"], "guide_metadata")

    def test_generate_record_can_skip_metadata_for_plain_chat_endpoints(self) -> None:
        client = FakeClient(["initial answer", "guided answer"])
        source = sample_source("Unsafe request")

        record = generate_record(
            init_client=client,
            guide_client=client,
            source=source,
            guide=sample_guide(),
            include_metadata=False,
        )

        self.assertEqual(record["init_response"], "initial answer")
        self.assertEqual(record["guide_response"], "guided answer")
        self.assertEqual(record["metadata_included"], False)
        self.assertEqual(record["applicable_section_ids"], [])
        self.assertEqual(record["primary_section_id"], "none")
        self.assertIsNone(record["critique"])
        self.assertEqual(record["guide_metadata_prompt"], "")
        self.assertEqual(len(client.calls), 2)

    def test_local_base_url_disables_default_reasoning_payload(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "generate",
                "--guide",
                "guide.md",
                "--output",
                "out.jsonl",
                "--base-url",
                "http://127.0.0.1:8080/v1",
            ]
        )

        self.assertIsNone(_reasoning_from_args(args, base_url=args.base_url))

    def test_split_model_args_keep_local_initial_plain_and_openrouter_guide_reasoning(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "generate",
                "--guide",
                "guide.md",
                "--output",
                "out.jsonl",
                "--init-base-url",
                "http://127.0.0.1:8080/v1",
                "--init-model",
                "qwen3.5-4b-heretic",
                "--guide-model",
                "deepseek/deepseek-v3.2",
            ]
        )

        self.assertIsNone(_reasoning_from_args(args, base_url=args.init_base_url))
        self.assertEqual(_reasoning_from_args(args, base_url=None), {"effort": "none", "exclude": True})

    def test_reasoning_effort_none_disables_all_reasoning_payloads(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "generate",
                "--guide",
                "guide.md",
                "--output",
                "out.jsonl",
                "--reasoning-effort",
                "none",
            ]
        )

        self.assertEqual(_reasoning_from_args(args, base_url=None), {"effort": "none", "exclude": True})


if __name__ == "__main__":
    unittest.main()
