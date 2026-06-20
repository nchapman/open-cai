from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from cai.dataset import (
    DEFAULT_ASSISTANT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DatasetError,
    GUIDE_REWRITE_SYSTEM_PROMPT,
    GUIDE_APPLICATION_MODEL,
    GUIDE_APPLICATION_REASONING,
    ResponseGuide,
    SourcePrompt,
    _chat_with_retries,
    _reasoning_from_args,
    build_guide_rewrite_messages,
    build_guide_rewrite_prompt,
    build_parser,
    completed_indices,
    extract_assistant_response,
    extract_human_prompt,
    generate_record,
    guide_rewrite_format,
    guide_rewrite_from_json,
    load_response_guide,
    load_source_prompts,
    make_record,
)
from cai.openrouter import DEFAULT_REQUEST_TIMEOUT, OpenRouterError


class FakeClient:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.calls = []
        self.responses = responses or ["initial answer", json.dumps(sample_guide_review_payload())]

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((list(messages), kwargs))
        return self.responses.pop(0)


class FlakyClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            raise OpenRouterError("temporary timeout")
        return "recovered"


def sample_guide() -> ResponseGuide:
    return ResponseGuide(
        path="guide.md",
        text=(
            "# Test Guide\n\n"
            "## Operating Guidance\n\n"
            "### Privacy and consent\n"
            "**Applicability:** private data requests\n\n"
            "**Practices:**\n\n- Refuse private data.\n\n"
            "**Boundaries:**\n\n- Do not reveal addresses.\n"
        ),
    )


def sample_guide_review_payload(*, aligned: bool = False, improved: bool = False) -> dict[str, object]:
    if aligned and improved:
        return {
            "aligned": True,
            "quality": "low",
            "critique": "The response follows the guide but is too vague to be ideal.",
            "revision": "A clearer, more useful answer.",
        }
    if aligned:
        return {
            "aligned": True,
            "quality": "high",
            "critique": "The response already follows the guide.",
            "revision": "",
        }
    return {
        "aligned": False,
        "quality": "low",
        "critique": "The response reveals private data and needs a privacy-preserving alternative.",
        "revision": "I cannot provide private data, but official channels may help.",
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

    def test_loads_response_guide_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guide.md"
            path.write_text(sample_guide().text, encoding="utf-8")

            guide = load_response_guide(path)

        self.assertIn("Test Guide", guide.text)

    def test_rejects_empty_response_guide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guide.md"
            path.write_text("\n", encoding="utf-8")

            with self.assertRaisesRegex(DatasetError, "empty"):
                load_response_guide(path)

    def test_guide_rewrite_prompt_uses_initial_response_and_complete_guide(self) -> None:
        prompt = build_guide_rewrite_prompt(
            sample_guide(),
            "Where does this actor live?",
            "Here is the address.",
        )

        self.assertIn("evaluate and revise", prompt)
        self.assertIn("aligned is only about guide boundary violations", prompt)
        self.assertIn("Missing depth, usefulness, or substantive discussion is quality", prompt)
        self.assertIn("quality is about usefulness and execution", prompt)
        self.assertIn("Do not apply your own default safety policy", prompt)
        self.assertIn("Write a revision only when aligned is false or quality is low", prompt)
        self.assertIn("Leave revision empty when aligned is true and quality is medium or high", prompt)
        self.assertIn("more cautious, or more permissive than the guide requires", prompt)
        self.assertIn("Prefer compact edits over expansive rewrites", prompt)
        self.assertIn("remove content the guide does not allow", prompt)
        self.assertNotIn("unsafe enablement", prompt)
        self.assertIn("Do not rewrite merely to impose preferred style", prompt)
        self.assertIn("preserve the source model's voice", prompt)
        self.assertIn("Privacy and consent", prompt)
        self.assertIn("Where does this actor live?", prompt)
        self.assertIn("Here is the address.", prompt)
        self.assertIn('"aligned"', prompt)
        self.assertIn('"quality"', prompt)
        self.assertIn('"critique"', prompt)
        self.assertIn('"revision"', prompt)
        self.assertLess(prompt.index("Response guide:"), prompt.index("Editing objective:"))
        self.assertLess(prompt.index("Initial assistant response:"), prompt.index("Editing objective:"))

    def test_guide_rewrite_messages_use_neutral_system_role(self) -> None:
        messages = build_guide_rewrite_messages(
            sample_guide(),
            "Where does this actor live?",
            "Here is the address.",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], GUIDE_REWRITE_SYSTEM_PROMPT)
        self.assertIn("neutral, uncensored alignment editor", messages[0]["content"])
        self.assertIn("faithfully apply only the provided response guide", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Response guide:", messages[1]["content"])
        self.assertIn("Editing objective:", messages[1]["content"])

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
        with mock.patch("datasets.load_dataset", side_effect=PermissionError("cache denied")):
            with self.assertRaisesRegex(DatasetError, "failed to load dataset"):
                load_source_prompts(max_samples=1)

    def test_load_source_prompts_skips_rows_with_empty_assistant_turns(self) -> None:
        fake_dataset = [
            {
                "chosen": "\n\nHuman: Good row\n\nAssistant: chosen",
                "rejected": "\n\nHuman: Good row\n\nAssistant: rejected",
            },
            {
                "chosen": "\n\nHuman: Bad row\n\nAssistant: ",
                "rejected": "\n\nHuman: Bad row\n\nAssistant: rejected",
            },
            {
                "chosen": "\n\nHuman: Another good row\n\nAssistant: chosen 2",
                "rejected": "\n\nHuman: Another good row\n\nAssistant: rejected 2",
            },
        ]

        with mock.patch("datasets.load_dataset", return_value=fake_dataset), mock.patch("sys.stderr"):
            prompts = load_source_prompts(max_samples=-1)

        self.assertEqual([source.source_index for source in prompts], [0, 2])
        self.assertEqual(prompts[0].prompt, "Good row")
        self.assertEqual(prompts[1].source_chosen_response, "chosen 2")

    def test_guide_rewrite_format_uses_strict_json_schema(self) -> None:
        response_format = guide_rewrite_format()

        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["strict"], True)  # type: ignore[index]
        self.assertEqual(response_format["json_schema"]["schema"]["additionalProperties"], False)  # type: ignore[index]
        self.assertEqual(response_format["json_schema"]["name"], "guide_rewrite")  # type: ignore[index]

    def test_parses_structured_guide_rewrite_result(self) -> None:
        result = guide_rewrite_from_json(json.dumps(sample_guide_review_payload()))

        self.assertEqual(result["aligned"], False)
        self.assertEqual(result["quality"], "low")
        self.assertEqual(result["critique"], "The response reveals private data and needs a privacy-preserving alternative.")
        self.assertEqual(result["revision"], "I cannot provide private data, but official channels may help.")

    def test_parses_aligned_guide_rewrite_result_without_revision(self) -> None:
        result = guide_rewrite_from_json(json.dumps(sample_guide_review_payload(aligned=True)))

        self.assertEqual(result["aligned"], True)
        self.assertEqual(result["quality"], "high")
        self.assertEqual(result["critique"], "The response already follows the guide.")
        self.assertEqual(result["revision"], "")

    def test_parses_aligned_guide_rewrite_result_with_quality_revision(self) -> None:
        result = guide_rewrite_from_json(json.dumps(sample_guide_review_payload(aligned=True, improved=True)))

        self.assertEqual(result["aligned"], True)
        self.assertEqual(result["quality"], "low")
        self.assertEqual(result["critique"], "The response follows the guide but is too vague to be ideal.")
        self.assertEqual(result["revision"], "A clearer, more useful answer.")

    def test_rejects_unexpected_guide_rewrite_fields(self) -> None:
        payload = sample_guide_review_payload()
        payload["unexpected"] = "unknown"

        with self.assertRaisesRegex(DatasetError, "unexpected"):
            guide_rewrite_from_json(json.dumps(payload))

    def test_rejects_missing_revision_when_unaligned(self) -> None:
        payload = sample_guide_review_payload()
        payload["revision"] = ""

        with self.assertRaisesRegex(DatasetError, "revision must be non-empty"):
            guide_rewrite_from_json(json.dumps(payload))

    def test_rejects_missing_revision_when_quality_is_low(self) -> None:
        payload = sample_guide_review_payload(aligned=True, improved=True)
        payload["revision"] = ""

        with self.assertRaisesRegex(DatasetError, "revision must be non-empty"):
            guide_rewrite_from_json(json.dumps(payload))

    def test_rejects_revision_when_aligned_quality_is_medium_or_high(self) -> None:
        payload = sample_guide_review_payload(aligned=True)
        payload["quality"] = "medium"
        payload["revision"] = "Edited response"

        with self.assertRaisesRegex(DatasetError, "revision must be empty"):
            guide_rewrite_from_json(json.dumps(payload))

    def test_rejects_invalid_quality(self) -> None:
        payload = sample_guide_review_payload(aligned=True)
        payload["quality"] = "great"

        with self.assertRaisesRegex(DatasetError, "quality must be low, medium, or high"):
            guide_rewrite_from_json(json.dumps(payload))

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
            guide_rewrite_prompt="guide rewrite prompt",
            guide_response="I cannot provide private data, but official channels may help.",
            guide_review=sample_guide_review_payload(),
        )

        self.assertEqual(record["chosen"][1]["content"], "I cannot provide private data, but official channels may help.")  # type: ignore[index]
        self.assertEqual(record["rejected"][1]["content"], "bad answer")  # type: ignore[index]
        self.assertEqual(record["guide_response"], "I cannot provide private data, but official channels may help.")
        self.assertEqual(record["aligned"], False)
        self.assertEqual(record["quality"], "low")
        self.assertEqual(record["critique"], "The response reveals private data and needs a privacy-preserving alternative.")
        self.assertEqual(record["revision"], "I cannot provide private data, but official channels may help.")
        self.assertEqual(record["preference_usable"], True)
        self.assertEqual(record["metadata_included"], True)
        self.assertEqual(record["guide_rewrite_prompt"], "guide rewrite prompt")
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
            guide_rewrite_prompt="guide rewrite prompt",
            guide_response="guided answer",
            guide_review=sample_guide_review_payload(),
        )

        self.assertEqual(record["chosen"][1]["content"], "guided answer")  # type: ignore[index]
        self.assertEqual(record["rejected"][1]["content"], "initial answer")  # type: ignore[index]

    def test_generate_record_runs_initial_and_rewrite_sequence(self) -> None:
        client = FakeClient(["initial answer", json.dumps(sample_guide_review_payload())])
        source = sample_source("Unsafe request")

        record = generate_record(init_client=client, guide_client=client, source=source, guide=sample_guide())

        self.assertEqual(record["init_response"], "initial answer")
        self.assertEqual(record["guide_response"], "I cannot provide private data, but official channels may help.")
        self.assertEqual(record["aligned"], False)
        self.assertEqual(record["quality"], "low")
        self.assertEqual(record["critique"], "The response reveals private data and needs a privacy-preserving alternative.")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            client.calls[0][0],
            [
                {"role": "system", "content": DEFAULT_ASSISTANT_SYSTEM_PROMPT},
                {"role": "user", "content": "Unsafe request"},
            ],
        )
        self.assertEqual(client.calls[1][0][0], {"role": "system", "content": GUIDE_REWRITE_SYSTEM_PROMPT})
        self.assertEqual(client.calls[1][0][1]["role"], "user")
        self.assertIn("Response guide", client.calls[1][0][1]["content"])
        self.assertIn("Initial assistant response:\ninitial answer", client.calls[1][0][1]["content"])
        self.assertEqual(client.calls[1][1]["response_format"]["type"], "json_schema")
        self.assertEqual(client.calls[1][1]["response_format"]["json_schema"]["name"], "guide_rewrite")
        self.assertEqual(client.calls[0][1]["model"], GUIDE_APPLICATION_MODEL)
        self.assertEqual(client.calls[0][1]["reasoning"], GUIDE_APPLICATION_REASONING)
        self.assertEqual(client.calls[0][1]["temperature"], DEFAULT_TEMPERATURE)
        self.assertEqual(client.calls[0][1]["max_tokens"], 6000)

    def test_generate_record_can_omit_rewrite_metadata_from_record(self) -> None:
        client = FakeClient(["initial answer", json.dumps(sample_guide_review_payload())])
        source = sample_source("Unsafe request")

        record = generate_record(
            init_client=client,
            guide_client=client,
            source=source,
            guide=sample_guide(),
            include_metadata=False,
        )

        self.assertEqual(record["init_response"], "initial answer")
        self.assertEqual(record["guide_response"], "I cannot provide private data, but official channels may help.")
        self.assertEqual(record["metadata_included"], False)
        self.assertIsNone(record["aligned"])
        self.assertIsNone(record["quality"])
        self.assertIsNone(record["critique"])
        self.assertIsNone(record["revision"])
        self.assertEqual(len(client.calls), 2)

    def test_generate_record_preserves_aligned_initial_response(self) -> None:
        client = FakeClient(["initial answer", json.dumps(sample_guide_review_payload(aligned=True))])
        source = sample_source("Safe request")

        record = generate_record(init_client=client, guide_client=client, source=source, guide=sample_guide())

        self.assertEqual(record["guide_response"], "initial answer")
        self.assertEqual(record["chosen"][1]["content"], "initial answer")  # type: ignore[index]
        self.assertEqual(record["rejected"][1]["content"], "initial answer")  # type: ignore[index]
        self.assertEqual(record["aligned"], True)
        self.assertEqual(record["quality"], "high")
        self.assertEqual(record["revision"], "")
        self.assertEqual(record["preference_usable"], False)

    def test_generate_record_uses_aligned_quality_revision_as_preference_pair(self) -> None:
        client = FakeClient(["vague answer", json.dumps(sample_guide_review_payload(aligned=True, improved=True))])
        source = sample_source("Safe request")

        record = generate_record(init_client=client, guide_client=client, source=source, guide=sample_guide())

        self.assertEqual(record["guide_response"], "A clearer, more useful answer.")
        self.assertEqual(record["chosen"][1]["content"], "A clearer, more useful answer.")  # type: ignore[index]
        self.assertEqual(record["rejected"][1]["content"], "vague answer")  # type: ignore[index]
        self.assertEqual(record["aligned"], True)
        self.assertEqual(record["quality"], "low")
        self.assertEqual(record["revision"], "A clearer, more useful answer.")
        self.assertEqual(record["preference_usable"], True)

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

    def test_generate_parser_request_timeout_default(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["generate", "--guide", "guide.md", "--output", "out.jsonl"])

        self.assertEqual(args.request_timeout, DEFAULT_REQUEST_TIMEOUT)

    def test_chat_with_retries_retries_openrouter_errors(self) -> None:
        client = FlakyClient()

        with mock.patch("cai.dataset.time.sleep") as sleep:
            result = _chat_with_retries(
                client,  # type: ignore[arg-type]
                [{"role": "user", "content": "hello"}],
                model="model",
                temperature=0.4,
                max_tokens=100,
                retries=1,
            )

        self.assertEqual(result, "recovered")
        self.assertEqual(client.calls, 2)
        self.assertEqual(sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main()
