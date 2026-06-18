# cai

A small, straightforward Constitutional AI pipeline playground.

This project follows the broad ideas from the Hugging Face Constitutional AI
recipe: generate an initial answer, produce a constitution-guided answer, then
use the guided output for supervised or preference data. We are not copying the
original implementation or constitution verbatim.

## Constitutions

Constitutions start as human-authored Markdown in `constitutions/*.md`. The
recommended shape is an opening paragraph followed by a normal bullet list of
principles written in plain language.

The compiler sends the full Markdown document to an OpenRouter model, asks for
strict JSON, validates the response-guide structure, and writes an editable
Markdown guide. The guide is meant to be readable enough for a human reviewer to
use directly: sections, applicability criteria, do/avoid guidance, and examples.

Validate a source constitution with:

```bash
uv run cai-constitution validate constitutions/balanced.md
```

Compile a constitution to a reviewable Markdown response guide with OpenRouter:

```bash
uv run cai-constitution compile constitutions/balanced.md -o constitutions/guides/balanced.guide.md
```

The main examples are a posture spectrum:

```bash
uv run cai-constitution compile constitutions/strict.md -o constitutions/guides/strict.guide.md
uv run cai-constitution compile constitutions/balanced.md -o constitutions/guides/balanced.guide.md
uv run cai-constitution compile constitutions/permissive.md -o constitutions/guides/permissive.guide.md
```

There is also a playful tone example:

```bash
uv run cai-constitution compile constitutions/playful.md -o constitutions/guides/playful.guide.md
```

The pipeline should consume the reviewed Markdown guide, not silently recompile
the source constitution during an experiment.

When tuning the compiler prompt, keep versioned outputs under
`constitutions/guides/versions/` before promoting a new guide to the canonical
`constitutions/guides/*.guide.md` path.

## OpenRouter

Inference uses OpenRouter's OpenAI-compatible chat completions API. Create a
local `.env` from `.env.example` and set `OPENROUTER_API_KEY`.

Model roles are code defaults, not `.env` settings. Guide compilation uses
`deepseek/deepseek-v4-pro` with high reasoning. Dataset generation uses
`deepseek/deepseek-v3.2` with reasoning disabled. Both CLIs expose `--model`,
`--reasoning-effort`, `--reasoning-max-tokens`, and `--max-tokens` overrides for
experiments.

Model calls that expect JSON use OpenRouter `json_schema` response formats, not
prompt-only JSON instructions. The current structured calls are guide compilation
and guide metadata auditing. The guided assistant response itself is generated as
a normal chat response with the reviewed guide in the system message.

Try a one-off request with:

```bash
uv run cai-openrouter chat "Say this is a test"
```

The same client can call a local OpenAI-compatible endpoint. For example, with a
separately managed vLLM server:

```bash
uv run cai-openrouter chat "Say this is a local test" \
  --base-url http://127.0.0.1:8080/v1 \
  --api-key local-token \
  --model qwen3.5-4b-heretic
```

## Dataset generation

Generate guide-optimized preference rows from Anthropic HH-RLHF harmless-base
with a reviewed Markdown response guide:

```bash
uv run cai-dataset generate \
  --guide constitutions/guides/balanced.guide.md \
  --output data/generated/balanced.jsonl \
  --max-samples 128 \
  --concurrency 16
```

Use `--max-samples -1` for the full split. The output is JSONL and includes the
initial response, applicable guide sections, critique, guide-optimized response,
change notes, original HH-RLHF chosen/rejected responses, and `chosen`/`rejected`
message pairs for later training steps. `chosen` is always the guide-optimized
response; `rejected` is always the generated initial response.

Rows also include `comparison_pairs` so the same run can compare the guided
response against the generated initial response, the original HH-RLHF chosen
response, the original HH-RLHF rejected response, or the original HH-RLHF pair.
Re-running the same command resumes by skipping source indices already present in
the output file.

For local model generation, point the dataset CLI at the local OpenAI-compatible
server and skip metadata if the server does not support strict JSON schema
responses:

```bash
uv run cai-dataset generate \
  --guide constitutions/guides/balanced.guide.md \
  --output data/generated/qwen-local-balanced.jsonl \
  --max-samples 128 \
  --concurrency 16 \
  --base-url http://127.0.0.1:8080/v1 \
  --api-key local-token \
  --model qwen3.5-4b-heretic \
  --skip-metadata
```

To generate target-model rejected responses locally while using a stronger
OpenRouter teacher for the guided chosen responses, override the initial and
guide providers separately:

```bash
uv run cai-dataset generate \
  --guide constitutions/guides/balanced.guide.md \
  --output data/generated/qwen-local-deepseek-teacher.jsonl \
  --max-samples 128 \
  --concurrency 8 \
  --init-base-url http://127.0.0.1:8080/v1 \
  --init-model qwen3.5-4b-heretic \
  --guide-model deepseek/deepseek-v3.2 \
  --reasoning-effort none \
  --skip-metadata \
  --max-tokens 1800
```

Run tests with:

```bash
uv run python -m unittest discover -s tests
```
