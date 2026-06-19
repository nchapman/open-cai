# Open CAI

Open CAI is a small Constitutional AI pipeline playground. It turns human-written
Markdown constitutions into reviewable response guides, then uses those guides to
generate preference-style training rows.

The project follows the broad shape of the Hugging Face Constitutional AI
example, but keeps the implementation intentionally simple and hackable.

## What It Does

- Write a constitution as normal Markdown in `constitutions/*.md`.
- Compile it into a human-reviewable guide in `constitutions/guides/*.guide.md`.
- Generate data from Anthropic HH-RLHF harmless-base.
- Pair a target model's initial answer with a stronger guide-following answer.
- Resume interrupted runs by skipping already generated source rows.

## Setup

This project uses Python with `uv`.

```bash
uv sync
cp .env.example .env
```

Add your OpenRouter key to `.env`:

```bash
OPENROUTER_API_KEY=...
```

Run the tests:

```bash
uv run python -m unittest discover -s tests
```

Training and evaluation use optional extras:

```bash
uv sync --extra train
uv sync --extra eval
```

## Constitutions

Constitutions are freeform Markdown. The sample files use a short opening
paragraph followed by compact principles such as "It should...".

Compile one into a response guide:

```bash
uv run cai-constitution compile \
  constitutions/balanced.md \
  -o constitutions/guides/balanced.guide.md
```

The included examples cover different postures:

- `strict.md`
- `balanced.md`
- `permissive.md`
- `playful.md`

The compiled guide is the artifact to review and tweak before generating data.
Version experimental guides under `constitutions/guides/versions/` before
promoting one to `constitutions/guides/*.guide.md`.

## Generate Data

Small OpenRouter-only run:

```bash
uv run cai-dataset generate \
  --guide constitutions/guides/balanced.guide.md \
  --output data/generated/balanced.jsonl \
  --max-samples 128 \
  --concurrency 16
```

Split run with a local target model for rejected responses and an OpenRouter
teacher for chosen responses:

```bash
uv run cai-dataset generate \
  --guide constitutions/guides/balanced.guide.md \
  --output data/generated/qwen-local-deepseek-balanced.jsonl \
  --max-samples 128 \
  --concurrency 16 \
  --init-base-url http://127.0.0.1:8080/v1 \
  --init-model qwen3.5-4b-heretic \
  --guide-model deepseek/deepseek-v3.2 \
  --skip-metadata \
  --max-tokens 1800 \
  --request-timeout 120 \
  --retries 3
```

Use `--max-samples -1` for the full harmless-base train split. Re-running the
same command resumes against the same output file.

Each output row includes the prompt, source HH-RLHF responses, generated initial
response, guide-optimized response, SFT `messages`, DPO `chosen`/`rejected`
message pairs, and comparison pairs for later analysis. Unless `--skip-metadata`
is set, rows also include guide-section metadata and critique notes from a JSON
schema response.

## Prepare Training Data

Training prep is config-driven and supports both local JSONL files and Hugging
Face datasets. Local CAI data is the default path for generated constitution
data; uploading to the Hub is optional for sharing.

Smoke test the prep pipeline:

```bash
uv run cai-train prepare --config configs/training/smoke.yaml
```

Prepare the balanced mix:

```bash
uv run cai-train prepare --config configs/training/balanced.yaml
```

The default balanced config uses:

- SFT: `HuggingFaceTB/smoltalk` / `smol-magpie-ultra` plus local CAI `messages`
- DPO: cleaned UltraFeedback plus local CAI `chosen` / `rejected`

Prepared files are written under `data/training/<name>/`:

```text
sft/train.jsonl
sft/test.jsonl
dpo/train.jsonl
dpo/test.jsonl
```

SFT rows use `messages`. DPO rows use explicit conversational `prompt`,
`chosen`, and `rejected` fields for TRL compatibility.

## Train

Start with a dry run to validate the config and prepared SFT rows without
loading a model:

```bash
uv run cai-train sft --config configs/training/sft-smoke.yaml --dry-run
```

Run the one-step SFT smoke train after installing training dependencies:

```bash
uv run --extra train cai-train sft --config configs/training/sft-smoke.yaml
```

The SFT command uses TRL's conversational `messages` format directly. LoRA is
enabled in the smoke config and can be disabled or tuned in YAML.

Run the one-step DPO smoke train from the SFT smoke adapter:

```bash
uv run --extra train cai-train dpo --config configs/training/dpo-smoke.yaml
```

The DPO command uses prepared conversational `prompt`, `chosen`, and `rejected`
rows. When `model.adapter_path` is set, DPO continues that adapter and uses the
pre-DPO adapter state as the reference policy.

## Evaluate

Run a small side-by-side local eval against base, SFT, and DPO variants:

```bash
uv run --extra eval cai-eval run --config configs/eval/smoke.yaml
```

The eval harness reads prompts and model variants from YAML, runs each model
sequentially to keep VRAM use predictable, and writes:

```text
outputs/eval/smoke/results.jsonl
outputs/eval/smoke/report.md
```

Run the fast evaluation suite for a broader signal:

```bash
uv sync --extra eval
uv run --extra eval cai-eval suite --config configs/eval/fast.yaml
```

The fast suite includes:

- capability checks via `lm-eval` (`hellaswag`, `arc_challenge`, `gsm8k`)
- refusal counts on 100 `mlabonne/harmful_behaviors` prompts
- behavior drift KL on 100 `mlabonne/harmless_alpaca` prompts
- constitution judging on 10 hand-written prompts against the compiled guide

It writes `summary.md` plus detailed JSON/JSONL artifacts under
`outputs/eval/fast/`.

The eval YAML supports `batch_size` for local generation and lm-eval tasks. Use
`max_gen_toks` on generated capability tasks as a generous runaway-response
safety cap, not as the main tuning dial for benchmark quality. Constitution
judge calls use bounded OpenRouter concurrency via `suite.judge.concurrency`.

## Model Defaults

Model choices live in code rather than `.env`:

- Guide compilation: `deepseek/deepseek-v4-pro`
- Guide application: `deepseek/deepseek-v3.2`

The CLIs expose overrides for model, temperature, max tokens, reasoning effort,
reasoning token budget, request timeout, retries, and provider base URLs.

Structured model calls use JSON schema response formats. Freeform assistant
responses are generated as normal chat completions with the reviewed guide in
the system message.

## License

MIT
