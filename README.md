# cai

A small, straightforward Constitutional AI pipeline playground.

This project follows the broad ideas from the Hugging Face Constitutional AI
recipe: generate an initial answer, critique it with a constitution, revise it,
then use the revised output for supervised or preference data. We are not
copying the original implementation or constitution verbatim.

## Constitutions

Constitutions start as human-authored Markdown in `constitutions/*.md`. The
recommended shape is an opening paragraph followed by a normal bullet list of
principles written in plain language.

The compiler sends the full Markdown document to an OpenRouter model, asks for
strict JSON, validates the rules, and writes an editable YAML ruleset. Each rule
has only `id`, `category`, `principle`, `critic`, and `revision`.

Validate a source constitution with:

```bash
uv run cai-constitution validate constitutions/balanced.md
```

Compile a constitution to a reviewable YAML ruleset with OpenRouter:

```bash
uv run cai-constitution compile constitutions/balanced.md -o constitutions/compiled/balanced.rules.yaml
```

The main examples are a posture spectrum:

```bash
uv run cai-constitution compile constitutions/strict.md -o constitutions/compiled/strict.rules.yaml
uv run cai-constitution compile constitutions/balanced.md -o constitutions/compiled/balanced.rules.yaml
uv run cai-constitution compile constitutions/permissive.md -o constitutions/compiled/permissive.rules.yaml
```

There is also a playful tone example:

```bash
uv run cai-constitution compile constitutions/playful.md -o constitutions/compiled/playful.rules.yaml
```

The pipeline should consume the reviewed YAML ruleset, not silently recompile
Markdown during an experiment.

When tuning the compiler prompt, keep versioned outputs under
`constitutions/compiled/versions/` before promoting a new ruleset to the
canonical `constitutions/compiled/*.rules.yaml` path.

## OpenRouter

Inference uses OpenRouter's OpenAI-compatible chat completions API. Create a
local `.env` from `.env.example` and set `OPENROUTER_API_KEY`.

Model roles are code defaults, not `.env` settings. Rule compilation uses
`deepseek/deepseek-v4-pro` with high reasoning. Dataset generation uses
`deepseek/deepseek-v4-flash` with medium reasoning. Both CLIs expose `--model`,
`--reasoning-effort`, `--reasoning-max-tokens`, and `--max-tokens` overrides for
experiments.

Model calls that expect JSON use OpenRouter `json_schema` response formats, not
prompt-only JSON instructions. The current structured calls are constitution
compilation and rule critique.

Try a one-off request with:

```bash
uv run cai-openrouter chat "Say this is a test"
```

## Dataset generation

Generate CAI-style critique/revision rows from Anthropic HH-RLHF harmless-base
with a reviewed YAML ruleset:

```bash
uv run cai-dataset generate \
  --rules constitutions/compiled/balanced.rules.yaml \
  --output data/generated/balanced.jsonl \
  --max-samples 128 \
  --concurrency 16
```

Use `--max-samples -1` for the full split. The output is JSONL and includes the
initial response, sampled rule, critique, revision, and `chosen`/`rejected`
message pairs for later training steps. Re-running the same command resumes by
skipping source indices already present in the output file.

Run tests with:

```bash
uv run python -m unittest discover -s tests
```
