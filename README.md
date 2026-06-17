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
uv run cai-constitution validate constitutions/core.md
```

Compile it to a reviewable YAML ruleset with OpenRouter:

```bash
uv run cai-constitution compile constitutions/core.md -o constitutions/compiled/core.rules.yaml
```

There is also a Grok-style example:

```bash
uv run cai-constitution compile constitutions/grok.md -o constitutions/compiled/grok.rules.yaml
```

The pipeline should consume the reviewed YAML ruleset, not silently recompile
Markdown during an experiment.

## OpenRouter

Inference uses OpenRouter's OpenAI-compatible chat completions API. Create a
local `.env` from `.env.example` and set `OPENROUTER_API_KEY`.

Try a one-off request with:

```bash
uv run cai-openrouter chat "Say this is a test"
```

Run tests with:

```bash
uv run python -m unittest discover -s tests
```
