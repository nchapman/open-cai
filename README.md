# cai

A small, straightforward Constitutional AI pipeline playground.

This project follows the broad ideas from the Hugging Face Constitutional AI
recipe: generate an initial answer, critique it with a constitution, revise it,
then use the revised output for supervised or preference data. We are not
copying the original implementation or constitution verbatim.

## Constitutions

Constitutions live in `constitutions/*.md`. They are Markdown documents with:

- TOML front matter for machine-readable document metadata.
- One `## <id>: <title>` section per principle.
- Simple `Tags:` and `Weight:` lines.
- `### Critique` and `### Revision` sections used by the pipeline.

Validate a constitution with:

```bash
uv run cai-constitution validate constitutions/core.md
```

Run tests with:

```bash
uv run python -m unittest discover -s tests
```
