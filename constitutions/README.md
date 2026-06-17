# Constitution Format

Constitutions are Markdown files designed for humans first.

Use freeform Markdown. For this project's samples, prefer:

- One `#` heading naming the constitution.
- One short opening paragraph describing the overall intent.
- A bullet list of compact principles, usually phrased as `It should...`.

Example:

```markdown
# Core Constitution

This constitution describes the baseline assistant behavior we want.

- It should be honest about uncertainty.
- It should avoid helping people cause harm or break the law.
- It should preserve safe helpfulness instead of over-refusing.
```

Generate the executable ruleset with:

```bash
uv run cai-constitution compile constitutions/core.md -o constitutions/compiled/core.rules.yaml
```

The repo includes two examples:

- `core.md`: baseline harmlessness and helpfulness.
- `grok.md`: a smaller, more playful style variant inspired by the HF Grok-style
  example.

The compiler sends the full Markdown document to an OpenRouter model, validates
the returned JSON, then writes YAML. The generated YAML is a single array of
rules. Each rule has:

- `id`
- `category`
- `principle`
- `critic`
- `revision`

Review and edit the YAML by hand before using it in a pipeline run.
