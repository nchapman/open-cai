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

Generate an executable ruleset with:

```bash
uv run cai-constitution compile constitutions/balanced.md -o constitutions/compiled/balanced.rules.yaml
```

The repo includes a risk-posture spectrum:

- `strict.md`: precautionary, ambiguity-sensitive, and conservative around
  borderline harmful requests.
- `balanced.md`: balanced harmlessness and helpfulness.
- `permissive.md`: high-helpfulness, anti-over-refusal, and still bounded by
  concrete harm limits.

It also includes `playful.md`, a smaller style variant for direct callouts with
light wit.

The compiler sends the full Markdown document to an OpenRouter model, validates
the returned JSON, then writes YAML. The generated YAML is a single array of
rules. Each rule has:

- `id`
- `category`
- `principle`
- `critic`
- `revision`

Review and edit the YAML by hand before using it in a pipeline run.

When changing the compiler prompt or source constitution, keep generated
comparison versions in `compiled/versions/` until the new output is reviewed.
