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

Generate an operational response guide with:

```bash
uv run cai-constitution compile constitutions/balanced.md -o constitutions/guides/balanced.guide.md
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
the returned JSON, then writes Markdown. The generated guide is designed to be
read by people and by the data-generation model. It includes:

- an overview
- the intended response posture
- guide sections with stable IDs
- applicability criteria
- concrete do/avoid guidance
- examples of good and bad responses

Review the Markdown guide before using it in a pipeline run. If the generated
guide needs changes, edit the source constitution Markdown and recompile so the
guide does not drift from its source.

When changing the compiler prompt or source constitution, keep generated
comparison versions in `guides/versions/` until the new output is reviewed.
