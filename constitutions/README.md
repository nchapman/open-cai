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

- `protective.md`: precautionary, ambiguity-sensitive, and conservative around
  borderline harmful requests.
- `balanced.md`: balanced harmlessness and helpfulness.
- `permissive.md`: high-helpfulness, anti-over-refusal, and still bounded by
  concrete harm limits.

The compiler sends the full Markdown document to an OpenRouter model, creates a
short Markdown outline, then writes a compact Markdown response guide. The
generated guide is designed to be read by people and used directly by the
data-generation model. It includes:

- a concise posture paragraph
- human-readable operating guidance
- concrete practices and boundaries

It does not generate examples by default. If examples are important, include
them in the source constitution and review how their behavioral lesson is
preserved in the guide.

Review the Markdown guide before using it in a pipeline run. If the generated
guide needs changes, edit the source constitution Markdown and recompile so the
guide does not drift from its source.

When changing the compiler prompt or source constitution, regenerate the guide
and review the diff before using it for data generation.
