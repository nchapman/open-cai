# Constitution Format

Constitutions are Markdown files designed for humans first and code second.

Each file starts with TOML front matter:

```toml
+++
id = "core"
title = "Core Constitution"
version = "0.1.0"
description = "Short description."
tags = ["harmlessness"]
+++
```

Then add one principle per second-level heading:

```markdown
## safety-legality: Safety and Legality

Tags: safety, law
Weight: 1.0

Plain-language notes can go here.

### Critique

Prompt used to critique the assistant response.

### Revision

Prompt used to revise the assistant response.
```

The parser requires each principle to have tags, a critique prompt, and a
revision prompt. The body text is optional and exists for reviewers.

