# Changelog convention

This directory holds the project's **per-push changelog** — a running set of
study notes, one file per GitHub push. The repository as a whole is a didactic
CUDA C++ implementation of a Multi-Layer Perceptron (forward + backward), so the
changelog itself is written as *learning material*: each entry explains not just
*what* changed but *why*, so a future reader (likely you, months later) can
reconstruct the reasoning behind every push.

This `README.md` documents the convention. The actual entries are the numbered
files alongside it (`0001-...md`, `0002-...md`, ...).

---

## The rule

> **Every time something new is pushed to GitHub, add one new numbered markdown
> file to `docs/changelog/` describing exactly what changed and why.**

We do *not* rewrite history: existing changelog files are append-only and are
never edited after their push (fix-ups go in a later entry). This keeps the
directory a faithful, chronological record of how the project evolved — the same
way Git history does, but in prose meant for studying rather than for `git log`.

---

## File naming

Each entry is named:

```
NNNN-short-title.md
```

| Part          | Meaning                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| `NNNN`        | Zero-padded 4-digit sequence number, starting at `0001`, +1 per push.   |
| `short-title` | Lowercase, hyphen-separated kebab-case slug summarizing the push.       |
| `.md`         | Markdown extension.                                                     |

Examples:

```
0001-initial-implementation.md
0002-add-dropout-regularization.md
0003-fix-softmax-overflow.md
```

The numeric prefix means the files **sort chronologically** in any file listing,
and the slug makes each push identifiable at a glance without opening it.

Numbers are never reused or reordered. If a push is later reverted, the revert
gets its *own* new number (e.g. `0004-revert-dropout.md`) rather than deleting
`0002`.

---

## Entry format

Each `NNNN-*.md` file follows this template. Keep it concise but complete enough
to study from without re-reading the diff:

```markdown
# NNNN — Short title

**Date:** YYYY-MM-DD
**Pushed by:** sora5801

## Summary
One or two sentences: what this push delivers, in plain language.

## What changed
- Bullet list of files added / modified / removed.
- For each, a 1–3 sentence note on its role and what CUDA/ML concept it teaches.

## Why
The motivation. What problem this solves, what it improves, or what concept it
was meant to demonstrate. This is the most important section for study notes.

## Notes / gotchas
Anything subtle: numerical-stability tricks, memory-layout assumptions, kernel
launch-config choices, things that surprised you, or follow-ups left as exercises.

## Build / run
If the build or run procedure changed, state the new steps; otherwise note
"unchanged — see top-level README.md".
```

Not every section must be long — for a tiny push, "Notes / gotchas" may be a
single line — but the headings stay so the entries are uniform and skimmable.

---

## Why bother

- **Traceability.** Each push has a human-readable rationale next to the code,
  decoupled from terse commit messages.
- **Study value.** Re-reading the entries in order is a guided tour of how the
  MLP and its CUDA kernels were built up, mistake by mistake and fix by fix.
- **Self-documenting history.** New numbered files never collide and always sort
  correctly, so the directory stays a clean timeline with zero merge friction.

See [`0001-initial-implementation.md`](0001-initial-implementation.md) for the
first entry, which records the initial implementation of the whole repo.
