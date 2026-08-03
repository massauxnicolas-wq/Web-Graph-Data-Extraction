# CLAUDE.md

> **Read [AGENTS.md](AGENTS.md) first.** It holds the project documentation: architecture,
> module map, coordinate spaces, hard invariants, pyqtgraph gotchas verified against the
> installed version, and how to test a Qt app without a display. This file only adds
> behavioural guidance and Claude Code operating notes on top of it.

---

## Session start checklist

1. Read `AGENTS.md` — especially **§8 Invariants** and **§9 pyqtgraph notes**. Both exist
   because the app was broken once by ignoring them.
2. `python -m pytest -q` — 92 tests should pass before you change anything.
3. New computation → `core/` (must stay Qt-free). New widget behaviour → `ui/` (disposable).

## Operating notes for this repo

- **Run the app / tests offscreen.** `QT_QPA_PLATFORM=offscreen`, set before the first PyQt6
  import. You cannot drive a real interactive window from the agent environment.
- **Say what you could not verify.** Offscreen tests prove wiring, not feel. Finish by listing
  exactly what the human should click through (drag precision, colour/opacity ranges, whether
  a click reliably hits its target). Never imply interaction was verified when it wasn't.
- **Prefer a real interaction test over another construction test** when touching mouse
  handling — `QTest.mousePress/mouseMove/mouseRelease` on `image_view.viewport()`. See
  `AGENTS.md` §10, including the window-sizing trap.
- **Commit messages carry no AI attribution** — no "Generated with", no `Co-Authored-By`.
  Explain why. Small, focused commits; the user frequently asks for one commit per change.
- Don't commit run output (`/*.csv`, `/session.json`, `scratch/` are gitignored).

## Diagnosing "it's broken"

When the user reports breakage, **find the root cause before redesigning.** The one real
outage in this project's history was misdiagnosed twice as a mouse-event conflict and
"fixed" by rewriting the wrong layer; the actual cause was a one-line early return that
silently swallowed canvas updates. Reproduce first — read the offending commit's diff, and
where behaviour is in question, measure it with synthetic events rather than reasoning about
what pyqtgraph *probably* does. If a fix ships and the user says it's still broken, revert
to a known-good state first, then investigate.

---

# Behavioural guidelines

Guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
