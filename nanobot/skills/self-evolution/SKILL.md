---
name: self-evolution
description: Improve agent capability through controlled skill evolution. Use when repeated failures, repeated manual fixes, or missing capabilities appear across tasks and you need to propose or apply small updates to workspace skills.
---

# Self Evolution

Use this workflow to evolve capabilities safely across skills and tools.

## Workflow

1. Identify one concrete gap from the current task.
2. Check existing workspace skills with `skill_manager(action="list")`.
3. Check existing workspace tools with `tool_manager(action="list")`.
4. Prefer `update` before creating a new skill or tool.
5. If no suitable component exists, create one minimal skill or tool with narrow scope.
6. Keep changes reversible. Use `deprecate` instead of deletion.

## Decision Rules

- Use `update` for missing steps, bad prompts, weak examples, or repetitive manual operations.
- Use `create` only when no existing skill/tool can absorb the gap.
- Use `deprecate` when a skill/tool is obsolete or replaced.
- Do not modify builtin skills under `nanobot/skills/`.
- Keep tool code under `workspace/tools/` and reload via `tool_manager(action="reload")` after changes.

## Quality Bar

- One change should solve one repeated problem.
- Keep skill body concise and procedural.
- Keep tool interfaces small and explicit (clear parameter schema).
- Validate by retrying the original task after change.
- If confidence is low, return a proposal first instead of applying changes.
