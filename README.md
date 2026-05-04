# Claude Code Orchestrator

Minimal, robust, token-efficient Claude Code control layer for advanced AI and research coding projects.

This repo is meant to be copied into another coding repo so Claude Code behaves more like a disciplined engineering team: inspect first, plan one phase, implement a small diff, run targeted checks, review the diff, compact state, and stop.

## What This Is

- A set of Claude Code agents with strict roles and stop conditions.
- A set of reusable slash commands for common engineering loops.
- Durable state files for compact handoffs between sessions.
- Small standard-library Python scripts for repo inspection, diff summaries, checks, and context budgeting.
- YAML workflow recipes for bug fixes, features, refactors, research code, and experiment reports.

## What This Is Not

- Not a runtime orchestration engine.
- Not a chatbot framework.
- Not an agent swarm.
- Not a web UI.
- Not a background worker system.
- Not based on LangGraph, CrewAI, AutoGen, vector databases, or vendored external code.

## Core Loop

1. Inspect repo.
2. Read state.
3. Plan one phase or task.
4. Implement a minimal diff.
5. Run targeted tests.
6. Review git diff.
7. Compact durable state.
8. Stop.

## Install And Use

Copy this control layer into a target repo:

```bash
cp -R .claude state scripts workflows templates README.md pyproject.toml /path/to/target-repo/
```

Then start with:

```bash
python scripts/validate_state.py
python scripts/repo_snapshot.py --max-depth 3
```

In Claude Code, use the commands in `.claude/commands` as slash commands. The recommended first commands are:

- `/inspect-repo`
- `/state-of-affairs`
- `/build-phased-plan`
- `/execute-next-task`
- `/review-diff`
- `/compact-state`

## State Files

- `state/AGENT_STATE.md` records the current phase, objective, changed files, tests, risks, and next task.
- `state/PROJECT_CONSTRAINTS.md` records global constraints and forbidden actions.
- `state/DECISIONS.md` records durable decisions only.
- `state/NEXT_TASK.md` keeps the next task small and testable.
- `state/TEST_COMMANDS.md` documents fast, targeted, and full checks.

Keep state concise. It should preserve what the next Claude session needs, not a full transcript.

## Scripts

All scripts use the Python standard library.

```bash
python scripts/repo_snapshot.py --max-depth 3
python scripts/changed_files.py
python scripts/diff_summary.py --stat
python scripts/run_checks.py --dry-run
python scripts/validate_state.py
python scripts/token_budget.py --top 20
```

Use `repo_snapshot.py` and `token_budget.py` before pasting context into Claude. Use `changed_files.py` and `diff_summary.py` before review or PR prep. Use `run_checks.py` to execute checks recorded in `state/TEST_COMMANDS.md` or a one-off `--command`.

`run_checks.py` is intended for simple one-line commands. Run complex shell commands manually or wrap them in a script when they need pipes, environment variables, `cd` chains, redirects, or multi-line shell logic.

## Example Workflows

- `workflows/bugfix.yaml`: reproduce, isolate, patch, verify.
- `workflows/feature.yaml`: deliver one feature slice with test boundaries.
- `workflows/refactor.yaml`: preserve behavior while changing structure.
- `workflows/research-code.yaml`: improve correctness or reproducibility without overbuilding.
- `workflows/experiment-report.yaml`: summarize evidence, metrics, limitations, and next steps.

These YAML files are recipes, not a runtime engine.

## Design Philosophy

Small, strict, token-efficient, test-gated.

The system favors explicit roles, narrow tasks, durable state, and practical scripts over heavy orchestration frameworks. Claude should inspect before acting, keep diffs reviewable, and stop at phase boundaries.
