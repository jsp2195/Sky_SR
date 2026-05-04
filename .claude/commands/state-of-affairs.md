# state-of-affairs

## Purpose
Read durable state, inspect git status, and summarize the current phase, task, risks, and next action.

## Required Inputs
- `state/AGENT_STATE.md`
- `state/NEXT_TASK.md`
- `state/PROJECT_CONSTRAINTS.md`

## Steps
1. Read required state files.
2. Run `git status --short`.
3. Run `python scripts/changed_files.py` when available.
4. Summarize current phase, objective, completed work, active risks, and next action.
5. Identify stale or missing state.

## Output Format
- Current phase
- Current task
- Git state
- Risks
- Recommended next action
- State maintenance needed

## Stop Condition
Stop after summarizing state.

## What Not To Do
- Do not edit code.
- Do not update state unless explicitly asked.
- Do not infer completed work without evidence.

