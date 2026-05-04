# execute-next-task

## Purpose
Execute only the next scoped task from state, then run targeted checks and review the diff.

## Required Inputs
- `state/NEXT_TASK.md`
- `state/PROJECT_CONSTRAINTS.md`
- Relevant source and test files

## Steps
1. Read `NEXT_TASK.md`, `PROJECT_CONSTRAINTS.md`, and `TEST_COMMANDS.md`.
2. Inspect likely files before editing.
3. Implement the smallest diff that satisfies the task.
4. Run targeted checks when possible.
5. Inspect `git diff` for scope creep.
6. Stop and report changed files, checks, and follow-up risks.

## Output Format
- Task completed
- Files changed
- Checks run
- Diff review result
- Remaining risks
- Exact next recommended task

## Stop Condition
Stop after the current task is implemented and reviewed.

## What Not To Do
- Do not proceed to later phases.
- Do not perform unrelated cleanup.
- Do not add dependencies unless explicitly required and approved.

