# review-diff

## Purpose
Inspect the current git diff, compare it to the task and constraints, and classify the result.

## Required Inputs
- Current task or `state/NEXT_TASK.md`
- `state/PROJECT_CONSTRAINTS.md`
- Current git diff

## Steps
1. Read task and constraints.
2. Run `git status --short`.
3. Inspect `git diff --stat` and relevant file diffs.
4. Compare changed files and behavior to intended scope.
5. Assess tests run against changed behavior.
6. Classify as PASS, PASS WITH WARNINGS, or FAIL.

## Output Format
- Verdict
- Findings
- Scope assessment
- Test assessment
- Required fixes or follow-up

## Stop Condition
Stop after the review verdict.

## What Not To Do
- Do not edit files unless explicitly asked.
- Do not approve unexplained broad rewrites.
- Do not ignore missing tests for behavioral changes.

