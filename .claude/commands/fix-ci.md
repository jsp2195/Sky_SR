# fix-ci

## Purpose
Inspect CI failure logs, identify the root cause, patch minimally, and rerun relevant checks.

## Required Inputs
- CI failure logs or failing command
- Current git diff
- Relevant source and test files

## Steps
1. Read the failure logs and identify the failing command.
2. Reproduce locally when feasible.
3. Isolate the smallest root cause.
4. Patch only the files needed for the failure.
5. Rerun the failing or closest local check.
6. Report fix, checks, and residual risk.

## Output Format
- Failure summary
- Root cause
- Files changed
- Checks rerun
- Result
- Follow-up risk

## Stop Condition
Stop after the failing check is addressed or a blocker is documented.

## What Not To Do
- Do not mask failures by deleting tests.
- Do not rewrite unrelated code.
- Do not guess without logs or reproduction evidence.

