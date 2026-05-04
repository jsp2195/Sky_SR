# prep-pr

## Purpose
Summarize the diff, checks, risks, and draft a pull request body without changing code.

## Required Inputs
- Current git diff
- Task objective
- Checks run
- Known risks

## Steps
1. Run `python scripts/diff_summary.py --stat` when available.
2. Inspect relevant diffs.
3. Summarize what changed and why.
4. List checks run and results.
5. Draft a concise PR body with risks and follow-up.

## Output Format
- PR title suggestion
- Summary
- Changed files
- Tests
- Risks
- Draft PR body

## Stop Condition
Stop after drafting PR content.

## What Not To Do
- Do not edit files.
- Do not commit, push, or open a PR unless explicitly asked.
- Do not hide known test gaps.

