# Next Task

## Task title
Validate Phase 1 scaffold.

## Task objective
Run the required validation and inspection scripts, then review the diff for scope and completeness.

## In scope
- Validate required state files and headings.
- Generate a compact repo snapshot.
- List changed files.
- Summarize diff stats.
- Estimate token budget hotspots.

## Out of scope
- Building Phase 2.
- Adding runtime orchestration.
- Adding third-party dependencies.

## Likely files to inspect
- `state/*.md`
- `scripts/*.py`
- `.claude/agents/*.md`
- `.claude/commands/*.md`

## Likely files to modify
- None unless validation finds a Phase 1 defect.

## Required checks
- `python scripts/validate_state.py`
- `python scripts/repo_snapshot.py --max-depth 3`
- `python scripts/changed_files.py`
- `python scripts/diff_summary.py --stat`
- `python scripts/token_budget.py --top 20`

## Exit criteria
- Required files exist.
- State headings validate.
- Scripts run successfully.
- Diff contains only Phase 1 scaffold work.

