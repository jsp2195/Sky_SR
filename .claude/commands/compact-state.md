# compact-state

## Purpose
Update durable state files and produce a concise handoff that preserves the next actionable context.

## Required Inputs
- Current task result
- Changed files
- Tests run
- Known risks

## Steps
1. Read state files.
2. Remove stale details and keep durable facts.
3. Update current phase, completed work, last changed files, tests, risks, and next task as needed.
4. Append decisions only when a real decision was made.
5. Produce a compact handoff.

## Output Format
- State files updated
- Handoff summary
- Next task
- Risks
- Checks

## Stop Condition
Stop after state is compacted and handoff is produced.

## What Not To Do
- Do not rewrite history or invent completed work.
- Do not add verbose logs to durable state.
- Do not proceed to implementation.

