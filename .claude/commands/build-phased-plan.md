# build-phased-plan

## Purpose
Turn a large goal into a phase-gated plan with deliverables, likely touched files, tests, and exit criteria.

## Required Inputs
- User goal
- Repository inspection summary
- `state/PROJECT_CONSTRAINTS.md`

## Steps
1. Inspect the repo before planning.
2. Read current state and constraints.
3. Break the goal into small phases with one clear next task.
4. Define deliverables, likely files to inspect, likely files to modify, tests, and exit criteria.
5. Update `state/NEXT_TASK.md` only if the user explicitly asks.

## Output Format
- Goal summary
- Phase plan
- Next task brief
- Out-of-scope items
- Tests and exit criteria

## Stop Condition
Stop after planning the next phase/task.

## What Not To Do
- Do not implement.
- Do not plan every future phase in low-level detail.
- Do not add forbidden frameworks or orchestration systems.

