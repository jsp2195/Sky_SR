# write-tests

## Purpose
Add minimal tests for a scoped behavior using the repository's existing test style.

## Required Inputs
- Behavior to test
- Relevant source files
- Existing test files
- `state/TEST_COMMANDS.md`

## Steps
1. Inspect existing test directories, naming, fixtures, and assertions.
2. Identify the smallest useful test surface.
3. Add or update minimal tests.
4. Run targeted checks when possible.
5. Report coverage and remaining gaps.

## Output Format
- Tests added or changed
- Commands run
- Results
- Coverage notes
- Remaining gaps

## Stop Condition
Stop after targeted tests are added and checked.

## What Not To Do
- Do not refactor production code broadly.
- Do not create a new test framework.
- Do not run expensive full suites without justification.

