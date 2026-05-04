# inspect-repo

## Purpose
Inspect the repository and summarize its structure, languages, frameworks, entrypoints, tests, and excluded directories without editing files.

## Required Inputs
- Optional user focus area or path
- Repository root

## Steps
1. Run `python scripts/repo_snapshot.py --max-depth 3` when available.
2. Inspect `git status --short`.
3. Identify languages, package files, entrypoints, test directories, and scripts.
4. Note excluded or high-volume directories such as data, checkpoints, artifacts, and build output.
5. Summarize findings with uncertainty called out.

## Output Format
- Repository summary
- Language and framework signals
- Entrypoints
- Test surface
- Excluded/high-volume directories
- Risks or unknowns

## Stop Condition
Stop after inspection and summary.

## What Not To Do
- Do not edit files.
- Do not create a plan unless explicitly asked.
- Do not paste large files into context.

