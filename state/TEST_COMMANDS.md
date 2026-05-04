# Test Commands

## Fast checks
```bash
python scripts/validate_state.py
python scripts/changed_files.py
```

## Targeted checks
```bash
python scripts/repo_snapshot.py --max-depth 3
python scripts/diff_summary.py --stat
python scripts/token_budget.py --top 20
```

## Full checks
```bash
python -m compileall scripts
```

## Notes on when to run each
- Fast checks should run after state or script edits.
- Targeted checks should run before handoff for this control layer.
- Full checks are appropriate before PR prep or release tagging.

