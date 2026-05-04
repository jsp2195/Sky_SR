#!/usr/bin/env python3
"""Validate required state files and headings."""

from __future__ import annotations

from pathlib import Path

REQUIRED = {
    "AGENT_STATE.md": [
        "Current phase",
        "Current objective",
        "Completed work",
        "Active constraints",
        "Known risks",
        "Last changed files",
        "Last tests run",
        "Next task",
    ],
    "PROJECT_CONSTRAINTS.md": [
        "Global constraints",
        "Forbidden actions",
        "Style rules",
        "Testing rules",
        "Data/checkpoint handling rules",
    ],
    "DECISIONS.md": ["Decision log"],
    "NEXT_TASK.md": [
        "Task title",
        "Task objective",
        "In scope",
        "Out of scope",
        "Likely files to inspect",
        "Likely files to modify",
        "Required checks",
        "Exit criteria",
    ],
    "TEST_COMMANDS.md": [
        "Fast checks",
        "Targeted checks",
        "Full checks",
        "Notes on when to run each",
    ],
}


def has_heading(text: str, heading: str) -> bool:
    return any(line.strip() == f"## {heading}" or line.strip() == f"# {heading}" for line in text.splitlines())


def main() -> int:
    state_dir = Path("state")
    errors: list[str] = []
    for file_name, headings in REQUIRED.items():
        path = state_dir / file_name
        if not path.exists():
            errors.append(f"Missing file: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for heading in headings:
            if not has_heading(text, heading):
                errors.append(f"Missing heading in {path}: {heading}")

    if errors:
        print("State validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("State validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

