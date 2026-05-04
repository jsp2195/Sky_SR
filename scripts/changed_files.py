#!/usr/bin/env python3
"""Print changed files grouped by git status."""

from __future__ import annotations

import subprocess
from collections import defaultdict

STATUS_LABELS = {
    "M": "modified",
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "?": "untracked",
}


def git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], text=True, capture_output=True, check=False)


def main() -> int:
    if git(["rev-parse", "--is-inside-work-tree"]).returncode != 0:
        print("Not a git repository.")
        return 0

    result = git(["status", "--porcelain"])
    groups: dict[str, list[str]] = defaultdict(list)
    for line in result.stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:]
        key = "untracked" if status == "??" else STATUS_LABELS.get(status.strip()[:1], "other")
        groups[key].append(path)

    if not groups:
        print("No changed files.")
        return 0

    for label in ("modified", "added", "deleted", "renamed", "copied", "untracked", "other"):
        files = groups.get(label, [])
        if not files:
            continue
        print(f"# {label}")
        for file_name in files:
            print(file_name)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

