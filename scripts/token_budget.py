#!/usr/bin/env python3
"""Estimate line and character counts to guide Claude context use."""

from __future__ import annotations

import argparse
from pathlib import Path

EXCLUDES = {
    ".git",
    ".venv",
    "__pycache__",
    "data",
    "datasets",
    "checkpoints",
    "artifacts",
    "runs",
    "wandb",
    "logs",
    "build",
    "dist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate file sizes by lines and characters.")
    parser.add_argument("--root", default=".", help="Root directory to inspect.")
    parser.add_argument("--top", type=int, default=20, help="Number of largest files to print.")
    return parser.parse_args()


def skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return any(part in EXCLUDES for part in rel.parts)


def measure(path: Path) -> tuple[int, int] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    return len(text.splitlines()), len(text)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    rows: list[tuple[int, int, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file() or skip(path, root):
            continue
        result = measure(path)
        if result is None:
            continue
        lines, chars = result
        rows.append((chars, lines, path))

    rows.sort(reverse=True)
    print(f"# Largest text files under {root}")
    print("| Characters | Lines | File |")
    print("| ---: | ---: | --- |")
    for chars, lines, path in rows[: args.top]:
        print(f"| {chars} | {lines} | {path.relative_to(root)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

