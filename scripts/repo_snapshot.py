#!/usr/bin/env python3
"""Print a compact repository tree."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_EXCLUDES = {
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
    parser = argparse.ArgumentParser(description="Print a compact repository tree.")
    parser.add_argument("--root", default=".", help="Repository root to inspect.")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth to print.")
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files except excluded directories.",
    )
    return parser.parse_args()


def should_skip(path: Path, include_hidden: bool) -> bool:
    if path.name in DEFAULT_EXCLUDES:
        return True
    if not include_hidden and path.name.startswith(".") and path.name != ".claude":
        return True
    return False


def visible_children(path: Path, include_hidden: bool) -> list[Path]:
    children = [child for child in path.iterdir() if not should_skip(child, include_hidden)]
    return sorted(children, key=lambda child: (child.is_file(), child.name.lower()))


def print_tree(path: Path, max_depth: int, include_hidden: bool, depth: int = 0) -> None:
    if depth >= max_depth:
        return
    for child in visible_children(path, include_hidden):
        marker = "/" if child.is_dir() else ""
        print(f"{'  ' * depth}- {child.name}{marker}")
        if child.is_dir():
            print_tree(child, max_depth, include_hidden, depth + 1)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root does not exist: {root}")
        return 2

    print(f"# Repo Snapshot: {root}")
    print()
    print_tree(root, args.max_depth, args.include_hidden)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
