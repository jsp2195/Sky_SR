#!/usr/bin/env python3
"""Summarize the current git diff."""

from __future__ import annotations

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize git diff.")
    parser.add_argument("--stat", action="store_true", help="Print git diff --stat.")
    parser.add_argument("--names-only", action="store_true", help="Print changed file names only.")
    return parser.parse_args()


def git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], text=True, capture_output=True, check=False)


def ensure_git_repo() -> bool:
    result = git(["rev-parse", "--is-inside-work-tree"])
    if result.returncode != 0:
        print("Not a git repository.")
        return False
    return True


def output_lines(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in result.stdout.splitlines() if line]


def untracked_files() -> list[str]:
    return output_lines(git(["ls-files", "--others", "--exclude-standard"]))


def print_lines(lines: list[str], empty_message: str) -> None:
    print("\n".join(lines) if lines else empty_message)


def main() -> int:
    args = parse_args()
    if not ensure_git_repo():
        return 0

    tracked_names = output_lines(git(["diff", "--name-only"]))
    untracked_names = untracked_files()

    if args.names_only:
        print_lines(tracked_names + untracked_names, "No changed files.")
        return 0

    print("# Changed files")
    print_lines(tracked_names + untracked_names, "No changed files.")

    if args.stat:
        print()
        print("# Diff stat")
        stat = git(["diff", "--stat"]).stdout.strip()
        print(stat or "No tracked file diff.")
        print()
        print("# Untracked files")
        print_lines(untracked_names, "No untracked files.")

    shortstat = git(["diff", "--shortstat"]).stdout.strip()
    if shortstat:
        print()
        print(f"# Insertions/deletions\n{shortstat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
