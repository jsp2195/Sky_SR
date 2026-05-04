#!/usr/bin/env python3
"""Run checks from state/TEST_COMMANDS.md or explicit commands."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured checks.")
    parser.add_argument("--file", default="state/TEST_COMMANDS.md", help="Markdown file with bash code blocks.")
    parser.add_argument("--command", action="append", help="One-off command to run. Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--continue-on-fail", action="store_true", help="Run all checks even if one fails.")
    return parser.parse_args()


def commands_from_markdown(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    commands: list[str] = []
    for block in re.findall(r"```(?:bash|sh)?\n(.*?)```", text, flags=re.DOTALL):
        for line in block.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                commands.append(stripped)
    return commands


def main() -> int:
    args = parse_args()
    commands = args.command or commands_from_markdown(Path(args.file))
    if not commands:
        print("No checks configured.")
        return 0

    failed = 0
    for command in commands:
        print(f"$ {command}")
        if args.dry_run:
            continue
        result = subprocess.run(shlex.split(command), check=False)
        if result.returncode == 0:
            print("PASS")
            continue
        print(f"FAIL exit={result.returncode}")
        failed = result.returncode
        if not args.continue_on_fail:
            return failed
    return failed


if __name__ == "__main__":
    raise SystemExit(main())

