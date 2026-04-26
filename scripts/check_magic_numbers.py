"""Lightweight magic-number reporter for GitHub governance checks.

The previous workflow depended on a non-existent flake8 plugin. This script keeps
the governance signal deterministic without adding another third-party package.
"""

from __future__ import annotations

import argparse
import json
import tokenize
from pathlib import Path

IGNORED_VALUES = {-1, 0, 1}


def _is_ignored_path(path: Path, exclude_parts: set[str]) -> bool:
    return any(part in exclude_parts for part in path.parts)


def scan_file(path: Path) -> list[dict[str, object]]:
    """Scan numeric tokens without building ASTs for thousands of repo files."""
    try:
        stream = path.open("rb")
    except OSError:
        return []

    findings: list[dict[str, object]] = []
    with stream:
        try:
            tokens = tokenize.tokenize(stream.readline)
            for token in tokens:
                if token.type != tokenize.NUMBER:
                    continue
                try:
                    value = float(token.string.replace("_", ""))
                except ValueError:
                    continue
                if value in IGNORED_VALUES:
                    continue
                findings.append(
                    {
                        "path": str(path).replace("\\", "/"),
                        "line": token.start[0],
                        "column": token.start[1],
                        "value": token.string,
                    }
                )
        except (SyntaxError, tokenize.TokenError, UnicodeDecodeError):
            return []
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--json-output", default="magic-numbers.json")
    args = parser.parse_args()

    exclude_parts = set(args.exclude)
    findings: list[dict[str, object]] = []
    for raw_path in args.paths:
        root = Path(raw_path)
        files = [root] if root.is_file() else root.rglob("*.py")
        for path in files:
            if _is_ignored_path(path, exclude_parts):
                continue
            findings.extend(scan_file(path))

    Path(args.json_output).write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"Magic-number findings: {len(findings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
