#!/usr/bin/env python3
"""
Brute-force repair: Remove ALL `from greenlang.schemas...` lines from
everywhere in each file, then add exactly one of each at the top.
Repeat until no broken files remain (handles cascading fixes).
"""

import py_compile
import re
from pathlib import Path

SCHEMA_RE = re.compile(r'^from greenlang\.schemas')
ROOT = Path(__file__).resolve().parent.parent


def is_broken(f: Path) -> bool:
    try:
        compile(f.read_text("utf-8"), str(f), "exec")
        return False
    except SyntaxError:
        return True


def find_broken() -> list:
    broken = []
    for d in ["greenlang/agents", "packs"]:
        dp = ROOT / d
        if not dp.exists():
            continue
        for f in dp.rglob("*.py"):
            if any(p in f.parts for p in ("__pycache__", ".git", ".venv")):
                continue
            if "schemas" in f.parts and "greenlang" in f.parts:
                continue
            if is_broken(f):
                broken.append(f)
    return broken


def fix_file(f: Path) -> bool:
    text = f.read_text("utf-8")
    lines = text.split('\n')

    # Collect all schema import line contents
    schema_lines = []
    other_lines = []
    for line in lines:
        s = line.strip()
        if SCHEMA_RE.match(s):
            schema_lines.append(s)
        else:
            other_lines.append(line)

    if not schema_lines:
        return False

    # Fix empty try blocks that resulted from removing _utcnow
    fixed = []
    i = 0
    while i < len(other_lines):
        line = other_lines[i]
        s = line.strip()
        if s.startswith('try:'):
            # Look ahead for empty body (next non-blank is except/finally)
            j = i + 1
            while j < len(other_lines) and other_lines[j].strip() == '':
                j += 1
            if j < len(other_lines) and (
                other_lines[j].strip().startswith('except')
                or other_lines[j].strip().startswith('finally')
            ):
                indent = len(line) - len(line.lstrip())
                fixed.append(line)
                fixed.append(' ' * (indent + 4) + 'pass')
                i += 1
                continue
        fixed.append(line)
        i += 1

    # Deduplicate schema imports
    unique = list(dict.fromkeys(schema_lines))

    # Find insertion point
    insert_idx = 0
    for idx, line in enumerate(fixed):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('from ') or s.startswith('import '):
            if not line.startswith(('    ', '\t')):
                insert_idx = idx + 1

    # Insert deduplicated imports
    for imp in reversed(unique):
        already = any(line.strip() == imp for line in fixed)
        if not already:
            fixed.insert(insert_idx, imp)

    new_text = '\n'.join(fixed)
    try:
        compile(new_text, str(f), "exec")
        f.write_text(new_text, "utf-8")
        return True
    except SyntaxError:
        return False


def main():
    print(f"Root: {ROOT}")
    iteration = 0

    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        broken = find_broken()
        print(f"Found {len(broken)} broken files")

        if not broken:
            print("All files clean!")
            break

        if iteration > 5:
            print(f"Giving up after {iteration} iterations. {len(broken)} files still broken.")
            for f in broken[:20]:
                print(f"  {f.relative_to(ROOT)}")
            break

        repaired = 0
        for f in broken:
            if fix_file(f):
                repaired += 1
                print(f"  FIXED: {f.relative_to(ROOT)}")

        print(f"Repaired {repaired} / {len(broken)} this iteration")

        if repaired == 0:
            print("No progress, stopping.")
            for f in broken[:20]:
                try:
                    compile(f.read_text("utf-8"), str(f), "exec")
                except SyntaxError as e:
                    print(f"  {f.relative_to(ROOT)}: {e.msg} line {e.lineno}")
            break


if __name__ == "__main__":
    main()
