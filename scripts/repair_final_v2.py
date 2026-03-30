#!/usr/bin/env python3
"""
Final Repair Script v2 - Comprehensive fix for ALL remaining broken files.

Handles:
1. `from greenlang.schemas import utcnow` inside multi-line `from X import (...)` blocks
2. `from greenlang.schemas import utcnow` inside `try:` blocks (between try and except)
3. `from greenlang.schemas import utcnow` at column 0 inside indented method bodies
4. Empty try blocks left after _utcnow removal
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

UTCNOW_RE = re.compile(r'^from greenlang\.schemas import utcnow$')
SCHEMA_RE = re.compile(r'^from greenlang\.schemas')

ROOT = Path(__file__).resolve().parent.parent


def find_broken(dirs: list[str]) -> list[tuple[Path, str, int]]:
    broken = []
    for d in dirs:
        dp = ROOT / d
        if not dp.exists():
            continue
        for f in dp.rglob("*.py"):
            if any(p in f.parts for p in ("__pycache__", ".git", ".venv")):
                continue
            if "schemas" in f.parts and "greenlang" in f.parts:
                continue
            try:
                compile(f.read_text("utf-8"), str(f), "exec")
            except SyntaxError as e:
                broken.append((f, e.msg, e.lineno or 0))
    return sorted(broken, key=lambda x: str(x[0]))


def repair_file(filepath: Path) -> tuple[bool, str]:
    """Remove ALL misplaced `from greenlang.schemas import utcnow` lines,
    then re-add at the correct top-level position if utcnow is used."""
    content = filepath.read_text("utf-8")
    lines = content.split('\n')

    # Step 1: Find and remove ALL `from greenlang.schemas import utcnow` lines
    removed_count = 0
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if UTCNOW_RE.match(stripped):
            removed_count += 1
            continue
        new_lines.append(line)

    if removed_count == 0:
        return False, "no utcnow import lines found"

    # Step 2: Check if utcnow() is actually used in the file
    remaining_text = '\n'.join(new_lines)
    uses_utcnow = bool(re.search(r'\butcnow\b', remaining_text))

    # Step 3: Fix empty try blocks
    # Pattern: try: body was removed, so next non-blank after try: is except/finally
    fixed = []
    i = 0
    while i < len(new_lines):
        line = new_lines[i]
        stripped = line.strip()

        # Check for try: with empty body
        if stripped.endswith('try:') or stripped == 'try:':
            # Look ahead past blanks
            j = i + 1
            while j < len(new_lines) and new_lines[j].strip() == '':
                j += 1
            if j < len(new_lines) and (
                new_lines[j].strip().startswith('except')
                or new_lines[j].strip().startswith('finally')
            ):
                indent = len(line) - len(line.lstrip())
                fixed.append(line)
                fixed.append(' ' * (indent + 4) + 'pass')
                i += 1
                continue

        fixed.append(line)
        i += 1

    # Step 4: If utcnow is used, add the import at the correct top-level position
    if uses_utcnow:
        # Check if import already exists (from previous partial repair)
        already_has = any(
            UTCNOW_RE.match(l.strip()) for l in fixed
        )
        if not already_has:
            insert_idx = _find_insert_point(fixed)
            fixed.insert(insert_idx, 'from greenlang.schemas import utcnow')

    new_content = '\n'.join(fixed)

    # Step 5: Verify
    try:
        compile(new_content, str(filepath), 'exec')
    except SyntaxError as e:
        return False, f"STILL BROKEN after repair: {e.msg} line {e.lineno}"

    if new_content == content:
        return False, "no changes"

    filepath.write_text(new_content, "utf-8")
    return True, f"removed {removed_count} misplaced import(s), {'re-added at top' if uses_utcnow else 'utcnow not used'}"


def _find_insert_point(lines: list[str]) -> int:
    """Find the correct top-level import insertion point."""
    insert_idx = 0
    in_paren = False
    in_docstring = False
    docstring_char = None

    for idx, line in enumerate(lines):
        s = line.strip()

        # Track multi-line strings (module docstrings)
        if not in_paren:
            for q in ('"""', "'''"):
                if q in s:
                    count = s.count(q)
                    if in_docstring and docstring_char == q:
                        in_docstring = False
                        docstring_char = None
                    elif not in_docstring and count == 1:
                        in_docstring = True
                        docstring_char = q
                    # count >= 2 means open+close on same line

        if in_docstring:
            continue

        # Track multi-line imports
        if in_paren:
            if ')' in s:
                in_paren = False
                # This line closes a top-level import
                if not line.startswith(('    ', '\t')):
                    insert_idx = idx + 1
            continue

        if ('import (' in s or 'import(' in s) and not s.startswith('#'):
            if not line.startswith(('    ', '\t')):
                in_paren = ')' not in s  # single-line parens
                if not in_paren:
                    insert_idx = idx + 1
                continue

        # Top-level import/from line
        if (s.startswith('from ') or s.startswith('import ')) and not s.startswith('#'):
            if not line.startswith(('    ', '\t')):
                insert_idx = idx + 1

    return insert_idx


def main():
    print(f"Root: {ROOT}")
    broken = find_broken(["greenlang/agents", "packs"])
    print(f"Found {len(broken)} broken files\n")

    repaired = 0
    still_broken = []

    for filepath, msg, lineno in broken:
        rel = filepath.relative_to(ROOT)
        ok, desc = repair_file(filepath)
        if ok:
            print(f"  REPAIRED: {rel} -- {desc}")
            repaired += 1
        else:
            still_broken.append((rel, desc, msg, lineno))

    print(f"\nSummary: {repaired} repaired, {len(still_broken)} remaining")
    if still_broken:
        print(f"\nStill broken ({len(still_broken)}):")
        for rel, desc, msg, lineno in still_broken:
            print(f"  {rel}: {desc} (original: {msg} line {lineno})")


if __name__ == "__main__":
    main()
