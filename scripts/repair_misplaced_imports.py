#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair Misplaced Import Lines
===============================

Fixes files where `from greenlang.schemas import utcnow` or
`from greenlang.schemas.enums import ...` was accidentally inserted
inside indented code blocks, multi-line imports, or try/except blocks.

Strategy:
  1. py_compile each file to detect syntax errors
  2. For broken files, find misplaced schema import lines
  3. Remove them from the wrong location
  4. Re-insert at the correct top-level import section
  5. Re-verify with py_compile

Usage:
    python scripts/repair_misplaced_imports.py --dry-run
    python scripts/repair_misplaced_imports.py

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import argparse
import glob
import os
import py_compile
import re
import sys
from pathlib import Path
from typing import List, Tuple


SCHEMAS_IMPORT_RE = re.compile(
    r'^(from greenlang\.schemas(?:\.(?:enums|base|fields))?\s+import\s+.+)$'
)


def find_all_python_files(dirs: List[str], root: Path) -> List[Path]:
    """Find all Python files recursively."""
    files = []
    for d in dirs:
        dir_path = root / d
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            skip = False
            for part in py_file.parts:
                if part in {"__pycache__", ".git", "node_modules", ".venv"}:
                    skip = True
                    break
            if "greenlang" in py_file.parts and "schemas" in py_file.parts:
                skip = True
            if not skip:
                files.append(py_file)
    return sorted(files)


def has_syntax_error(filepath: Path) -> bool:
    """Check if file has a syntax error."""
    try:
        py_compile.compile(str(filepath), doraise=True)
        return False
    except (py_compile.PyCompileError, SyntaxError):
        return True


def repair_file(filepath: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Repair misplaced schema imports in a file."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError) as e:
        return False, f"SKIP: {e}"

    lines = content.split('\n')
    misplaced = []
    clean_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Check if this is a schema import line that's NOT at column 0
        if SCHEMAS_IMPORT_RE.match(stripped) and not SCHEMAS_IMPORT_RE.match(line):
            # This line is indented (misplaced inside a block)
            misplaced.append(stripped)
            continue

        # Also check if a schema import line is inside a multi-line import
        # (preceded by a line ending with `(` or a line with incomplete import)
        if SCHEMAS_IMPORT_RE.match(stripped) and i > 0:
            prev = lines[i-1].rstrip()
            # Check if prev line is an incomplete multi-line import
            if prev.endswith('(') and 'import' in prev:
                misplaced.append(stripped)
                continue
            # Check if prev line starts with whitespace (we're inside a block)
            if prev.startswith('    ') or prev.startswith('\t'):
                # But only if our line is also indented
                if line.startswith('    ') or line.startswith('\t'):
                    misplaced.append(stripped)
                    continue

        clean_lines.append(line)

    if not misplaced:
        return False, "no misplaced imports found"

    # Find insertion point for the imports
    insert_idx = 0
    last_gl_import = -1
    last_import = -1

    for idx, line in enumerate(clean_lines):
        stripped = line.strip()
        if stripped.startswith('from greenlang') and not stripped.startswith('#'):
            last_gl_import = idx
        if (stripped.startswith('import ') or stripped.startswith('from ')) and not stripped.startswith('#'):
            last_import = idx

    if last_gl_import >= 0:
        insert_idx = last_gl_import + 1
    elif last_import >= 0:
        insert_idx = last_import + 1
    else:
        # After module docstring
        in_docstring = False
        for idx, line in enumerate(clean_lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    insert_idx = idx + 1
                    break
                elif stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    insert_idx = idx + 1
                    continue
                else:
                    in_docstring = True

    # Deduplicate imports
    unique_imports = list(dict.fromkeys(misplaced))

    # Insert each unique import at the correct position
    for imp in reversed(unique_imports):
        # Check if this exact import already exists in clean_lines
        already_exists = any(line.strip() == imp for line in clean_lines)
        if not already_exists:
            clean_lines.insert(insert_idx, imp)

    new_content = '\n'.join(clean_lines)

    if new_content == content:
        return False, "no changes needed"

    desc = f"moved {len(unique_imports)} import(s): {', '.join(unique_imports[:3])}"

    if not dry_run:
        filepath.write_text(new_content, encoding='utf-8')

        # Verify fix worked
        if has_syntax_error(filepath):
            # Revert and report
            filepath.write_text(content, encoding='utf-8')
            return False, f"REVERT: still broken after repair"

    return True, desc


def main():
    parser = argparse.ArgumentParser(description="Repair misplaced schema imports")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    print(f"Project root: {root}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")
    print()

    dirs = [args.path] if args.path else ["greenlang/agents", "packs"]
    files = find_all_python_files(dirs, root)
    print(f"Scanning {len(files)} Python files for syntax errors...")

    broken = []
    for f in files:
        if has_syntax_error(f):
            broken.append(f)

    print(f"Found {len(broken)} files with syntax errors")
    print()

    if not broken:
        print("No repairs needed!")
        return

    repaired = 0
    failed = 0

    for filepath in broken:
        rel = filepath.relative_to(root)
        changed, desc = repair_file(filepath, dry_run=args.dry_run)
        if changed:
            print(f"  REPAIRED: {rel}")
            print(f"    {desc}")
            repaired += 1
        else:
            print(f"  FAILED:   {rel}")
            print(f"    {desc}")
            failed += 1

    print()
    print(f"Summary: {repaired} repaired, {failed} failed, {len(broken)} total broken")


if __name__ == "__main__":
    main()
