#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_utcnow() → shared utcnow() Migration Script
===============================================

Replaces local _utcnow() function definitions across the codebase with
imports from the shared greenlang.schemas module.

Handles these patterns:
  1. Removes `def _utcnow() -> datetime:` function definitions (+ body)
  2. Adds `from greenlang.schemas import utcnow` (if not already present)
  3. Replaces `_utcnow()` call sites with `utcnow()`
  4. Adds `_utcnow = utcnow` alias ONLY if there are non-call usages

Usage:
    python scripts/migrate_utcnow.py --dry-run          # Preview changes
    python scripts/migrate_utcnow.py                     # Apply changes
    python scripts/migrate_utcnow.py --path greenlang/agents/data  # Specific dir

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


# Directories to process
DEFAULT_DIRS = [
    "greenlang/agents",
    "packs",
]

# Files/dirs to skip
SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    "dist", "build", ".eggs", ".tox", ".mypy_cache",
}

SKIP_FILES = {
    "migrate_utcnow.py",  # Don't migrate self
    "base.py",  # Don't touch the source definition in schemas/base.py
}

# Pattern to match _utcnow function definition (def + body)
# Handles both:
#   def _utcnow() -> datetime:
#       """..."""
#       return datetime.now(timezone.utc).replace(microsecond=0)
#
#   def _utcnow():
#       return datetime.now(timezone.utc).replace(microsecond=0)
FUNC_DEF_PATTERN = re.compile(
    r'^def _utcnow\(\)[^:]*:.*?(?=\n(?:def |class |\S)|\Z)',
    re.MULTILINE | re.DOTALL,
)

# More precise: match def _utcnow block including docstring and return
FUNC_DEF_BLOCK = re.compile(
    r'\ndef _utcnow\(\)[^\n]*:\n(?:    [^\n]*\n)*',
    re.MULTILINE,
)

# Pattern to match import of schemas utcnow
SCHEMAS_IMPORT = re.compile(
    r'from greenlang\.schemas(?:\.base)?\s+import\s+.*\butcnow\b'
)

# Pattern to match _utcnow = utcnow alias
ALIAS_PATTERN = re.compile(r'^_utcnow\s*=\s*utcnow\s*$', re.MULTILINE)

# Pattern for _utcnow() call sites
CALL_PATTERN = re.compile(r'\b_utcnow\(\)')

# Pattern for default_factory=_utcnow
FACTORY_PATTERN = re.compile(r'default_factory\s*=\s*_utcnow\b')

# Pattern for the import line to add to
IMPORT_FROM_SCHEMAS = re.compile(
    r'^from greenlang\.schemas import \(([^)]*)\)',
    re.MULTILINE | re.DOTALL,
)

IMPORT_FROM_SCHEMAS_SINGLE = re.compile(
    r'^from greenlang\.schemas import (.+)$',
    re.MULTILINE,
)


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    parts = path.parts
    for part in parts:
        if part in SKIP_DIRS:
            return True
    if path.name in SKIP_FILES:
        return True
    # Don't touch the schemas module itself
    if "greenlang" in parts and "schemas" in parts:
        return True
    return False


def find_python_files(dirs: List[str], root: Path) -> List[Path]:
    """Find all Python files in given directories."""
    files = []
    for d in dirs:
        dir_path = root / d
        if not dir_path.exists():
            print(f"  Warning: {dir_path} does not exist, skipping")
            continue
        for py_file in dir_path.rglob("*.py"):
            if not should_skip(py_file):
                files.append(py_file)
    return sorted(files)


def has_utcnow_def(content: str) -> bool:
    """Check if file has a _utcnow function definition."""
    return bool(re.search(r'^def _utcnow\(\)', content, re.MULTILINE))


def has_utcnow_calls(content: str) -> bool:
    """Check if file has _utcnow() call sites."""
    return bool(CALL_PATTERN.search(content))


def has_utcnow_factory(content: str) -> bool:
    """Check if file has default_factory=_utcnow references."""
    return bool(FACTORY_PATTERN.search(content))


def has_schemas_import(content: str) -> bool:
    """Check if file already imports utcnow from greenlang.schemas."""
    return bool(SCHEMAS_IMPORT.search(content))


def remove_utcnow_def(content: str) -> str:
    """Remove the _utcnow() function definition and its body."""
    lines = content.split('\n')
    result = []
    skip_block = False
    blank_after_def = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith('def _utcnow('):
            skip_block = True
            i += 1
            continue

        if skip_block:
            # Still in the function body (indented or docstring or blank)
            if stripped == '' or line.startswith('    ') or line.startswith('\t'):
                i += 1
                continue
            else:
                # Exited the function block
                skip_block = False
                # Don't add extra blank lines
                # Remove trailing blank lines before next content
                while result and result[-1].strip() == '':
                    result.pop()
                result.append('')  # Single blank separator

        result.append(line)
        i += 1

    # Clean up multiple consecutive blank lines
    cleaned = []
    prev_blank = False
    for line in result:
        is_blank = line.strip() == ''
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    return '\n'.join(cleaned)


def add_schemas_import(content: str) -> str:
    """Add 'from greenlang.schemas import utcnow' to the file."""
    # Check if there's already a greenlang.schemas import we can extend
    match_multi = IMPORT_FROM_SCHEMAS.search(content)
    if match_multi:
        # Extend existing multi-line import
        imports_block = match_multi.group(1)
        if 'utcnow' not in imports_block:
            # Add utcnow to the import list
            new_imports = imports_block.rstrip() + '\n    utcnow,\n'
            content = content[:match_multi.start(1)] + new_imports + content[match_multi.end(1):]
        return content

    match_single = IMPORT_FROM_SCHEMAS_SINGLE.search(content)
    if match_single:
        existing = match_single.group(1).strip()
        if 'utcnow' not in existing:
            new_line = f"from greenlang.schemas import {existing}, utcnow"
            content = content[:match_single.start()] + new_line + content[match_single.end():]
        return content

    # No existing schemas import - add new one
    # Find best insertion point: after last 'from greenlang' import or after imports block
    lines = content.split('\n')
    insert_idx = 0

    # Strategy: insert after the last `from greenlang.*` import line
    last_gl_import = -1
    last_any_import = -1
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('from greenlang'):
            last_gl_import = idx
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_any_import = idx

    if last_gl_import >= 0:
        insert_idx = last_gl_import + 1
    elif last_any_import >= 0:
        insert_idx = last_any_import + 1
        # Add blank line before greenlang import if coming after stdlib
        lines.insert(insert_idx, '')
        insert_idx += 1
    else:
        # No imports found, insert after module docstring
        in_docstring = False
        for idx, line in enumerate(lines):
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
            elif not in_docstring and stripped and not stripped.startswith('#'):
                insert_idx = idx
                break

    lines.insert(insert_idx, 'from greenlang.schemas import utcnow')
    return '\n'.join(lines)


def replace_utcnow_calls(content: str) -> str:
    """Replace _utcnow() calls with utcnow()."""
    content = CALL_PATTERN.sub('utcnow()', content)
    return content


def replace_utcnow_factory(content: str) -> str:
    """Replace default_factory=_utcnow with default_factory=utcnow."""
    content = FACTORY_PATTERN.sub('default_factory=utcnow', content)
    return content


def remove_alias(content: str) -> str:
    """Remove _utcnow = utcnow alias lines."""
    content = ALIAS_PATTERN.sub('', content)
    return content


def migrate_file(filepath: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Migrate a single file. Returns (changed, description)."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError) as e:
        return False, f"  SKIP (read error): {e}"

    original = content
    changes = []

    has_def = has_utcnow_def(content)
    has_calls = has_utcnow_calls(content)
    has_factory = has_utcnow_factory(content)
    has_import = has_schemas_import(content)

    if not has_def and not has_calls and not has_factory:
        return False, ""

    # Step 1: Remove _utcnow function definition
    if has_def:
        content = remove_utcnow_def(content)
        changes.append("removed _utcnow() def")

    # Step 2: Add schemas import if not present
    if not has_import:
        content = add_schemas_import(content)
        changes.append("added schemas import")

    # Step 3: Replace _utcnow() calls with utcnow()
    if has_calls:
        content = replace_utcnow_calls(content)
        changes.append("replaced _utcnow() calls")

    # Step 4: Replace default_factory=_utcnow
    if has_factory:
        content = replace_utcnow_factory(content)
        changes.append("replaced default_factory=_utcnow")

    # Step 5: Remove any existing _utcnow = utcnow alias
    if ALIAS_PATTERN.search(content):
        content = remove_alias(content)
        changes.append("removed alias")

    # Step 6: Clean up unnecessary datetime imports if _utcnow was the only user
    # (Don't remove - other code may still need datetime/timezone)

    if content == original:
        return False, ""

    desc = f"  {'DRY-RUN' if dry_run else 'MIGRATED'}: {', '.join(changes)}"

    if not dry_run:
        filepath.write_text(content, encoding='utf-8')

    return True, desc


def main():
    parser = argparse.ArgumentParser(
        description="Migrate _utcnow() definitions to shared greenlang.schemas.utcnow"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing files"
    )
    parser.add_argument(
        "--path", type=str, default=None,
        help="Specific directory to migrate (relative to project root)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show all files processed, not just changed"
    )
    args = parser.parse_args()

    # Find project root (directory containing greenlang/)
    root = Path(__file__).resolve().parent.parent
    if not (root / "greenlang").exists():
        print(f"Error: Cannot find greenlang/ in {root}")
        sys.exit(1)

    print(f"Project root: {root}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")
    print()

    dirs = [args.path] if args.path else DEFAULT_DIRS
    files = find_python_files(dirs, root)
    print(f"Found {len(files)} Python files to check")
    print()

    changed_count = 0
    error_count = 0

    for filepath in files:
        try:
            changed, desc = migrate_file(filepath, dry_run=args.dry_run)
            if changed:
                rel = filepath.relative_to(root)
                print(f"  {rel}")
                if desc:
                    print(f"    {desc}")
                changed_count += 1
            elif args.verbose:
                rel = filepath.relative_to(root)
                print(f"  {rel} (no changes)")
        except Exception as e:
            rel = filepath.relative_to(root)
            print(f"  ERROR: {rel}: {e}")
            error_count += 1

    print()
    print(f"Summary: {changed_count} files {'would be' if args.dry_run else ''} migrated, "
          f"{error_count} errors, {len(files)} total checked")

    if args.dry_run and changed_count > 0:
        print(f"\nRun without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
