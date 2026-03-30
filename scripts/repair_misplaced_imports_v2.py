#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair Misplaced Schema Import Lines (v2)
==========================================

Fixes ALL cases where `from greenlang.schemas import ...` was inserted:
  1. Inside try/except blocks (between try: and except:)
  2. Inside multi-line from ... import (...) parenthesized blocks
  3. Inside indented function/class bodies
  4. At column 0 but structurally inside a block

Strategy: Remove ALL schema import lines, then re-add them at the top.

Usage:
    python scripts/repair_misplaced_imports_v2.py
"""

from __future__ import annotations

import argparse
import glob
import py_compile
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple


SCHEMA_IMPORT_RE = re.compile(
    r'^from greenlang\.schemas(?:\.(?:enums|base|fields))?\s+import\s+.+$'
)


def find_broken_files(dirs: List[str], root: Path) -> List[Path]:
    """Find all Python files with syntax errors."""
    broken = []
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
            if skip:
                continue
            try:
                py_compile.compile(str(py_file), doraise=True)
            except (py_compile.PyCompileError, SyntaxError):
                broken.append(py_file)
    return sorted(broken)


def repair_file(filepath: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Repair a file by extracting and relocating all schema imports."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError):
        return False, "SKIP: read error"

    original = content
    lines = content.split('\n')

    # Phase 1: Find all schema import lines and their locations
    schema_imports: List[str] = []
    lines_to_remove: Set[int] = set()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if SCHEMA_IMPORT_RE.match(stripped):
            schema_imports.append(stripped)
            lines_to_remove.add(i)

    if not schema_imports:
        return False, "no schema imports found"

    # Phase 2: Remove all schema import lines
    new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]

    # Phase 3: Check if removal alone fixes the syntax (some lines may be correctly placed)
    test_content = '\n'.join(new_lines)
    try:
        compile(test_content, str(filepath), 'exec')
        # Removal alone works - now re-add imports at the correct location
    except SyntaxError:
        # Removal creates new issues - try harder
        pass

    # Phase 4: Find the correct insertion point
    insert_idx = 0
    last_gl_import = -1
    last_import = -1
    in_multiline = False

    for idx, line in enumerate(new_lines):
        stripped = line.strip()

        # Track multi-line import blocks
        if 'import (' in stripped or 'import(' in stripped:
            in_multiline = True
        if in_multiline:
            if ')' in stripped:
                in_multiline = False
                if stripped.startswith('from greenlang'):
                    last_gl_import = idx
                elif stripped.startswith('from ') or stripped.startswith('import '):
                    last_import = idx
            continue

        if stripped.startswith('from greenlang') and not stripped.startswith('#'):
            last_gl_import = idx
        if (stripped.startswith('import ') or stripped.startswith('from ')) and not stripped.startswith('#'):
            if not any(new_lines[idx].startswith(ws) for ws in ('    ', '\t')):
                last_import = idx

    if last_gl_import >= 0:
        insert_idx = last_gl_import + 1
    elif last_import >= 0:
        insert_idx = last_import + 1
    else:
        # After module docstring
        in_docstring = False
        for idx, line in enumerate(new_lines):
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

    # Phase 5: Deduplicate and merge schema imports
    unique_imports = list(dict.fromkeys(schema_imports))

    # Try to merge multiple `from greenlang.schemas import X` lines
    schemas_names: Set[str] = set()
    enums_names: Set[str] = set()
    other_imports: List[str] = []

    for imp in unique_imports:
        if 'greenlang.schemas.enums' in imp:
            match = re.search(r'import\s+(.+)$', imp)
            if match:
                for name in match.group(1).split(','):
                    enums_names.add(name.strip())
        elif 'greenlang.schemas.base' in imp or 'greenlang.schemas.fields' in imp:
            other_imports.append(imp)
        elif 'greenlang.schemas' in imp:
            match = re.search(r'import\s+(.+)$', imp)
            if match:
                for name in match.group(1).split(','):
                    name = name.strip()
                    if name:
                        schemas_names.add(name)

    # Build final import lines
    final_imports = []
    if schemas_names:
        # Check if any of these already exist in the remaining content
        existing = set()
        for line in new_lines:
            stripped = line.strip()
            if stripped.startswith('from greenlang.schemas import') and 'enums' not in stripped:
                match = re.search(r'import\s+(.+)$', stripped)
                if match:
                    for n in match.group(1).replace('(', '').replace(')', '').split(','):
                        n = n.strip()
                        if n:
                            existing.add(n)
        new_names = schemas_names - existing
        if new_names:
            final_imports.append(f"from greenlang.schemas import {', '.join(sorted(new_names))}")

    if enums_names:
        # Check existing
        existing = set()
        for line in new_lines:
            stripped = line.strip()
            if 'greenlang.schemas.enums' in stripped:
                match = re.search(r'import\s+(.+)$', stripped)
                if match:
                    for n in match.group(1).replace('(', '').replace(')', '').split(','):
                        n = n.strip()
                        if n:
                            existing.add(n)
        new_names = enums_names - existing
        if new_names:
            final_imports.append(f"from greenlang.schemas.enums import {', '.join(sorted(new_names))}")

    final_imports.extend(other_imports)

    # Phase 6: Insert at correct position
    for imp in reversed(final_imports):
        # Double-check it doesn't already exist at column 0
        if not any(line.strip() == imp for line in new_lines):
            new_lines.insert(insert_idx, imp)

    new_content = '\n'.join(new_lines)

    if new_content == original:
        return False, "no changes"

    # Phase 7: Verify the fix
    try:
        compile(new_content, str(filepath), 'exec')
    except SyntaxError as e:
        return False, f"STILL BROKEN after repair: {e}"

    if not dry_run:
        filepath.write_text(new_content, encoding='utf-8')

    desc = f"relocated {len(schema_imports)} import(s)"
    return True, desc


def main():
    parser = argparse.ArgumentParser(description="Repair misplaced schema imports v2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    print(f"Project root: {root}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")
    print()

    dirs = [args.path] if args.path else ["greenlang/agents", "packs"]
    print("Scanning for broken files...")
    broken = find_broken_files(dirs, root)
    print(f"Found {len(broken)} files with syntax errors")
    print()

    repaired = 0
    failed = 0
    still_broken = []

    for filepath in broken:
        rel = filepath.relative_to(root)
        changed, desc = repair_file(filepath, dry_run=args.dry_run)
        if changed:
            print(f"  REPAIRED: {rel}")
            repaired += 1
        else:
            if "STILL BROKEN" in desc or "no schema imports" in desc:
                still_broken.append((rel, desc))
            failed += 1

    print()
    print(f"Summary: {repaired} repaired, {failed} not-repairable, {len(broken)} total")

    if still_broken:
        print(f"\nFiles that may have pre-existing errors ({len(still_broken)}):")
        for rel, desc in still_broken[:10]:
            print(f"  {rel}: {desc}")
        if len(still_broken) > 10:
            print(f"  ... and {len(still_broken) - 10} more")


if __name__ == "__main__":
    main()
