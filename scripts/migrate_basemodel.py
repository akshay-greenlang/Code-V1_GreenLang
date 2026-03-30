#!/usr/bin/env python3
"""
Migrate BaseModel → GreenLangBase across all agents and packs.

Strategy:
1. In `from pydantic import BaseModel, Field, ...` → remove BaseModel from the list
2. Add `from greenlang.schemas import GreenLangBase` at the correct top-level position
3. Replace `class Foo(BaseModel):` → `class Foo(GreenLangBase):`
4. Verify with compile()
5. If already has `from greenlang.schemas import X`, merge GreenLangBase into it

Avoids the insertion bugs from previous scripts by finding the correct
top-level position OUTSIDE parenthesized blocks and try/except bodies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# Patterns
PYDANTIC_IMPORT_RE = re.compile(
    r'^from pydantic import (.+)$'
)
CLASS_BASEMODEL_RE = re.compile(
    r'^(class \w+)\(BaseModel\):'
)
# Matches `from greenlang.schemas import X, Y, Z`
SCHEMAS_IMPORT_RE = re.compile(
    r'^from greenlang\.schemas import (.+)$'
)


def find_insert_point(lines: list[str]) -> int:
    """Find the correct top-level import insertion point.

    Returns index AFTER the last top-level import line, correctly skipping
    parenthesized import blocks and indented code.
    """
    insert_idx = 0
    in_paren = False

    for idx, line in enumerate(lines):
        s = line.strip()

        # Track multi-line parenthesized imports
        if in_paren:
            if ')' in s:
                in_paren = False
                insert_idx = idx + 1
            continue

        # Detect start of multi-line import
        if not s.startswith('#') and ('import (' in s or 'import(' in s):
            if not line.startswith(('    ', '\t')):  # top-level only
                if ')' not in s:  # multi-line
                    in_paren = True
                else:
                    insert_idx = idx + 1
                continue

        # Single-line top-level import
        if (s.startswith('from ') or s.startswith('import ')) and not s.startswith('#'):
            if not line.startswith(('    ', '\t')):
                insert_idx = idx + 1

    return insert_idx


def migrate_file(filepath: Path, dry_run: bool = False) -> tuple[bool, str]:
    """Migrate a single file from BaseModel to GreenLangBase."""
    try:
        content = filepath.read_text("utf-8")
    except (UnicodeDecodeError, PermissionError):
        return False, "SKIP: read error"

    lines = content.split('\n')

    # Check if file uses BaseModel
    has_basemodel_import = False
    has_class_basemodel = False
    for line in lines:
        s = line.strip()
        if PYDANTIC_IMPORT_RE.match(s) and 'BaseModel' in s:
            has_basemodel_import = True
        if CLASS_BASEMODEL_RE.match(s):
            has_class_basemodel = True

    if not has_basemodel_import:
        return False, "no BaseModel import"

    # Step 1: Modify the pydantic import line to remove BaseModel
    new_lines = []
    schemas_import_idx = -1  # Index of existing `from greenlang.schemas import ...` line
    schemas_import_names: set[str] = set()

    for idx, line in enumerate(lines):
        s = line.strip()

        # Check for existing schemas import line
        m = SCHEMAS_IMPORT_RE.match(s)
        if m and not line.startswith(('    ', '\t')):
            schemas_import_idx = len(new_lines)
            for name in m.group(1).split(','):
                name = name.strip()
                if name:
                    schemas_import_names.add(name)
            new_lines.append(line)
            continue

        # Process pydantic import line
        m = PYDANTIC_IMPORT_RE.match(s)
        if m and 'BaseModel' in s and not line.startswith(('    ', '\t')):
            imports_str = m.group(1)
            # Parse individual imports
            names = [n.strip() for n in imports_str.split(',')]
            remaining = [n for n in names if n != 'BaseModel']

            if remaining:
                new_import = f"from pydantic import {', '.join(remaining)}"
                new_lines.append(new_import)
            # If BaseModel was the only import, skip the line entirely
            continue

        new_lines.append(line)

    # Step 2: Replace class declarations
    final_lines = []
    for line in new_lines:
        s = line.strip()
        m = CLASS_BASEMODEL_RE.match(s)
        if m:
            # Preserve original indentation
            indent = len(line) - len(line.lstrip())
            prefix = ' ' * indent
            final_lines.append(f"{prefix}{m.group(1)}(GreenLangBase):")
        else:
            final_lines.append(line)

    # Step 3: Add GreenLangBase to schemas import
    if schemas_import_idx >= 0 and 'GreenLangBase' not in schemas_import_names:
        # Merge into existing schemas import
        schemas_import_names.add('GreenLangBase')
        sorted_names = sorted(schemas_import_names)
        new_import_line = f"from greenlang.schemas import {', '.join(sorted_names)}"
        final_lines[schemas_import_idx] = new_import_line
    elif 'GreenLangBase' not in schemas_import_names:
        # Add new schemas import at the correct position
        insert_idx = find_insert_point(final_lines)
        final_lines.insert(insert_idx, "from greenlang.schemas import GreenLangBase")

    new_content = '\n'.join(final_lines)

    if new_content == content:
        return False, "no changes"

    # Step 4: Verify
    try:
        compile(new_content, str(filepath), 'exec')
    except SyntaxError as e:
        return False, f"SYNTAX ERROR after migration: {e.msg} line {e.lineno}"

    if not dry_run:
        filepath.write_text(new_content, "utf-8")

    changes = []
    if has_class_basemodel:
        changes.append("class→GreenLangBase")
    changes.append("import updated")
    return True, ", ".join(changes)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Migrate BaseModel to GreenLangBase")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--path", type=str, default=None, help="Specific subdir to migrate")
    args = parser.parse_args()

    print(f"Root: {ROOT}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")

    dirs = [args.path] if args.path else ["greenlang/agents", "packs"]
    total = 0
    migrated = 0
    errors = 0
    skipped = 0

    for d in dirs:
        dp = ROOT / d
        if not dp.exists():
            print(f"  SKIP: {d} not found")
            continue
        for f in sorted(dp.rglob("*.py")):
            if any(p in f.parts for p in ("__pycache__", ".git", ".venv")):
                continue
            if "schemas" in f.parts and "greenlang" in f.parts:
                continue
            total += 1
            ok, desc = migrate_file(f, dry_run=args.dry_run)
            if ok:
                migrated += 1
            elif "SYNTAX ERROR" in desc:
                errors += 1
                rel = f.relative_to(ROOT)
                print(f"  ERROR: {rel}: {desc}")
            else:
                skipped += 1

    print(f"\nSummary: {migrated} migrated, {skipped} skipped, {errors} errors, {total} total")

    # Final verification
    if not args.dry_run and errors == 0:
        print("\nVerifying all files...")
        broken = 0
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
                except SyntaxError:
                    broken += 1
                    print(f"  BROKEN: {f.relative_to(ROOT)}")
        print(f"Post-migration verification: {broken} broken files")


if __name__ == "__main__":
    main()
