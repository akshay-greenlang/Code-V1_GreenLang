#!/usr/bin/env python3
"""
Final Comprehensive Import Repair Script
==========================================

Strategy: For each broken file:
1. Remove ALL lines matching `from greenlang.schemas...` pattern
2. Also remove lines that are clearly the BODY of a removed _utcnow (leftover blank try blocks)
3. Re-add deduplicated imports at the correct top-level location
4. Fix empty try blocks by adding `pass`
5. Verify with compile()
"""

from __future__ import annotations

import glob
import py_compile
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple


SCHEMA_LINE_RE = re.compile(r'from greenlang\.schemas')


def find_broken(dirs: List[str], root: Path) -> List[Path]:
    broken = []
    for d in dirs:
        dp = root / d
        if not dp.exists():
            continue
        for f in dp.rglob("*.py"):
            skip = any(p in f.parts for p in ("__pycache__", ".git", ".venv"))
            if "schemas" in f.parts and "greenlang" in f.parts:
                skip = True
            if skip:
                continue
            try:
                compile(f.read_text("utf-8"), str(f), "exec")
            except SyntaxError:
                broken.append(f)
    return sorted(broken)


def repair(filepath: Path) -> Tuple[bool, str]:
    content = filepath.read_text("utf-8")
    original = content
    lines = content.split('\n')

    # Step 1: Extract all schema import lines and remove them
    extracted: List[str] = []
    keep: List[str] = []
    for line in lines:
        stripped = line.strip()
        if SCHEMA_LINE_RE.search(stripped) and (
            stripped.startswith('from greenlang.schemas')
        ):
            extracted.append(stripped)
        else:
            keep.append(line)

    if not extracted:
        return False, "no schema lines found"

    # Step 2: Fix empty try blocks (try: \n except: where the body was removed)
    # Pattern: try: followed by except/finally on the very next non-blank line
    fixed = []
    i = 0
    while i < len(keep):
        line = keep[i]
        stripped = line.strip()

        if stripped == 'try:':
            # Check if next non-blank line is except/finally
            j = i + 1
            while j < len(keep) and keep[j].strip() == '':
                j += 1
            if j < len(keep) and (keep[j].strip().startswith('except') or keep[j].strip().startswith('finally')):
                # Get indentation
                indent = len(line) - len(line.lstrip())
                fixed.append(line)
                fixed.append(' ' * (indent + 4) + 'pass  # _utcnow removed by migration')
                i += 1
                continue

        fixed.append(line)
        i += 1

    # Step 3: Find insertion point for imports
    insert_idx = 0
    last_gl = -1
    last_imp = -1
    in_paren = False

    for idx, line in enumerate(fixed):
        s = line.strip()
        if in_paren:
            if ')' in s:
                in_paren = False
            continue
        if ('import (' in s or 'import(' in s) and not s.startswith('#'):
            in_paren = True
            if s.startswith('from greenlang'):
                last_gl = idx
            elif s.startswith('from ') or s.startswith('import '):
                last_imp = idx
            continue
        if s.startswith('from greenlang') and not s.startswith('#'):
            last_gl = idx
        elif (s.startswith('import ') or s.startswith('from ')) and not s.startswith('#'):
            if not line.startswith(('    ', '\t')):
                last_imp = idx

    if last_gl >= 0:
        insert_idx = last_gl + 1
    elif last_imp >= 0:
        insert_idx = last_imp + 1

    # Step 4: Deduplicate imports
    unique = list(dict.fromkeys(extracted))

    # Merge same-source imports
    schemas_names: Set[str] = set()
    enums_names: Set[str] = set()

    for imp in unique:
        m = re.search(r'from greenlang\.schemas\.enums\s+import\s+(.+)$', imp)
        if m:
            for n in m.group(1).split(','):
                n = n.strip().rstrip(',')
                if n:
                    enums_names.add(n)
            continue
        m = re.search(r'from greenlang\.schemas\s+import\s+(.+)$', imp)
        if m:
            for n in m.group(1).split(','):
                n = n.strip().rstrip(',')
                if n:
                    schemas_names.add(n)

    # Check which names already exist in the file's imports
    existing_schemas = set()
    existing_enums = set()
    for line in fixed:
        s = line.strip()
        if 'greenlang.schemas' in s and not 'enums' in s and s.startswith('from'):
            m = re.search(r'import\s+(.+)$', s)
            if m:
                for n in re.split(r'[,\s]+', m.group(1)):
                    n = n.strip('() ,')
                    if n:
                        existing_schemas.add(n)
        if 'greenlang.schemas.enums' in s and s.startswith('from'):
            m = re.search(r'import\s+(.+)$', s)
            if m:
                for n in re.split(r'[,\s]+', m.group(1)):
                    n = n.strip('() ,')
                    if n:
                        existing_enums.add(n)

    new_schemas = schemas_names - existing_schemas
    new_enums = enums_names - existing_enums

    imports_to_add = []
    if new_schemas:
        imports_to_add.append(f"from greenlang.schemas import {', '.join(sorted(new_schemas))}")
    if new_enums:
        imports_to_add.append(f"from greenlang.schemas.enums import {', '.join(sorted(new_enums))}")

    for imp in reversed(imports_to_add):
        fixed.insert(insert_idx, imp)

    new_content = '\n'.join(fixed)

    # Step 5: Verify
    try:
        compile(new_content, str(filepath), 'exec')
    except SyntaxError as e:
        return False, f"STILL BROKEN: {e.msg} (line {e.lineno})"

    filepath.write_text(new_content, "utf-8")
    return True, f"relocated {len(extracted)} import(s)"


def main():
    root = Path(__file__).resolve().parent.parent
    print(f"Root: {root}")
    print("Scanning for broken files...")

    broken = find_broken(["greenlang/agents", "packs"], root)
    print(f"Found {len(broken)} broken files\n")

    repaired = 0
    still_broken = []

    for f in broken:
        rel = f.relative_to(root)
        ok, desc = repair(f)
        if ok:
            print(f"  REPAIRED: {rel}")
            repaired += 1
        else:
            still_broken.append((rel, desc))

    print(f"\nSummary: {repaired} repaired, {len(still_broken)} remaining")
    if still_broken:
        print(f"\nStill broken ({len(still_broken)}):")
        for rel, desc in still_broken:
            print(f"  {rel}: {desc}")


if __name__ == "__main__":
    main()
