#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duplicate Enum → Shared Enums Migration Script
================================================

Replaces locally-defined enum classes that duplicate the shared enums
in greenlang.schemas.enums with imports from the shared module.

Targets these commonly-duplicated enums:
  - CalculationStatus, JobStatus, ProcessingStatus
  - Severity, ValidationSeverity, Priority
  - DataQualityLevel, MatchStatus, ResolutionStatus
  - ReportingPeriod, RegulatoryFramework
  - RiskLevel, ComplianceStatus
  - ControlApproach, EmissionUnit

Usage:
    python scripts/migrate_enums.py --dry-run    # Preview changes
    python scripts/migrate_enums.py              # Apply changes

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Shared enums and their canonical member→value mappings
# (Only replace if the local definition matches exactly)
SHARED_ENUMS: Dict[str, Dict[str, str]] = {
    "CalculationStatus": {
        "PENDING": "pending", "RUNNING": "running",
        "COMPLETED": "completed", "FAILED": "failed",
    },
    "JobStatus": {
        "PENDING": "pending", "PROCESSING": "processing",
        "COMPLETED": "completed", "FAILED": "failed", "CANCELLED": "cancelled",
    },
    "ProcessingStatus": {
        "QUEUED": "queued", "IN_PROGRESS": "in_progress",
        "COMPLETED": "completed", "FAILED": "failed",
        "SKIPPED": "skipped", "CANCELLED": "cancelled",
    },
    "Severity": {
        "CRITICAL": "critical", "ERROR": "error",
        "WARNING": "warning", "INFO": "info",
    },
    "ValidationSeverity": {
        "ERROR": "error", "WARNING": "warning", "INFO": "info",
    },
    "Priority": {
        "CRITICAL": "critical", "HIGH": "high",
        "MEDIUM": "medium", "LOW": "low",
    },
    "DataQualityLevel": {
        "HIGH": "high", "MEDIUM": "medium", "LOW": "low",
        "ESTIMATED": "estimated", "DEFAULT": "default",
    },
    "MatchStatus": {
        "MATCHED": "matched", "UNMATCHED": "unmatched",
        "PARTIAL": "partial", "CONFLICT": "conflict", "PENDING": "pending",
    },
    "ResolutionStatus": {
        "PENDING": "pending", "RESOLVED": "resolved",
        "REJECTED": "rejected", "ESCALATED": "escalated",
    },
    "ReportingPeriod": {
        "DAILY": "daily", "WEEKLY": "weekly", "MONTHLY": "monthly",
        "QUARTERLY": "quarterly", "ANNUAL": "annual",
    },
    "RegulatoryFramework": {
        "GHG_PROTOCOL": "ghg_protocol", "ISO_14064": "iso_14064",
        "CSRD_ESRS_E1": "csrd_esrs_e1", "EPA_40CFR98": "epa_40cfr98",
        "UK_SECR": "uk_secr", "EU_ETS": "eu_ets", "EUDR": "eudr",
        "CBAM": "cbam", "SBTi": "sbti", "CDP": "cdp", "TCFD": "tcfd",
    },
    "RiskLevel": {
        "CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium",
        "LOW": "low", "NEGLIGIBLE": "negligible",
    },
    "ComplianceStatus": {
        "COMPLIANT": "compliant", "NON_COMPLIANT": "non_compliant",
        "PARTIALLY_COMPLIANT": "partially_compliant",
        "UNDER_REVIEW": "under_review", "NOT_ASSESSED": "not_assessed",
    },
    "ControlApproach": {
        "OPERATIONAL": "operational", "FINANCIAL": "financial",
        "EQUITY_SHARE": "equity_share",
    },
    "EmissionUnit": {
        "KG_CO2E": "kg_co2e", "TONNES_CO2E": "tonnes_co2e",
        "MT_CO2E": "mt_co2e", "KG": "kg", "TONNES": "tonnes",
    },
}

# Directories to process
DEFAULT_DIRS = [
    "greenlang/agents",
    "packs",
]

SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    "dist", "build", ".eggs", ".tox", ".mypy_cache",
}

SKIP_FILES = {
    "migrate_enums.py",
    "enums.py",  # Don't touch the canonical source
}


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    parts = path.parts
    for part in parts:
        if part in SKIP_DIRS:
            return True
    if path.name in SKIP_FILES:
        return True
    if "greenlang" in parts and "schemas" in parts:
        return True
    return False


def find_python_files(dirs: List[str], root: Path) -> List[Path]:
    """Find all Python files in given directories."""
    files = []
    for d in dirs:
        dir_path = root / d
        if not dir_path.exists():
            continue
        for py_file in dir_path.rglob("*.py"):
            if not should_skip(py_file):
                files.append(py_file)
    return sorted(files)


def extract_enum_definitions(content: str) -> Dict[str, Tuple[int, int, Dict[str, str]]]:
    """Extract enum class definitions and their members from source content.

    Returns dict: {class_name: (start_line, end_line, {member: value})}
    Lines are 0-indexed.
    """
    enums = {}
    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Match class definition like: class CalculationStatus(str, Enum):
        match = re.match(
            r'^class\s+(\w+)\s*\(\s*(?:str\s*,\s*)?Enum\s*\)\s*:',
            stripped
        )
        if match:
            class_name = match.group(1)
            start_line = i
            members = {}
            i += 1

            # Parse body
            while i < len(lines):
                body_line = lines[i]
                body_stripped = body_line.strip()

                # Empty line or docstring - continue
                if body_stripped == '' or body_stripped.startswith('"""') or body_stripped.startswith("'''"):
                    # Handle multiline docstring
                    if body_stripped.startswith('"""') or body_stripped.startswith("'''"):
                        quote = body_stripped[:3]
                        if body_stripped.count(quote) == 1:
                            # Multiline docstring
                            i += 1
                            while i < len(lines) and quote not in lines[i]:
                                i += 1
                    i += 1
                    continue

                # Check if still in the class body (indented)
                if not body_line.startswith('    ') and not body_line.startswith('\t') and body_stripped != '':
                    break

                # Parse member assignment: PENDING = "pending"
                member_match = re.match(
                    r'\s+(\w+)\s*=\s*["\'](\w+)["\']',
                    body_line
                )
                if member_match:
                    members[member_match.group(1)] = member_match.group(2)

                # Comment lines are ok
                i += 1

            end_line = i
            if members:  # Only track if we found members
                enums[class_name] = (start_line, end_line, members)
        else:
            i += 1

    return enums


def enum_matches_shared(class_name: str, members: Dict[str, str]) -> bool:
    """Check if a local enum definition matches a shared enum (subset or exact)."""
    if class_name not in SHARED_ENUMS:
        return False

    shared = SHARED_ENUMS[class_name]

    # Local enum must be a subset of or equal to shared enum
    for member, value in members.items():
        if member not in shared:
            return False
        if shared[member] != value:
            return False

    return True


def migrate_file(filepath: Path, root: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Migrate a single file's duplicate enums. Returns (changed, description)."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError) as e:
        return False, f"SKIP (read error): {e}"

    original = content

    # Find local enum definitions
    enum_defs = extract_enum_definitions(content)
    if not enum_defs:
        return False, ""

    # Check which ones match shared enums
    matching = {}
    for class_name, (start, end, members) in enum_defs.items():
        if enum_matches_shared(class_name, members):
            matching[class_name] = (start, end, members)

    if not matching:
        return False, ""

    # Remove matching enum definitions (process in reverse order to preserve line numbers)
    lines = content.split('\n')
    sorted_matches = sorted(matching.items(), key=lambda x: x[1][0], reverse=True)

    for class_name, (start, end, _) in sorted_matches:
        # Remove lines from start to end
        del lines[start:end]

    content = '\n'.join(lines)

    # Clean up multiple consecutive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Add import from greenlang.schemas.enums
    enum_names = sorted(matching.keys())

    # Check if there's already a schemas.enums import
    existing_import = re.search(
        r'^from greenlang\.schemas\.enums import (.+)$',
        content, re.MULTILINE
    )
    existing_import_multi = re.search(
        r'^from greenlang\.schemas\.enums import \(([^)]*)\)',
        content, re.MULTILINE | re.DOTALL
    )

    if existing_import_multi:
        # Extend multi-line import
        existing_names = {n.strip().rstrip(',') for n in existing_import_multi.group(1).split('\n') if n.strip()}
        all_names = sorted(existing_names | set(enum_names))
        new_import = "from greenlang.schemas.enums import (\n"
        for name in all_names:
            new_import += f"    {name},\n"
        new_import += ")"
        content = content[:existing_import_multi.start()] + new_import + content[existing_import_multi.end():]
    elif existing_import:
        # Extend single-line import
        existing_names = {n.strip() for n in existing_import.group(1).split(',')}
        all_names = sorted(existing_names | set(enum_names))
        new_import = f"from greenlang.schemas.enums import {', '.join(all_names)}"
        content = content[:existing_import.start()] + new_import + content[existing_import.end():]
    else:
        # Add new import line after last greenlang import or after imports block
        import_line = f"from greenlang.schemas.enums import {', '.join(enum_names)}"

        lines = content.split('\n')
        insert_idx = 0
        last_gl_import = -1
        last_import = -1

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('from greenlang'):
                last_gl_import = idx
            if stripped.startswith('import ') or stripped.startswith('from '):
                last_import = idx

        if last_gl_import >= 0:
            insert_idx = last_gl_import + 1
        elif last_import >= 0:
            insert_idx = last_import + 1
            lines.insert(insert_idx, '')
            insert_idx += 1
        else:
            insert_idx = 0

        lines.insert(insert_idx, import_line)
        content = '\n'.join(lines)

    # Remove now-unnecessary `from enum import Enum` if no other enums remain
    remaining_enums = extract_enum_definitions(content)
    if not remaining_enums:
        # Check if Enum is still used anywhere else
        if not re.search(r'\bEnum\b', content.replace('from enum import Enum', '')):
            content = re.sub(r'^from enum import Enum\n', '', content, flags=re.MULTILINE)

    if content == original:
        return False, ""

    changes = f"replaced {len(matching)} enum(s): {', '.join(enum_names)}"
    desc = f"{'DRY-RUN' if dry_run else 'MIGRATED'}: {changes}"

    if not dry_run:
        filepath.write_text(content, encoding='utf-8')

    return True, desc


def main():
    parser = argparse.ArgumentParser(
        description="Migrate duplicate enums to shared greenlang.schemas.enums"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes")
    parser.add_argument("--path", type=str, default=None, help="Specific directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all files")
    args = parser.parse_args()

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
            changed, desc = migrate_file(filepath, root, dry_run=args.dry_run)
            if changed:
                rel = filepath.relative_to(root)
                print(f"  {rel}")
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
