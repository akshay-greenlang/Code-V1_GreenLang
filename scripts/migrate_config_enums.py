#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config Enum Consolidation Migration Script
============================================
Removes duplicate enum class definitions from individual files and replaces
them with imports from greenlang.schemas.enums.

Target enums (15 new shared enums):
  Environment, LogLevel, HealthStatus, AlertSeverity, AlertStatus,
  ReportFormat, FileFormat, NotificationChannel, ExecutionStatus,
  ScheduleFrequency, SortOrder, ProtocolType, GeographicRegion,
  StorageBackend, LanguageCode

Strategy:
  1. Find files containing duplicate enum class definitions
  2. Remove the inline class definition (class + all members + docstring)
  3. Add import from greenlang.schemas.enums
  4. Handle files that already import from greenlang.schemas.enums (merge)

Author: GreenLang Platform Team
Date: 2026-03-30
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Enums to migrate: name -> canonical values (for verification)
TARGET_ENUMS = [
    "HealthStatus",
    "AlertSeverity",
    "ReportFormat",
    "ExportFormat",      # alias -> ReportFormat
    "LogLevel",
    "NotificationChannel",
    "Environment",
    "AlertStatus",
    "FileFormat",
    "SortOrder",
    "ExecutionStatus",
]

# ExportFormat is commonly used as an alias for ReportFormat
ENUM_RENAMES = {
    "ExportFormat": "ReportFormat",
}

# Skip these files (the canonical source)
SKIP_FILES = {
    "greenlang/schemas/enums.py",
    "greenlang\\schemas\\enums.py",
}


def remove_enum_class(content: str, enum_name: str) -> tuple[str, bool]:
    """Remove an inline enum class definition from file content.

    Returns (new_content, was_removed).
    """
    # Pattern matches:
    #   class EnumName(str, Enum):
    #       """docstring"""
    #       MEMBER = "value"
    #       ...
    # Until the next class/function/top-level code or double newline + non-indent
    pattern = re.compile(
        r'^class\s+' + re.escape(enum_name) + r'\((?:str,\s*)?Enum\):\s*\n'
        r'(?:'
        r'    .*\n|'        # indented lines
        r'\s*\n'             # blank lines within the class
        r')*',
        re.MULTILINE
    )

    match = pattern.search(content)
    if not match:
        return content, False

    # Remove the matched block
    new_content = content[:match.start()] + content[match.end():]
    # Clean up excessive blank lines left behind
    new_content = re.sub(r'\n{4,}', '\n\n\n', new_content)
    return new_content, True


def add_enum_import(content: str, enum_names: list[str]) -> str:
    """Add or merge enum imports from greenlang.schemas.enums."""
    # Apply renames
    final_names = set()
    for name in enum_names:
        final_names.add(ENUM_RENAMES.get(name, name))

    # Check if there's already an import from greenlang.schemas.enums
    existing_import = re.search(
        r'from greenlang\.schemas\.enums import\s+(.+?)$',
        content,
        re.MULTILINE
    )

    if existing_import:
        # Parse existing imports
        existing = {i.strip() for i in existing_import.group(1).split(",")}
        existing = {e for e in existing if e}
        merged = sorted(existing | final_names)
        new_import = f"from greenlang.schemas.enums import {', '.join(merged)}"
        content = content[:existing_import.start()] + new_import + content[existing_import.end():]
    else:
        # Also check for multi-line import
        existing_block = re.search(
            r'from greenlang\.schemas\.enums import \((.*?)\)',
            content,
            re.DOTALL
        )
        if existing_block:
            # Parse existing
            block = existing_block.group(1)
            existing = {i.strip().rstrip(",") for i in block.split("\n") if i.strip() and not i.strip().startswith("#")}
            merged = sorted(existing | final_names)
            if len(merged) <= 4:
                new_import = f"from greenlang.schemas.enums import {', '.join(merged)}"
            else:
                inner = ",\n    ".join(merged)
                new_import = f"from greenlang.schemas.enums import (\n    {inner},\n)"
            content = content[:existing_block.start()] + new_import + content[existing_block.end():]
        else:
            # No existing import - add one
            import_line = f"from greenlang.schemas.enums import {', '.join(sorted(final_names))}\n"

            # Find insertion point - after greenlang.schemas imports or after other imports
            schemas_pos = content.rfind("from greenlang.schemas import")
            if schemas_pos >= 0:
                eol = content.find("\n", schemas_pos)
                if eol >= 0:
                    content = content[:eol + 1] + import_line + content[eol + 1:]
            else:
                # After last import
                last_import = 0
                for m in re.finditer(r'^(?:from|import)\s+\S+', content, re.MULTILINE):
                    last_import = m.end()
                eol = content.find("\n", last_import)
                if eol >= 0:
                    content = content[:eol + 1] + import_line + content[eol + 1:]

    return content


def replace_enum_references(content: str, old_name: str, new_name: str) -> str:
    """Replace references to renamed enums (e.g. ExportFormat -> ReportFormat)."""
    if old_name == new_name:
        return content
    # Replace class usage: ExportFormat.JSON -> ReportFormat.JSON
    content = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, content)
    return content


def migrate_file(filepath: Path, dry_run: bool = False) -> dict:
    """Migrate a single file."""
    rel = str(filepath.relative_to(ROOT))
    stats = {
        "file": rel,
        "enums_removed": [],
        "imports_added": [],
        "error": None,
    }

    if rel.replace("\\", "/") in SKIP_FILES:
        return stats

    if not filepath.exists():
        stats["error"] = "File not found"
        return stats

    content = filepath.read_text(encoding="utf-8")
    original = content
    enums_found = []

    for enum_name in TARGET_ENUMS:
        # Check if this file defines this enum
        if re.search(r'^class\s+' + re.escape(enum_name) + r'\((?:str,\s*)?Enum\):', content, re.MULTILINE):
            content, removed = remove_enum_class(content, enum_name)
            if removed:
                enums_found.append(enum_name)
                stats["enums_removed"].append(enum_name)

    if enums_found:
        # Add import for the removed enums
        content = add_enum_import(content, enums_found)
        stats["imports_added"] = [ENUM_RENAMES.get(e, e) for e in enums_found]

        # Apply renames
        for old_name, new_name in ENUM_RENAMES.items():
            if old_name in enums_found:
                content = replace_enum_references(content, old_name, new_name)

        # Also remove now-unused `from enum import Enum` if no other enums remain
        if not re.search(r'class\s+\w+\((?:str,\s*)?Enum\):', content):
            # No more enum class definitions - check if Enum is still used
            if not re.search(r'\bEnum\b', content.replace("from enum import Enum", "")):
                content = re.sub(r'^from enum import Enum\s*\n?', '', content, flags=re.MULTILINE)

        content = re.sub(r'\n{4,}', '\n\n\n', content)

    if content != original:
        if not dry_run:
            filepath.write_text(content, encoding="utf-8")
        return stats

    return stats


def find_target_files() -> list[Path]:
    """Find all Python files containing target enum definitions."""
    files = set()
    for enum_name in TARGET_ENUMS:
        pattern = f"class {enum_name}(str, Enum):"
        # Search in greenlang/
        for f in ROOT.joinpath("greenlang").rglob("*.py"):
            try:
                if pattern in f.read_text(encoding="utf-8"):
                    files.add(f)
            except (UnicodeDecodeError, PermissionError):
                pass
        # Search in packs/
        for f in ROOT.joinpath("packs").rglob("*.py"):
            try:
                if pattern in f.read_text(encoding="utf-8"):
                    files.add(f)
            except (UnicodeDecodeError, PermissionError):
                pass
        # Also check for (Enum) without str,
        pattern2 = f"class {enum_name}(Enum):"
        for base in ["greenlang", "packs"]:
            for f in ROOT.joinpath(base).rglob("*.py"):
                try:
                    if pattern2 in f.read_text(encoding="utf-8"):
                        files.add(f)
                except (UnicodeDecodeError, PermissionError):
                    pass
    return sorted(files)


def main():
    dry_run = "--dry-run" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print(f"{'DRY RUN - ' if dry_run else ''}Config Enum Consolidation Migration")
    print(f"{'=' * 65}")
    print("Finding files with duplicate enum definitions...")

    files = find_target_files()
    print(f"Found {len(files)} files to process\n")

    total_removed = 0
    migrated = 0
    skipped = 0

    for filepath in files:
        stats = migrate_file(filepath, dry_run=dry_run)
        if stats.get("error"):
            print(f"  ERROR  {stats['file']}: {stats['error']}")
        elif stats["enums_removed"]:
            enums_str = ", ".join(stats["enums_removed"])
            print(f"  DONE   {stats['file']}: removed {enums_str}")
            total_removed += len(stats["enums_removed"])
            migrated += 1
        else:
            if verbose:
                print(f"  SKIP   {stats['file']}")
            skipped += 1

    print(f"\n{'=' * 65}")
    print(f"Summary:")
    print(f"  Files migrated:   {migrated}")
    print(f"  Files skipped:    {skipped}")
    print(f"  Enums removed:    {total_removed} inline definitions")
    print(f"  Target enums:     {len(TARGET_ENUMS)}")
    if dry_run:
        print(f"\n  DRY RUN - no files modified. Remove --dry-run to apply.")


if __name__ == "__main__":
    main()
