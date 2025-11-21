#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Import Migration Helper
================================

This script helps identify and migrate old import patterns to new ones.
It can scan code for deprecated imports, suggest replacements, and
optionally auto-update imports.

Usage:
    python scripts/migration/check_imports.py --scan
    python scripts/migration/check_imports.py --fix --dry-run
    python scripts/migration/check_imports.py --fix --confirm
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import patterns to detect and their replacements
IMPORT_MIGRATIONS = {
    # Core module migrations
    'from greenlang.core': 'from core.greenlang',
    'import greenlang.core': 'import core.greenlang',

    # Specific submodule migrations
    'from greenlang.packs': 'from core.greenlang.packs',
    'from greenlang.policy': 'from core.greenlang.policy',
    'from greenlang.runtime': 'from core.greenlang.runtime',
    'from greenlang.sdk': 'from core.greenlang.sdk',
    'from greenlang.cli': 'from core.greenlang.cli',
    'from greenlang.hub': 'from core.greenlang.hub',
    'from greenlang.utils': 'from core.greenlang.utils',

    # Test migrations
    'from greenlang.test_utils': 'from tests.utils',
    'from greenlang.testing': 'from tests.framework',

    # Agent imports that should use compat layer
    'from greenlang.agents import Agent': 'from greenlang.compat import Agent',
    'from greenlang.sdk import Pipeline': 'from greenlang.compat import Pipeline',
}

class ImportScanner(ast.NodeVisitor):
    """AST visitor to scan for import statements."""

    def __init__(self):
        self.imports: List[Tuple[str, int, str]] = []  # (import_line, line_number, full_line)
        self.current_file: Optional[str] = None

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        line_num = node.lineno
        for alias in node.names:
            import_name = alias.name
            if self._should_migrate(f"import {import_name}"):
                self.imports.append((f"import {import_name}", line_num, f"import {import_name}"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements."""
        if node.module:
            line_num = node.lineno
            import_line = f"from {node.module}"
            if self._should_migrate(import_line):
                # Reconstruct the full import line
                names = [alias.name for alias in node.names]
                full_line = f"from {node.module} import {', '.join(names)}"
                self.imports.append((import_line, line_num, full_line))
        self.generic_visit(node)

    def _should_migrate(self, import_line: str) -> bool:
        """Check if an import line should be migrated."""
        for old_pattern in IMPORT_MIGRATIONS:
            if import_line.startswith(old_pattern):
                return True
        return False

class ImportMigrator:
    """Main class for scanning and migrating imports."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.scanner = ImportScanner()
        self.found_imports: Dict[str, List[Tuple[str, int, str]]] = {}

    def scan_file(self, file_path: Path) -> None:
        """Scan a single Python file for deprecated imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            self.scanner.current_file = str(file_path)
            self.scanner.imports = []
            self.scanner.visit(tree)

            if self.scanner.imports:
                rel_path = str(file_path.relative_to(self.project_root))
                self.found_imports[rel_path] = self.scanner.imports

        except Exception as e:
            print(f"Error scanning {file_path}: {e}")

    def scan_directory(self, directory: Path = None) -> None:
        """Scan a directory recursively for Python files."""
        if directory is None:
            directory = self.project_root

        for py_file in directory.rglob("*.py"):
            # Skip virtual environments and build directories
            if any(part in str(py_file) for part in ['.venv', 'venv', '__pycache__', 'build', 'dist']):
                continue
            self.scan_file(py_file)

    def get_migration_suggestion(self, import_line: str) -> str:
        """Get migration suggestion for an import line."""
        for old_pattern, new_pattern in IMPORT_MIGRATIONS.items():
            if import_line.startswith(old_pattern):
                return import_line.replace(old_pattern, new_pattern, 1)
        return import_line

    def generate_report(self) -> str:
        """Generate a migration report."""
        if not self.found_imports:
            return "No deprecated imports found!"

        report = []
        report.append("GreenLang Import Migration Report")
        report.append("=" * 35)
        report.append("")

        total_files = len(self.found_imports)
        total_imports = sum(len(imports) for imports in self.found_imports.values())

        report.append(f"Found {total_imports} deprecated imports in {total_files} files:")
        report.append("")

        for file_path, imports in sorted(self.found_imports.items()):
            report.append(f"File: {file_path}")
            for import_line, line_num, full_line in imports:
                suggestion = self.get_migration_suggestion(import_line)
                report.append(f"   Line {line_num}: {full_line}")
                if suggestion != import_line:
                    report.append(f"   Suggestion: {suggestion}")
                report.append("")

        report.append("Migration Suggestions:")
        report.append("-" * 20)
        for old, new in IMPORT_MIGRATIONS.items():
            report.append(f"  {old} => {new}")

        return "\n".join(report)

    def fix_file(self, file_path: Path, dry_run: bool = True) -> Dict[str, any]:
        """Fix imports in a single file."""
        changes = {"file": str(file_path), "modifications": [], "errors": []}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            modified_lines = lines.copy()
            modifications = 0

            for i, line in enumerate(lines):
                original_line = line.strip()
                new_line = line

                # Check each migration pattern
                for old_pattern, new_pattern in IMPORT_MIGRATIONS.items():
                    if original_line.startswith(old_pattern):
                        new_line = line.replace(old_pattern, new_pattern, 1)
                        if new_line != line:
                            modified_lines[i] = new_line
                            modifications += 1
                            changes["modifications"].append({
                                "line": i + 1,
                                "old": line.strip(),
                                "new": new_line.strip()
                            })
                            break

            if modifications > 0 and not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(modified_lines)
                changes["written"] = True
            else:
                changes["written"] = False

            changes["modifications_count"] = modifications

        except Exception as e:
            changes["errors"].append(str(e))

        return changes

    def fix_imports(self, dry_run: bool = True) -> Dict[str, any]:
        """Fix deprecated imports across all found files."""
        results = {"files_modified": 0, "total_changes": 0, "changes": [], "errors": []}

        for file_path in self.found_imports:
            full_path = self.project_root / file_path
            result = self.fix_file(full_path, dry_run)

            if result["modifications_count"] > 0:
                results["files_modified"] += 1
                results["total_changes"] += result["modifications_count"]
                results["changes"].append(result)

            if result["errors"]:
                results["errors"].extend(result["errors"])

        return results

def main():
    parser = argparse.ArgumentParser(description="GreenLang Import Migration Helper")
    parser.add_argument("--scan", action="store_true", help="Scan for deprecated imports")
    parser.add_argument("--fix", action="store_true", help="Fix deprecated imports")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument("--confirm", action="store_true", help="Actually modify files (use with --fix)")
    parser.add_argument("--path", type=str, help="Specific path to scan/fix")
    parser.add_argument("--output", type=str, help="Output file for report")

    args = parser.parse_args()

    if not (args.scan or args.fix):
        parser.print_help()
        return

    # Initialize migrator
    migrator = ImportMigrator(PROJECT_ROOT)

    # Scan for imports
    print("Scanning for deprecated imports...")
    if args.path:
        scan_path = Path(args.path)
        if scan_path.is_file():
            migrator.scan_file(scan_path)
        else:
            migrator.scan_directory(scan_path)
    else:
        migrator.scan_directory()

    if args.scan:
        # Generate and show report
        report = migrator.generate_report()
        print(report)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")

    if args.fix:
        if not migrator.found_imports:
            print("No deprecated imports found to fix.")
            return

        # Fix imports
        dry_run = not args.confirm
        if dry_run:
            print("\n--- DRY RUN MODE ---")
            print("Showing what would be changed. Use --confirm to actually modify files.")

        results = migrator.fix_imports(dry_run)

        print(f"\nResults:")
        print(f"Files with changes: {results['files_modified']}")
        print(f"Total modifications: {results['total_changes']}")

        if results["changes"]:
            print(f"\nChanges {'(preview)' if dry_run else '(applied)'}:")
            for change in results["changes"]:
                print(f"\nFile: {change['file']}")
                for mod in change["modifications"]:
                    print(f"   Line {mod['line']}:")
                    print(f"   - {mod['old']}")
                    print(f"   + {mod['new']}")

        if results["errors"]:
            print(f"\nErrors:")
            for error in results["errors"]:
                print(f"   ERROR: {error}")

if __name__ == "__main__":
    main()