# -*- coding: utf-8 -*-
"""
Security Hardening Refactoring Script
Team 7: Input Validation & Security Hardening

Automated refactoring to:
1. Replace logging with StructuredLogger
2. Add validate_safe_path() to all file operations
3. Add ValidationFramework to agents
4. Add XSS/SQLi validators to API endpoints

Usage:
    python scripts/security_hardening_refactor.py --task all
    python scripts/security_hardening_refactor.py --task logging
    python scripts/security_hardening_refactor.py --task path-validation
    python scripts/security_hardening_refactor.py --task validation-framework
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class SecurityRefactorTool:
    """Automated security refactoring tool."""

    def __init__(self, root_dir: Path, dry_run: bool = False):
        """
        Initialize refactoring tool.

        Args:
            root_dir: Root directory of the codebase
            dry_run: If True, only print changes without modifying files
        """
        self.root_dir = root_dir
        self.dry_run = dry_run
        self.stats = {
            "files_scanned": 0,
            "files_modified": 0,
            "logging_replacements": 0,
            "path_validations_added": 0,
            "import_updates": 0,
            "errors": []
        }

    def refactor_logging_to_structured(self, file_path: Path) -> bool:
        """
        Replace logging with StructuredLogger in a file.

        Args:
            file_path: Path to Python file

        Returns:
            True if file was modified, False otherwise
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            modified = False

            # Check if file uses logging
            if 'import logging' not in content:
                return False

            # Skip if already using StructuredLogger
            if 'from greenlang.telemetry import' in content and 'StructuredLogger' in content:
                return False

            # Pattern 1: Replace "import logging"
            if re.search(r'^import logging$', content, re.MULTILINE):
                content = re.sub(
                    r'^import logging$',
                    'from greenlang.telemetry import StructuredLogger, get_logger',
                    content,
                    count=1,
                    flags=re.MULTILINE
                )
                modified = True
                self.stats["import_updates"] += 1

            # Pattern 2: Replace "logger = logging.getLogger(__name__)"
            if 'logger = logging.getLogger' in content:
                content = re.sub(
                    r'logger\s*=\s*logging\.getLogger\(__name__\)',
                    'logger = get_logger(__name__)',
                    content
                )
                modified = True
                self.stats["logging_replacements"] += 1

            # Pattern 3: Replace "logging.getLogger(__name__)" inline
            content = re.sub(
                r'logging\.getLogger\(__name__\)',
                'get_logger(__name__)',
                content
            )

            if modified:
                if not self.dry_run:
                    file_path.write_text(content, encoding='utf-8')
                    print(f"✓ Updated logging in: {file_path.relative_to(self.root_dir)}")
                else:
                    print(f"[DRY RUN] Would update logging in: {file_path.relative_to(self.root_dir)}")
                return True

            return False

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.stats["errors"].append(error_msg)
            print(f"✗ {error_msg}")
            return False

    def add_path_validation(self, file_path: Path) -> bool:
        """
        Add validate_safe_path() to file operations.

        Args:
            file_path: Path to Python file

        Returns:
            True if file was modified, False otherwise
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            modified = False

            # Check if file has file operations
            has_file_ops = any(pattern in content for pattern in [
                'open(', 'Path(', 'with open', 'pathlib.Path'
            ])

            if not has_file_ops:
                return False

            # Skip if already has path validation
            if 'from greenlang.security' in content and 'PathTraversalValidator' in content:
                return False

            # Add import if needed
            if 'from greenlang.security' not in content:
                # Find import section (after docstring, before class/function definitions)
                import_section_end = 0
                lines = content.split('\n')

                in_docstring = False
                for i, line in enumerate(lines):
                    stripped = line.strip()

                    # Track docstrings
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        in_docstring = not in_docstring
                        continue

                    if in_docstring:
                        continue

                    # Find last import
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        import_section_end = i + 1
                    elif stripped and not stripped.startswith('#'):
                        # First non-import, non-comment line
                        break

                if import_section_end > 0:
                    lines.insert(import_section_end, 'from greenlang.security.validators import PathTraversalValidator')
                    content = '\n'.join(lines)
                    modified = True
                    self.stats["import_updates"] += 1

            # Pattern 1: Add validation to "with open(file_path)" patterns
            # Look for: with open(some_path, ...):
            pattern = r'(\s+)(with\s+open\s*\(\s*)([^,\)]+)(\s*,\s*["\'][rwba+]+["\']\s*\)\s*as\s+\w+:)'

            def add_validation_before_open(match):
                indent = match.group(1)
                with_open = match.group(2)
                path_var = match.group(3).strip()
                rest = match.group(4)

                # Add validation line before with open
                validation_line = f'{indent}PathTraversalValidator.validate_path({path_var})\n'
                return f'{validation_line}{indent}{with_open}{path_var}{rest}'

            new_content = re.sub(pattern, add_validation_before_open, content)
            if new_content != content:
                content = new_content
                modified = True
                self.stats["path_validations_added"] += content.count('PathTraversalValidator.validate_path')

            if modified:
                if not self.dry_run:
                    file_path.write_text(content, encoding='utf-8')
                    print(f"✓ Added path validation to: {file_path.relative_to(self.root_dir)}")
                else:
                    print(f"[DRY RUN] Would add path validation to: {file_path.relative_to(self.root_dir)}")
                return True

            return False

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.stats["errors"].append(error_msg)
            print(f"✗ {error_msg}")
            return False

    def find_python_files(self, exclude_patterns: List[str] = None) -> List[Path]:
        """
        Find all Python files in the codebase.

        Args:
            exclude_patterns: List of patterns to exclude

        Returns:
            List of Python file paths
        """
        exclude_patterns = exclude_patterns or [
            'test-v030-audit-install',
            '__pycache__',
            '.git',
            'venv',
            'env',
            '.pytest_cache',
            'node_modules'
        ]

        python_files = []
        for file_path in self.root_dir.rglob('*.py'):
            # Check if file should be excluded
            should_exclude = any(pattern in str(file_path) for pattern in exclude_patterns)
            if not should_exclude:
                python_files.append(file_path)

        return python_files

    def refactor_all_logging(self, target_dir: Path = None) -> Dict:
        """Refactor all logging to StructuredLogger."""
        target_path = target_dir or self.root_dir
        python_files = self.find_python_files()

        print(f"\n{'='*80}")
        print(f"TASK 1: Refactoring logging to StructuredLogger")
        print(f"{'='*80}")
        print(f"Scanning {len(python_files)} Python files...\n")

        for file_path in python_files:
            self.stats["files_scanned"] += 1
            if self.refactor_logging_to_structured(file_path):
                self.stats["files_modified"] += 1

        return self.stats

    def add_all_path_validations(self, target_dir: Path = None) -> Dict:
        """Add path validation to all file operations."""
        target_path = target_dir or self.root_dir
        python_files = self.find_python_files()

        print(f"\n{'='*80}")
        print(f"TASK 2: Adding path validation to file operations")
        print(f"{'='*80}")
        print(f"Scanning {len(python_files)} Python files...\n")

        for file_path in python_files:
            if self.add_path_validation(file_path):
                self.stats["files_modified"] += 1

        return self.stats

    def print_summary(self):
        """Print refactoring summary."""
        print(f"\n{'='*80}")
        print(f"REFACTORING SUMMARY")
        print(f"{'='*80}")
        print(f"Files scanned:              {self.stats['files_scanned']}")
        print(f"Files modified:             {self.stats['files_modified']}")
        print(f"Logging replacements:       {self.stats['logging_replacements']}")
        print(f"Path validations added:     {self.stats['path_validations_added']}")
        print(f"Import updates:             {self.stats['import_updates']}")
        print(f"Errors:                     {len(self.stats['errors'])}")

        if self.stats['errors']:
            print(f"\nErrors encountered:")
            for error in self.stats['errors'][:10]:  # Show first 10
                print(f"  - {error}")
            if len(self.stats['errors']) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more")

        print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Security Hardening Refactoring Tool')
    parser.add_argument('--task', choices=['all', 'logging', 'path-validation'],
                        default='all', help='Refactoring task to run')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run mode (no file modifications)')
    parser.add_argument('--target-dir', type=str, default=None,
                        help='Target directory (default: repository root)')

    args = parser.parse_args()

    # Determine root directory
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    if args.target_dir:
        target_dir = Path(args.target_dir)
        if not target_dir.exists():
            print(f"Error: Target directory does not exist: {target_dir}")
            sys.exit(1)
    else:
        target_dir = None

    print(f"Security Hardening Refactoring Tool")
    print(f"Root directory: {root_dir}")
    print(f"Dry run: {args.dry_run}")
    print(f"Task: {args.task}\n")

    # Create refactoring tool
    tool = SecurityRefactorTool(root_dir, dry_run=args.dry_run)

    # Run selected task
    if args.task == 'logging':
        tool.refactor_all_logging(target_dir)
    elif args.task == 'path-validation':
        tool.add_all_path_validations(target_dir)
    elif args.task == 'all':
        tool.refactor_all_logging(target_dir)
        tool.stats["files_scanned"] = 0  # Reset for next task
        tool.add_all_path_validations(target_dir)

    # Print summary
    tool.print_summary()

    # Exit code
    sys.exit(0 if len(tool.stats['errors']) == 0 else 1)


if __name__ == '__main__':
    main()
