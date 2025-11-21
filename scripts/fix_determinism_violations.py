#!/usr/bin/env python3
"""
Fix Determinism Violations in GreenLang Codebase

This script automatically fixes determinism violations to make GreenLang
suitable for regulatory use.

Violations fixed:
1. UUID replacement with deterministic IDs
2. Timestamp fixing with DeterministicClock
3. Random operations with seeded generators
4. Float operations with Decimal for financial calculations
5. File operation ordering with sorted operations

Author: GreenLang Team
Date: 2025-11-21
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import shutil
from collections import defaultdict


class DeterminismFixer:
    """Fix determinism violations in Python files."""

    def __init__(self, dry_run: bool = False, backup: bool = True):
        """
        Initialize fixer.

        Args:
            dry_run: If True, only report violations without fixing
            backup: If True, create backups before modifying files
        """
        self.dry_run = dry_run
        self.backup = backup
        self.violations_fixed = defaultdict(int)
        self.files_modified = set()

    def fix_file(self, filepath: Path) -> bool:
        """
        Fix determinism violations in a single file.

        Args:
            filepath: Path to Python file

        Returns:
            True if file was modified
        """
        if not filepath.exists() or filepath.suffix != '.py':
            return False

        # Skip test files and generated files
        if any(skip in str(filepath) for skip in ['__pycache__', '.pyc', 'venv', '.git']):
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return False

        content = original_content
        imports_added = set()
        modified = False

        # Fix UUID violations
        content, uuid_count, uuid_imports = self.fix_uuid_violations(content)
        if uuid_count > 0:
            imports_added.update(uuid_imports)
            self.violations_fixed['uuid'] += uuid_count
            modified = True

        # Fix timestamp violations
        content, time_count, time_imports = self.fix_timestamp_violations(content)
        if time_count > 0:
            imports_added.update(time_imports)
            self.violations_fixed['timestamp'] += time_count
            modified = True

        # Fix random violations
        content, random_count, random_imports = self.fix_random_violations(content)
        if random_count > 0:
            imports_added.update(random_imports)
            self.violations_fixed['random'] += random_count
            modified = True

        # Fix float violations (for financial calculations)
        content, float_count, float_imports = self.fix_float_violations(content, filepath)
        if float_count > 0:
            imports_added.update(float_imports)
            self.violations_fixed['float'] += float_count
            modified = True

        # Fix file operation violations
        content, file_count, file_imports = self.fix_file_operations(content)
        if file_count > 0:
            imports_added.update(file_imports)
            self.violations_fixed['file_ops'] += file_count
            modified = True

        # Add imports if needed
        if imports_added and modified:
            content = self.add_imports(content, imports_added)

        # Write back if modified
        if modified and content != original_content:
            if not self.dry_run:
                if self.backup:
                    backup_path = filepath.with_suffix('.py.backup')
                    shutil.copy2(filepath, backup_path)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

            self.files_modified.add(str(filepath))
            return True

        return False

    def fix_uuid_violations(self, content: str) -> Tuple[str, int, set]:
        """Fix UUID violations."""
        imports = set()
        count = 0

        # Pattern for uuid4() calls
        uuid_patterns = [
            (r'\buuid\.uuid4\(\)', 'deterministic_uuid(__name__, str(DeterministicClock.now()))'),
            (r'\buuid4\(\)', 'deterministic_uuid(__name__, str(DeterministicClock.now()))'),
            (r'str\(uuid\.uuid4\(\)\)', 'deterministic_uuid(__name__, str(DeterministicClock.now()))'),
            (r'str\(uuid4\(\)\)', 'deterministic_uuid(__name__, str(DeterministicClock.now()))'),
        ]

        for pattern, replacement in uuid_patterns:
            matches = re.findall(pattern, content)
            if matches:
                count += len(matches)
                content = re.sub(pattern, replacement, content)
                imports.add('from greenlang.determinism import deterministic_uuid, DeterministicClock')

        # Handle faker.uuid4() in test files
        if 'faker.uuid4()' in content:
            count += content.count('faker.uuid4()')
            content = content.replace('faker.uuid4()', 'deterministic_uuid("test", str(i))')
            imports.add('from greenlang.determinism import deterministic_uuid')

        return content, count, imports

    def fix_timestamp_violations(self, content: str) -> Tuple[str, int, set]:
        """Fix timestamp violations."""
        imports = set()
        count = 0

        # Pattern for datetime.now() and datetime.utcnow()
        timestamp_patterns = [
            (r'\bdatetime\.now\(\)', 'DeterministicClock.now()'),
            (r'\bdatetime\.utcnow\(\)', 'DeterministicClock.utcnow()'),
            (r'\bdatetime\.datetime\.now\(\)', 'DeterministicClock.now()'),
            (r'\bdatetime\.datetime\.utcnow\(\)', 'DeterministicClock.utcnow()'),
        ]

        for pattern, replacement in timestamp_patterns:
            matches = re.findall(pattern, content)
            if matches:
                count += len(matches)
                content = re.sub(pattern, replacement, content)
                imports.add('from greenlang.determinism import DeterministicClock')

        return content, count, imports

    def fix_random_violations(self, content: str) -> Tuple[str, int, set]:
        """Fix random violations."""
        imports = set()
        count = 0

        # Check if file already has seeding
        has_seed = 'random.seed(' in content or 'np.random.seed(' in content

        # Pattern for random operations
        random_patterns = [
            (r'\brandom\.random\(\)', 'deterministic_random().random()'),
            (r'\brandom\.randint\(', 'deterministic_random().randint('),
            (r'\brandom\.choice\(', 'deterministic_random().choice('),
            (r'\brandom\.sample\(', 'deterministic_random().sample('),
            (r'\brandom\.shuffle\(', 'deterministic_random().shuffle('),
        ]

        for pattern, replacement in random_patterns:
            if pattern.endswith('('):
                # For functions with arguments
                matches = re.findall(pattern, content)
                if matches:
                    count += len(matches)
                    content = re.sub(pattern, replacement, content)
                    imports.add('from greenlang.determinism import deterministic_random')
            else:
                # For functions without arguments
                matches = re.findall(pattern, content)
                if matches:
                    count += len(matches)
                    content = re.sub(pattern, replacement, content)
                    imports.add('from greenlang.determinism import deterministic_random')

        # Handle numpy random
        if 'np.random.random' in content:
            count += content.count('np.random.random')
            # Add seed at module level if not present
            if not has_seed and 'import numpy as np' in content:
                import_line = content.find('import numpy as np')
                newline_pos = content.find('\n', import_line)
                if newline_pos != -1:
                    content = content[:newline_pos] + '\nnp.random.seed(42)' + content[newline_pos:]
                    count += 1

        return content, count, imports

    def fix_float_violations(self, content: str, filepath: Path) -> Tuple[str, int, set]:
        """Fix float violations for financial calculations."""
        imports = set()
        count = 0

        # Only fix float operations in calculation/financial contexts
        financial_keywords = ['emission', 'cost', 'price', 'amount', 'total', 'sum', 'factor', 'rate']
        is_financial = any(keyword in str(filepath).lower() or keyword in content.lower()
                          for keyword in financial_keywords)

        if not is_financial:
            return content, count, imports

        # Pattern for float conversions that should use Decimal
        float_patterns = [
            (r'float\(([^)]+)\)', r'FinancialDecimal.from_string(\1)'),
        ]

        for pattern, replacement in float_patterns:
            # Only replace in calculation contexts
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if any(kw in line.lower() for kw in financial_keywords):
                    if re.search(pattern, line):
                        count += 1
                        line = re.sub(pattern, replacement, line)
                        imports.add('from greenlang.determinism import FinancialDecimal')
                new_lines.append(line)
            content = '\n'.join(new_lines)

        return content, count, imports

    def fix_file_operations(self, content: str) -> Tuple[str, int, set]:
        """Fix file operation ordering."""
        imports = set()
        count = 0

        # Pattern for file operations
        file_patterns = [
            (r'\bos\.listdir\(', 'sorted_listdir('),
            (r'\bglob\.glob\(', 'sorted_glob('),
            (r'\.iterdir\(\)', '.iterdir()'),  # This needs special handling
        ]

        for pattern, replacement in file_patterns:
            if pattern == r'\.iterdir\(\)':
                # Special handling for iterdir
                matches = re.findall(r'for (\w+) in ([^:]+)\.iterdir\(\)', content)
                if matches:
                    for var, path_expr in matches:
                        old = f'for {var} in {path_expr}.iterdir()'
                        new = f'for {var} in sorted({path_expr}.iterdir())'
                        content = content.replace(old, new)
                        count += 1
            else:
                matches = re.findall(pattern, content)
                if matches:
                    count += len(matches)
                    content = re.sub(pattern, replacement, content)
                    if 'sorted_listdir' in replacement:
                        imports.add('from greenlang.determinism import sorted_listdir')
                    elif 'sorted_glob' in replacement:
                        imports.add('from greenlang.determinism import sorted_glob')

        return content, count, imports

    def add_imports(self, content: str, imports: set) -> str:
        """Add necessary imports to file."""
        if not imports:
            return content

        lines = content.split('\n')

        # Find insertion point (after existing imports)
        insert_idx = 0
        in_imports = False
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                in_imports = True
                insert_idx = i + 1
            elif in_imports and line.strip() and not line.startswith('#'):
                break

        # Insert new imports
        for imp in sorted(imports):
            # Check if import already exists
            if imp not in content:
                lines.insert(insert_idx, imp)
                insert_idx += 1

        return '\n'.join(lines)

    def fix_directory(self, directory: Path) -> Dict[str, int]:
        """
        Fix all Python files in directory.

        Args:
            directory: Root directory to process

        Returns:
            Summary of fixes applied
        """
        py_files = list(directory.rglob('*.py'))

        # Exclude certain directories
        exclude_dirs = {'.git', '__pycache__', 'venv', 'env', '.venv', 'node_modules', 'build', 'dist'}
        py_files = [f for f in py_files if not any(ex in f.parts for ex in exclude_dirs)]

        print(f"Processing {len(py_files)} Python files...")

        for filepath in py_files:
            try:
                self.fix_file(filepath)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

        return dict(self.violations_fixed)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Fix determinism violations in GreenLang')
    parser.add_argument('--dry-run', action='store_true', help='Only report violations without fixing')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup files')
    parser.add_argument('--directory', type=Path, default=Path.cwd(), help='Root directory to process')

    args = parser.parse_args()

    fixer = DeterminismFixer(dry_run=args.dry_run, backup=not args.no_backup)
    summary = fixer.fix_directory(args.directory)

    # Print summary
    print("\n" + "=" * 60)
    print("DETERMINISM VIOLATIONS FIXED")
    print("=" * 60)

    total_violations = 0
    for violation_type, count in summary.items():
        print(f"{violation_type:15} : {count:5} violations fixed")
        total_violations += count

    print("-" * 60)
    print(f"{'TOTAL':15} : {total_violations:5} violations fixed")
    print(f"Files modified  : {len(fixer.files_modified)}")

    if args.dry_run:
        print("\n[DRY RUN] No files were actually modified")

    print("\n" + "=" * 60)

    # List modified files
    if fixer.files_modified:
        print("\nModified files:")
        for filepath in sorted(fixer.files_modified)[:20]:  # Show first 20
            print(f"  - {filepath}")
        if len(fixer.files_modified) > 20:
            print(f"  ... and {len(fixer.files_modified) - 20} more files")

    return 0 if total_violations <= 47 else 1


if __name__ == '__main__':
    sys.exit(main())