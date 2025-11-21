#!/usr/bin/env python3
"""
Script to fix linting errors in the GreenLang codebase.
Fixes syntax errors, undefined names, unused imports, and other issues.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import ast
import black
from collections import defaultdict

class LintFixer:
    def __init__(self, base_path: Path = Path("greenlang")):
        self.base_path = base_path
        self.fixes_count = defaultdict(int)
        self.errors_fixed = []

    def fix_syntax_errors(self, file_path: Path) -> bool:
        """Fix syntax errors in imports, particularly the broken import pattern."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            modified = False
            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]

                # Fix the specific pattern: incomplete import followed by misplaced import
                if i < len(lines) - 1:
                    next_line = lines[i + 1] if i + 1 < len(lines) else ""

                    # Pattern 1: from ..something import (incomplete) followed by from greenlang.determinism import
                    if ("from " in line and "import (" in line and line.strip().endswith("(")):
                        if "from greenlang.determinism import" in next_line:
                            # Skip the incomplete import line
                            self.fixes_count["syntax_errors"] += 1
                            self.errors_fixed.append(f"{file_path}: Fixed broken import pattern")
                            modified = True
                            i += 1  # Skip the incomplete line
                            continue

                # Fix standalone broken imports after intelligence imports
                if "from greenlang.intelligence import (" in line and line.strip().endswith("("):
                    # Find the matching closing parenthesis
                    j = i + 1
                    while j < len(lines) and ")" not in lines[j]:
                        j += 1

                    if j < len(lines):
                        # Collect all the imports
                        import_lines = [line]
                        for k in range(i + 1, j + 1):
                            import_lines.append(lines[k])

                        # Join and fix the import
                        full_import = "".join(import_lines)
                        # If there's a malformed import, fix it
                        if "from greenlang.determinism" in full_import:
                            # Extract the intelligence imports before the error
                            parts = full_import.split("from greenlang.determinism")
                            if parts[0].strip().endswith("("):
                                # Add closing parenthesis to intelligence import
                                new_lines.append(parts[0].rstrip("(\n") + ")\n")
                                # Add the determinism import properly
                                new_lines.append("from greenlang.determinism import DeterministicClock\n")
                                modified = True
                                self.fixes_count["syntax_errors"] += 1
                                i = j + 1
                                continue

                new_lines.append(line)
                i += 1

            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                return True

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        return False

    def add_missing_imports(self, file_path: Path) -> bool:
        """Add missing imports for undefined names."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Common missing imports patterns
            missing_imports = []

            # Check for undefined DeterministicClock
            if "DeterministicClock" in content and "from greenlang.determinism import DeterministicClock" not in content:
                if "import DeterministicClock" not in content:
                    missing_imports.append("from greenlang.determinism import DeterministicClock")
                    self.fixes_count["undefined_names"] += 1

            # Check for undefined logger
            if re.search(r'\blogger\.', content) and "import logging" not in content:
                missing_imports.append("import logging")
                missing_imports.append("logger = logging.getLogger(__name__)")
                self.fixes_count["undefined_names"] += 1

            # Check for undefined deterministic_uuid
            if "deterministic_uuid" in content and "from greenlang.determinism import deterministic_uuid" not in content:
                missing_imports.append("from greenlang.determinism import deterministic_uuid")
                self.fixes_count["undefined_names"] += 1

            # Check for undefined np (numpy)
            if re.search(r'\bnp\.', content) and "import numpy as np" not in content:
                missing_imports.append("import numpy as np")
                self.fixes_count["undefined_names"] += 1

            # Check for undefined Tuple, Optional, etc. from typing
            typing_imports = []
            if re.search(r'\bTuple\[', content) and "from typing import" not in content:
                typing_imports.append("Tuple")
            if re.search(r'\bOptional\[', content) and "from typing import" not in content:
                typing_imports.append("Optional")
            if re.search(r'\bAny\b', content) and "from typing import" not in content:
                typing_imports.append("Any")

            if typing_imports:
                # Check if there's already a typing import to extend
                if "from typing import" in content:
                    # Find and extend existing import
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith("from typing import"):
                            existing = re.findall(r'from typing import (.+)', line)[0]
                            existing_items = [item.strip() for item in existing.split(',')]
                            all_items = list(set(existing_items + typing_imports))
                            lines[i] = f"from typing import {', '.join(sorted(all_items))}"
                            content = '\n'.join(lines)
                            self.fixes_count["undefined_names"] += len(typing_imports)
                            break
                else:
                    missing_imports.append(f"from typing import {', '.join(sorted(typing_imports))}")
                    self.fixes_count["undefined_names"] += len(typing_imports)

            if missing_imports:
                # Add imports after the docstring and encoding
                lines = content.split('\n')
                insert_pos = 0

                # Find position after docstring
                for i, line in enumerate(lines):
                    if i == 0 and line.startswith('#'):
                        insert_pos = i + 1
                    elif '"""' in line or "'''" in line:
                        # Find end of docstring
                        quote = '"""' if '"""' in line else "'''"
                        if line.count(quote) == 2:  # Single line docstring
                            insert_pos = i + 1
                        else:
                            for j in range(i + 1, len(lines)):
                                if quote in lines[j]:
                                    insert_pos = j + 1
                                    break
                        break
                    elif line.strip() and not line.startswith('#'):
                        insert_pos = i
                        break

                # Add imports
                for imp in missing_imports:
                    lines.insert(insert_pos, imp)
                    insert_pos += 1

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))

                return True

        except Exception as e:
            print(f"Error adding imports to {file_path}: {e}")

        return False

    def remove_unused_imports(self, file_path: Path) -> bool:
        """Remove unused imports using AST analysis."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the file
            try:
                tree = ast.parse(content)
            except SyntaxError:
                # Can't parse, skip this file
                return False

            # Find all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, alias.asname or alias.name.split('.')[-1], node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append((f"{module}.{alias.name}", alias.asname or alias.name, node.lineno))

            # Find all name usages
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    # Get the base name
                    base = node
                    while isinstance(base, ast.Attribute):
                        base = base.value
                    if isinstance(base, ast.Name):
                        used_names.add(base.id)

            # Find unused imports
            unused_lines = set()
            for full_name, import_name, line_no in imports:
                if import_name not in used_names:
                    # Special cases - don't remove these
                    if import_name in ['__version__', 'logger', 'logging']:
                        continue
                    if '__init__.py' in str(file_path):
                        continue  # Don't remove from __init__ files

                    unused_lines.add(line_no - 1)  # Convert to 0-indexed
                    self.fixes_count["unused_imports"] += 1

            if unused_lines:
                lines = content.split('\n')
                new_lines = [line for i, line in enumerate(lines) if i not in unused_lines]

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))

                return True

        except Exception as e:
            print(f"Error removing unused imports from {file_path}: {e}")

        return False

    def fix_unused_variables(self, file_path: Path) -> bool:
        """Fix unused local variables by prefixing with underscore."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Common patterns for unused variables
            patterns = [
                (r'(\s+)([a-z_][a-z0-9_]*) = .+  # F841', r'\1_\2 = \3  # Fixed: was unused'),
                (r'for ([a-z_][a-z0-9_]*) in .+ if \1 not used', r'for _\1 in \2'),
                (r'except .+ as ([a-z_][a-z0-9_]*):', r'except \1 as _\2:'),
            ]

            modified = False
            for pattern, replacement in patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    modified = True
                    self.fixes_count["unused_variables"] += 1

            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True

        except Exception as e:
            print(f"Error fixing unused variables in {file_path}: {e}")

        return False

    def fix_line_length(self, file_path: Path) -> bool:
        """Fix line length violations."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            modified = False
            new_lines = []

            for line in lines:
                if len(line.rstrip()) > 120:
                    # Try to split long lines intelligently
                    if ',' in line and not line.strip().startswith('#'):
                        # Split at commas
                        parts = line.split(',')
                        if len(parts) > 1:
                            indent = len(line) - len(line.lstrip())
                            new_line = parts[0] + ',\n'
                            for part in parts[1:-1]:
                                new_line += ' ' * (indent + 4) + part.strip() + ',\n'
                            new_line += ' ' * (indent + 4) + parts[-1].strip()
                            new_lines.append(new_line)
                            modified = True
                            self.fixes_count["line_length"] += 1
                            continue

                new_lines.append(line)

            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                return True

        except Exception as e:
            print(f"Error fixing line length in {file_path}: {e}")

        return False

    def process_directory(self, directory: Path):
        """Process all Python files in a directory."""
        for py_file in directory.glob("**/*.py"):
            if "__pycache__" in str(py_file):
                continue

            print(f"Processing {py_file}...")

            # Fix in priority order
            if self.fix_syntax_errors(py_file):
                print(f"  Fixed syntax errors in {py_file}")

            if self.add_missing_imports(py_file):
                print(f"  Added missing imports to {py_file}")

            if self.remove_unused_imports(py_file):
                print(f"  Removed unused imports from {py_file}")

            if self.fix_unused_variables(py_file):
                print(f"  Fixed unused variables in {py_file}")

            if self.fix_line_length(py_file):
                print(f"  Fixed line length in {py_file}")

    def generate_report(self):
        """Generate a summary report of fixes."""
        print("\n" + "="*60)
        print("LINTING FIXES SUMMARY")
        print("="*60)

        total_fixes = 0
        for error_type, count in self.fixes_count.items():
            print(f"{error_type}: {count} fixes")
            total_fixes += count

        print(f"\nTotal fixes applied: {total_fixes}")

        if self.errors_fixed:
            print("\nDetailed fixes:")
            for fix in self.errors_fixed[:20]:  # Show first 20
                print(f"  - {fix}")
            if len(self.errors_fixed) > 20:
                print(f"  ... and {len(self.errors_fixed) - 20} more")


def main():
    """Main entry point."""
    print("Starting lint fixing process...")

    fixer = LintFixer(Path("greenlang"))
    fixer.process_directory(fixer.base_path)
    fixer.generate_report()

    print("\nLint fixing complete!")


if __name__ == "__main__":
    main()