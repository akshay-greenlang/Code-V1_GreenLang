#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to automatically replace print() statements with proper logging
in Python files throughout the codebase.
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple, Set

# Files to skip (user-facing CLI output, test files, etc.)
SKIP_PATTERNS = [
    '_test.py',
    'test_',
    '/tests/',
    '/test/',
    '__pycache__',
    '.pyc',
    '/examples/',
    'setup.py',
    'conftest.py',
]

# Directories to prioritize
PRIORITY_DIRS = [
    'greenlang/cli',
    'greenlang/agents',
    'GL-CSRD-APP/CSRD-Reporting-Platform/services',
    'GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services',
    '.greenlang/scripts',
    '.greenlang/tools',
]


def should_skip_file(file_path: str) -> bool:
    """Check if file should be skipped."""
    for pattern in SKIP_PATTERNS:
        if pattern in file_path:
            return True
    return False


def has_logging_import(content: str) -> bool:
    """Check if file already imports logging."""
    patterns = [
        r'^import logging',
        r'^from logging import',
        r'logger\s*=\s*logging\.getLogger',
    ]
    for pattern in patterns:
        if re.search(pattern, content, re.MULTILINE):
            return True
    return False


def add_logging_import(content: str) -> str:
    """Add logging import and logger initialization if not present."""
    lines = content.split('\n')

    # Find the position to insert imports (after docstring and before first code)
    insert_pos = 0
    in_docstring = False
    docstring_char = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle docstrings
        if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
            docstring_char = stripped[:3]
            in_docstring = True
            if stripped.endswith(docstring_char) and len(stripped) > 6:
                in_docstring = False
            continue
        elif in_docstring and docstring_char in stripped:
            in_docstring = False
            insert_pos = i + 1
            continue

        # Skip empty lines and comments at top
        if not stripped or stripped.startswith('#'):
            if not in_docstring:
                insert_pos = i + 1
            continue

        # Found first import or code
        if stripped.startswith('import ') or stripped.startswith('from '):
            insert_pos = i
            break
        elif stripped and not in_docstring:
            insert_pos = i
            break

    # Check if logging is already imported
    import_section_end = insert_pos
    for i in range(insert_pos, len(lines)):
        if lines[i].strip().startswith('import ') or lines[i].strip().startswith('from '):
            import_section_end = i + 1
        elif lines[i].strip() and not lines[i].strip().startswith('#'):
            break

    # Check if logging import exists
    has_logging = any('import logging' in line or 'from logging' in line
                      for line in lines[insert_pos:import_section_end])

    # Check if logger is defined
    has_logger = any('logger = logging.getLogger' in line
                     for line in lines)

    # Add import and logger if needed
    if not has_logging:
        lines.insert(import_section_end, 'import logging')
        import_section_end += 1

    if not has_logger:
        # Find position after all imports
        logger_pos = import_section_end
        for i in range(import_section_end, len(lines)):
            if not (lines[i].strip().startswith('import ') or
                   lines[i].strip().startswith('from ') or
                   lines[i].strip() == ''):
                logger_pos = i
                break

        # Add logger definition
        if logger_pos > 0 and lines[logger_pos - 1].strip():
            lines.insert(logger_pos, '')
        lines.insert(logger_pos, 'logger = logging.getLogger(__name__)')
        lines.insert(logger_pos + 1, '')

    return '\n'.join(lines)


def replace_print_statements(content: str) -> Tuple[str, int]:
    """Replace print() statements with logger calls."""
    count = 0

    # Pattern to match print statements
    # This handles: print("msg"), print(f"msg"), print(f"msg: {var}")
    patterns = [
        # print(f"Error: ...")  -> logger.error("...")
        (r'print\(f?"Error:([^"]*)"', r'logger.error(f"\1"', 'error'),
        (r'print\(f?\'Error:([^\']*)\'', r'logger.error(f\'\1\'', 'error'),

        # print(f"Warning: ...") -> logger.warning("...")
        (r'print\(f?"Warning:([^"]*)"', r'logger.warning(f"\1"', 'warning'),
        (r'print\(f?\'Warning:([^\']*)\'', r'logger.warning(f\'\1\'', 'warning'),

        # print(f"Failed: ...") -> logger.error("...")
        (r'print\(f?"Failed:([^"]*)"', r'logger.error(f"\1"', 'error'),
        (r'print\(f?\'Failed:([^\']*)\'', r'logger.error(f\'\1\'', 'error'),

        # print(f"DEBUG: ...") -> logger.debug("...")
        (r'print\(f?"DEBUG:([^"]*)"', r'logger.debug(f"\1"', 'debug'),
        (r'print\(f?\'DEBUG:([^\']*)\'', r'logger.debug(f\'\1\'', 'debug'),

        # Remaining print() -> logger.info()
        (r'print\(f?"([^"]*)"', r'logger.info(f"\1"', 'info'),
        (r'print\(f?\'([^\']*)\'', r'logger.info(f\'\1\'', 'info'),
    ]

    modified = content
    for pattern, replacement, level in patterns:
        new_content = re.sub(pattern, replacement, modified)
        if new_content != modified:
            count += modified.count('print(') - new_content.count('print(')
            modified = new_content

    return modified, count


def process_file(file_path: Path) -> Tuple[bool, int]:
    """Process a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Skip if no print statements
        if 'print(' not in original_content:
            return False, 0

        # Check if logging is already set up
        needs_import = not has_logging_import(original_content)

        # Replace print statements
        modified_content, count = replace_print_statements(original_content)

        # Add logging import if needed
        if needs_import and count > 0:
            modified_content = add_logging_import(modified_content)

        # Write back if changes were made
        if modified_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            return True, count

        return False, 0

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0


def find_python_files(root_dir: str, priority_dirs: List[str]) -> List[Path]:
    """Find Python files to process."""
    files = []
    root_path = Path(root_dir)

    # Process priority directories first
    for priority_dir in priority_dirs:
        dir_path = root_path / priority_dir
        if dir_path.exists():
            for py_file in dir_path.rglob('*.py'):
                if not should_skip_file(str(py_file)):
                    files.append(py_file)

    return files


def main():
    """Main function."""
    root_dir = Path(__file__).parent

    print("=" * 80)
    print("CODE QUALITY FIX - STRUCTURED LOGGING")
    print("=" * 80)
    print(f"\nScanning directory: {root_dir}")
    print(f"Priority directories: {', '.join(PRIORITY_DIRS)}\n")

    # Find files
    files = find_python_files(root_dir, PRIORITY_DIRS)
    print(f"Found {len(files)} Python files to process\n")

    # Process files
    total_modified = 0
    total_replacements = 0
    modified_files = []

    for file_path in files:
        was_modified, count = process_file(file_path)
        if was_modified:
            total_modified += 1
            total_replacements += count
            modified_files.append((file_path, count))
            print(f"✓ {file_path.relative_to(root_dir)} ({count} replacements)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files modified: {total_modified}")
    print(f"Total print() statements replaced: {total_replacements}")

    if modified_files:
        print("\nModified files:")
        for file_path, count in modified_files:
            print(f"  - {file_path.relative_to(root_dir)}: {count} replacements")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
