#!/usr/bin/env python3
"""
GreenLang Dependency Updater

Scans requirements.txt and updates dependencies to use GreenLang packages.
Adds missing greenlang-* packages and removes redundant ones.
"""

import argparse
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import subprocess


@dataclass
class Dependency:
    """Represents a package dependency."""
    name: str
    version_spec: str
    extras: List[str]
    original_line: str


class DependencyUpdater:
    """Updates requirements.txt with GreenLang packages."""

    def __init__(self):
        # GreenLang packages to add
        self.greenlang_packages = {
            'greenlang-core': '>=1.0.0',
            'greenlang-intelligence': '>=1.0.0',
            'greenlang-validation': '>=1.0.0',
            'greenlang-cache': '>=1.0.0',
            'greenlang-sdk': '>=1.0.0',
        }

        # Packages that can be removed if using GreenLang
        self.redundant_packages = {
            'openai': 'greenlang-intelligence',
            'anthropic': 'greenlang-intelligence',
            'langchain': 'greenlang-sdk',
            'redis': 'greenlang-cache',
            'jsonschema': 'greenlang-validation',
        }

        # Package version updates
        self.version_updates = {
            'pydantic': '>=2.0.0',
            'fastapi': '>=0.100.0',
            'uvicorn': '>=0.23.0',
        }

    def parse_requirement(self, line: str) -> Dependency:
        """Parse a requirement line."""
        # Remove comments
        if '#' in line:
            line = line.split('#')[0]

        line = line.strip()

        if not line:
            return None

        # Parse package name, version spec, and extras
        # Format: package[extra1,extra2]>=1.0.0
        match = re.match(r'^([a-zA-Z0-9_-]+)(\[.*?\])?(.*)$', line)

        if not match:
            return None

        name = match.group(1)
        extras_str = match.group(2) or ''
        version_spec = match.group(3).strip()

        # Parse extras
        extras = []
        if extras_str:
            extras_str = extras_str.strip('[]')
            extras = [e.strip() for e in extras_str.split(',')]

        return Dependency(
            name=name,
            version_spec=version_spec,
            extras=extras,
            original_line=line
        )

    def update_requirements(self, file_path: str, dry_run: bool = True, remove_redundant: bool = False) -> Tuple[str, str, Dict]:
        """
        Update a requirements.txt file.

        Returns:
            (original_content, new_content, stats)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                lines = original_content.split('\n')

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return "", "", {}

        new_lines = []
        existing_packages = set()
        stats = {
            'added': [],
            'removed': [],
            'updated': [],
            'unchanged': 0
        }

        # Process existing lines
        for line in lines:
            original_line = line
            line = line.strip()

            # Keep comments and empty lines
            if not line or line.startswith('#'):
                new_lines.append(original_line)
                continue

            # Parse dependency
            dep = self.parse_requirement(line)

            if not dep:
                new_lines.append(original_line)
                continue

            existing_packages.add(dep.name)

            # Check if package is redundant
            if remove_redundant and dep.name in self.redundant_packages:
                replacement = self.redundant_packages[dep.name]
                new_lines.append(f"# {original_line}  # Replaced by {replacement}")
                stats['removed'].append(dep.name)
                continue

            # Check if version should be updated
            if dep.name in self.version_updates:
                new_version = self.version_updates[dep.name]
                if dep.version_spec != new_version:
                    extras_str = f"[{','.join(dep.extras)}]" if dep.extras else ""
                    new_line = f"{dep.name}{extras_str}{new_version}"
                    new_lines.append(new_line)
                    stats['updated'].append(f"{dep.name}: {dep.version_spec} -> {new_version}")
                    continue

            # Keep unchanged
            new_lines.append(original_line)
            stats['unchanged'] += 1

        # Add missing GreenLang packages
        greenlang_section = []
        for package, version in sorted(self.greenlang_packages.items()):
            if package not in existing_packages:
                greenlang_section.append(f"{package}{version}")
                stats['added'].append(package)

        if greenlang_section:
            # Add GreenLang section
            new_lines.append("")
            new_lines.append("# GreenLang Infrastructure")
            new_lines.extend(greenlang_section)

        # Generate new content
        new_content = '\n'.join(new_lines)

        # Write back if not dry run
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

        return original_content, new_content, stats

    def generate_report(self, stats: Dict) -> str:
        """Generate update report."""
        lines = []
        lines.append("=" * 80)
        lines.append("GreenLang Dependency Update Report")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        total_changes = len(stats['added']) + len(stats['removed']) + len(stats['updated'])
        lines.append("SUMMARY:")
        lines.append(f"  Total changes: {total_changes}")
        lines.append(f"  Packages added: {len(stats['added'])}")
        lines.append(f"  Packages removed: {len(stats['removed'])}")
        lines.append(f"  Packages updated: {len(stats['updated'])}")
        lines.append(f"  Packages unchanged: {stats['unchanged']}")
        lines.append("")

        # Details
        if stats['added']:
            lines.append("ADDED:")
            for package in stats['added']:
                lines.append(f"  + {package}")
            lines.append("")

        if stats['removed']:
            lines.append("REMOVED:")
            for package in stats['removed']:
                lines.append(f"  - {package}")
            lines.append("")

        if stats['updated']:
            lines.append("UPDATED:")
            for update in stats['updated']:
                lines.append(f"  ~ {update}")
            lines.append("")

        return '\n'.join(lines)

    def scan_imports(self, directory: str) -> Set[str]:
        """Scan Python files to detect which packages are actually used."""
        import ast

        used_packages = set()
        exclude_patterns = ['__pycache__', '.git', 'venv', 'env', 'node_modules', '.greenlang']

        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        tree = ast.parse(content)

                        # Extract imports
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    # Get top-level package
                                    package = alias.name.split('.')[0]
                                    used_packages.add(package)

                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    package = node.module.split('.')[0]
                                    used_packages.add(package)

                    except Exception:
                        # Skip files that can't be parsed
                        pass

        return used_packages

    def suggest_packages(self, directory: str) -> Dict[str, str]:
        """Suggest GreenLang packages based on actual usage."""
        used_packages = self.scan_imports(directory)

        suggestions = {}

        # Map used packages to GreenLang packages
        if 'openai' in used_packages or 'anthropic' in used_packages:
            suggestions['greenlang-intelligence'] = '>=1.0.0'

        if 'redis' in used_packages:
            suggestions['greenlang-cache'] = '>=1.0.0'

        if 'jsonschema' in used_packages or 'pydantic' in used_packages:
            suggestions['greenlang-validation'] = '>=1.0.0'

        if 'langchain' in used_packages:
            suggestions['greenlang-sdk'] = '>=1.0.0'

        # Always suggest core
        suggestions['greenlang-core'] = '>=1.0.0'

        return suggestions

    def install_packages(self, file_path: str):
        """Install packages from requirements.txt."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', file_path],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("✓ Packages installed successfully")
                return True
            else:
                print(f"✗ Error installing packages:\n{result.stderr}")
                return False

        except Exception as e:
            print(f"✗ Error running pip install: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="GreenLang Dependency Updater - Update requirements.txt with GreenLang packages"
    )

    parser.add_argument(
        '--file',
        default='requirements.txt',
        help='Path to requirements.txt (default: requirements.txt)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )

    parser.add_argument(
        '--remove-redundant',
        action='store_true',
        help='Remove redundant packages replaced by GreenLang'
    )

    parser.add_argument(
        '--scan',
        help='Scan directory to detect which packages are used'
    )

    parser.add_argument(
        '--install',
        action='store_true',
        help='Install packages after updating'
    )

    parser.add_argument(
        '--show-diff',
        action='store_true',
        help='Show diff of changes'
    )

    args = parser.parse_args()

    # Initialize tool
    tool = DependencyUpdater()

    # Scan for used packages if requested
    if args.scan:
        print(f"Scanning {args.scan} for package usage...")
        suggestions = tool.suggest_packages(args.scan)

        print("\nSuggested GreenLang packages based on usage:")
        for package, version in sorted(suggestions.items()):
            print(f"  {package}{version}")
        print()

    # Update requirements
    print(f"Updating {args.file}...")

    original, new, stats = tool.update_requirements(
        args.file,
        dry_run=args.dry_run,
        remove_redundant=args.remove_redundant
    )

    # Generate report
    print("\n" + tool.generate_report(stats))

    # Show diff if requested
    if args.show_diff:
        import difflib
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=args.file,
            tofile=args.file
        )
        print("\nDiff:")
        print(''.join(diff))

    # Install if requested
    if args.install and not args.dry_run:
        print("\nInstalling packages...")
        tool.install_packages(args.file)

    if args.dry_run:
        print("\n(Dry run - no changes applied)")
    else:
        print("\n✓ Dependencies updated!")


if __name__ == '__main__':
    main()
