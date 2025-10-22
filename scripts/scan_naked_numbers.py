#!/usr/bin/env python
"""
Scanner for naked numbers in test fixtures and AI response files.

This script scans for numeric values that appear without proper tool provenance
or {{claim:i}} macro references, helping enforce the "no naked numbers" policy.
"""

import re
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Violation:
    """Represents a naked number violation."""
    file: Path
    line_number: int
    line_content: str
    numeric_value: str
    field_path: str


# Whitelist patterns (same as in runtime/tools.py)
WHITELIST_PATTERNS = [
    r'(?:^|\n)\d+\.\s',  # Ordered lists (e.g., "1. Item")
    r'\b\d{4}-\d{2}-\d{2}\b',  # ISO dates (e.g., "2024-01-15")
    r'\bv?\d+\.\d+(\.\d+)?\b',  # Version strings (e.g., "v1.2.3", "1.0")
    r'\bID[-_]?\d+\b',  # ID patterns (e.g., "ID-123", "ID_456")
    r'\b\d{2}:\d{2}(:\d{2})?\b',  # Time stamps (e.g., "12:30", "14:25:10")
    r'\bstep[-_]?\d+\b',  # Step numbers (e.g., "step-1", "step_2")
    r'\btc[-_]?\d+\b',  # Tool call IDs (e.g., "tc_1", "tc-2")
]


def is_whitelisted(text: str, position: int) -> bool:
    """Check if a numeric at given position is whitelisted."""
    # Get context around the position
    start = max(0, position - 20)
    end = min(len(text), position + 20)
    context = text[start:end]

    for pattern in WHITELIST_PATTERNS:
        if re.search(pattern, context):
            return True

    return False


def scan_json_for_naked_numbers(file_path: Path) -> List[Violation]:
    """
    Scan a JSON file for naked numbers.

    Looks for numeric values in JSON structures that don't have proper provenance.
    """
    violations = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Look for numeric values in specific contexts
        def check_object(obj, path="$"):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}"

                    # Check if this is a numeric field without source
                    if isinstance(value, (int, float)):
                        # Check if there's a corresponding source or tool_call_id
                        has_source = 'source' in obj or 'tool_call_id' in obj
                        has_unit = 'unit' in obj  # Quantity schema
                        has_claim = 'claims' in obj or 'claim' in str(key)

                        if not (has_source or has_unit or has_claim):
                            violations.append(Violation(
                                file=file_path,
                                line_number=0,  # JSON doesn't have line numbers easily
                                line_content=json.dumps({key: value}),
                                numeric_value=str(value),
                                field_path=current_path
                            ))

                    # Recurse
                    check_object(value, current_path)

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_object(item, f"{path}[{i}]")

        check_object(data)

    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON in {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error scanning {file_path}: {e}", file=sys.stderr)

    return violations


def scan_text_for_naked_numbers(file_path: Path) -> List[Violation]:
    """
    Scan a text file for naked numbers.

    Looks for numeric patterns that aren't whitelisted.
    """
    violations = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Pattern to find numbers (including decimals and negatives)
        number_pattern = r'-?\d+\.?\d*'

        for line_num, line in enumerate(lines, start=1):
            # Skip lines with {{claim:i}} macros
            if re.search(r'\{\{claim:\d+\}\}', line):
                continue

            # Skip lines that look like tool outputs
            if 'source' in line or 'tool_call_id' in line or '"unit"' in line:
                continue

            # Find all numbers in the line
            for match in re.finditer(number_pattern, line):
                position = match.start()

                # Skip if whitelisted
                if is_whitelisted(line, position):
                    continue

                # This might be a naked number
                violations.append(Violation(
                    file=file_path,
                    line_number=line_num,
                    line_content=line.strip(),
                    numeric_value=match.group(),
                    field_path="N/A"
                ))

    except Exception as e:
        print(f"Warning: Error scanning {file_path}: {e}", file=sys.stderr)

    return violations


def scan_directory(directory: Path, patterns: List[str]) -> Dict[Path, List[Violation]]:
    """
    Scan a directory for naked numbers in matching files.

    Args:
        directory: Directory to scan
        patterns: List of glob patterns to match

    Returns:
        Dictionary mapping file paths to violations
    """
    all_violations = {}

    for pattern in patterns:
        for file_path in directory.rglob(pattern):
            if file_path.is_file():
                if file_path.suffix == '.json':
                    violations = scan_json_for_naked_numbers(file_path)
                else:
                    violations = scan_text_for_naked_numbers(file_path)

                if violations:
                    all_violations[file_path] = violations

    return all_violations


def print_violations(violations: Dict[Path, List[Violation]], verbose: bool = False):
    """Print violations in a human-readable format."""
    if not violations:
        print("✅ No naked number violations found!")
        return 0

    total_violations = sum(len(v) for v in violations.values())
    print(f"❌ Found {total_violations} naked number violation(s) in {len(violations)} file(s)\n")

    for file_path, file_violations in sorted(violations.items()):
        print(f"\n📄 {file_path}")
        print("=" * 80)

        for violation in file_violations:
            if violation.line_number > 0:
                print(f"  Line {violation.line_number}:")
                print(f"    {violation.line_content}")
                print(f"    ^ Naked number: {violation.numeric_value}")
            else:
                print(f"  Field: {violation.field_path}")
                print(f"    {violation.line_content}")
                print(f"    ^ Naked number: {violation.numeric_value}")

            if verbose:
                print(f"    Remediation: Ensure this value comes from a tool or is referenced via {{{{claim:i}}}}")

            print()

    return 1  # Exit code indicating violations found


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan for naked numbers in test fixtures and AI responses"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="tests",
        help="Directory to scan (default: tests)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with remediation suggestions"
    )
    parser.add_argument(
        "-p", "--patterns",
        nargs="+",
        default=["*.json", "*.txt", "*.md"],
        help="File patterns to scan (default: *.json *.txt *.md)"
    )

    args = parser.parse_args()

    scan_dir = Path(args.directory)

    if not scan_dir.exists():
        print(f"Error: Directory not found: {scan_dir}", file=sys.stderr)
        return 1

    print(f"🔍 Scanning for naked numbers in: {scan_dir}")
    print(f"📝 Patterns: {', '.join(args.patterns)}\n")

    violations = scan_directory(scan_dir, args.patterns)
    return print_violations(violations, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
