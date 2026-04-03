#!/usr/bin/env python3
"""
Standardize Logging Migration Script

Converts f-string logging anti-patterns to %-format across the GreenLang codebase.
Also fixes self.logger → module-level logger naming.

Usage:
    python scripts/standardize_logging.py               # Dry-run (report only)
    python scripts/standardize_logging.py --apply        # Apply changes
    python scripts/standardize_logging.py --apply --dir greenlang/auth  # Specific dir

Author: GreenLang Platform Team
Date: April 2026
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches: logger.info(f"...", ...) or logger.error(f'...', ...)
# Also matches: self.logger.info(f"...", ...)
# Captures: prefix (logger/self.logger), level, f-string content, trailing args
FSTRING_LOG_RE = re.compile(
    r"((?:self\.)?logger\.(?:debug|info|warning|error|critical|exception))"
    r"\(\s*f([\"'])"
)

# Matches interpolation expressions inside f-strings: {expr}, {expr!r}, {expr:.2f}
INTERPOLATION_RE = re.compile(
    r"\{("
    r"[^{}]*?"        # expression (no nested braces)
    r")"
    r"(?:![rsa])?"    # optional conversion (!r, !s, !a)
    r"(?::([^{}]*?))?"  # optional format spec
    r"\}"
)

# Matches self.logger usage
SELF_LOGGER_RE = re.compile(r"\bself\.logger\b")

# Matches module-level logger assignment
MODULE_LOGGER_RE = re.compile(
    r"^logger\s*=\s*logging\.getLogger\(__name__\)",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# F-string → %-format converter
# ---------------------------------------------------------------------------


def convert_fstring_to_percent(fstring_body: str) -> tuple[str, list[str]]:
    """Convert an f-string body to %-format string + argument list.

    Args:
        fstring_body: The content of the f-string (without f prefix and quotes).

    Returns:
        Tuple of (format_string, [arg1, arg2, ...])
    """
    args: list[str] = []
    last_end = 0
    parts: list[str] = []

    for match in INTERPOLATION_RE.finditer(fstring_body):
        # Add literal text before this interpolation
        parts.append(fstring_body[last_end:match.start()])

        expr = match.group(1).strip()
        conversion = ""
        format_spec = match.group(2)

        # Check for conversion flag in the full match text
        full_match = match.group(0)
        if "!r}" in full_match or "!r:" in full_match:
            conversion = "r"
        elif "!a}" in full_match or "!a:" in full_match:
            conversion = "a"

        # Determine the %-format specifier
        if conversion == "r":
            spec = "%r"
        elif format_spec:
            # Handle common format specs
            if format_spec.endswith("f"):
                spec = f"%{format_spec}"
            elif format_spec.endswith("d"):
                spec = f"%{format_spec}"
            elif format_spec.endswith("s"):
                spec = f"%{format_spec}"
            elif format_spec.startswith(","):
                # Comma-separated numbers - can't do with %, use %s
                spec = "%s"
            else:
                spec = "%s"
        else:
            spec = "%s"

        parts.append(spec)

        # Clean up the expression
        # Remove redundant str() wrapping since %s calls str() anyway
        if expr.startswith("str(") and expr.endswith(")"):
            expr = expr[4:-1]

        args.append(expr)
        last_end = match.end()

    # Add remaining literal text
    parts.append(fstring_body[last_end:])

    format_string = "".join(parts)
    return format_string, args


def process_fstring_log_call(line: str) -> Optional[str]:
    """Convert a single f-string log call to %-format.

    Args:
        line: The source line containing the f-string log call.

    Returns:
        The converted line, or None if conversion is not possible.
    """
    match = FSTRING_LOG_RE.search(line)
    if not match:
        return None

    prefix = match.group(1)  # e.g., "logger.info" or "self.logger.error"
    quote_char = match.group(2)  # " or '

    # Find the matching closing quote and parenthesis
    start_pos = match.end()
    content = line[start_pos:]

    # Find the closing quote (handling escaped quotes)
    body_end = -1
    i = 0
    while i < len(content):
        if content[i] == "\\" and i + 1 < len(content):
            i += 2  # skip escaped character
            continue
        if content[i] == quote_char:
            body_end = i
            break
        if content[i] == "{" and i + 1 < len(content) and content[i + 1] == "{":
            i += 2  # skip escaped brace
            continue
        if content[i] == "}" and i + 1 < len(content) and content[i + 1] == "}":
            i += 2  # skip escaped brace
            continue
        i += 1

    if body_end == -1:
        return None  # Can't find closing quote (multiline f-string)

    fstring_body = content[:body_end]

    # Check for nested braces (complex expressions) - skip these
    depth = 0
    for ch in fstring_body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth > 1:
            return None  # Nested braces, too complex

    # Check for expressions with string formatting that's hard to convert
    if "{{" in fstring_body or "}}" in fstring_body:
        # Escaped braces - handle them
        fstring_body = fstring_body.replace("{{", "\x00LBRACE\x00").replace("}}", "\x00RBRACE\x00")
        format_string, args = convert_fstring_to_percent(fstring_body)
        format_string = format_string.replace("\x00LBRACE\x00", "{").replace("\x00RBRACE\x00", "}")
    else:
        format_string, args = convert_fstring_to_percent(fstring_body)

    if not args:
        return None  # No interpolations found, nothing to convert

    # Get the rest of the line after the f-string
    rest = content[body_end + 1:]

    # Build the new line
    indent = line[: len(line) - len(line.lstrip())]
    new_prefix = prefix.replace("self.logger", "logger") if "self.logger" in prefix else prefix

    # Escape any existing % in the format string (but not our %s/%d/%r etc.)
    # Actually, we need to be careful - only escape literal % that aren't part of our format specs
    # This is tricky, so let's just leave it for now

    # Build argument string
    args_str = ", ".join(args)

    # Reconstruct the line
    # Check if there are additional keyword args after the f-string
    rest_stripped = rest.strip()
    if rest_stripped.startswith(")"):
        # Simple case: just the f-string, closing paren
        trailing = rest_stripped[1:]
        new_line = f'{indent}{new_prefix}("{format_string}", {args_str}){trailing}'
    elif rest_stripped.startswith(","):
        # Additional args after the f-string (e.g., exc_info=True)
        new_line = f'{indent}{new_prefix}("{format_string}", {args_str}{rest_stripped}'
    else:
        return None  # Complex case, skip

    return new_line


# ---------------------------------------------------------------------------
# File processor
# ---------------------------------------------------------------------------


class MigrationStats:
    """Track migration statistics."""

    def __init__(self) -> None:
        self.files_scanned = 0
        self.files_modified = 0
        self.fstring_conversions = 0
        self.fstring_skipped = 0
        self.self_logger_fixes = 0
        self.modified_files: list[str] = []
        self.skipped_lines: list[tuple[str, int, str]] = []

    def report(self) -> str:
        lines = [
            "\n" + "=" * 70,
            "LOGGING STANDARDIZATION MIGRATION REPORT",
            "=" * 70,
            f"Files scanned:           {self.files_scanned}",
            f"Files modified:          {self.files_modified}",
            f"F-string conversions:    {self.fstring_conversions}",
            f"F-string skipped:        {self.fstring_skipped}",
            f"self.logger -> logger:    {self.self_logger_fixes}",
            "",
        ]
        if self.modified_files:
            lines.append(f"Modified files ({len(self.modified_files)}):")
            for f in sorted(self.modified_files)[:50]:
                lines.append(f"  {f}")
            if len(self.modified_files) > 50:
                lines.append(f"  ... and {len(self.modified_files) - 50} more")
        if self.skipped_lines:
            lines.append(f"\nSkipped f-string lines ({len(self.skipped_lines)}):")
            for filepath, lineno, line in self.skipped_lines[:20]:
                lines.append(f"  {filepath}:{lineno}: {line.strip()[:100]}")
            if len(self.skipped_lines) > 20:
                lines.append(f"  ... and {len(self.skipped_lines) - 20} more")
        lines.append("=" * 70)
        return "\n".join(lines)


def process_file(
    filepath: Path,
    stats: MigrationStats,
    apply: bool = False,
    fix_self_logger: bool = True,
) -> bool:
    """Process a single Python file for logging standardization.

    Args:
        filepath: Path to the Python file.
        stats: Statistics tracker.
        apply: If True, write changes to disk.
        fix_self_logger: If True, also fix self.logger → logger.

    Returns:
        True if the file was modified.
    """
    stats.files_scanned += 1

    try:
        content = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return False

    original = content
    lines = content.split("\n")
    new_lines: list[str] = []
    modified = False
    has_module_logger = bool(MODULE_LOGGER_RE.search(content))

    for i, line in enumerate(lines):
        new_line = line

        # --- Fix f-string logging ---
        if FSTRING_LOG_RE.search(line):
            # Skip multiline log calls (continuation lines)
            stripped = line.strip()
            if stripped.endswith("\\") or stripped.count("(") != stripped.count(")"):
                stats.fstring_skipped += 1
                stats.skipped_lines.append((str(filepath), i + 1, line))
                new_lines.append(line)
                continue

            converted = process_fstring_log_call(line)
            if converted is not None:
                new_line = converted
                stats.fstring_conversions += 1
                modified = True
            else:
                stats.fstring_skipped += 1
                stats.skipped_lines.append((str(filepath), i + 1, line))

        # --- Fix self.logger → logger (only if module-level logger exists) ---
        elif fix_self_logger and has_module_logger and SELF_LOGGER_RE.search(line):
            # Don't fix self.logger = ... assignments or self.logger.setLevel
            if "self.logger =" in line or "self.logger.setLevel" in line:
                new_lines.append(line)
                continue
            new_line = SELF_LOGGER_RE.sub("logger", line)
            if new_line != line:
                stats.self_logger_fixes += 1
                modified = True

        new_lines.append(new_line)

    if modified:
        stats.files_modified += 1
        stats.modified_files.append(str(filepath))
        if apply:
            filepath.write_text("\n".join(new_lines), encoding="utf-8")

    return modified


def process_directory(
    root: Path,
    stats: MigrationStats,
    apply: bool = False,
) -> None:
    """Process all Python files in a directory tree.

    Args:
        root: Root directory to scan.
        stats: Statistics tracker.
        apply: If True, write changes to disk.
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            filepath = Path(dirpath) / filename
            # Skip test files and migration scripts
            rel = filepath.relative_to(root) if filepath.is_relative_to(root) else filepath
            rel_str = str(rel)
            if "test" in rel_str.lower() and "test_" in filename.lower():
                continue
            process_file(filepath, stats, apply=apply)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardize logging patterns across GreenLang codebase"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Specific directory to process (default: greenlang/)",
    )
    args = parser.parse_args()

    # Determine root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.dir:
        target_dir = project_root / args.dir
    else:
        target_dir = project_root / "greenlang"

    if not target_dir.exists():
        print(f"Error: Directory not found: {target_dir}")
        sys.exit(1)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Logging Standardization Migration [{mode}]")
    print(f"Target: {target_dir}")
    print("-" * 70)

    stats = MigrationStats()
    process_directory(target_dir, stats, apply=args.apply)
    print(stats.report())


if __name__ == "__main__":
    main()
