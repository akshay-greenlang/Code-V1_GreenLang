#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

logger = logging.getLogger(__name__)
GreenLang Code Migration Tool

Automatically detects custom code patterns and suggests GreenLang infrastructure replacements.
Supports dry-run mode and auto-fix mode for safe migrations.
"""

import logging
import ast
import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import difflib
from collections import defaultdict


@dataclass
class MigrationPattern:
    """Represents a migration pattern to detect and replace."""
    name: str
    description: str
    detect_pattern: str  # Regex or AST pattern
    detect_imports: List[str]  # Required imports to trigger
    suggest_code: str
    suggest_imports: List[str]
    category: str  # llm, agent, cache, validation, etc.
    confidence: float  # 0.0 to 1.0


@dataclass
class MigrationSuggestion:
    """A single migration suggestion for a file."""
    file_path: str
    line_number: int
    pattern: MigrationPattern
    original_code: str
    suggested_code: str
    context: str  # Surrounding code for context


class CodeMigrationTool:
    """Main migration tool class."""

    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.suggestions = []
        self.stats = defaultdict(int)

    def _initialize_patterns(self) -> List[MigrationPattern]:
        """Initialize all migration patterns."""
        return [
            # Pattern 1: OpenAI Client
            MigrationPattern(
                name="openai_client",
                description="Replace OpenAI client with GreenLang ChatSession",
                detect_pattern=r"OpenAI\s*\(\s*api_key\s*=",
                detect_imports=["openai", "OpenAI"],
                suggest_code="""from greenlang.intelligence import ChatSession

# Replace OpenAI client with ChatSession
session = ChatSession(provider="openai")""",
                suggest_imports=["from greenlang.intelligence import ChatSession"],
                category="llm",
                confidence=0.95
            ),

            # Pattern 2: Custom Agent Class
            MigrationPattern(
                name="custom_agent",
                description="Migrate custom agent to inherit from Agent base class",
                detect_pattern=r"class\s+\w+Agent\s*(?:\([^)]*\))?\s*:",
                detect_imports=[],
                suggest_code="""from greenlang.integration.sdk.base import Agent

class YourAgent(Agent):
    def execute(self, input_data):
        # Your agent logic here
        return self.process(input_data)""",
                suggest_imports=["from greenlang.integration.sdk.base import Agent"],
                category="agent",
                confidence=0.85
            ),

            # Pattern 3: JSONSchema Validation
            MigrationPattern(
                name="jsonschema_validation",
                description="Replace jsonschema with GreenLang ValidationFramework",
                detect_pattern=r"jsonschema\.validate\s*\(",
                detect_imports=["jsonschema"],
                suggest_code="""from greenlang.validation import ValidationFramework

validator = ValidationFramework()
validator.validate_schema(data, schema)""",
                suggest_imports=["from greenlang.validation import ValidationFramework"],
                category="validation",
                confidence=0.9
            ),

            # Pattern 4: Direct Redis Usage
            MigrationPattern(
                name="redis_client",
                description="Replace Redis client with GreenLang CacheManager",
                detect_pattern=r"redis\.Redis\s*\(",
                detect_imports=["redis"],
                suggest_code="""from greenlang.cache import CacheManager

cache = CacheManager()
# Use cache.get(), cache.set(), cache.delete(), etc.""",
                suggest_imports=["from greenlang.cache import CacheManager"],
                category="cache",
                confidence=0.9
            ),

            # Pattern 5: Anthropic Client
            MigrationPattern(
                name="anthropic_client",
                description="Replace Anthropic client with GreenLang ChatSession",
                detect_pattern=r"Anthropic\s*\(\s*api_key\s*=",
                detect_imports=["anthropic", "Anthropic"],
                suggest_code="""from greenlang.intelligence import ChatSession

session = ChatSession(provider="anthropic")""",
                suggest_imports=["from greenlang.intelligence import ChatSession"],
                category="llm",
                confidence=0.95
            ),

            # Pattern 6: LangChain
            MigrationPattern(
                name="langchain_chain",
                description="Replace LangChain with GreenLang Pipeline",
                detect_pattern=r"from\s+langchain\s+import",
                detect_imports=["langchain"],
                suggest_code="""from greenlang.integration.sdk.base import Pipeline, Agent

# Use GreenLang Pipeline instead of LangChain
pipeline = Pipeline(agents=[agent1, agent2])""",
                suggest_imports=["from greenlang.integration.sdk.base import Pipeline, Agent"],
                category="pipeline",
                confidence=0.8
            ),

            # Pattern 7: Custom Logging
            MigrationPattern(
                name="custom_logging",
                description="Use GreenLang structured logging",
                detect_pattern=r"logging\.basicConfig\s*\(",
                detect_imports=["logging"],
                suggest_code="""from greenlang.utilities.utils.logging import StructuredLogger

logger = StructuredLogger(__name__)
# Use logger.info(), logger.error(), etc.""",
                suggest_imports=["from greenlang.utilities.utils.logging import StructuredLogger"],
                category="logging",
                confidence=0.7
            ),

            # Pattern 8: Custom HTTP Client
            MigrationPattern(
                name="requests_client",
                description="Consider using GreenLang HTTP utilities",
                detect_pattern=r"requests\.(?:get|post|put|delete)\s*\(",
                detect_imports=["requests"],
                suggest_code="""from greenlang.utilities.utils.http import HTTPClient

client = HTTPClient()
# Use client.get(), client.post(), etc. with built-in retries and monitoring""",
                suggest_imports=["from greenlang.utilities.utils.http import HTTPClient"],
                category="http",
                confidence=0.6
            ),

            # Pattern 9: Custom Batch Processing
            MigrationPattern(
                name="batch_processing",
                description="Use Agent.batch_process() for batch operations",
                detect_pattern=r"def\s+process_batch\s*\(",
                detect_imports=[],
                suggest_code="""from greenlang.integration.sdk.base import Agent

class YourAgent(Agent):
    def execute(self, input_data):
        # Process single item
        return result

# Use built-in batch processing
results = agent.batch_process(items, max_workers=10)""",
                suggest_imports=["from greenlang.integration.sdk.base import Agent"],
                category="agent",
                confidence=0.75
            ),

            # Pattern 10: Environment Variables
            MigrationPattern(
                name="env_variables",
                description="Use GreenLang config management",
                detect_pattern=r"os\.getenv\s*\(\s*['\"](?:API_KEY|SECRET|TOKEN)",
                detect_imports=["os"],
                suggest_code="""from greenlang.config import Config

config = Config()
api_key = config.get_secret('API_KEY')""",
                suggest_imports=["from greenlang.config import Config"],
                category="config",
                confidence=0.7
            ),
        ]

    def scan_file(self, file_path: str) -> List[MigrationSuggestion]:
        """Scan a single Python file for migration opportunities."""
        suggestions = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Parse AST
            try:
                tree = ast.parse(content)
                imports = self._extract_imports(tree)
            except SyntaxError:
                logger.warning(f"Could not parse {file_path} (syntax error)")
                return suggestions

            # Check each pattern
            for pattern in self.patterns:
                # Check if required imports are present (if any)
                if pattern.detect_imports:
                    has_import = any(
                        any(imp in str(existing_imp) for imp in pattern.detect_imports)
                        for existing_imp in imports
                    )
                    if not has_import:
                        continue

                # Search for pattern matches
                for i, line in enumerate(lines, 1):
                    if re.search(pattern.detect_pattern, line):
                        # Get context (5 lines before and after)
                        context_start = max(0, i - 6)
                        context_end = min(len(lines), i + 5)
                        context = '\n'.join(lines[context_start:context_end])

                        suggestion = MigrationSuggestion(
                            file_path=file_path,
                            line_number=i,
                            pattern=pattern,
                            original_code=line,
                            suggested_code=pattern.suggest_code,
                            context=context
                        )
                        suggestions.append(suggestion)
                        self.stats[pattern.category] += 1

        except Exception as e:
            print(f"Error scanning {file_path}: {e}")

        return suggestions

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                for alias in node.names:
                    imports.append(alias.name)
        return imports

    def scan_directory(self, directory: str, exclude_patterns: List[str] = None) -> List[MigrationSuggestion]:
        """Scan all Python files in a directory."""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'venv', 'env', 'node_modules', '.greenlang']

        all_suggestions = []

        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    suggestions = self.scan_file(file_path)
                    all_suggestions.extend(suggestions)
                    self.stats['files_scanned'] += 1

        return all_suggestions

    def generate_diff(self, suggestion: MigrationSuggestion) -> str:
        """Generate unified diff for a suggestion."""
        original = f"# Original code at line {suggestion.line_number}\n{suggestion.original_code}\n"
        suggested = f"# Suggested replacement\n{suggestion.suggested_code}\n"

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            suggested.splitlines(keepends=True),
            fromfile=suggestion.file_path,
            tofile=suggestion.file_path,
            lineterm=''
        )

        return ''.join(diff)

    def generate_report(self, suggestions: List[MigrationSuggestion], format: str = 'text') -> str:
        """Generate migration report."""
        if format == 'text':
            return self._generate_text_report(suggestions)
        elif format == 'json':
            return self._generate_json_report(suggestions)
        elif format == 'html':
            return self._generate_html_report(suggestions)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_text_report(self, suggestions: List[MigrationSuggestion]) -> str:
        """Generate text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("GreenLang Migration Report")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append("SUMMARY:")
        lines.append(f"  Files scanned: {self.stats['files_scanned']}")
        lines.append(f"  Migration opportunities found: {len(suggestions)}")
        lines.append("")

        # By category
        lines.append("BY CATEGORY:")
        for category, count in sorted(self.stats.items()):
            if category != 'files_scanned':
                lines.append(f"  {category}: {count}")
        lines.append("")

        # Detailed suggestions
        lines.append("DETAILED SUGGESTIONS:")
        lines.append("-" * 80)

        for i, suggestion in enumerate(suggestions, 1):
            lines.append(f"\n{i}. {suggestion.pattern.name}")
            lines.append(f"   File: {suggestion.file_path}:{suggestion.line_number}")
            lines.append(f"   Category: {suggestion.pattern.category}")
            lines.append(f"   Confidence: {suggestion.pattern.confidence * 100:.0f}%")
            lines.append(f"   Description: {suggestion.pattern.description}")
            lines.append("")
            lines.append("   Original code:")
            lines.append(f"   >>> {suggestion.original_code.strip()}")
            lines.append("")
            lines.append("   Suggested replacement:")
            for line in suggestion.suggested_code.split('\n'):
                lines.append(f"   {line}")
            lines.append("")
            lines.append("-" * 80)

        return '\n'.join(lines)

    def _generate_json_report(self, suggestions: List[MigrationSuggestion]) -> str:
        """Generate JSON report."""
        import json

        data = {
            'summary': {
                'files_scanned': self.stats['files_scanned'],
                'opportunities_found': len(suggestions),
                'by_category': {k: v for k, v in self.stats.items() if k != 'files_scanned'}
            },
            'suggestions': [
                {
                    'file': s.file_path,
                    'line': s.line_number,
                    'pattern': s.pattern.name,
                    'category': s.pattern.category,
                    'confidence': s.pattern.confidence,
                    'description': s.pattern.description,
                    'original': s.original_code,
                    'suggested': s.suggested_code
                }
                for s in suggestions
            ]
        }

        return json.dumps(data, indent=2)

    def _generate_html_report(self, suggestions: List[MigrationSuggestion]) -> str:
        """Generate HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>GreenLang Migration Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c5f2d; }}
        .summary {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .category {{ display: inline-block; margin: 5px; padding: 5px 10px; background: #4caf50; color: white; border-radius: 3px; }}
        .suggestion {{ border: 1px solid #ddd; margin: 15px 0; padding: 15px; border-radius: 5px; }}
        .suggestion-header {{ background: #f9f9f9; padding: 10px; margin: -15px -15px 10px -15px; border-radius: 5px 5px 0 0; }}
        .code {{ background: #f5f5f5; padding: 10px; border-left: 3px solid #4caf50; margin: 10px 0; font-family: monospace; overflow-x: auto; }}
        .original {{ border-left-color: #ff9800; }}
        .confidence {{ color: #666; font-size: 0.9em; }}
        .high {{ color: #4caf50; font-weight: bold; }}
        .medium {{ color: #ff9800; font-weight: bold; }}
        .low {{ color: #f44336; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>GreenLang Migration Report</h1>

        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Files scanned:</strong> {self.stats['files_scanned']}</p>
            <p><strong>Migration opportunities found:</strong> {len(suggestions)}</p>
            <div>
                <strong>By category:</strong><br>
"""

        for category, count in sorted(self.stats.items()):
            if category != 'files_scanned':
                html += f'                <span class="category">{category}: {count}</span>\n'

        html += """            </div>
        </div>

        <h2>Detailed Suggestions</h2>
"""

        for i, suggestion in enumerate(suggestions, 1):
            confidence_class = 'high' if suggestion.pattern.confidence >= 0.8 else 'medium' if suggestion.pattern.confidence >= 0.6 else 'low'
            suggested_code_html = '<br>'.join(suggestion.suggested_code.split('\n'))

            html += f"""        <div class="suggestion">
            <div class="suggestion-header">
                <strong>{i}. {suggestion.pattern.name}</strong>
                <span class="confidence">
                    Confidence: <span class="{confidence_class}">{suggestion.pattern.confidence * 100:.0f}%</span>
                </span>
            </div>
            <p><strong>File:</strong> {suggestion.file_path}:{suggestion.line_number}</p>
            <p><strong>Category:</strong> {suggestion.pattern.category}</p>
            <p><strong>Description:</strong> {suggestion.pattern.description}</p>

            <p><strong>Original code:</strong></p>
            <div class="code original">{suggestion.original_code.strip()}</div>

            <p><strong>Suggested replacement:</strong></p>
            <div class="code">{suggested_code_html}</div>
        </div>
"""

        html += """    </div>
</body>
</html>"""

        return html

    def apply_suggestions(self, suggestions: List[MigrationSuggestion], auto_fix: bool = False):
        """Apply migration suggestions (with confirmation unless auto_fix)."""
        if not auto_fix:
            print("Auto-fix mode is disabled. Use --auto-fix to apply changes.")
            return

        print(f"Applying {len(suggestions)} suggestions...")

        # Group by file
        by_file = defaultdict(list)
        for suggestion in suggestions:
            by_file[suggestion.file_path].append(suggestion)

        # Apply changes file by file
        for file_path, file_suggestions in by_file.items():
            print(f"\nProcessing {file_path}...")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Sort suggestions by line number (reverse) to avoid offset issues
                file_suggestions.sort(key=lambda s: s.line_number, reverse=True)

                lines = content.split('\n')
                modified = False

                for suggestion in file_suggestions:
                    line_idx = suggestion.line_number - 1
                    if 0 <= line_idx < len(lines):
                        # Simple replacement (more sophisticated logic could be added)
                        lines[line_idx] = f"# MIGRATED: {lines[line_idx]}\n{suggestion.suggested_code}"
                        modified = True
                        print(f"  - Applied {suggestion.pattern.name} at line {suggestion.line_number}")

                if modified:
                    # Write back
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    print(f"  ✓ Updated {file_path}")

            except Exception as e:
                print(f"  ✗ Error applying changes to {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="GreenLang Code Migration Tool - Migrate custom code to GreenLang infrastructure"
    )

    parser.add_argument(
        'path',
        help='File or directory to scan'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them (default)'
    )

    parser.add_argument(
        '--auto-fix',
        action='store_true',
        help='Automatically apply migration suggestions'
    )

    parser.add_argument(
        '--format',
        choices=['text', 'json', 'html'],
        default='text',
        help='Report format (default: text)'
    )

    parser.add_argument(
        '--output',
        help='Output file for report (default: stdout)'
    )

    parser.add_argument(
        '--category',
        help='Only show suggestions for specific category'
    )

    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.0,
        help='Minimum confidence threshold (0.0-1.0)'
    )

    args = parser.parse_args()

    # Initialize tool
    tool = CodeMigrationTool()

    # Scan
    print(f"Scanning {args.path}...")

    if os.path.isfile(args.path):
        suggestions = tool.scan_file(args.path)
    elif os.path.isdir(args.path):
        suggestions = tool.scan_directory(args.path)
    else:
        logger.error(f"{args.path} is not a valid file or directory")
        sys.exit(1)

    # Filter by category if specified
    if args.category:
        suggestions = [s for s in suggestions if s.pattern.category == args.category]

    # Filter by confidence
    suggestions = [s for s in suggestions if s.pattern.confidence >= args.min_confidence]

    print(f"Found {len(suggestions)} migration opportunities")

    # Generate report
    report = tool.generate_report(suggestions, format=args.format)

    # Output report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print("\n" + report)

    # Apply suggestions if requested
    if args.auto_fix and not args.dry_run:
        tool.apply_suggestions(suggestions, auto_fix=True)
        print("\n✓ Migration complete!")
    elif not args.dry_run:
        print("\nTo apply changes, run with --auto-fix")


if __name__ == '__main__':
    main()
