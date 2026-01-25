#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Infrastructure-First Static Linter
=============================================

AST-based static analysis to enforce infrastructure-first principles.

Usage:
    python infrastructure_first.py [--path PATH] [--output FORMAT]

Features:
    - Detects forbidden direct imports
    - Identifies custom agent classes
    - Flags custom LLM clients
    - Detects direct database usage
    - Checks for missing infrastructure imports
"""

import ast
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, asdict


@dataclass
class Violation:
    """Represents a single violation"""
    file: str
    line: int
    column: int
    severity: str
    code: str
    message: str
    suggestion: str
    category: str


class InfrastructureFirstLinter(ast.NodeVisitor):
    """AST-based linter for infrastructure-first enforcement"""

    # Forbidden imports with their GreenLang alternatives
    FORBIDDEN_IMPORTS = {
        'openai': ('greenlang.intelligence', 'Use ChatSession for LLM calls'),
        'anthropic': ('greenlang.intelligence', 'Use ChatSession for LLM calls'),
        'redis': ('greenlang.cache', 'Use CacheManager for caching'),
        'pymongo': ('greenlang.db', 'Use GreenLang database connectors'),
        'motor': ('greenlang.db', 'Use GreenLang async database connectors'),
        'sqlalchemy': ('greenlang.db', 'Use GreenLang ORM layer'),
        'jose': ('greenlang.auth', 'Use AuthManager for JWT'),
        'jwt': ('greenlang.auth', 'Use AuthManager for JWT'),
        'pyjwt': ('greenlang.auth', 'Use AuthManager for JWT'),
        'passlib': ('greenlang.auth', 'Use AuthManager for password hashing'),
        'bcrypt': ('greenlang.auth', 'Use AuthManager for password hashing'),
        'requests': ('greenlang.http', 'Use GreenLang HTTP client (with retries, timeouts)'),
    }

    # Patterns that suggest LLM usage
    LLM_PATTERNS = {
        'create_completion', 'chat.completions', 'embeddings.create',
        'generate_text', 'get_completion', 'call_llm', 'llm_response'
    }

    # Patterns that suggest auth usage
    AUTH_PATTERNS = {
        'hash_password', 'verify_password', 'create_token', 'decode_token',
        'authenticate_user', 'check_permission', 'get_current_user'
    }

    # Database operation patterns
    DB_PATTERNS = {
        '.find(', '.insert(', '.update(', '.delete(',
        '.find_one(', '.insert_one(', '.update_one(', '.delete_one(',
        'execute(', 'query(', 'session.add(', 'session.commit('
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.violations: List[Violation] = []
        self.imports: Set[str] = set()
        self.greenlang_imports: Set[str] = set()
        self.has_agent_import = False

    def visit_Import(self, node: ast.Import):
        """Check import statements"""
        for alias in node.names:
            module = alias.name
            root_module = module.split('.')[0]

            self.imports.add(module)

            # Track GreenLang imports
            if module.startswith('greenlang'):
                self.greenlang_imports.add(module)
                if 'sdk.base' in module or alias.asname == 'Agent':
                    self.has_agent_import = True

            # Check forbidden imports
            if root_module in self.FORBIDDEN_IMPORTS:
                alternative, reason = self.FORBIDDEN_IMPORTS[root_module]
                self.add_violation(
                    node.lineno,
                    node.col_offset,
                    'ERROR',
                    'FORBIDDEN_IMPORT',
                    f"Direct import of '{root_module}' is forbidden",
                    f"Use {alternative} instead. {reason}",
                    'imports'
                )

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Check from-import statements"""
        if not node.module:
            self.generic_visit(node)
            return

        module = node.module
        root_module = module.split('.')[0]

        self.imports.add(module)

        # Track GreenLang imports
        if module.startswith('greenlang'):
            self.greenlang_imports.add(module)
            # Check if importing Agent
            for alias in node.names:
                if alias.name == 'Agent':
                    self.has_agent_import = True

        # Check forbidden imports
        if root_module in self.FORBIDDEN_IMPORTS:
            alternative, reason = self.FORBIDDEN_IMPORTS[root_module]
            self.add_violation(
                node.lineno,
                node.col_offset,
                'ERROR',
                'FORBIDDEN_IMPORT',
                f"Direct import from '{root_module}' is forbidden",
                f"Use {alternative} instead. {reason}",
                'imports'
            )

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Check class definitions"""
        class_name = node.name

        # Check if this looks like an Agent class
        if 'agent' in class_name.lower():
            # Get base classes
            base_names = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(self._get_attr_name(base))

            # Check if inherits from Agent
            has_agent_base = any('Agent' in base for base in base_names)

            if not has_agent_base:
                self.add_violation(
                    node.lineno,
                    node.col_offset,
                    'ERROR',
                    'CUSTOM_AGENT',
                    f"Agent class '{class_name}' does not inherit from greenlang.integration.sdk.base.Agent",
                    "Add 'from greenlang.integration.sdk.base import Agent' and inherit from Agent",
                    'architecture'
                )
            elif not self.has_agent_import:
                self.add_violation(
                    node.lineno,
                    node.col_offset,
                    'WARNING',
                    'MISSING_IMPORT',
                    f"Agent base class used but greenlang.sdk.base not imported",
                    "Add 'from greenlang.integration.sdk.base import Agent'",
                    'imports'
                )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Check function calls for patterns"""
        call_str = ast.unparse(node)

        # Check for LLM patterns
        if any(pattern in call_str for pattern in self.LLM_PATTERNS):
            if not any('intelligence' in imp for imp in self.greenlang_imports):
                self.add_violation(
                    node.lineno,
                    node.col_offset,
                    'ERROR',
                    'CUSTOM_LLM',
                    f"Custom LLM call detected: {call_str[:50]}...",
                    "Use greenlang.intelligence.ChatSession instead",
                    'llm'
                )

        # Check for auth patterns
        if any(pattern in call_str for pattern in self.AUTH_PATTERNS):
            if not any('auth' in imp for imp in self.greenlang_imports):
                self.add_violation(
                    node.lineno,
                    node.col_offset,
                    'ERROR',
                    'CUSTOM_AUTH',
                    f"Custom auth operation detected: {call_str[:50]}...",
                    "Use greenlang.auth.AuthManager instead",
                    'auth'
                )

        # Check for database patterns
        if any(pattern in call_str for pattern in self.DB_PATTERNS):
            if not any('db' in imp or 'database' in imp for imp in self.greenlang_imports):
                self.add_violation(
                    node.lineno,
                    node.col_offset,
                    'WARNING',
                    'DIRECT_DB',
                    f"Direct database operation detected: {call_str[:50]}...",
                    "Consider using greenlang.db connectors",
                    'database'
                )

        self.generic_visit(node)

    def add_violation(self, line: int, column: int, severity: str,
                     code: str, message: str, suggestion: str, category: str):
        """Add a violation to the list"""
        self.violations.append(Violation(
            file=self.file_path,
            line=line,
            column=column,
            severity=severity,
            code=code,
            message=message,
            suggestion=suggestion,
            category=category
        ))

    @staticmethod
    def _get_attr_name(node):
        """Get full attribute name"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{InfrastructureFirstLinter._get_attr_name(node.value)}.{node.attr}"
        return ""


def lint_file(file_path: Path) -> List[Violation]:
    """Lint a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))
        linter = InfrastructureFirstLinter(str(file_path))
        linter.visit(tree)

        return linter.violations

    except SyntaxError as e:
        return [Violation(
            file=str(file_path),
            line=e.lineno or 0,
            column=e.offset or 0,
            severity='ERROR',
            code='SYNTAX_ERROR',
            message=f"Syntax error: {e.msg}",
            suggestion="Fix syntax errors before linting",
            category='syntax'
        )]
    except Exception as e:
        return [Violation(
            file=str(file_path),
            line=0,
            column=0,
            severity='ERROR',
            code='LINT_ERROR',
            message=f"Error linting file: {str(e)}",
            suggestion="Check file for issues",
            category='error'
        )]


def lint_directory(directory: Path) -> List[Violation]:
    """Lint all Python files in directory"""
    violations = []

    for py_file in directory.rglob('*.py'):
        # Skip virtual environments and build directories
        if any(part.startswith('.') or part in {'venv', 'env', 'build', 'dist', '__pycache__'}
               for part in py_file.parts):
            continue

        violations.extend(lint_file(py_file))

    return violations


def format_violations_text(violations: List[Violation]) -> str:
    """Format violations as text"""
    if not violations:
        return "✓ No violations found"

    output = []
    output.append(f"\nFound {len(violations)} violation(s):\n")
    output.append("=" * 80)

    # Group by file
    by_file = {}
    for v in violations:
        by_file.setdefault(v.file, []).append(v)

    for file_path, file_violations in sorted(by_file.items()):
        output.append(f"\n{file_path}:")

        for v in sorted(file_violations, key=lambda x: x.line):
            severity_marker = "✗" if v.severity == "ERROR" else "⚠"
            output.append(f"  {severity_marker} Line {v.line}:{v.column} [{v.code}] {v.message}")
            output.append(f"    → {v.suggestion}")

    output.append("\n" + "=" * 80)
    return "\n".join(output)


def format_violations_json(violations: List[Violation]) -> str:
    """Format violations as JSON"""
    return json.dumps({
        'violations': [asdict(v) for v in violations],
        'summary': {
            'total': len(violations),
            'errors': len([v for v in violations if v.severity == 'ERROR']),
            'warnings': len([v for v in violations if v.severity == 'WARNING']),
            'by_category': {
                cat: len([v for v in violations if v.category == cat])
                for cat in set(v.category for v in violations)
            }
        }
    }, indent=2)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='GreenLang Infrastructure-First Static Linter'
    )
    parser.add_argument(
        '--path',
        default='.',
        help='Path to file or directory to lint (default: current directory)'
    )
    parser.add_argument(
        '--output',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--fail-on',
        choices=['error', 'warning', 'any'],
        default='error',
        help='Exit with error code when violations found (default: error)'
    )

    args = parser.parse_args()

    # Lint files
    path = Path(args.path)
    if path.is_file():
        violations = lint_file(path)
    elif path.is_dir():
        violations = lint_directory(path)
    else:
        print(f"Error: {path} is not a valid file or directory", file=sys.stderr)
        return 1

    # Format output
    if args.output == 'json':
        print(format_violations_json(violations))
    else:
        print(format_violations_text(violations))

    # Determine exit code
    if args.fail_on == 'any' and violations:
        return 1
    elif args.fail_on == 'warning' and any(v.severity in ['ERROR', 'WARNING'] for v in violations):
        return 1
    elif args.fail_on == 'error' and any(v.severity == 'ERROR' for v in violations):
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
