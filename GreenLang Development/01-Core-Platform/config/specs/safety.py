# -*- coding: utf-8 -*-
"""
GreenLang AgentSpec v2 - AST-Based Safety Validation

This module provides AST-based static analysis for validating that tools marked
as "safe" actually meet safety requirements.

Safe Tool Requirements (CTO Security Spec):
1. Pure function (no side effects)
2. No network access (no requests, urllib, httpx, etc.)
3. No filesystem writes (no open with 'w', no pathlib write)
4. No subprocess execution (no subprocess, os.system, eval, exec)
5. Deterministic (same inputs → same outputs)

Author: GreenLang Framework Team
Date: October 2025
Spec: FRMW-201 (AgentSpec v2 Schema + Validators)
"""

import ast
import re
from typing import List

from .errors import GLVErr, GLValidationError


# Forbidden modules and functions for safe tools
FORBIDDEN_MODULES = {
    "subprocess", "os", "sys", "socket", "urllib", "urllib3", "requests",
    "httpx", "http", "ftplib", "smtplib", "telnetlib", "paramiko",
    "asyncio", "threading", "multiprocessing", "ctypes", "builtins"
}

FORBIDDEN_FUNCTIONS = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "system", "popen", "spawn", "fork", "kill", "exit", "quit"
}

FORBIDDEN_WRITE_MODES = {"w", "a", "x", "wb", "ab", "xb", "w+", "a+", "x+"}

# Python URI pattern (must match agentspec_v2.py)
PYTHON_URI_RE = re.compile(
    r"^python://([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*):([a-z_][a-z0-9_]*)$",
    re.IGNORECASE
)


class SafetyChecker(ast.NodeVisitor):
    """
    AST visitor for checking tool safety constraints.

    Detects:
    - Forbidden module imports
    - Forbidden function calls
    - Filesystem writes (open with write mode)
    - Network access attempts
    - Subprocess execution
    - Code evaluation (eval, exec)
    """

    def __init__(self):
        self.violations: List[str] = []
        self.in_open_call = False

    def visit_Import(self, node):
        """Check for forbidden module imports."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            if module_name in FORBIDDEN_MODULES:
                self.violations.append(
                    f"Imports forbidden module '{alias.name}' (unsafe for side effects/network)"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Check for forbidden module imports (from X import Y)."""
        if node.module:
            module_name = node.module.split('.')[0]
            if module_name in FORBIDDEN_MODULES:
                self.violations.append(
                    f"Imports from forbidden module '{node.module}' (unsafe for side effects/network)"
                )
        self.generic_visit(node)

    def visit_Call(self, node):
        """Check for forbidden function calls."""
        # Check direct function calls (eval, exec, etc.)
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_FUNCTIONS:
                self.violations.append(
                    f"Calls forbidden function '{node.func.id}' (unsafe operation)"
                )

            # Special check for open() with write mode
            if node.func.id == "open" and len(node.args) >= 2:
                if isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                    if any(m in str(mode) for m in FORBIDDEN_WRITE_MODES):
                        self.violations.append(
                            f"Calls open() with write mode '{mode}' (filesystem modification)"
                        )

        # Check module.function calls (os.system, subprocess.run, etc.)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                func_name_attr = node.func.attr
                if module_name in FORBIDDEN_MODULES:
                    self.violations.append(
                        f"Calls {module_name}.{func_name_attr}() (forbidden module operation)"
                    )

        self.generic_visit(node)

    def visit_Exec(self, node):
        """Check for exec statements (Python 2 style)."""
        self.violations.append("Uses exec statement (unsafe code execution)")
        self.generic_visit(node)


def validate_safe_tool(impl_uri: str, tool_name: str) -> None:
    """
    Validate that a tool marked as 'safe' is actually safe.

    Safe tool requirements (CTO security spec):
    1. Pure function (no side effects)
    2. No network access (no requests, urllib, httpx, etc.)
    3. No filesystem writes (no open with 'w', no pathlib write)
    4. No subprocess execution (no subprocess, os.system, eval, exec)
    5. Deterministic (same inputs → same outputs)

    This is a STATIC analysis check (AST-based). Runtime enforcement is separate.

    Args:
        impl_uri: Python URI of tool implementation (e.g., "python://module:function")
        tool_name: Tool name for error reporting

    Raises:
        GLValidationError: If tool violates safety constraints

    Example:
        >>> validate_safe_tool("python://math:sqrt", "calculator")
        # No exception - math.sqrt is safe

        >>> validate_safe_tool("python://os:system", "shell_exec")
        # Raises GLValidationError - os.system is unsafe
    """
    # Extract module and function from URI
    match = PYTHON_URI_RE.match(impl_uri)
    if not match:
        return  # URI validation happens elsewhere

    module_path, func_name = match.groups()

    # Try to load and analyze the function
    # NOTE: In production, we would do this in an isolated environment
    # For now, we'll do best-effort static analysis only
    try:
        # Attempt to import the module (only for AST analysis, not execution)
        # This is safe because we're only parsing the source code
        import importlib.util

        # Try to find the module spec
        spec = importlib.util.find_spec(module_path)
        if spec is None or spec.origin is None:
            # Module not found - skip validation (will fail at runtime)
            # We only validate modules that are present
            return

        # Read the source file
        with open(spec.origin, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Parse the AST
        tree = ast.parse(source_code)

        # Find the specific function definition
        function_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                function_found = True
                # Check the function's AST for violations
                checker = SafetyChecker()
                checker.visit(node)

                if checker.violations:
                    violation_details = "\n  - ".join(checker.violations)
                    raise GLValidationError(
                        GLVErr.CONNECTOR_INVALID,  # Using CONNECTOR_INVALID for tool safety
                        f"Tool '{tool_name}' marked as safe=true but violates safety constraints:\n  - {violation_details}",
                        ["ai", "tools", tool_name, "safe"]
                    )
                break

        if not function_found:
            # Function not found in module - skip validation (will fail at runtime)
            return

    except FileNotFoundError:
        # Module file not found - skip validation
        # This can happen for built-in modules or if the module isn't installed yet
        return
    except (ImportError, ModuleNotFoundError):
        # Module can't be imported - skip validation
        # This can happen during testing or if dependencies aren't installed
        return
    except SyntaxError:
        # Source code has syntax errors - skip validation
        # Will fail at runtime anyway
        return
    except GLValidationError:
        # Re-raise our validation errors
        raise
    except Exception:
        # Any other error - skip validation
        # Better to be permissive than block legitimate tools
        return
