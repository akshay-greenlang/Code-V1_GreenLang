# -*- coding: utf-8 -*-
"""
Agent Validator

Validates agent code for structure, security, and compliance.
Includes static analysis, security scanning, and sandbox testing.
"""

import ast
import re
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ValidationResultData:
    """Validation result data"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityIssue:
    """Security vulnerability"""
    severity: SecurityLevel
    message: str
    code: str
    line: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class SecurityScanResult:
    """Security scan result"""
    passed: bool
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 100.0  # 0-100, 100 is perfect


class CodeValidator:
    """
    Static code validator.

    Validates Python code structure and compliance.
    """

    FORBIDDEN_IMPORTS = {
        'os.system', 'subprocess', 'eval', 'exec', '__import__',
        'compile', 'open', 'file', 'input', 'raw_input'
    }

    FORBIDDEN_BUILTINS = {
        'eval', 'exec', 'compile', '__import__', 'globals', 'locals',
        'vars', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr'
    }

    ALLOWED_MODULES = {
        # Standard library
        'json', 'math', 'datetime', 'decimal', 'typing', 'dataclasses',
        'collections', 'itertools', 'functools', 're', 'string',
        # Data science
        'numpy', 'pandas', 'scipy', 'sklearn',
        # GreenLang
        'greenlang', 'greenlang.agents', 'greenlang.sdk'
    }

    MAX_CODE_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_LINES = 10000

    def validate_structure(self, source: str) -> ValidationResultData:
        """
        Validate code structure.

        Args:
            source: Python source code

        Returns:
            Validation result
        """
        result = ValidationResultData(passed=False)

        try:
            # Parse AST
            tree = ast.parse(source)

            # Check code size
            if len(source) > self.MAX_CODE_SIZE:
                result.errors.append(
                    f"Code size exceeds maximum ({self.MAX_CODE_SIZE} bytes)"
                )

            lines = source.split('\n')
            if len(lines) > self.MAX_LINES:
                result.errors.append(
                    f"Code exceeds maximum lines ({self.MAX_LINES})"
                )

            # Find agent class
            agent_class = self._find_agent_class(tree)

            if not agent_class:
                result.errors.append(
                    "No class inheriting from BaseAgent found"
                )
                return result

            # Check required methods
            has_execute = self._has_method(agent_class, 'execute')
            if not has_execute:
                result.errors.append(
                    "Agent class must implement execute() method"
                )

            # Check docstring
            docstring = ast.get_docstring(agent_class)
            if not docstring or len(docstring) < 20:
                result.warnings.append(
                    "Agent class should have a descriptive docstring (minimum 20 characters)"
                )

            # Check execute method signature
            execute_method = self._get_method(agent_class, 'execute')
            if execute_method:
                if not execute_method.args.args or len(execute_method.args.args) < 1:
                    result.errors.append(
                        "execute() method must accept 'self' parameter"
                    )

                # Check for type hints
                if not execute_method.returns:
                    result.warnings.append(
                        "execute() method should have return type annotation"
                    )

            # Validate imports
            import_issues = self._validate_imports(tree)
            result.errors.extend(import_issues)

            # Check for forbidden patterns
            forbidden_issues = self._check_forbidden_patterns(tree)
            result.errors.extend(forbidden_issues)

            # Calculate metrics
            result.metadata = {
                "class_name": agent_class.name,
                "has_execute": has_execute,
                "has_docstring": bool(docstring),
                "lines_of_code": len(lines),
                "complexity": self._calculate_complexity(tree)
            }

            result.passed = len(result.errors) == 0

        except SyntaxError as e:
            result.errors.append(f"Syntax error: {e}")

        except Exception as e:
            result.errors.append(f"Validation error: {e}")
            logger.error(f"Validation error: {e}", exc_info=True)

        return result

    def _find_agent_class(self, tree: ast.AST) -> Optional[ast.ClassDef]:
        """Find class inheriting from BaseAgent"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'BaseAgent':
                        return node
                    if isinstance(base, ast.Attribute) and base.attr == 'BaseAgent':
                        return node
        return None

    def _has_method(self, class_node: ast.ClassDef, method_name: str) -> bool:
        """Check if class has a method"""
        return any(
            isinstance(node, ast.FunctionDef) and node.name == method_name
            for node in class_node.body
        )

    def _get_method(self, class_node: ast.ClassDef, method_name: str) -> Optional[ast.FunctionDef]:
        """Get method from class"""
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return node
        return None

    def _validate_imports(self, tree: ast.AST) -> List[str]:
        """Validate import statements"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_allowed_module(alias.name):
                        issues.append(
                            f"Import of '{alias.name}' is not allowed"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module and not self._is_allowed_module(node.module):
                    issues.append(
                        f"Import from '{node.module}' is not allowed"
                    )

        return issues

    def _is_allowed_module(self, module_name: str) -> bool:
        """Check if module is allowed"""
        # Check exact match
        if module_name in self.ALLOWED_MODULES:
            return True

        # Check prefix match
        for allowed in self.ALLOWED_MODULES:
            if module_name.startswith(allowed + '.'):
                return True

        return False

    def _check_forbidden_patterns(self, tree: ast.AST) -> List[str]:
        """Check for forbidden code patterns"""
        issues = []

        for node in ast.walk(tree):
            # Check for eval/exec
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.FORBIDDEN_BUILTINS:
                        issues.append(
                            f"Use of '{node.func.id}' is forbidden"
                        )

            # Check for file operations
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    issues.append(
                        "Direct file operations are not allowed. Use GreenLang SDK instead."
                    )

        return issues

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity


class SecurityScanner:
    """
    Security vulnerability scanner.

    Scans code for security issues and vulnerabilities.
    """

    DANGEROUS_PATTERNS = [
        (r'os\.system', SecurityLevel.CRITICAL, "System command execution"),
        (r'subprocess\.(call|run|Popen)', SecurityLevel.CRITICAL, "Subprocess execution"),
        (r'eval\(', SecurityLevel.CRITICAL, "Dynamic code evaluation"),
        (r'exec\(', SecurityLevel.CRITICAL, "Dynamic code execution"),
        (r'__import__', SecurityLevel.HIGH, "Dynamic import"),
        (r'pickle\.loads?', SecurityLevel.HIGH, "Unsafe deserialization"),
        (r'yaml\.load\(', SecurityLevel.HIGH, "Unsafe YAML loading"),
        (r'input\(', SecurityLevel.MEDIUM, "User input"),
        (r'open\(.+[\'"]w', SecurityLevel.MEDIUM, "File write operation"),
        (r'requests\.(get|post)', SecurityLevel.LOW, "Network request"),
        (r'urllib\.request', SecurityLevel.LOW, "Network request"),
        (r'socket\.', SecurityLevel.MEDIUM, "Socket operation"),
        (r'sql.*=.*format', SecurityLevel.HIGH, "Potential SQL injection"),
        (r'password.*=.*[\'"]', SecurityLevel.MEDIUM, "Hardcoded credential"),
        (r'api_key.*=.*[\'"]', SecurityLevel.MEDIUM, "Hardcoded API key"),
    ]

    def scan(self, source: str) -> SecurityScanResult:
        """
        Scan code for security vulnerabilities.

        Args:
            source: Python source code

        Returns:
            Security scan result
        """
        result = SecurityScanResult(passed=True)

        # Pattern matching
        for pattern, severity, message in self.DANGEROUS_PATTERNS:
            matches = list(re.finditer(pattern, source, re.MULTILINE))

            for match in matches:
                # Find line number
                line_num = source[:match.start()].count('\n') + 1

                vulnerability = {
                    "severity": severity.value,
                    "message": message,
                    "code": pattern,
                    "line": line_num,
                    "match": match.group(0)
                }

                result.vulnerabilities.append(vulnerability)

                # Deduct from score based on severity
                if severity == SecurityLevel.CRITICAL:
                    result.score -= 25
                elif severity == SecurityLevel.HIGH:
                    result.score -= 15
                elif severity == SecurityLevel.MEDIUM:
                    result.score -= 10
                elif severity == SecurityLevel.LOW:
                    result.score -= 5

        # Fail if critical or high severity issues found
        critical_or_high = any(
            v["severity"] in ["critical", "high"]
            for v in result.vulnerabilities
        )

        result.passed = not critical_or_high
        result.score = max(0, result.score)

        return result

    def check_dependencies(self, dependencies: Dict[str, str]) -> List[SecurityIssue]:
        """
        Check dependencies for known vulnerabilities.

        Args:
            dependencies: Dict of {package: version}

        Returns:
            List of security issues
        """
        issues = []

        # Known vulnerable packages (simplified)
        KNOWN_VULNERABILITIES = {
            'requests': {
                '<2.20.0': 'Known security vulnerability in requests < 2.20.0'
            },
            'pyyaml': {
                '<5.4': 'Unsafe YAML loading in PyYAML < 5.4'
            },
            'pillow': {
                '<8.3.2': 'Known vulnerabilities in Pillow < 8.3.2'
            }
        }

        for package, version in dependencies.items():
            if package in KNOWN_VULNERABILITIES:
                vulns = KNOWN_VULNERABILITIES[package]
                # Simple version check (in production, use proper version parsing)
                for vuln_version, message in vulns.items():
                    issues.append(SecurityIssue(
                        severity=SecurityLevel.HIGH,
                        message=f"{package} {version}: {message}",
                        code="VULNERABLE_DEPENDENCY",
                        details={"package": package, "version": version}
                    ))

        return issues


class AgentValidator:
    """
    Main agent validator.

    Combines code validation and security scanning.
    """

    def __init__(self):
        self.code_validator = CodeValidator()
        self.security_scanner = SecurityScanner()

    def validate(self, source: str, dependencies: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Perform complete validation.

        Args:
            source: Python source code
            dependencies: Optional dependencies dict

        Returns:
            Complete validation result
        """
        # Code validation
        code_result = self.code_validator.validate_structure(source)

        # Security scan
        security_result = self.security_scanner.scan(source)

        # Dependency check
        dependency_issues = []
        if dependencies:
            dependency_issues = self.security_scanner.check_dependencies(dependencies)

        # Combine results
        return {
            "passed": code_result.passed and security_result.passed,
            "code_validation": {
                "passed": code_result.passed,
                "errors": code_result.errors,
                "warnings": code_result.warnings,
                "metadata": code_result.metadata
            },
            "security_scan": {
                "passed": security_result.passed,
                "score": security_result.score,
                "vulnerabilities": security_result.vulnerabilities
            },
            "dependencies": {
                "issues": [
                    {"severity": i.severity.value, "message": i.message}
                    for i in dependency_issues
                ]
            }
        }

    def validate_structure(self, source: str) -> ValidationResultData:
        """Validate code structure only"""
        return self.code_validator.validate_structure(source)

    def scan_security(self, source: str) -> SecurityScanResult:
        """Security scan only"""
        return self.security_scanner.scan(source)
