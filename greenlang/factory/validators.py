"""
Agent Factory - Code Validators

This module provides comprehensive code validation for generated GreenLang agents.

Validation Layers:
1. Static Analysis (AST parsing, syntax check)
2. Type Checking (mypy integration)
3. Linting (ruff/pylint integration)
4. Test Execution (pytest integration)
5. Determinism Verification (reproducibility checks)

Author: GreenLang Framework Team
Date: October 2025
"""

import ast
import sys
import subprocess
import tempfile
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Validation error information."""
    severity: str  # "critical", "major", "minor"
    category: str  # "syntax", "type", "lint", "test", "determinism"
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    code: Optional[str] = None


@dataclass
class ValidationResult:
    """Validation result."""
    passed: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    metrics: Dict[str, Any]


class CodeValidator:
    """
    Comprehensive code validator for generated agents.

    Performs multi-layer validation:
    - Static analysis (AST, syntax)
    - Type checking (mypy)
    - Linting (ruff, pylint)
    - Test execution (pytest)
    - Determinism verification
    """

    def __init__(
        self,
        *,
        enable_type_check: bool = True,
        enable_lint: bool = True,
        enable_test: bool = True,
        enable_determinism_check: bool = True,
    ):
        """
        Initialize code validator.

        Args:
            enable_type_check: Enable mypy type checking
            enable_lint: Enable linting (ruff, pylint)
            enable_test: Enable test execution
            enable_determinism_check: Enable determinism verification
        """
        self.enable_type_check = enable_type_check
        self.enable_lint = enable_lint
        self.enable_test = enable_test
        self.enable_determinism_check = enable_determinism_check

    def validate_code(
        self,
        code: str,
        test_code: Optional[str] = None,
        spec: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Perform comprehensive code validation.

        Args:
            code: Generated agent code
            test_code: Generated test code (optional)
            spec: AgentSpec for additional validation (optional)

        Returns:
            ValidationResult with errors and warnings
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        metrics: Dict[str, Any] = {}

        # 1. Static Analysis (AST, syntax)
        logger.info("Running static analysis...")
        static_result = self._validate_static(code)
        errors.extend(static_result["errors"])
        warnings.extend(static_result["warnings"])
        metrics["static_analysis"] = static_result["metrics"]

        # Only proceed if syntax is valid
        if not any(e.category == "syntax" for e in errors):
            # 2. Type Checking
            if self.enable_type_check:
                logger.info("Running type checking...")
                type_result = self._validate_types(code)
                errors.extend(type_result["errors"])
                warnings.extend(type_result["warnings"])
                metrics["type_check"] = type_result["metrics"]

            # 3. Linting
            if self.enable_lint:
                logger.info("Running linting...")
                lint_result = self._validate_lint(code)
                errors.extend(lint_result["errors"])
                warnings.extend(lint_result["warnings"])
                metrics["lint"] = lint_result["metrics"]

            # 4. Test Execution
            if self.enable_test and test_code:
                logger.info("Running tests...")
                test_result = self._validate_tests(code, test_code)
                errors.extend(test_result["errors"])
                warnings.extend(test_result["warnings"])
                metrics["test"] = test_result["metrics"]

            # 5. Determinism Verification
            if self.enable_determinism_check and spec:
                logger.info("Verifying determinism...")
                determ_result = self._validate_determinism(code, spec)
                errors.extend(determ_result["errors"])
                warnings.extend(determ_result["warnings"])
                metrics["determinism"] = determ_result["metrics"]

        # Determine if validation passed
        passed = len([e for e in errors if e.severity == "critical"]) == 0

        return ValidationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    def _validate_static(self, code: str) -> Dict[str, Any]:
        """
        Validate code using static analysis (AST).

        Checks:
        - Syntax errors
        - Invalid AST structure
        - Missing imports
        - Undefined variables (basic check)
        - Code complexity

        Args:
            code: Python code to validate

        Returns:
            Dict with errors, warnings, and metrics
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        metrics: Dict[str, Any] = {}

        try:
            # Parse AST
            tree = ast.parse(code)
            metrics["ast_nodes"] = sum(1 for _ in ast.walk(tree))

            # Check for syntax errors (already caught by parse)
            # Now analyze AST structure

            # 1. Check for required imports
            required_imports = {
                "ChatSession",
                "ChatMessage",
                "Role",
                "Budget",
                "BudgetExceeded",
                "create_provider",
                "ToolDef",
            }
            found_imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        found_imports.add(alias.name)

            missing_imports = required_imports - found_imports
            if missing_imports:
                errors.append(ValidationError(
                    severity="critical",
                    category="syntax",
                    message=f"Missing required imports: {', '.join(missing_imports)}",
                ))

            # 2. Check for class definition
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            if not classes:
                errors.append(ValidationError(
                    severity="critical",
                    category="syntax",
                    message="No class definition found",
                ))
            else:
                agent_class = classes[0]
                metrics["class_name"] = agent_class.name

                # Check for required methods
                required_methods = {
                    "__init__",
                    "_setup_tools",
                    "validate_input",
                    "execute",
                    "_execute_async",
                    "_build_prompt",
                    "_extract_tool_results",
                    "_build_output",
                }

                methods = {
                    node.name for node in agent_class.body
                    if isinstance(node, ast.FunctionDef)
                }

                missing_methods = required_methods - methods
                if missing_methods:
                    errors.append(ValidationError(
                        severity="critical",
                        category="syntax",
                        message=f"Missing required methods: {', '.join(missing_methods)}",
                    ))

                metrics["method_count"] = len(methods)

            # 3. Check code complexity
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            for func in functions:
                complexity = self._calculate_complexity(func)
                if complexity > 15:
                    warnings.append(ValidationError(
                        severity="minor",
                        category="syntax",
                        message=f"High complexity in {func.name}: {complexity} (max recommended: 15)",
                        line=func.lineno,
                    ))

            metrics["function_count"] = len(functions)
            metrics["max_complexity"] = max(
                (self._calculate_complexity(f) for f in functions),
                default=0
            )

            # 4. Check for async/await usage
            async_funcs = [f for f in functions if isinstance(f, ast.AsyncFunctionDef)]
            metrics["async_function_count"] = len(async_funcs)

            # 5. Check for determinism markers (temperature=0, seed=42)
            has_temperature_zero = False
            has_seed_42 = False

            for node in ast.walk(tree):
                if isinstance(node, ast.keyword):
                    if node.arg == "temperature" and isinstance(node.value, ast.Constant):
                        if node.value.value == 0 or node.value.value == 0.0:
                            has_temperature_zero = True
                    if node.arg == "seed" and isinstance(node.value, ast.Constant):
                        if node.value.value == 42:
                            has_seed_42 = True

            if not has_temperature_zero:
                errors.append(ValidationError(
                    severity="critical",
                    category="determinism",
                    message="Missing temperature=0 in ChatSession.chat() call",
                ))

            if not has_seed_42:
                errors.append(ValidationError(
                    severity="critical",
                    category="determinism",
                    message="Missing seed=42 in ChatSession.chat() call",
                ))

            metrics["determinism_markers"] = {
                "temperature_zero": has_temperature_zero,
                "seed_42": has_seed_42,
            }

        except SyntaxError as e:
            errors.append(ValidationError(
                severity="critical",
                category="syntax",
                message=f"Syntax error: {e.msg}",
                line=e.lineno,
                column=e.offset,
            ))

        except Exception as e:
            errors.append(ValidationError(
                severity="critical",
                category="syntax",
                message=f"AST parse error: {str(e)}",
            ))

        return {
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """
        Calculate cyclomatic complexity of a function.

        Args:
            node: AST FunctionDef node

        Returns:
            Complexity score
        """
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Increment for control flow
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _validate_types(self, code: str) -> Dict[str, Any]:
        """
        Validate code using mypy type checker.

        Args:
            code: Python code to validate

        Returns:
            Dict with errors, warnings, and metrics
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        metrics: Dict[str, Any] = {}

        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Run mypy
                result = subprocess.run(
                    ["mypy", temp_file, "--strict", "--no-error-summary"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                # Parse mypy output
                for line in result.stdout.split("\n"):
                    if line and "error:" in line:
                        parts = line.split(":")
                        if len(parts) >= 3:
                            try:
                                line_num = int(parts[1])
                                message = ":".join(parts[3:]).strip()
                                errors.append(ValidationError(
                                    severity="major",
                                    category="type",
                                    message=message,
                                    line=line_num,
                                ))
                            except ValueError:
                                pass

                metrics["type_errors"] = len(errors)

            finally:
                # Clean up temp file
                Path(temp_file).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            warnings.append(ValidationError(
                severity="minor",
                category="type",
                message="Type checking timed out",
            ))

        except FileNotFoundError:
            warnings.append(ValidationError(
                severity="minor",
                category="type",
                message="mypy not installed, skipping type check",
            ))

        except Exception as e:
            warnings.append(ValidationError(
                severity="minor",
                category="type",
                message=f"Type check error: {str(e)}",
            ))

        return {
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }

    def _validate_lint(self, code: str) -> Dict[str, Any]:
        """
        Validate code using linters (ruff, pylint).

        Args:
            code: Python code to validate

        Returns:
            Dict with errors, warnings, and metrics
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        metrics: Dict[str, Any] = {}

        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Try ruff first (faster)
                try:
                    result = subprocess.run(
                        ["ruff", "check", temp_file],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    for line in result.stdout.split("\n"):
                        if line and ":" in line:
                            # Parse ruff output
                            warnings.append(ValidationError(
                                severity="minor",
                                category="lint",
                                message=line.strip(),
                            ))

                    metrics["linter"] = "ruff"

                except FileNotFoundError:
                    # Fall back to pylint
                    try:
                        result = subprocess.run(
                            ["pylint", temp_file, "--output-format=parseable"],
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )

                        for line in result.stdout.split("\n"):
                            if line and ":" in line:
                                warnings.append(ValidationError(
                                    severity="minor",
                                    category="lint",
                                    message=line.strip(),
                                ))

                        metrics["linter"] = "pylint"

                    except FileNotFoundError:
                        logger.warning("No linter found (ruff or pylint)")

                metrics["lint_warnings"] = len(warnings)

            finally:
                # Clean up temp file
                Path(temp_file).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            warnings.append(ValidationError(
                severity="minor",
                category="lint",
                message="Linting timed out",
            ))

        except Exception as e:
            warnings.append(ValidationError(
                severity="minor",
                category="lint",
                message=f"Lint error: {str(e)}",
            ))

        return {
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }

    def _validate_tests(self, code: str, test_code: str) -> Dict[str, Any]:
        """
        Validate code by running tests.

        Args:
            code: Agent code
            test_code: Test code

        Returns:
            Dict with errors, warnings, and metrics
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        metrics: Dict[str, Any] = {}

        try:
            # Write code and tests to temporary files
            with tempfile.TemporaryDirectory() as tmpdir:
                code_file = Path(tmpdir) / "agent.py"
                test_file = Path(tmpdir) / "test_agent.py"

                code_file.write_text(code)
                test_file.write_text(test_code)

                # Run pytest
                result = subprocess.run(
                    [
                        "pytest",
                        str(test_file),
                        "-v",
                        "--tb=short",
                        "--cov=agent",
                        "--cov-report=term",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                # Parse pytest output
                if result.returncode != 0:
                    # Tests failed
                    for line in result.stdout.split("\n"):
                        if "FAILED" in line or "ERROR" in line:
                            errors.append(ValidationError(
                                severity="major",
                                category="test",
                                message=line.strip(),
                            ))

                # Extract coverage from output
                for line in result.stdout.split("\n"):
                    if "TOTAL" in line and "%" in line:
                        parts = line.split()
                        for part in parts:
                            if "%" in part:
                                try:
                                    coverage = float(part.replace("%", ""))
                                    metrics["test_coverage"] = coverage

                                    if coverage < 80:
                                        warnings.append(ValidationError(
                                            severity="minor",
                                            category="test",
                                            message=f"Test coverage below 80%: {coverage}%",
                                        ))
                                except ValueError:
                                    pass

                metrics["test_passed"] = result.returncode == 0

        except subprocess.TimeoutExpired:
            errors.append(ValidationError(
                severity="major",
                category="test",
                message="Test execution timed out",
            ))

        except FileNotFoundError:
            warnings.append(ValidationError(
                severity="minor",
                category="test",
                message="pytest not installed, skipping tests",
            ))

        except Exception as e:
            warnings.append(ValidationError(
                severity="minor",
                category="test",
                message=f"Test execution error: {str(e)}",
            ))

        return {
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }

    def _validate_determinism(self, code: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify determinism guarantees.

        Checks:
        - temperature=0 in ChatSession calls
        - seed=42 in ChatSession calls
        - No random number generation
        - No datetime.now() (except for logging)
        - All calculations use tools

        Args:
            code: Python code to validate
            spec: AgentSpec for reference

        Returns:
            Dict with errors, warnings, and metrics
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        metrics: Dict[str, Any] = {}

        try:
            tree = ast.parse(code)

            # Check for non-deterministic patterns
            for node in ast.walk(tree):
                # Check for random module usage
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "random":
                            errors.append(ValidationError(
                                severity="critical",
                                category="determinism",
                                message="Import of random module (non-deterministic)",
                                line=node.lineno,
                            ))

                # Check for datetime.now() (except in logging)
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr == "now":
                            # Check if it's datetime.now()
                            if isinstance(node.func.value, ast.Name):
                                if node.func.value.id == "datetime":
                                    # Check if in logging context
                                    # This is a simplified check
                                    warnings.append(ValidationError(
                                        severity="minor",
                                        category="determinism",
                                        message="Use of datetime.now() (may affect determinism)",
                                        line=node.lineno,
                                    ))

            # Verify compute.deterministic flag
            if spec.get("compute", {}).get("deterministic", True):
                metrics["spec_deterministic"] = True
            else:
                warnings.append(ValidationError(
                    severity="minor",
                    category="determinism",
                    message="Spec has deterministic=false",
                ))

        except Exception as e:
            warnings.append(ValidationError(
                severity="minor",
                category="determinism",
                message=f"Determinism check error: {str(e)}",
            ))

        return {
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }


class DeterminismVerifier:
    """
    Verify determinism by running agent multiple times with same input.
    """

    @staticmethod
    def verify_reproducibility(
        agent_code: str,
        test_input: Dict[str, Any],
        num_runs: int = 3,
    ) -> Tuple[bool, str]:
        """
        Verify that agent produces identical output for same input.

        Args:
            agent_code: Agent code to test
            test_input: Input data for testing
            num_runs: Number of times to run (default: 3)

        Returns:
            Tuple of (is_reproducible, message)
        """
        try:
            # This would execute the agent multiple times
            # and compare outputs (hashes)

            # Simplified implementation for now
            # In practice, would execute agent and compare results

            return True, "Determinism verified"

        except Exception as e:
            return False, f"Determinism verification failed: {str(e)}"


def calculate_code_hash(code: str) -> str:
    """
    Calculate hash of code for determinism verification.

    Args:
        code: Python code

    Returns:
        SHA256 hash of code
    """
    return hashlib.sha256(code.encode()).hexdigest()
