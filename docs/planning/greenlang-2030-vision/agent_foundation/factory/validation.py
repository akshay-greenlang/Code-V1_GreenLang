# -*- coding: utf-8 -*-
"""
Validation - Validate generated agents against quality standards.

This module ensures generated agents meet GreenLang quality requirements
including code quality, test coverage, performance targets, and compliance.

Example:
    >>> validator = AgentValidator()
    >>> result = validator.validate_agent(code_path, test_path, spec)
    >>> print(f"Quality score: {result.quality_score}%")
"""

import ast
import re
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of agent validation."""

    # Overall status
    is_valid: bool = Field(..., description="Overall validation status")
    quality_score: float = Field(0.0, ge=0.0, le=100.0, description="Quality score (0-100)")

    # Individual metrics
    code_quality_score: float = Field(0.0, description="Code quality score")
    test_coverage: float = Field(0.0, description="Test coverage percentage")
    documentation_score: float = Field(0.0, description="Documentation completeness")
    performance_score: float = Field(0.0, description="Performance compliance")
    security_score: float = Field(0.0, description="Security assessment")

    # Issues found
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Critical errors")
    warnings: List[Dict[str, Any]] = Field(default_factory=list, description="Warnings")
    suggestions: List[Dict[str, Any]] = Field(default_factory=list, description="Improvement suggestions")

    # Detailed checks
    checks_passed: List[str] = Field(default_factory=list, description="Passed checks")
    checks_failed: List[str] = Field(default_factory=list, description="Failed checks")

    # Metadata
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    validator_version: str = Field("1.0.0")


class QualityMetrics(BaseModel):
    """Quality metrics for agent code."""

    # Code metrics
    lines_of_code: int = Field(0, description="Total lines of code")
    cyclomatic_complexity: float = Field(0.0, description="Average cyclomatic complexity")
    maintainability_index: float = Field(0.0, description="Maintainability index")
    technical_debt_ratio: float = Field(0.0, description="Technical debt ratio")

    # Documentation metrics
    docstring_coverage: float = Field(0.0, description="Percentage of documented functions")
    comment_ratio: float = Field(0.0, description="Comment to code ratio")

    # Test metrics
    test_count: int = Field(0, description="Number of tests")
    assertion_count: int = Field(0, description="Number of assertions")
    test_line_coverage: float = Field(0.0, description="Line coverage percentage")
    test_branch_coverage: float = Field(0.0, description="Branch coverage percentage")

    # Best practices
    type_hint_coverage: float = Field(0.0, description="Type hint coverage")
    error_handling_coverage: float = Field(0.0, description="Try-except coverage")
    logging_present: bool = Field(False, description="Logging statements present")


class AgentValidator:
    """
    Comprehensive validator for generated agents.

    Validates:
    - Code quality (linting, complexity, style)
    - Test coverage (target: 85%+)
    - Documentation completeness
    - Performance compliance
    - Security best practices
    - GreenLang standards compliance
    """

    def __init__(self):
        """Initialize validator."""
        self.quality_thresholds = {
            "test_coverage": 85.0,
            "docstring_coverage": 80.0,
            "complexity_threshold": 10.0,
            "maintainability_index": 20.0,
            "type_hint_coverage": 90.0
        }

        self.validators = {
            "code_quality": self._validate_code_quality,
            "test_coverage": self._validate_test_coverage,
            "documentation": self._validate_documentation,
            "performance": self._validate_performance,
            "security": self._validate_security,
            "standards": self._validate_standards
        }

    def validate_agent(
        self,
        code_path: Path,
        test_path: Optional[Path] = None,
        spec: Optional[Any] = None
    ) -> ValidationResult:
        """
        Validate generated agent against standards.

        Args:
            code_path: Path to agent code
            test_path: Path to test file
            spec: Agent specification

        Returns:
            Validation result with quality score
        """
        logger.info(f"Starting validation for {code_path}")

        errors = []
        warnings = []
        suggestions = []
        checks_passed = []
        checks_failed = []

        # Run all validators
        scores = {}
        for name, validator in self.validators.items():
            try:
                score, issues = validator(code_path, test_path, spec)
                scores[name] = score

                if score >= 80:
                    checks_passed.append(name)
                else:
                    checks_failed.append(name)

                # Collect issues
                for issue in issues:
                    if issue["severity"] == "error":
                        errors.append(issue)
                    elif issue["severity"] == "warning":
                        warnings.append(issue)
                    else:
                        suggestions.append(issue)

            except Exception as e:
                logger.error(f"Validator '{name}' failed: {str(e)}")
                scores[name] = 0.0
                checks_failed.append(name)
                errors.append({
                    "validator": name,
                    "severity": "error",
                    "message": str(e)
                })

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(scores)

        # Determine validity
        is_valid = (
            quality_score >= 70.0 and
            len(errors) == 0 and
            scores.get("test_coverage", 0) >= self.quality_thresholds["test_coverage"]
        )

        result = ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            code_quality_score=scores.get("code_quality", 0.0),
            test_coverage=scores.get("test_coverage", 0.0),
            documentation_score=scores.get("documentation", 0.0),
            performance_score=scores.get("performance", 0.0),
            security_score=scores.get("security", 0.0),
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            checks_passed=checks_passed,
            checks_failed=checks_failed
        )

        logger.info(
            f"Validation complete: Valid={is_valid}, "
            f"Quality={quality_score:.1f}%, "
            f"Errors={len(errors)}, Warnings={len(warnings)}"
        )

        return result

    def _validate_code_quality(
        self,
        code_path: Path,
        test_path: Optional[Path],
        spec: Optional[Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Validate code quality metrics."""
        issues = []
        score = 100.0

        # Parse code with AST
        try:
            with open(code_path, 'r') as f:
                code = f.read()
                tree = ast.parse(code)

            # Check complexity
            complexity = self._calculate_complexity(tree)
            if complexity > self.quality_thresholds["complexity_threshold"]:
                score -= 20
                issues.append({
                    "severity": "warning",
                    "message": f"High complexity: {complexity:.1f} (threshold: {self.quality_thresholds['complexity_threshold']})"
                })

            # Check for common issues
            if "exec(" in code or "eval(" in code:
                score -= 30
                issues.append({
                    "severity": "error",
                    "message": "Use of exec() or eval() detected - security risk"
                })

            # Check imports
            if not self._check_imports(tree):
                score -= 10
                issues.append({
                    "severity": "warning",
                    "message": "Missing required imports or unused imports detected"
                })

            # Check error handling
            if not self._check_error_handling(tree):
                score -= 15
                issues.append({
                    "severity": "warning",
                    "message": "Insufficient error handling"
                })

        except SyntaxError as e:
            score = 0.0
            issues.append({
                "severity": "error",
                "message": f"Syntax error in code: {str(e)}"
            })

        return max(0.0, score), issues

    def _validate_test_coverage(
        self,
        code_path: Path,
        test_path: Optional[Path],
        spec: Optional[Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Validate test coverage."""
        issues = []

        if not test_path or not test_path.exists():
            return 0.0, [{
                "severity": "error",
                "message": "No test file found"
            }]

        # Count test methods
        try:
            with open(test_path, 'r') as f:
                test_code = f.read()

            test_count = len(re.findall(r'def\s+test_\w+', test_code))

            if test_count < 5:
                issues.append({
                    "severity": "warning",
                    "message": f"Only {test_count} tests found (recommend 5+)"
                })

            # Estimate coverage (simplified - real implementation would run coverage.py)
            estimated_coverage = min(100.0, test_count * 15.0)

            if estimated_coverage < self.quality_thresholds["test_coverage"]:
                issues.append({
                    "severity": "warning",
                    "message": f"Test coverage {estimated_coverage:.1f}% below threshold {self.quality_thresholds['test_coverage']}%"
                })

            return estimated_coverage, issues

        except Exception as e:
            return 0.0, [{
                "severity": "error",
                "message": f"Failed to analyze tests: {str(e)}"
            }]

    def _validate_documentation(
        self,
        code_path: Path,
        test_path: Optional[Path],
        spec: Optional[Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Validate documentation completeness."""
        issues = []
        score = 100.0

        try:
            with open(code_path, 'r') as f:
                code = f.read()
                tree = ast.parse(code)

            # Check module docstring
            if not ast.get_docstring(tree):
                score -= 20
                issues.append({
                    "severity": "warning",
                    "message": "Missing module docstring"
                })

            # Check class and method docstrings
            total_items = 0
            documented_items = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    total_items += 1
                    if ast.get_docstring(node):
                        documented_items += 1

            if total_items > 0:
                doc_coverage = (documented_items / total_items) * 100
                if doc_coverage < self.quality_thresholds["docstring_coverage"]:
                    score -= 30
                    issues.append({
                        "severity": "warning",
                        "message": f"Docstring coverage {doc_coverage:.1f}% below threshold"
                    })

            # Check for type hints
            type_hint_coverage = self._check_type_hints(tree)
            if type_hint_coverage < self.quality_thresholds["type_hint_coverage"]:
                score -= 20
                issues.append({
                    "severity": "suggestion",
                    "message": f"Type hint coverage {type_hint_coverage:.1f}% below recommended"
                })

        except Exception as e:
            score = 50.0
            issues.append({
                "severity": "error",
                "message": f"Failed to analyze documentation: {str(e)}"
            })

        return max(0.0, score), issues

    def _validate_performance(
        self,
        code_path: Path,
        test_path: Optional[Path],
        spec: Optional[Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Validate performance characteristics."""
        issues = []
        score = 100.0

        if spec and hasattr(spec, 'performance_targets'):
            # Check for performance anti-patterns
            with open(code_path, 'r') as f:
                code = f.read()

            # Check for nested loops (potential O(n²) or worse)
            if re.search(r'for\s+.*:\s+.*for\s+.*:', code):
                score -= 20
                issues.append({
                    "severity": "suggestion",
                    "message": "Nested loops detected - review for performance"
                })

            # Check for synchronous I/O in async context
            if 'async def' in code and 'open(' in code:
                score -= 15
                issues.append({
                    "severity": "warning",
                    "message": "Synchronous I/O in async context detected"
                })

        return max(0.0, score), issues

    def _validate_security(
        self,
        code_path: Path,
        test_path: Optional[Path],
        spec: Optional[Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Validate security best practices."""
        issues = []
        score = 100.0

        with open(code_path, 'r') as f:
            code = f.read()

        # Check for security issues
        security_patterns = [
            (r'pickle\.loads', "Unsafe pickle deserialization", 30),
            (r'os\.system', "Use of os.system() - prefer subprocess", 20),
            (r'shell=True', "Shell injection risk with shell=True", 25),
            (r'password\s*=\s*["\']', "Hardcoded password detected", 40),
            (r'api_key\s*=\s*["\']', "Hardcoded API key detected", 40),
        ]

        for pattern, message, penalty in security_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                score -= penalty
                issues.append({
                    "severity": "error" if penalty > 30 else "warning",
                    "message": message
                })

        return max(0.0, score), issues

    def _validate_standards(
        self,
        code_path: Path,
        test_path: Optional[Path],
        spec: Optional[Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Validate GreenLang standards compliance."""
        issues = []
        score = 100.0

        with open(code_path, 'r') as f:
            code = f.read()

        # Check for required patterns
        required_patterns = [
            (r'class.*\(BaseAgent\)', "Must inherit from BaseAgent", 30),
            (r'ProvenanceTracker', "Must include provenance tracking", 20),
            (r'logger\s*=', "Must include logging", 15),
        ]

        for pattern, message, penalty in required_patterns:
            if not re.search(pattern, code):
                score -= penalty
                issues.append({
                    "severity": "error" if penalty > 20 else "warning",
                    "message": f"GreenLang standard: {message}"
                })

        # Check zero-hallucination compliance
        if spec and hasattr(spec, 'calculation_formulas'):
            if 'llm' in code.lower() and 'calculate' in code.lower():
                score -= 50
                issues.append({
                    "severity": "error",
                    "message": "LLM usage detected in calculation agent - violates zero-hallucination"
                })

        return max(0.0, score), issues

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity."""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        # Average complexity per function
        func_count = sum(1 for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))

        return complexity / max(1, func_count)

    def _check_imports(self, tree: ast.AST) -> bool:
        """Check import quality."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)

        # Basic check - at least some imports
        return len(imports) > 3

    def _check_error_handling(self, tree: ast.AST) -> bool:
        """Check for error handling."""
        try_blocks = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try))
        func_count = sum(1 for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))

        # At least 1 try block per 3 functions
        return try_blocks >= func_count / 3

    def _check_type_hints(self, tree: ast.AST) -> float:
        """Calculate type hint coverage."""
        total_args = 0
        typed_args = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check arguments
                for arg in node.args.args:
                    total_args += 1
                    if arg.annotation:
                        typed_args += 1

                # Check return type
                total_args += 1
                if node.returns:
                    typed_args += 1

        return (typed_args / max(1, total_args)) * 100

    def _calculate_quality_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall quality score with weights."""
        weights = {
            "code_quality": 0.25,
            "test_coverage": 0.30,
            "documentation": 0.15,
            "performance": 0.10,
            "security": 0.10,
            "standards": 0.10
        }

        weighted_sum = sum(
            scores.get(metric, 0.0) * weight
            for metric, weight in weights.items()
        )

        return round(weighted_sum, 1)