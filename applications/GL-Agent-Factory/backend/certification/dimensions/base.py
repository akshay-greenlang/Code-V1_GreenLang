"""
Base Dimension Evaluator

This module provides the base class and common types for all certification
dimension evaluators.

Example:
    >>> class MyDimension(BaseDimension):
    ...     DIMENSION_ID = "D99"
    ...     DIMENSION_NAME = "My Dimension"
    ...     def evaluate(self, agent_path, agent, config):
    ...         return DimensionResult(...)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DimensionStatus(str, Enum):
    """Status of a dimension evaluation."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class CheckResult:
    """Result of a single check within a dimension."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class DimensionResult:
    """Result of evaluating a single certification dimension."""

    dimension_id: str
    dimension_name: str
    status: DimensionStatus
    score: float  # 0.0 to 100.0
    checks_passed: int
    checks_failed: int
    checks_total: int
    execution_time_ms: float
    check_results: List[CheckResult] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    remediation: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def passed(self) -> bool:
        """Check if dimension passed."""
        return self.status == DimensionStatus.PASS

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        if self.checks_total == 0:
            return 100.0
        return (self.checks_passed / self.checks_total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dimension_id": self.dimension_id,
            "dimension_name": self.dimension_name,
            "status": self.status.value,
            "score": self.score,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_total": self.checks_total,
            "execution_time_ms": self.execution_time_ms,
            "check_results": [
                {
                    "name": cr.name,
                    "passed": cr.passed,
                    "message": cr.message,
                    "severity": cr.severity,
                    "details": cr.details,
                }
                for cr in self.check_results
            ],
            "details": self.details,
            "remediation": self.remediation,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseDimension(ABC):
    """
    Base class for certification dimension evaluators.

    All dimension evaluators must inherit from this class and implement
    the evaluate() method.

    Attributes:
        DIMENSION_ID: Short identifier (e.g., "D01")
        DIMENSION_NAME: Human-readable name
        DESCRIPTION: Detailed description
        WEIGHT: Weight in overall score calculation (default 1.0)
        REQUIRED_FOR_CERTIFICATION: If True, must pass for certification
    """

    DIMENSION_ID: str = "D00"
    DIMENSION_NAME: str = "Base Dimension"
    DESCRIPTION: str = "Base dimension evaluator"
    WEIGHT: float = 1.0
    REQUIRED_FOR_CERTIFICATION: bool = True

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dimension evaluator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._checks: List[CheckResult] = []

        logger.debug(f"Initialized {self.DIMENSION_NAME} evaluator")

    @abstractmethod
    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate the dimension for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Optional agent instance
            sample_input: Optional sample input for testing

        Returns:
            DimensionResult with evaluation results
        """
        pass

    def get_remediation_suggestions(
        self,
        result: DimensionResult,
    ) -> List[str]:
        """
        Get remediation suggestions for failed checks.

        Args:
            result: Dimension result to analyze

        Returns:
            List of remediation suggestions
        """
        suggestions = []

        for check in result.check_results:
            if not check.passed:
                suggestion = self._get_check_remediation(check)
                if suggestion:
                    suggestions.append(suggestion)

        return suggestions

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """
        Get remediation for a specific check.

        Override in subclasses for specific remediation advice.

        Args:
            check: Failed check result

        Returns:
            Remediation suggestion or None
        """
        return None

    def _add_check(
        self,
        name: str,
        passed: bool,
        message: str,
        severity: str = "error",
        details: Optional[Dict[str, Any]] = None,
    ) -> CheckResult:
        """
        Add a check result.

        Args:
            name: Check name
            passed: Whether check passed
            message: Result message
            severity: Severity level
            details: Additional details

        Returns:
            The created CheckResult
        """
        check = CheckResult(
            name=name,
            passed=passed,
            message=message,
            severity=severity,
            details=details or {},
        )
        self._checks.append(check)
        return check

    def _reset_checks(self) -> None:
        """Reset check results for new evaluation."""
        self._checks = []

    def _calculate_score(self) -> float:
        """
        Calculate dimension score from checks.

        Returns:
            Score from 0.0 to 100.0
        """
        if not self._checks:
            return 100.0

        # Weight checks by severity
        weights = {"error": 1.0, "warning": 0.5, "info": 0.1}

        total_weight = sum(weights.get(c.severity, 1.0) for c in self._checks)
        passed_weight = sum(
            weights.get(c.severity, 1.0)
            for c in self._checks
            if c.passed
        )

        if total_weight == 0:
            return 100.0

        return (passed_weight / total_weight) * 100

    def _determine_status(self) -> DimensionStatus:
        """
        Determine dimension status from checks.

        Returns:
            Dimension status
        """
        if not self._checks:
            return DimensionStatus.PASS

        errors = [c for c in self._checks if not c.passed and c.severity == "error"]
        warnings = [c for c in self._checks if not c.passed and c.severity == "warning"]

        if errors:
            return DimensionStatus.FAIL
        elif warnings:
            return DimensionStatus.WARNING
        else:
            return DimensionStatus.PASS

    def _create_result(
        self,
        execution_time_ms: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> DimensionResult:
        """
        Create dimension result from accumulated checks.

        Args:
            execution_time_ms: Evaluation execution time
            details: Additional result details

        Returns:
            Complete dimension result
        """
        status = self._determine_status()
        score = self._calculate_score()

        checks_passed = sum(1 for c in self._checks if c.passed)
        checks_failed = len(self._checks) - checks_passed

        result = DimensionResult(
            dimension_id=self.DIMENSION_ID,
            dimension_name=self.DIMENSION_NAME,
            status=status,
            score=score,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            checks_total=len(self._checks),
            execution_time_ms=execution_time_ms,
            check_results=self._checks.copy(),
            details=details or {},
        )

        # Add remediation suggestions
        result.remediation = self.get_remediation_suggestions(result)

        return result
