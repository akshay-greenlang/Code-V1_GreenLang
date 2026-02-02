"""
ResponseValidator - <1s Response Time Validation

This module implements response time validation for Emergency Shutdown
Systems per IEC 61511. ESD response time is critical - typically
required to be <1 second from initiation to safe state.

Key requirements:
- Total response time < Process Safety Time
- Typical ESD response < 1 second
- All components must contribute to budget
- Periodic validation testing required

Reference: IEC 61511-1 Clause 11.5, Clause 16

Example:
    >>> from greenlang.safety.esd.response_validator import ResponseValidator
    >>> validator = ResponseValidator(max_response_ms=1000)
    >>> result = validator.validate_response(test_data)
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime
import statistics
import uuid

logger = logging.getLogger(__name__)


class TestType(str, Enum):
    """Types of response time tests."""

    FULL_FUNCTION = "full_function"  # Complete SIF test
    SENSOR_ONLY = "sensor_only"  # Sensor response
    LOGIC_ONLY = "logic_only"  # Logic solver response
    ACTUATOR_ONLY = "actuator_only"  # Final element response
    COMMUNICATION = "communication"  # Communication latency
    END_TO_END = "end_to_end"  # Full end-to-end test


class ValidationStatus(str, Enum):
    """Validation result status."""

    PASS = "pass"
    FAIL = "fail"
    MARGINAL = "marginal"  # Within 90-100% of limit
    NOT_TESTED = "not_tested"


class ResponseTest(BaseModel):
    """Individual response time test record."""

    test_id: str = Field(
        default_factory=lambda: f"RT-{uuid.uuid4().hex[:8].upper()}",
        description="Test identifier"
    )
    test_type: TestType = Field(
        ...,
        description="Type of test"
    )
    equipment_id: str = Field(
        ...,
        description="Equipment tested"
    )
    sif_id: Optional[str] = Field(
        None,
        description="SIF identifier"
    )
    test_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Test date"
    )
    response_time_ms: float = Field(
        ...,
        ge=0,
        description="Measured response time (ms)"
    )
    requirement_ms: float = Field(
        ...,
        gt=0,
        description="Required response time (ms)"
    )
    sensor_time_ms: Optional[float] = Field(
        None,
        description="Sensor component time (ms)"
    )
    logic_time_ms: Optional[float] = Field(
        None,
        description="Logic solver time (ms)"
    )
    actuator_time_ms: Optional[float] = Field(
        None,
        description="Actuator time (ms)"
    )
    communication_time_ms: Optional[float] = Field(
        None,
        description="Communication time (ms)"
    )
    tester: str = Field(
        default="",
        description="Person conducting test"
    )
    notes: str = Field(
        default="",
        description="Test notes"
    )
    environmental_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environmental conditions during test"
    )


class ResponseResult(BaseModel):
    """Response time validation result."""

    validation_id: str = Field(
        default_factory=lambda: f"VAL-{uuid.uuid4().hex[:8].upper()}",
        description="Validation identifier"
    )
    test_id: str = Field(
        ...,
        description="Associated test ID"
    )
    status: ValidationStatus = Field(
        ...,
        description="Validation status"
    )
    measured_ms: float = Field(
        ...,
        description="Measured response time (ms)"
    )
    requirement_ms: float = Field(
        ...,
        description="Required response time (ms)"
    )
    margin_ms: float = Field(
        ...,
        description="Margin (requirement - measured)"
    )
    margin_percent: float = Field(
        ...,
        description="Margin as percentage"
    )
    component_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Time breakdown by component"
    )
    meets_requirement: bool = Field(
        ...,
        description="Does response meet requirement"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Improvement recommendations"
    )
    validation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Validation timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ResponseValidator:
    """
    Response Time Validator for ESD Systems.

    Validates that ESD response times meet requirements per IEC 61511.
    Provides:
    - Single test validation
    - Trend analysis
    - Component budget analysis
    - Improvement recommendations

    The validator follows zero-hallucination principles:
    - All calculations deterministic
    - Clear pass/fail criteria
    - Complete audit trail

    Attributes:
        max_response_ms: Maximum allowed response time
        test_history: Historical test records

    Example:
        >>> validator = ResponseValidator(max_response_ms=1000)
        >>> test = ResponseTest(test_type=TestType.FULL_FUNCTION, ...)
        >>> result = validator.validate_response(test)
    """

    # Typical component budget allocation (percentage)
    TYPICAL_BUDGET_ALLOCATION = {
        "sensor": 0.10,  # 10%
        "communication_in": 0.05,  # 5%
        "logic": 0.10,  # 10%
        "communication_out": 0.05,  # 5%
        "actuator": 0.50,  # 50%
        "margin": 0.20,  # 20%
    }

    def __init__(
        self,
        max_response_ms: float = 1000.0,
        marginal_threshold: float = 0.9
    ):
        """
        Initialize ResponseValidator.

        Args:
            max_response_ms: Maximum allowed response time (ms)
            marginal_threshold: Threshold for marginal status (0.9 = 90%)
        """
        self.max_response_ms = max_response_ms
        self.marginal_threshold = marginal_threshold
        self.test_history: List[ResponseTest] = []
        self.validation_history: List[ResponseResult] = []

        logger.info(
            f"ResponseValidator initialized: max={max_response_ms}ms"
        )

    def validate_response(
        self,
        test: ResponseTest
    ) -> ResponseResult:
        """
        Validate a response time test.

        Args:
            test: ResponseTest to validate

        Returns:
            ResponseResult with validation outcome
        """
        logger.info(
            f"Validating response time for {test.equipment_id}: "
            f"{test.response_time_ms}ms vs {test.requirement_ms}ms"
        )

        # Store test
        self.test_history.append(test)

        # Calculate margin
        margin_ms = test.requirement_ms - test.response_time_ms
        margin_percent = (margin_ms / test.requirement_ms) * 100

        # Determine status
        meets_requirement = test.response_time_ms <= test.requirement_ms

        if meets_requirement:
            if test.response_time_ms >= test.requirement_ms * self.marginal_threshold:
                status = ValidationStatus.MARGINAL
            else:
                status = ValidationStatus.PASS
        else:
            status = ValidationStatus.FAIL

        # Build component breakdown
        breakdown = {}
        if test.sensor_time_ms is not None:
            breakdown["sensor_ms"] = test.sensor_time_ms
        if test.logic_time_ms is not None:
            breakdown["logic_ms"] = test.logic_time_ms
        if test.actuator_time_ms is not None:
            breakdown["actuator_ms"] = test.actuator_time_ms
        if test.communication_time_ms is not None:
            breakdown["communication_ms"] = test.communication_time_ms

        # Generate recommendations
        recommendations = self._generate_recommendations(
            test, status, breakdown
        )

        # Build result
        result = ResponseResult(
            test_id=test.test_id,
            status=status,
            measured_ms=test.response_time_ms,
            requirement_ms=test.requirement_ms,
            margin_ms=margin_ms,
            margin_percent=margin_percent,
            component_breakdown=breakdown,
            meets_requirement=meets_requirement,
            recommendations=recommendations,
        )

        # Calculate provenance
        result.provenance_hash = self._calculate_provenance(result)

        # Store result
        self.validation_history.append(result)

        logger.info(
            f"Validation {result.validation_id}: {status.value}, "
            f"margin={margin_ms:.0f}ms ({margin_percent:.1f}%)"
        )

        return result

    def validate_budget(
        self,
        total_requirement_ms: float,
        sensor_ms: float,
        logic_ms: float,
        actuator_ms: float,
        communication_ms: float = 0
    ) -> Dict[str, Any]:
        """
        Validate component time budget allocation.

        Args:
            total_requirement_ms: Total time requirement
            sensor_ms: Sensor component time
            logic_ms: Logic solver time
            actuator_ms: Actuator time
            communication_ms: Communication time

        Returns:
            Budget validation result
        """
        total_used = sensor_ms + logic_ms + actuator_ms + communication_ms
        margin_ms = total_requirement_ms - total_used
        margin_percent = (margin_ms / total_requirement_ms) * 100

        # Calculate percentages
        components = {
            "sensor": {
                "time_ms": sensor_ms,
                "percent": (sensor_ms / total_requirement_ms) * 100,
                "typical_percent": self.TYPICAL_BUDGET_ALLOCATION["sensor"] * 100,
            },
            "logic": {
                "time_ms": logic_ms,
                "percent": (logic_ms / total_requirement_ms) * 100,
                "typical_percent": self.TYPICAL_BUDGET_ALLOCATION["logic"] * 100,
            },
            "actuator": {
                "time_ms": actuator_ms,
                "percent": (actuator_ms / total_requirement_ms) * 100,
                "typical_percent": self.TYPICAL_BUDGET_ALLOCATION["actuator"] * 100,
            },
            "communication": {
                "time_ms": communication_ms,
                "percent": (communication_ms / total_requirement_ms) * 100,
                "typical_percent": (
                    self.TYPICAL_BUDGET_ALLOCATION["communication_in"] +
                    self.TYPICAL_BUDGET_ALLOCATION["communication_out"]
                ) * 100,
            },
        }

        # Identify components exceeding typical allocation
        concerns = []
        for name, data in components.items():
            if data["percent"] > data["typical_percent"] * 1.5:
                concerns.append(
                    f"{name} using {data['percent']:.1f}% "
                    f"(typical: {data['typical_percent']:.1f}%)"
                )

        return {
            "total_requirement_ms": total_requirement_ms,
            "total_used_ms": total_used,
            "margin_ms": margin_ms,
            "margin_percent": margin_percent,
            "is_valid": margin_ms >= 0,
            "has_adequate_margin": margin_percent >= 20,
            "components": components,
            "concerns": concerns,
        }

    def analyze_trends(
        self,
        equipment_id: str,
        lookback_count: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze response time trends for equipment.

        Args:
            equipment_id: Equipment to analyze
            lookback_count: Number of tests to analyze

        Returns:
            Trend analysis results
        """
        # Filter tests for equipment
        tests = [
            t for t in self.test_history
            if t.equipment_id == equipment_id
        ][-lookback_count:]

        if not tests:
            return {"error": f"No tests found for {equipment_id}"}

        if len(tests) < 3:
            return {
                "equipment_id": equipment_id,
                "test_count": len(tests),
                "warning": "Insufficient data for trend analysis"
            }

        times = [t.response_time_ms for t in tests]

        # Calculate statistics
        mean_time = statistics.mean(times)
        stdev_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)

        # Calculate trend (simple linear)
        trend = "stable"
        if len(times) >= 3:
            first_half = statistics.mean(times[:len(times)//2])
            second_half = statistics.mean(times[len(times)//2:])
            change = (second_half - first_half) / first_half * 100

            if change > 10:
                trend = "increasing"  # Degradation
            elif change < -10:
                trend = "decreasing"  # Improvement
            else:
                trend = "stable"

        # Predict margin erosion
        latest_margin = tests[-1].requirement_ms - tests[-1].response_time_ms
        margin_concern = latest_margin < (tests[-1].requirement_ms * 0.2)

        return {
            "equipment_id": equipment_id,
            "test_count": len(tests),
            "latest_test": tests[-1].test_date.isoformat(),
            "statistics": {
                "mean_ms": round(mean_time, 1),
                "stdev_ms": round(stdev_time, 1),
                "min_ms": round(min_time, 1),
                "max_ms": round(max_time, 1),
            },
            "trend": trend,
            "margin_concern": margin_concern,
            "latest_margin_ms": round(latest_margin, 1),
            "recommendation": (
                "Monitor closely - margin degradation detected"
                if trend == "increasing" or margin_concern
                else "Response time within acceptable range"
            ),
        }

    def get_compliance_report(
        self,
        equipment_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report for response time validation.

        Args:
            equipment_ids: Specific equipment (all if None)

        Returns:
            Compliance report
        """
        # Filter validations
        if equipment_ids:
            tests = [
                t for t in self.test_history
                if t.equipment_id in equipment_ids
            ]
        else:
            tests = self.test_history

        if not tests:
            return {"error": "No test data available"}

        # Aggregate results
        pass_count = sum(
            1 for t in tests
            if t.response_time_ms <= t.requirement_ms
        )
        fail_count = len(tests) - pass_count

        # Equipment summary
        equipment_summary = {}
        for test in tests:
            eq = test.equipment_id
            if eq not in equipment_summary:
                equipment_summary[eq] = {
                    "test_count": 0,
                    "pass_count": 0,
                    "latest_time_ms": 0,
                    "requirement_ms": test.requirement_ms,
                }

            equipment_summary[eq]["test_count"] += 1
            if test.response_time_ms <= test.requirement_ms:
                equipment_summary[eq]["pass_count"] += 1
            equipment_summary[eq]["latest_time_ms"] = test.response_time_ms

        # Calculate compliance rate
        compliance_rate = (pass_count / len(tests)) * 100 if tests else 0

        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "total_tests": len(tests),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "compliance_rate_percent": round(compliance_rate, 1),
            "equipment_count": len(equipment_summary),
            "equipment_summary": equipment_summary,
            "max_response_requirement_ms": self.max_response_ms,
            "provenance_hash": hashlib.sha256(
                f"{datetime.utcnow().isoformat()}|{len(tests)}|{compliance_rate}".encode()
            ).hexdigest()
        }

    def _generate_recommendations(
        self,
        test: ResponseTest,
        status: ValidationStatus,
        breakdown: Dict[str, float]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        if status == ValidationStatus.FAIL:
            recommendations.append(
                f"CRITICAL: Response time {test.response_time_ms}ms exceeds "
                f"requirement {test.requirement_ms}ms"
            )

        if status == ValidationStatus.MARGINAL:
            recommendations.append(
                "Response time is marginal. Monitor for degradation."
            )

        # Analyze component breakdown
        if breakdown:
            if breakdown.get("actuator_ms", 0) > test.requirement_ms * 0.6:
                recommendations.append(
                    "Actuator time is dominant. Consider faster actuator."
                )

            if breakdown.get("communication_ms", 0) > test.requirement_ms * 0.1:
                recommendations.append(
                    "Communication time is high. Review network architecture."
                )

        return recommendations

    def _calculate_provenance(self, result: ResponseResult) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{result.validation_id}|"
            f"{result.test_id}|"
            f"{result.status.value}|"
            f"{result.measured_ms}|"
            f"{result.validation_timestamp.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
