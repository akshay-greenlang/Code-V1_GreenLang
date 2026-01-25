"""
ResponseValidation - Response Time Validation Framework (<1s)

This module implements comprehensive response time validation for Emergency
Shutdown Systems per IEC 61511-1 Clause 11.5 and Clause 16. ESD response
time is critical and typically required to be <1 second from initiation
to safe state achievement.

Key features:
- Response time measurement framework
- <1 second validation for safety functions
- Statistical response time analysis
- Periodic validation scheduling
- Response time degradation detection
- Compliance reporting with provenance

Reference: IEC 61511-1 Clause 11.5, Clause 16, ISA TR84.00.04

Example:
    >>> from greenlang.safety.esd.response_validation import ResponseTimeValidator
    >>> validator = ResponseTimeValidator(max_response_ms=1000)
    >>> result = validator.measure_response_time(measurement)
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import statistics
import uuid
import time

logger = logging.getLogger(__name__)


class ValidationResult(str, Enum):
    """Response time validation result."""

    PASS = "pass"  # Response time within requirement
    FAIL = "fail"  # Response time exceeds requirement
    MARGINAL = "marginal"  # Within 90-100% of limit
    DEGRADED = "degraded"  # Trending toward failure
    NOT_TESTED = "not_tested"  # Not yet tested


class ComponentType(str, Enum):
    """SIF component types for response time budget."""

    SENSOR = "sensor"
    INPUT_MODULE = "input_module"
    COMMUNICATION_IN = "communication_in"
    LOGIC_SOLVER = "logic_solver"
    COMMUNICATION_OUT = "communication_out"
    OUTPUT_MODULE = "output_module"
    ACTUATOR = "actuator"
    FINAL_ELEMENT = "final_element"


class ResponseMeasurement(BaseModel):
    """Individual response time measurement."""

    measurement_id: str = Field(
        default_factory=lambda: f"RTM-{uuid.uuid4().hex[:8].upper()}",
        description="Measurement identifier"
    )
    sif_id: str = Field(
        ...,
        description="SIF being tested"
    )
    equipment_id: str = Field(
        ...,
        description="Equipment tag"
    )
    test_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Measurement date"
    )
    tester: str = Field(
        default="",
        description="Person conducting measurement"
    )
    total_response_ms: float = Field(
        ...,
        ge=0,
        description="Total measured response time (ms)"
    )
    requirement_ms: float = Field(
        default=1000.0,
        gt=0,
        description="Required response time (ms)"
    )
    component_times: Dict[str, float] = Field(
        default_factory=dict,
        description="Response time by component"
    )
    process_safety_time_ms: float = Field(
        default=0,
        description="Process safety time (ms)"
    )
    test_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Test conditions (temp, pressure, etc.)"
    )
    notes: str = Field(
        default="",
        description="Test notes"
    )


class ValidationSchedule(BaseModel):
    """Response time validation schedule."""

    schedule_id: str = Field(
        default_factory=lambda: f"RTS-{uuid.uuid4().hex[:8].upper()}",
        description="Schedule identifier"
    )
    sif_id: str = Field(
        ...,
        description="SIF to validate"
    )
    validation_interval_days: int = Field(
        default=365,
        gt=0,
        description="Validation interval (days)"
    )
    last_validation_date: Optional[datetime] = Field(
        None,
        description="Last validation date"
    )
    next_validation_due: Optional[datetime] = Field(
        None,
        description="Next validation due date"
    )
    is_overdue: bool = Field(
        default=False,
        description="Is validation overdue"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Validation priority (1=highest)"
    )
    responsible_person: str = Field(
        default="",
        description="Person responsible for validation"
    )


class DegradationAnalysis(BaseModel):
    """Response time degradation analysis."""

    analysis_id: str = Field(
        default_factory=lambda: f"RDA-{uuid.uuid4().hex[:8].upper()}",
        description="Analysis identifier"
    )
    sif_id: str = Field(
        ...,
        description="SIF analyzed"
    )
    analysis_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Analysis date"
    )
    measurement_count: int = Field(
        default=0,
        description="Number of measurements analyzed"
    )
    trend: str = Field(
        default="stable",
        description="Trend (improving, stable, degrading)"
    )
    rate_of_change_ms_per_month: float = Field(
        default=0.0,
        description="Rate of change (ms/month)"
    )
    predicted_failure_date: Optional[datetime] = Field(
        None,
        description="Predicted failure date (if degrading)"
    )
    confidence_percent: float = Field(
        default=0.0,
        description="Confidence in prediction"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )
    component_analysis: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-component analysis"
    )


class ComplianceReport(BaseModel):
    """Response time compliance report."""

    report_id: str = Field(
        default_factory=lambda: f"RCR-{uuid.uuid4().hex[:8].upper()}",
        description="Report identifier"
    )
    report_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Report date"
    )
    reporting_period_start: datetime = Field(
        ...,
        description="Period start date"
    )
    reporting_period_end: datetime = Field(
        ...,
        description="Period end date"
    )
    total_sifs: int = Field(
        default=0,
        description="Total SIFs in scope"
    )
    compliant_count: int = Field(
        default=0,
        description="Number of compliant SIFs"
    )
    non_compliant_count: int = Field(
        default=0,
        description="Number of non-compliant SIFs"
    )
    marginal_count: int = Field(
        default=0,
        description="Number of marginal SIFs"
    )
    not_tested_count: int = Field(
        default=0,
        description="Number of untested SIFs"
    )
    compliance_rate_percent: float = Field(
        default=0.0,
        description="Overall compliance rate"
    )
    sif_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-SIF details"
    )
    overdue_validations: List[str] = Field(
        default_factory=list,
        description="SIFs with overdue validations"
    )
    degradation_warnings: List[str] = Field(
        default_factory=list,
        description="SIFs with degradation warnings"
    )
    certification_statement: str = Field(
        default="",
        description="Compliance certification statement"
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


class ResponseTimeValidator:
    """
    Response Time Validator for ESD/SIS Systems.

    Provides comprehensive response time validation per IEC 61511.
    Features:
    - Response time measurement framework
    - <1 second validation for safety functions
    - Statistical analysis of response times
    - Trend analysis and degradation detection
    - Periodic validation scheduling
    - Compliance reporting

    The validator follows IEC 61511 principles:
    - All measurements traceable
    - Clear pass/fail criteria
    - Complete audit trail

    Attributes:
        max_response_ms: Maximum allowed response time
        measurements: Historical measurements
        schedules: Validation schedules

    Example:
        >>> validator = ResponseTimeValidator(max_response_ms=1000)
        >>> measurement = ResponseMeasurement(sif_id="SIF-001", ...)
        >>> result = validator.validate_measurement(measurement)
    """

    # Standard response time budgets (percentage of total)
    STANDARD_BUDGET: Dict[ComponentType, float] = {
        ComponentType.SENSOR: 0.05,  # 5%
        ComponentType.INPUT_MODULE: 0.02,  # 2%
        ComponentType.COMMUNICATION_IN: 0.03,  # 3%
        ComponentType.LOGIC_SOLVER: 0.05,  # 5%
        ComponentType.COMMUNICATION_OUT: 0.03,  # 3%
        ComponentType.OUTPUT_MODULE: 0.02,  # 2%
        ComponentType.ACTUATOR: 0.60,  # 60%
        ComponentType.FINAL_ELEMENT: 0.20,  # 20% margin
    }

    # Marginal threshold (percentage of requirement)
    MARGINAL_THRESHOLD = 0.90

    def __init__(
        self,
        max_response_ms: float = 1000.0,
        marginal_threshold: float = 0.90,
        degradation_threshold_percent: float = 10.0
    ):
        """
        Initialize ResponseTimeValidator.

        Args:
            max_response_ms: Maximum allowed response time (ms)
            marginal_threshold: Threshold for marginal status (0.9 = 90%)
            degradation_threshold_percent: Threshold for degradation warning
        """
        self.max_response_ms = max_response_ms
        self.marginal_threshold = marginal_threshold
        self.degradation_threshold = degradation_threshold_percent

        self.measurements: Dict[str, List[ResponseMeasurement]] = {}
        self.schedules: Dict[str, ValidationSchedule] = {}
        self.validation_results: Dict[str, ValidationResult] = {}

        logger.info(
            f"ResponseTimeValidator initialized: max={max_response_ms}ms"
        )

    def measure_response_time(
        self,
        sif_id: str,
        equipment_id: str,
        trigger_callback: Callable[[], None],
        completion_callback: Callable[[], bool],
        tester: str = "",
        timeout_ms: float = 5000.0
    ) -> ResponseMeasurement:
        """
        Measure actual response time using callbacks.

        Args:
            sif_id: SIF being tested
            equipment_id: Equipment tag
            trigger_callback: Function to trigger the SIF
            completion_callback: Function to check if action completed
            tester: Person conducting test
            timeout_ms: Timeout in milliseconds

        Returns:
            ResponseMeasurement with results
        """
        logger.info(f"Measuring response time for {sif_id}")

        start_time = time.time()

        # Trigger the SIF
        trigger_callback()

        # Wait for completion
        elapsed_ms = 0.0
        while elapsed_ms < timeout_ms:
            if completion_callback():
                break
            time.sleep(0.001)  # 1ms poll
            elapsed_ms = (time.time() - start_time) * 1000

        total_response_ms = (time.time() - start_time) * 1000

        measurement = ResponseMeasurement(
            sif_id=sif_id,
            equipment_id=equipment_id,
            tester=tester,
            total_response_ms=total_response_ms,
            requirement_ms=self.max_response_ms,
        )

        # Store measurement
        if sif_id not in self.measurements:
            self.measurements[sif_id] = []
        self.measurements[sif_id].append(measurement)

        logger.info(
            f"Response time measured: {sif_id} = {total_response_ms:.1f}ms"
        )

        return measurement

    def validate_measurement(
        self,
        measurement: ResponseMeasurement
    ) -> Dict[str, Any]:
        """
        Validate a response time measurement.

        Args:
            measurement: ResponseMeasurement to validate

        Returns:
            Validation result dictionary
        """
        total_ms = measurement.total_response_ms
        requirement_ms = measurement.requirement_ms

        # Determine result
        if total_ms > requirement_ms:
            result = ValidationResult.FAIL
        elif total_ms >= requirement_ms * self.marginal_threshold:
            result = ValidationResult.MARGINAL
        else:
            result = ValidationResult.PASS

        self.validation_results[measurement.sif_id] = result

        # Calculate margin
        margin_ms = requirement_ms - total_ms
        margin_percent = (margin_ms / requirement_ms) * 100

        # Analyze component times
        component_analysis = {}
        for component, time_ms in measurement.component_times.items():
            budget_percent = self.STANDARD_BUDGET.get(
                ComponentType(component),
                0.05
            )
            budget_ms = requirement_ms * budget_percent
            within_budget = time_ms <= budget_ms

            component_analysis[component] = {
                "measured_ms": time_ms,
                "budget_ms": budget_ms,
                "within_budget": within_budget,
                "percent_of_total": (time_ms / total_ms) * 100 if total_ms > 0 else 0,
            }

        # Generate recommendations
        recommendations = []
        if result == ValidationResult.FAIL:
            recommendations.append(
                f"CRITICAL: Response time {total_ms:.0f}ms exceeds "
                f"requirement {requirement_ms:.0f}ms"
            )
            recommendations.append(
                "Immediate investigation required per IEC 61511"
            )

        if result == ValidationResult.MARGINAL:
            recommendations.append(
                "Response time is marginal. Consider component optimization."
            )

        # Check which components are over budget
        for component, analysis in component_analysis.items():
            if not analysis["within_budget"]:
                recommendations.append(
                    f"{component}: {analysis['measured_ms']:.0f}ms exceeds "
                    f"budget {analysis['budget_ms']:.0f}ms"
                )

        # Store measurement
        if measurement.sif_id not in self.measurements:
            self.measurements[measurement.sif_id] = []
        self.measurements[measurement.sif_id].append(measurement)

        # Update schedule if exists
        if measurement.sif_id in self.schedules:
            schedule = self.schedules[measurement.sif_id]
            schedule.last_validation_date = measurement.test_date
            schedule.next_validation_due = measurement.test_date + timedelta(
                days=schedule.validation_interval_days
            )
            schedule.is_overdue = False

        logger.info(
            f"Validation {measurement.measurement_id}: {result.value}, "
            f"margin={margin_ms:.0f}ms ({margin_percent:.1f}%)"
        )

        return {
            "measurement_id": measurement.measurement_id,
            "sif_id": measurement.sif_id,
            "result": result.value,
            "measured_ms": total_ms,
            "requirement_ms": requirement_ms,
            "margin_ms": margin_ms,
            "margin_percent": margin_percent,
            "meets_requirement": result in [ValidationResult.PASS, ValidationResult.MARGINAL],
            "component_analysis": component_analysis,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat(),
            "provenance_hash": hashlib.sha256(
                f"{measurement.measurement_id}|{result.value}|{total_ms}".encode()
            ).hexdigest()
        }

    def validate_component_budget(
        self,
        sif_id: str,
        requirement_ms: float,
        component_times: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Validate response time budget allocation by component.

        Args:
            sif_id: SIF identifier
            requirement_ms: Total response time requirement
            component_times: Measured times per component

        Returns:
            Budget validation result
        """
        total_measured = sum(component_times.values())
        margin_ms = requirement_ms - total_measured
        margin_percent = (margin_ms / requirement_ms) * 100

        component_details = {}
        concerns = []

        for component_name, measured_ms in component_times.items():
            try:
                component_type = ComponentType(component_name)
                budget_percent = self.STANDARD_BUDGET.get(component_type, 0.05)
            except ValueError:
                budget_percent = 0.05

            budget_ms = requirement_ms * budget_percent
            within_budget = measured_ms <= budget_ms
            utilization = (measured_ms / budget_ms) * 100 if budget_ms > 0 else 100

            component_details[component_name] = {
                "measured_ms": measured_ms,
                "budget_ms": budget_ms,
                "within_budget": within_budget,
                "utilization_percent": utilization,
            }

            if not within_budget:
                concerns.append(
                    f"{component_name}: {measured_ms:.0f}ms exceeds "
                    f"budget {budget_ms:.0f}ms ({utilization:.0f}% utilized)"
                )

        is_valid = margin_ms >= 0
        has_adequate_margin = margin_percent >= 20  # IEC 61511 recommends 20%

        return {
            "sif_id": sif_id,
            "requirement_ms": requirement_ms,
            "total_measured_ms": total_measured,
            "margin_ms": margin_ms,
            "margin_percent": margin_percent,
            "is_valid": is_valid,
            "has_adequate_margin": has_adequate_margin,
            "component_details": component_details,
            "concerns": concerns,
            "recommendation": (
                "Budget allocation acceptable"
                if is_valid and has_adequate_margin
                else "Budget optimization required"
            )
        }

    def analyze_degradation(
        self,
        sif_id: str,
        min_measurements: int = 5
    ) -> DegradationAnalysis:
        """
        Analyze response time degradation trends.

        Args:
            sif_id: SIF to analyze
            min_measurements: Minimum measurements for analysis

        Returns:
            DegradationAnalysis with trend information
        """
        measurements = self.measurements.get(sif_id, [])

        if len(measurements) < min_measurements:
            return DegradationAnalysis(
                sif_id=sif_id,
                measurement_count=len(measurements),
                recommendations=[
                    f"Insufficient data: {len(measurements)} measurements "
                    f"(need {min_measurements})"
                ]
            )

        # Sort by date
        measurements = sorted(measurements, key=lambda m: m.test_date)

        # Extract response times
        times_ms = [m.total_response_ms for m in measurements]

        # Calculate statistics
        mean_time = statistics.mean(times_ms)
        stdev_time = statistics.stdev(times_ms) if len(times_ms) > 1 else 0

        # Calculate trend (linear regression approximation)
        first_half = times_ms[:len(times_ms)//2]
        second_half = times_ms[len(times_ms)//2:]

        first_mean = statistics.mean(first_half)
        second_mean = statistics.mean(second_half)

        # Calculate rate of change
        time_span_months = (
            measurements[-1].test_date - measurements[0].test_date
        ).days / 30.0

        if time_span_months > 0:
            rate_per_month = (second_mean - first_mean) / (time_span_months / 2)
        else:
            rate_per_month = 0

        # Determine trend
        if rate_per_month > (self.max_response_ms * self.degradation_threshold / 100):
            trend = "degrading"
        elif rate_per_month < -(self.max_response_ms * self.degradation_threshold / 100):
            trend = "improving"
        else:
            trend = "stable"

        # Predict failure if degrading
        predicted_failure_date = None
        confidence = 0.0

        if trend == "degrading" and rate_per_month > 0:
            current_time = times_ms[-1]
            remaining_margin = self.max_response_ms - current_time
            if remaining_margin > 0:
                months_to_failure = remaining_margin / rate_per_month
                predicted_failure_date = datetime.utcnow() + timedelta(
                    days=months_to_failure * 30
                )
                # Confidence based on R-squared (simplified)
                confidence = min(90.0, 50.0 + len(measurements) * 2)

        # Analyze by component
        component_analysis = {}
        if measurements[-1].component_times:
            for component, time_ms in measurements[-1].component_times.items():
                historical = [
                    m.component_times.get(component, 0)
                    for m in measurements
                    if component in m.component_times
                ]

                if len(historical) >= 2:
                    component_analysis[component] = {
                        "latest_ms": time_ms,
                        "mean_ms": statistics.mean(historical),
                        "trend": (
                            "degrading"
                            if historical[-1] > historical[0] * 1.1
                            else "stable"
                        )
                    }

        # Generate recommendations
        recommendations = []
        if trend == "degrading":
            recommendations.append(
                f"Response time degradation detected: "
                f"+{rate_per_month:.1f}ms/month"
            )
            if predicted_failure_date:
                recommendations.append(
                    f"Predicted to exceed requirement by: "
                    f"{predicted_failure_date.strftime('%Y-%m-%d')}"
                )
            recommendations.append(
                "Investigate actuator/valve performance"
            )

        if stdev_time > mean_time * 0.1:
            recommendations.append(
                f"High variability detected: stdev={stdev_time:.0f}ms"
            )

        analysis = DegradationAnalysis(
            sif_id=sif_id,
            measurement_count=len(measurements),
            trend=trend,
            rate_of_change_ms_per_month=rate_per_month,
            predicted_failure_date=predicted_failure_date,
            confidence_percent=confidence,
            recommendations=recommendations,
            component_analysis=component_analysis,
        )

        return analysis

    def create_validation_schedule(
        self,
        sif_id: str,
        interval_days: int = 365,
        priority: int = 1,
        responsible_person: str = ""
    ) -> ValidationSchedule:
        """
        Create a validation schedule for a SIF.

        Args:
            sif_id: SIF identifier
            interval_days: Validation interval in days
            priority: Priority (1=highest)
            responsible_person: Person responsible

        Returns:
            ValidationSchedule
        """
        schedule = ValidationSchedule(
            sif_id=sif_id,
            validation_interval_days=interval_days,
            priority=priority,
            responsible_person=responsible_person,
            next_validation_due=datetime.utcnow() + timedelta(days=interval_days),
        )

        self.schedules[sif_id] = schedule

        logger.info(
            f"Validation schedule created for {sif_id}: "
            f"every {interval_days} days"
        )

        return schedule

    def check_overdue_validations(self) -> List[ValidationSchedule]:
        """
        Check for overdue validation schedules.

        Returns:
            List of overdue schedules
        """
        now = datetime.utcnow()
        overdue = []

        for schedule in self.schedules.values():
            if schedule.next_validation_due and now > schedule.next_validation_due:
                schedule.is_overdue = True
                overdue.append(schedule)
                logger.warning(
                    f"Validation overdue for {schedule.sif_id}: "
                    f"due {schedule.next_validation_due.isoformat()}"
                )

        return overdue

    def generate_compliance_report(
        self,
        period_start: datetime,
        period_end: datetime,
        sif_ids: Optional[List[str]] = None
    ) -> ComplianceReport:
        """
        Generate a compliance report for response time validation.

        Args:
            period_start: Reporting period start
            period_end: Reporting period end
            sif_ids: Specific SIFs to include (all if None)

        Returns:
            ComplianceReport
        """
        # Determine SIFs in scope
        if sif_ids:
            scope_sifs = sif_ids
        else:
            scope_sifs = list(set(
                list(self.measurements.keys()) +
                list(self.schedules.keys())
            ))

        compliant = 0
        non_compliant = 0
        marginal = 0
        not_tested = 0
        sif_details = []
        overdue_validations = []
        degradation_warnings = []

        for sif_id in scope_sifs:
            # Get measurements in period
            measurements = [
                m for m in self.measurements.get(sif_id, [])
                if period_start <= m.test_date <= period_end
            ]

            if not measurements:
                not_tested += 1
                status = "not_tested"
                latest_ms = None
            else:
                latest = measurements[-1]
                latest_ms = latest.total_response_ms
                requirement = latest.requirement_ms

                if latest_ms > requirement:
                    non_compliant += 1
                    status = "non_compliant"
                elif latest_ms >= requirement * self.marginal_threshold:
                    marginal += 1
                    status = "marginal"
                else:
                    compliant += 1
                    status = "compliant"

            # Check schedule
            schedule = self.schedules.get(sif_id)
            if schedule and schedule.is_overdue:
                overdue_validations.append(sif_id)

            # Check degradation
            if len(self.measurements.get(sif_id, [])) >= 5:
                analysis = self.analyze_degradation(sif_id)
                if analysis.trend == "degrading":
                    degradation_warnings.append(sif_id)

            sif_details.append({
                "sif_id": sif_id,
                "status": status,
                "latest_response_ms": latest_ms,
                "measurement_count": len(measurements),
                "is_overdue": sif_id in overdue_validations,
                "is_degrading": sif_id in degradation_warnings,
            })

        # Calculate compliance rate
        tested = compliant + non_compliant + marginal
        compliance_rate = (compliant / tested * 100) if tested > 0 else 0

        # Generate certification statement
        if compliance_rate == 100 and not overdue_validations:
            certification = (
                f"All SIFs ({tested}) meet IEC 61511 response time requirements. "
                f"No validations are overdue."
            )
        elif compliance_rate >= 90:
            certification = (
                f"Response time compliance rate: {compliance_rate:.1f}%. "
                f"Non-compliant SIFs require immediate attention."
            )
        else:
            certification = (
                f"Response time compliance rate: {compliance_rate:.1f}%. "
                f"Significant remediation required per IEC 61511."
            )

        report = ComplianceReport(
            reporting_period_start=period_start,
            reporting_period_end=period_end,
            total_sifs=len(scope_sifs),
            compliant_count=compliant,
            non_compliant_count=non_compliant,
            marginal_count=marginal,
            not_tested_count=not_tested,
            compliance_rate_percent=compliance_rate,
            sif_details=sif_details,
            overdue_validations=overdue_validations,
            degradation_warnings=degradation_warnings,
            certification_statement=certification,
        )

        report.provenance_hash = hashlib.sha256(
            f"{report.report_id}|{report.total_sifs}|{compliance_rate}|{period_end.isoformat()}".encode()
        ).hexdigest()

        logger.info(
            f"Compliance report generated: {compliance_rate:.1f}% compliant"
        )

        return report

    def get_sif_statistics(
        self,
        sif_id: str
    ) -> Dict[str, Any]:
        """
        Get response time statistics for a SIF.

        Args:
            sif_id: SIF identifier

        Returns:
            Statistics dictionary
        """
        measurements = self.measurements.get(sif_id, [])

        if not measurements:
            return {"error": f"No measurements for {sif_id}"}

        times_ms = [m.total_response_ms for m in measurements]

        return {
            "sif_id": sif_id,
            "measurement_count": len(measurements),
            "latest_date": measurements[-1].test_date.isoformat(),
            "latest_response_ms": times_ms[-1],
            "requirement_ms": measurements[-1].requirement_ms,
            "statistics": {
                "mean_ms": round(statistics.mean(times_ms), 1),
                "stdev_ms": round(statistics.stdev(times_ms), 1) if len(times_ms) > 1 else 0,
                "min_ms": round(min(times_ms), 1),
                "max_ms": round(max(times_ms), 1),
                "median_ms": round(statistics.median(times_ms), 1),
            },
            "current_status": self.validation_results.get(sif_id, ValidationResult.NOT_TESTED).value,
            "schedule": {
                "next_due": self.schedules[sif_id].next_validation_due.isoformat()
                if sif_id in self.schedules and self.schedules[sif_id].next_validation_due
                else None,
                "is_overdue": self.schedules[sif_id].is_overdue
                if sif_id in self.schedules
                else False,
            } if sif_id in self.schedules else None,
        }

    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Get overall response time validation statistics.

        Returns:
            Overall statistics dictionary
        """
        total_measurements = sum(
            len(m) for m in self.measurements.values()
        )

        status_counts = {
            status.value: 0 for status in ValidationResult
        }
        for status in self.validation_results.values():
            status_counts[status.value] += 1

        overdue = self.check_overdue_validations()

        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "max_response_requirement_ms": self.max_response_ms,
            "total_sifs": len(self.measurements),
            "total_measurements": total_measurements,
            "status_counts": status_counts,
            "overdue_count": len(overdue),
            "schedules_count": len(self.schedules),
        }
