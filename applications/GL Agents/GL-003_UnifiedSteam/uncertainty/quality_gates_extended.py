"""
Extended Quality Gates for GL-003 UNIFIEDSTEAM SteamSystemOptimizer.

This module provides additional quality gate classes including:
- QualityGate - Generic configurable gate
- UncertaintyQualityChecker - Application-specific uncertainty checking
- DataQualityValidator - Input data validation
- ComplianceChecker - Regulatory compliance checking
- Report generation functions

Zero-Hallucination Guarantee:
- All gate decisions are based on deterministic threshold comparisons
- No LLM inference in gating logic
- Complete audit trail for all gate decisions
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import logging

from .uncertainty_models import (
    UncertainValue,
    PropagatedUncertainty,
    ConfidenceLevel
)
from .quality_gates import (
    GateStatus,
    GateResult,
    RiskLevel,
    WarningPriority
)


logger = logging.getLogger(__name__)


# =============================================================================
# QUALITY GATE CLASS - Generic Configurable Gate
# =============================================================================

@dataclass
class QualityGateConfig:
    """
    Configuration for a quality gate.

    Attributes:
        gate_id: Unique gate identifier
        gate_name: Human-readable gate name
        pass_threshold: Value below which check passes
        warning_threshold: Value below which generates warning
        fail_threshold: Value above which check fails
        comparison_type: How to compare ("less_than", "greater_than", "in_range")
        unit: Unit for display
        description: Gate description
    """
    gate_id: str
    gate_name: str
    pass_threshold: float
    warning_threshold: float
    fail_threshold: float
    comparison_type: str = "less_than"
    unit: str = "%"
    description: str = ""


@dataclass
class QualityGateResult:
    """
    Result of a quality gate evaluation.

    Attributes:
        gate_id: Gate identifier
        status: Pass/warning/fail status
        measured_value: The measured/computed value
        pass_threshold: Threshold for passing
        fail_threshold: Threshold for failing
        message: Result message
        timestamp: When check was performed
        provenance_hash: SHA-256 hash for audit trail
    """
    gate_id: str
    status: GateStatus
    measured_value: float
    pass_threshold: float
    fail_threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash."""
        if not self.provenance_hash:
            hash_data = {
                "gate_id": self.gate_id,
                "status": self.status.value,
                "measured_value": self.measured_value,
                "timestamp": self.timestamp.isoformat()
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


class QualityGate:
    """
    Generic configurable quality gate with pass/warning/fail thresholds.

    Provides flexible threshold-based checking with configurable
    comparison modes and detailed result reporting.

    Example:
        gate = QualityGate(QualityGateConfig(
            gate_id="uncertainty_gate",
            gate_name="Uncertainty Quality Gate",
            pass_threshold=5.0,
            warning_threshold=10.0,
            fail_threshold=20.0,
            comparison_type="less_than"
        ))

        result = gate.evaluate(uncertainty_percent)
        if result.status == GateStatus.PASSED:
            proceed_with_action()
    """

    def __init__(self, config: QualityGateConfig):
        """
        Initialize quality gate.

        Args:
            config: Gate configuration
        """
        self.config = config
        self._evaluation_history: List[QualityGateResult] = []

    def evaluate(self, value: float) -> QualityGateResult:
        """
        Evaluate a value against the gate thresholds.

        Args:
            value: Value to evaluate

        Returns:
            QualityGateResult with evaluation outcome
        """
        if self.config.comparison_type == "less_than":
            status = self._evaluate_less_than(value)
        elif self.config.comparison_type == "greater_than":
            status = self._evaluate_greater_than(value)
        elif self.config.comparison_type == "in_range":
            status = self._evaluate_in_range(value)
        else:
            status = self._evaluate_less_than(value)

        # Generate message
        if status == GateStatus.PASSED:
            message = (
                f"{self.config.gate_name}: PASSED - "
                f"Value {value:.2f}{self.config.unit} within acceptable limit "
                f"({self.config.pass_threshold:.2f}{self.config.unit})"
            )
        elif status == GateStatus.WARNING:
            message = (
                f"{self.config.gate_name}: WARNING - "
                f"Value {value:.2f}{self.config.unit} approaching limit "
                f"(warning at {self.config.warning_threshold:.2f}{self.config.unit})"
            )
        else:
            message = (
                f"{self.config.gate_name}: FAILED - "
                f"Value {value:.2f}{self.config.unit} exceeds limit "
                f"({self.config.fail_threshold:.2f}{self.config.unit})"
            )

        result = QualityGateResult(
            gate_id=self.config.gate_id,
            status=status,
            measured_value=value,
            pass_threshold=self.config.pass_threshold,
            fail_threshold=self.config.fail_threshold,
            message=message
        )

        self._evaluation_history.append(result)

        return result

    def _evaluate_less_than(self, value: float) -> GateStatus:
        """Evaluate using less-than comparison (lower is better)."""
        if value <= self.config.pass_threshold:
            return GateStatus.PASSED
        elif value <= self.config.warning_threshold:
            return GateStatus.WARNING
        elif value < self.config.fail_threshold:
            return GateStatus.REQUIRES_CONFIRMATION
        else:
            return GateStatus.BLOCKED

    def _evaluate_greater_than(self, value: float) -> GateStatus:
        """Evaluate using greater-than comparison (higher is better)."""
        if value >= self.config.pass_threshold:
            return GateStatus.PASSED
        elif value >= self.config.warning_threshold:
            return GateStatus.WARNING
        elif value > self.config.fail_threshold:
            return GateStatus.REQUIRES_CONFIRMATION
        else:
            return GateStatus.BLOCKED

    def _evaluate_in_range(self, value: float) -> GateStatus:
        """Evaluate if value is within acceptable range."""
        # For in_range, pass_threshold is the center, warning is deviation, fail is max deviation
        deviation = abs(value - self.config.pass_threshold)

        if deviation <= self.config.warning_threshold:
            return GateStatus.PASSED
        elif deviation <= self.config.fail_threshold:
            return GateStatus.WARNING
        else:
            return GateStatus.BLOCKED

    def get_history(self) -> List[QualityGateResult]:
        """Get evaluation history."""
        return self._evaluation_history.copy()

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self._evaluation_history.clear()


# =============================================================================
# UNCERTAINTY QUALITY CHECKER
# =============================================================================

@dataclass
class UncertaintyCheckResult:
    """
    Result of uncertainty quality check.

    Attributes:
        is_acceptable: Whether uncertainty is acceptable
        uncertainty_percent: Measured uncertainty percentage
        max_acceptable: Maximum acceptable uncertainty
        status: Gate status
        contributors: Breakdown by contributor
        recommendations: Improvement recommendations
        provenance_hash: SHA-256 hash for audit trail
    """
    is_acceptable: bool
    uncertainty_percent: float
    max_acceptable: float
    status: GateStatus
    contributors: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash."""
        if not self.provenance_hash:
            hash_data = {
                "is_acceptable": self.is_acceptable,
                "uncertainty_percent": self.uncertainty_percent,
                "timestamp": self.timestamp.isoformat()
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


class UncertaintyQualityChecker:
    """
    Checks if uncertainty is acceptable for various applications.

    Provides application-specific uncertainty thresholds and
    recommendations for uncertainty reduction.

    Example:
        checker = UncertaintyQualityChecker()

        # Check for energy accounting
        result = checker.check_for_application(
            propagated_uncertainty,
            application="energy_accounting"
        )

        if not result.is_acceptable:
            print(f"Uncertainty too high: {result.uncertainty_percent:.1f}%")
            for rec in result.recommendations:
                print(f"  - {rec}")
    """

    # Default thresholds by application type
    APPLICATION_THRESHOLDS: Dict[str, float] = {
        "energy_accounting": 5.0,      # 5% max for energy reporting
        "emission_reporting": 3.0,     # 3% max for emissions
        "performance_monitoring": 10.0, # 10% for trend analysis
        "optimization": 15.0,          # 15% for optimization suggestions
        "billing": 2.0,                # 2% for billing accuracy
        "safety_critical": 1.0,        # 1% for safety systems
        "regulatory_compliance": 5.0,  # 5% for regulatory reporting
    }

    def __init__(
        self,
        custom_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize uncertainty quality checker.

        Args:
            custom_thresholds: Override default thresholds
        """
        self.thresholds = self.APPLICATION_THRESHOLDS.copy()
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

    def check_for_application(
        self,
        propagated: PropagatedUncertainty,
        application: str = "performance_monitoring"
    ) -> UncertaintyCheckResult:
        """
        Check if uncertainty is acceptable for given application.

        Args:
            propagated: Propagated uncertainty result
            application: Application type (see APPLICATION_THRESHOLDS)

        Returns:
            UncertaintyCheckResult with acceptance status
        """
        max_acceptable = self.thresholds.get(application, 10.0)
        uncertainty_percent = propagated.relative_uncertainty_percent()

        is_acceptable = uncertainty_percent <= max_acceptable

        # Determine status
        if uncertainty_percent <= max_acceptable * 0.5:
            status = GateStatus.PASSED
        elif uncertainty_percent <= max_acceptable:
            status = GateStatus.WARNING
        elif uncertainty_percent <= max_acceptable * 2:
            status = GateStatus.REQUIRES_CONFIRMATION
        else:
            status = GateStatus.BLOCKED

        # Get contributor breakdown
        contributors = propagated.get_contribution_breakdown()

        # Generate recommendations
        recommendations = []
        sorted_contributors = sorted(
            contributors.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if not is_acceptable:
            # Recommend improving top contributors
            for name, contrib in sorted_contributors[:3]:
                if contrib > 20.0:
                    recommendations.append(
                        f"PRIORITY: Reduce uncertainty in {name} "
                        f"(contributes {contrib:.1f}% of total)"
                    )
                elif contrib > 10.0:
                    recommendations.append(
                        f"Consider improving measurement of {name} "
                        f"({contrib:.1f}% contribution)"
                    )

            if len(recommendations) == 0:
                recommendations.append(
                    "Uncertainty distributed across many inputs - "
                    "may require multiple improvements"
                )

        return UncertaintyCheckResult(
            is_acceptable=is_acceptable,
            uncertainty_percent=uncertainty_percent,
            max_acceptable=max_acceptable,
            status=status,
            contributors=contributors,
            recommendations=recommendations
        )

    def check_multiple(
        self,
        propagated_results: Dict[str, PropagatedUncertainty],
        application: str = "performance_monitoring"
    ) -> Dict[str, UncertaintyCheckResult]:
        """
        Check multiple propagated uncertainties.

        Args:
            propagated_results: Dictionary of name to PropagatedUncertainty
            application: Application type

        Returns:
            Dictionary of name to UncertaintyCheckResult
        """
        results = {}
        for name, propagated in propagated_results.items():
            results[name] = self.check_for_application(propagated, application)
        return results

    def get_overall_status(
        self,
        check_results: Dict[str, UncertaintyCheckResult]
    ) -> GateStatus:
        """
        Get overall status from multiple check results.

        Args:
            check_results: Dictionary of check results

        Returns:
            Worst status among all checks
        """
        if not check_results:
            return GateStatus.PASSED

        status_priority = {
            GateStatus.PASSED: 0,
            GateStatus.WARNING: 1,
            GateStatus.REQUIRES_CONFIRMATION: 2,
            GateStatus.BLOCKED: 3
        }

        worst_status = GateStatus.PASSED
        for result in check_results.values():
            if status_priority[result.status] > status_priority[worst_status]:
                worst_status = result.status

        return worst_status


# =============================================================================
# DATA QUALITY VALIDATOR
# =============================================================================

@dataclass
class DataValidationResult:
    """
    Result of data quality validation.

    Attributes:
        is_valid: Whether data passed validation
        validation_checks: Results of individual checks
        failed_checks: List of failed check names
        warnings: List of warning messages
        data_quality_score: Overall quality score (0-100)
        provenance_hash: SHA-256 hash for audit trail
    """
    is_valid: bool
    validation_checks: Dict[str, bool]
    failed_checks: List[str]
    warnings: List[str]
    data_quality_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash."""
        if not self.provenance_hash:
            hash_data = {
                "is_valid": self.is_valid,
                "data_quality_score": self.data_quality_score,
                "failed_checks": self.failed_checks,
                "timestamp": self.timestamp.isoformat()
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


class DataQualityValidator:
    """
    Validates input data quality before uncertainty propagation.

    Performs checks on:
    - Value ranges (physical feasibility)
    - Uncertainty bounds (not too large or too small)
    - Missing data detection
    - Outlier detection
    - Consistency checks

    Example:
        validator = DataQualityValidator()

        measurements = {
            "temperature": UncertainValue.from_measurement(450, 2.0),
            "pressure": UncertainValue.from_measurement(10.0, 0.5)
        }

        result = validator.validate(measurements, context="steam_system")

        if not result.is_valid:
            for check in result.failed_checks:
                print(f"Failed: {check}")
    """

    # Physical limits for common measurements
    PHYSICAL_LIMITS: Dict[str, Tuple[float, float]] = {
        "temperature_c": (-273.15, 1000.0),
        "temperature_k": (0.0, 1273.15),
        "pressure_mpa": (0.0, 50.0),
        "pressure_bar": (0.0, 500.0),
        "mass_flow_kg_s": (0.0, 10000.0),
        "enthalpy_kj_kg": (0.0, 5000.0),
        "efficiency": (0.0, 1.0),
        "percentage": (0.0, 100.0)
    }

    def __init__(
        self,
        max_uncertainty_percent: float = 50.0,
        min_uncertainty_percent: float = 0.001
    ):
        """
        Initialize data quality validator.

        Args:
            max_uncertainty_percent: Maximum acceptable uncertainty
            min_uncertainty_percent: Minimum plausible uncertainty
        """
        self.max_uncertainty_percent = max_uncertainty_percent
        self.min_uncertainty_percent = min_uncertainty_percent

    def validate(
        self,
        measurements: Dict[str, UncertainValue],
        context: str = "general",
        custom_limits: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> DataValidationResult:
        """
        Validate measurement data quality.

        Args:
            measurements: Dictionary of measurements to validate
            context: Context for validation (affects which limits apply)
            custom_limits: Custom physical limits

        Returns:
            DataValidationResult with validation status
        """
        validation_checks = {}
        failed_checks = []
        warnings = []

        limits = self.PHYSICAL_LIMITS.copy()
        if custom_limits:
            limits.update(custom_limits)

        for name, value in measurements.items():
            # Check 1: Value is finite
            check_name = f"{name}_finite"
            is_finite = not (
                value.mean != value.mean or  # NaN check
                abs(value.mean) == float('inf')
            )
            validation_checks[check_name] = is_finite
            if not is_finite:
                failed_checks.append(check_name)

            # Check 2: Uncertainty is finite and positive
            check_name = f"{name}_uncertainty_valid"
            uncertainty_valid = (
                value.std >= 0 and
                value.std != float('inf') and
                value.std == value.std  # NaN check
            )
            validation_checks[check_name] = uncertainty_valid
            if not uncertainty_valid:
                failed_checks.append(check_name)

            # Check 3: Uncertainty not too large
            check_name = f"{name}_uncertainty_not_excessive"
            relative_unc = value.relative_uncertainty()
            uncertainty_reasonable = relative_unc <= self.max_uncertainty_percent
            validation_checks[check_name] = uncertainty_reasonable
            if not uncertainty_reasonable:
                warnings.append(
                    f"{name}: Uncertainty ({relative_unc:.1f}%) exceeds "
                    f"maximum ({self.max_uncertainty_percent}%)"
                )

            # Check 4: Uncertainty not suspiciously small
            check_name = f"{name}_uncertainty_not_zero"
            uncertainty_nonzero = relative_unc >= self.min_uncertainty_percent
            validation_checks[check_name] = uncertainty_nonzero
            if not uncertainty_nonzero:
                warnings.append(
                    f"{name}: Uncertainty ({relative_unc:.4f}%) suspiciously low - "
                    f"verify measurement source"
                )

            # Check 5: Physical limits (if known)
            matched_limit = None
            name_lower = name.lower()
            for limit_key, (low, high) in limits.items():
                if limit_key in name_lower:
                    matched_limit = (low, high)
                    break

            if matched_limit:
                check_name = f"{name}_physical_limits"
                low, high = matched_limit
                within_limits = low <= value.mean <= high
                validation_checks[check_name] = within_limits
                if not within_limits:
                    failed_checks.append(check_name)

            # Check 6: Confidence interval consistency
            check_name = f"{name}_ci_consistent"
            ci_consistent = value.lower_95 < value.mean < value.upper_95
            validation_checks[check_name] = ci_consistent
            if not ci_consistent:
                warnings.append(
                    f"{name}: Confidence interval inconsistent with mean"
                )

        # Calculate quality score
        total_checks = len(validation_checks)
        passed_checks = sum(1 for v in validation_checks.values() if v)
        data_quality_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0.0

        is_valid = len(failed_checks) == 0

        return DataValidationResult(
            is_valid=is_valid,
            validation_checks=validation_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            data_quality_score=data_quality_score
        )

    def check_outliers(
        self,
        values: List[float],
        n_sigma: float = 3.0
    ) -> List[int]:
        """
        Check for outliers using n-sigma rule.

        Args:
            values: List of values to check
            n_sigma: Number of standard deviations for outlier threshold

        Returns:
            List of indices of outlier values
        """
        import numpy as np

        if len(values) < 3:
            return []

        values_arr = np.array(values)
        mean = np.mean(values_arr)
        std = np.std(values_arr)

        if std < 1e-10:
            return []

        outlier_indices = []
        for i, v in enumerate(values):
            if abs(v - mean) > n_sigma * std:
                outlier_indices.append(i)

        return outlier_indices


# =============================================================================
# COMPLIANCE CHECKER FOR REGULATORY THRESHOLDS
# =============================================================================

@dataclass
class ComplianceCheckResult:
    """
    Result of regulatory compliance check.

    Attributes:
        is_compliant: Whether measurement is compliant
        regulation: Regulation being checked
        measured_value: Measured value with uncertainty
        regulatory_limit: Regulatory limit
        margin: Distance from limit (negative if exceeding)
        confidence_of_compliance: Probability of being compliant
        exceedance_risk: Probability of exceeding limit
        recommendations: Compliance recommendations
        provenance_hash: SHA-256 hash for audit trail
    """
    is_compliant: bool
    regulation: str
    measured_value: UncertainValue
    regulatory_limit: float
    margin: float
    confidence_of_compliance: float
    exceedance_risk: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash."""
        if not self.provenance_hash:
            hash_data = {
                "is_compliant": self.is_compliant,
                "regulation": self.regulation,
                "confidence_of_compliance": self.confidence_of_compliance,
                "timestamp": self.timestamp.isoformat()
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


class ComplianceChecker:
    """
    Checks measurements against regulatory thresholds considering uncertainty.

    Computes the probability of compliance accounting for measurement
    uncertainty, and provides risk assessment for regulatory reporting.

    Example:
        checker = ComplianceChecker()

        # Check emission against regulatory limit
        result = checker.check_compliance(
            measured=emission_value,
            limit=regulatory_limit,
            regulation="EPA_40CFR60",
            limit_type="upper"
        )

        print(f"Compliance confidence: {result.confidence_of_compliance:.1%}")
        print(f"Exceedance risk: {result.exceedance_risk:.1%}")
    """

    def __init__(
        self,
        confidence_threshold: float = 0.95,
        margin_safety_factor: float = 1.0
    ):
        """
        Initialize compliance checker.

        Args:
            confidence_threshold: Required confidence for compliance declaration
            margin_safety_factor: Safety factor for margin calculation
        """
        self.confidence_threshold = confidence_threshold
        self.margin_safety_factor = margin_safety_factor

    def check_compliance(
        self,
        measured: UncertainValue,
        limit: float,
        regulation: str,
        limit_type: str = "upper"
    ) -> ComplianceCheckResult:
        """
        Check if measurement is compliant with regulatory limit.

        Args:
            measured: Measured value with uncertainty
            limit: Regulatory limit value
            regulation: Name/identifier of regulation
            limit_type: Type of limit ("upper" or "lower")

        Returns:
            ComplianceCheckResult with compliance assessment
        """
        from scipy import stats

        # Calculate probability of compliance
        if limit_type == "upper":
            # Probability that true value is below limit
            z = (limit - measured.mean) / measured.std if measured.std > 0 else float('inf')
            confidence_of_compliance = float(stats.norm.cdf(z))
            exceedance_risk = 1.0 - confidence_of_compliance
            margin = limit - measured.mean
        else:
            # Probability that true value is above limit
            z = (measured.mean - limit) / measured.std if measured.std > 0 else float('inf')
            confidence_of_compliance = float(stats.norm.cdf(z))
            exceedance_risk = 1.0 - confidence_of_compliance
            margin = measured.mean - limit

        # Determine compliance status
        is_compliant = confidence_of_compliance >= self.confidence_threshold

        # Generate recommendations
        recommendations = []

        if not is_compliant:
            if exceedance_risk > 0.5:
                recommendations.append(
                    f"HIGH RISK: {exceedance_risk:.1%} probability of exceeding "
                    f"{regulation} limit. Immediate action required."
                )
            else:
                recommendations.append(
                    f"CAUTION: {exceedance_risk:.1%} probability of exceeding limit. "
                    f"Monitor closely and consider mitigation."
                )

            # Suggest uncertainty reduction if margin is close
            if abs(margin) < 3 * measured.std:
                recommendations.append(
                    f"Consider reducing measurement uncertainty to improve "
                    f"compliance confidence (current: {measured.std:.2f}, "
                    f"margin: {margin:.2f})"
                )

        elif confidence_of_compliance < 0.99:
            recommendations.append(
                f"Compliant with {confidence_of_compliance:.1%} confidence. "
                f"Monitor for changes that could affect compliance."
            )

        return ComplianceCheckResult(
            is_compliant=is_compliant,
            regulation=regulation,
            measured_value=measured,
            regulatory_limit=limit,
            margin=margin,
            confidence_of_compliance=confidence_of_compliance,
            exceedance_risk=exceedance_risk,
            recommendations=recommendations
        )

    def check_multiple_regulations(
        self,
        measured: UncertainValue,
        regulations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ComplianceCheckResult]:
        """
        Check measurement against multiple regulatory limits.

        Args:
            measured: Measured value with uncertainty
            regulations: Dictionary of regulation name to limit details

        Returns:
            Dictionary of regulation name to ComplianceCheckResult
        """
        results = {}
        for reg_name, reg_details in regulations.items():
            limit = reg_details.get("limit")
            limit_type = reg_details.get("limit_type", "upper")

            results[reg_name] = self.check_compliance(
                measured=measured,
                limit=limit,
                regulation=reg_name,
                limit_type=limit_type
            )

        return results


# =============================================================================
# QUALITY GATE REPORT GENERATION
# =============================================================================

@dataclass
class QualityGateReport:
    """
    Comprehensive quality gate report.

    Attributes:
        report_id: Unique report identifier
        report_timestamp: When report was generated
        overall_status: Overall quality status
        gate_results: Individual gate results
        data_validation: Data validation result
        uncertainty_checks: Uncertainty check results
        compliance_checks: Compliance check results
        summary: Executive summary
        recommendations: Prioritized recommendations
        provenance_hash: SHA-256 hash for audit trail
    """
    report_id: str
    report_timestamp: datetime
    overall_status: GateStatus
    gate_results: List[GateResult]
    data_validation: Optional[DataValidationResult]
    uncertainty_checks: Dict[str, UncertaintyCheckResult]
    compliance_checks: Dict[str, ComplianceCheckResult]
    summary: str
    recommendations: List[str]
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash."""
        if not self.provenance_hash:
            hash_data = {
                "report_id": self.report_id,
                "overall_status": self.overall_status.value,
                "timestamp": self.report_timestamp.isoformat()
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


def generate_quality_gate_report(
    gate_results: List[GateResult],
    data_validation: Optional[DataValidationResult] = None,
    uncertainty_checks: Optional[Dict[str, UncertaintyCheckResult]] = None,
    compliance_checks: Optional[Dict[str, ComplianceCheckResult]] = None,
    report_id: Optional[str] = None
) -> QualityGateReport:
    """
    Generate comprehensive quality gate report.

    Combines results from multiple quality checks into a single
    audit-ready report with executive summary and recommendations.

    Args:
        gate_results: List of gate check results
        data_validation: Optional data validation result
        uncertainty_checks: Optional uncertainty check results
        compliance_checks: Optional compliance check results
        report_id: Optional custom report ID

    Returns:
        QualityGateReport with complete assessment
    """
    report_timestamp = datetime.utcnow()
    report_id = report_id or f"qgr_{report_timestamp.strftime('%Y%m%d%H%M%S')}"

    # Determine overall status
    all_statuses = [r.status for r in gate_results]

    if data_validation and not data_validation.is_valid:
        all_statuses.append(GateStatus.BLOCKED)

    if uncertainty_checks:
        all_statuses.extend(r.status for r in uncertainty_checks.values())

    if compliance_checks:
        for result in compliance_checks.values():
            if not result.is_compliant:
                all_statuses.append(GateStatus.BLOCKED)
            elif result.exceedance_risk > 0.1:
                all_statuses.append(GateStatus.WARNING)

    # Get worst status
    status_priority = {
        GateStatus.PASSED: 0,
        GateStatus.WARNING: 1,
        GateStatus.REQUIRES_CONFIRMATION: 2,
        GateStatus.BLOCKED: 3
    }

    overall_status = GateStatus.PASSED
    for status in all_statuses:
        if status_priority[status] > status_priority[overall_status]:
            overall_status = status

    # Generate summary
    n_passed = sum(1 for s in all_statuses if s == GateStatus.PASSED)
    n_warning = sum(1 for s in all_statuses if s == GateStatus.WARNING)
    n_blocked = sum(1 for s in all_statuses if s in [GateStatus.BLOCKED, GateStatus.REQUIRES_CONFIRMATION])

    if overall_status == GateStatus.PASSED:
        summary = (
            f"Quality assessment PASSED. All {len(all_statuses)} checks passed. "
            f"Data quality is acceptable for proceeding with analysis."
        )
    elif overall_status == GateStatus.WARNING:
        summary = (
            f"Quality assessment completed with WARNINGS. "
            f"{n_passed} passed, {n_warning} warnings, {n_blocked} blocked. "
            f"Review warnings before proceeding."
        )
    else:
        summary = (
            f"Quality assessment FAILED. "
            f"{n_passed} passed, {n_warning} warnings, {n_blocked} blocked. "
            f"Address blocked items before proceeding."
        )

    # Collect recommendations
    recommendations = []

    # From gate results
    for result in gate_results:
        if result.required_action:
            recommendations.append(result.required_action)

    # From data validation
    if data_validation and data_validation.warnings:
        recommendations.extend(data_validation.warnings)

    # From uncertainty checks
    if uncertainty_checks:
        for name, result in uncertainty_checks.items():
            recommendations.extend(result.recommendations)

    # From compliance checks
    if compliance_checks:
        for name, result in compliance_checks.items():
            recommendations.extend(result.recommendations)

    # Deduplicate and limit
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)

    return QualityGateReport(
        report_id=report_id,
        report_timestamp=report_timestamp,
        overall_status=overall_status,
        gate_results=gate_results,
        data_validation=data_validation,
        uncertainty_checks=uncertainty_checks or {},
        compliance_checks=compliance_checks or {},
        summary=summary,
        recommendations=unique_recommendations[:10]  # Top 10 recommendations
    )


def format_quality_gate_report(report: QualityGateReport) -> str:
    """
    Format quality gate report as human-readable text.

    Args:
        report: QualityGateReport to format

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("QUALITY GATE REPORT")
    lines.append("=" * 70)
    lines.append(f"Report ID: {report.report_id}")
    lines.append(f"Generated: {report.report_timestamp.isoformat()}")
    lines.append(f"Overall Status: {report.overall_status.value.upper()}")
    lines.append(f"Provenance Hash: {report.provenance_hash[:16]}...")
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(report.summary)
    lines.append("")

    if report.gate_results:
        lines.append("GATE RESULTS")
        lines.append("-" * 70)
        for result in report.gate_results:
            status_str = result.status.value.upper()
            lines.append(
                f"  [{status_str:20}] {result.recommendation_id}: "
                f"uncertainty={result.uncertainty_level:.1f}%"
            )
        lines.append("")

    if report.data_validation:
        lines.append("DATA VALIDATION")
        lines.append("-" * 70)
        valid_str = "VALID" if report.data_validation.is_valid else "INVALID"
        lines.append(f"  Status: {valid_str}")
        lines.append(f"  Quality Score: {report.data_validation.data_quality_score:.1f}/100")
        if report.data_validation.failed_checks:
            lines.append(f"  Failed Checks: {', '.join(report.data_validation.failed_checks)}")
        lines.append("")

    if report.compliance_checks:
        lines.append("COMPLIANCE CHECKS")
        lines.append("-" * 70)
        for name, result in report.compliance_checks.items():
            status_str = "COMPLIANT" if result.is_compliant else "NON-COMPLIANT"
            lines.append(
                f"  [{status_str:15}] {name}: "
                f"confidence={result.confidence_of_compliance:.1%}, "
                f"risk={result.exceedance_risk:.1%}"
            )
        lines.append("")

    if report.recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)
