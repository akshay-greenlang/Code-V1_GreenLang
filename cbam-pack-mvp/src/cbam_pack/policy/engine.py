"""
Policy Engine for CBAM Compliance

Implements PASS/WARN/FAIL logic for policy-driven compliance validation.
Supports configurable thresholds for default factor usage caps.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from cbam_pack.calculators.emissions_calculator import CalculationResult
from cbam_pack.models import CBAMConfig, Quarter, MethodType


class PolicyStatus(str, Enum):
    """Policy evaluation status."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class PolicyViolation:
    """A single policy violation."""
    rule_id: str
    rule_name: str
    severity: PolicyStatus
    message: str
    affected_lines: list[str] = field(default_factory=list)
    threshold: Optional[float] = None
    actual_value: Optional[float] = None
    remediation: str = ""


@dataclass
class PolicyResult:
    """Result of policy evaluation."""
    status: PolicyStatus
    overall_score: float  # 0-100
    violations: list[PolicyViolation] = field(default_factory=list)
    warnings: list[PolicyViolation] = field(default_factory=list)
    passed_rules: list[str] = field(default_factory=list)
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    can_export: bool = True
    summary: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "overall_score": self.overall_score,
            "can_export": self.can_export,
            "summary": self.summary,
            "evaluated_at": self.evaluated_at.isoformat() + "Z",
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "rule_name": v.rule_name,
                    "severity": v.severity.value,
                    "message": v.message,
                    "affected_lines": v.affected_lines[:10],  # Limit for brevity
                    "affected_line_count": len(v.affected_lines),
                    "threshold": v.threshold,
                    "actual_value": v.actual_value,
                    "remediation": v.remediation,
                }
                for v in self.violations
            ],
            "warnings": [
                {
                    "rule_id": w.rule_id,
                    "rule_name": w.rule_name,
                    "severity": w.severity.value,
                    "message": w.message,
                    "affected_lines": w.affected_lines[:10],
                    "affected_line_count": len(w.affected_lines),
                    "threshold": w.threshold,
                    "actual_value": w.actual_value,
                    "remediation": w.remediation,
                }
                for w in self.warnings
            ],
            "passed_rules": self.passed_rules,
        }


@dataclass
class PolicyConfig:
    """Policy configuration."""
    # Default factor usage caps (percentage)
    default_usage_cap_transitional: float = 100.0  # No cap during transitional
    default_usage_cap_q3_2024_plus: float = 20.0   # 20% cap from Q3 2024
    default_usage_cap_definitive: float = 0.0      # No defaults in definitive period

    # Warning thresholds
    default_usage_warn_threshold: float = 50.0     # Warn if over 50%

    # Authorization threshold (tonnes/year)
    authorization_threshold_tonnes: float = 50.0

    # Enforcement mode
    block_export_on_fail: bool = False             # If True, prevent export on FAIL

    # Period-aware rules
    enable_period_aware_rules: bool = True


class PolicyEngine:
    """
    Policy Engine for CBAM compliance validation.

    Evaluates calculation results against configurable policy rules
    and returns PASS/WARN/FAIL status with detailed violations.
    """

    def __init__(self, policy_config: Optional[PolicyConfig] = None):
        """
        Initialize the policy engine.

        Args:
            policy_config: Policy configuration. Uses defaults if not provided.
        """
        self.config = policy_config or PolicyConfig()

    @classmethod
    def from_yaml_config(cls, cbam_config: CBAMConfig) -> "PolicyEngine":
        """
        Create policy engine from CBAM config.

        Extracts policy settings from the CBAM configuration.
        """
        policy_config = PolicyConfig()

        # Check if config has policy section
        if hasattr(cbam_config, 'policy'):
            policy = cbam_config.policy
            if hasattr(policy, 'default_usage_cap'):
                policy_config.default_usage_cap_q3_2024_plus = policy.default_usage_cap
            if hasattr(policy, 'block_export_on_fail'):
                policy_config.block_export_on_fail = policy.block_export_on_fail

        return cls(policy_config)

    def evaluate(
        self,
        calc_result: CalculationResult,
        config: CBAMConfig,
        total_annual_quantity_tonnes: Optional[float] = None,
    ) -> PolicyResult:
        """
        Evaluate calculation results against policy rules.

        Args:
            calc_result: Calculation results to evaluate
            config: CBAM configuration
            total_annual_quantity_tonnes: Optional annual quantity for authorization check

        Returns:
            PolicyResult with status and violations
        """
        violations: list[PolicyViolation] = []
        warnings: list[PolicyViolation] = []
        passed_rules: list[str] = []

        stats = calc_result.statistics

        # Rule 1: Default factor usage cap
        self._check_default_usage_cap(
            stats, config, calc_result, violations, warnings, passed_rules
        )

        # Rule 2: Period-aware rules (Q3 2024+ restrictions)
        if self.config.enable_period_aware_rules:
            self._check_period_rules(
                stats, config, calc_result, violations, warnings, passed_rules
            )

        # Rule 3: Authorization readiness (50t threshold)
        if total_annual_quantity_tonnes is not None:
            self._check_authorization_threshold(
                total_annual_quantity_tonnes, config, calc_result,
                violations, warnings, passed_rules
            )

        # Rule 4: Data quality checks
        self._check_data_quality(
            calc_result, violations, warnings, passed_rules
        )

        # Determine overall status
        if violations:
            status = PolicyStatus.FAIL
            can_export = not self.config.block_export_on_fail
        elif warnings:
            status = PolicyStatus.WARN
            can_export = True
        else:
            status = PolicyStatus.PASS
            can_export = True

        # Calculate score (100 = perfect, deduct for violations/warnings)
        score = 100.0
        score -= len(violations) * 25  # -25 per violation
        score -= len(warnings) * 10    # -10 per warning
        score = max(0.0, score)

        # Build summary
        if status == PolicyStatus.PASS:
            summary = "All policy checks passed. Report is compliant."
        elif status == PolicyStatus.WARN:
            summary = f"Report generated with {len(warnings)} warning(s). Review recommended."
        else:
            summary = f"Policy validation failed with {len(violations)} violation(s)."
            if not can_export:
                summary += " Export blocked per policy configuration."

        return PolicyResult(
            status=status,
            overall_score=score,
            violations=violations,
            warnings=warnings,
            passed_rules=passed_rules,
            can_export=can_export,
            summary=summary,
        )

    def _check_default_usage_cap(
        self,
        stats: dict,
        config: CBAMConfig,
        calc_result: CalculationResult,
        violations: list[PolicyViolation],
        warnings: list[PolicyViolation],
        passed_rules: list[str],
    ) -> None:
        """Check default factor usage against caps."""
        default_usage = stats.get("default_usage_percent", 0)

        # Determine applicable cap based on period
        quarter = config.reporting_period.quarter
        year = config.reporting_period.year

        # Q3 2024 onwards has stricter rules
        is_post_q3_2024 = (year > 2024) or (year == 2024 and quarter in [Quarter.Q3, Quarter.Q4])

        if is_post_q3_2024:
            cap = self.config.default_usage_cap_q3_2024_plus
        else:
            cap = self.config.default_usage_cap_transitional

        # Get lines using defaults
        default_lines = [
            r.line_id for r in calc_result.line_results
            if r.method_direct == MethodType.DEFAULT
        ]

        if default_usage > cap:
            violations.append(PolicyViolation(
                rule_id="POL-001",
                rule_name="Default Factor Usage Cap",
                severity=PolicyStatus.FAIL,
                message=f"Default factor usage ({default_usage:.1f}%) exceeds cap ({cap:.1f}%)",
                affected_lines=default_lines,
                threshold=cap,
                actual_value=default_usage,
                remediation=(
                    f"Obtain supplier-specific emission data for at least "
                    f"{len(default_lines) - int(len(calc_result.line_results) * cap / 100)} lines "
                    f"to reduce default usage below {cap:.0f}%."
                ),
            ))
        elif default_usage > self.config.default_usage_warn_threshold:
            warnings.append(PolicyViolation(
                rule_id="POL-001",
                rule_name="Default Factor Usage Cap",
                severity=PolicyStatus.WARN,
                message=f"Default factor usage ({default_usage:.1f}%) is high (threshold: {self.config.default_usage_warn_threshold:.1f}%)",
                affected_lines=default_lines,
                threshold=self.config.default_usage_warn_threshold,
                actual_value=default_usage,
                remediation="Consider obtaining supplier-specific data to improve accuracy.",
            ))
        else:
            passed_rules.append("POL-001: Default Factor Usage Cap")

    def _check_period_rules(
        self,
        stats: dict,
        config: CBAMConfig,
        calc_result: CalculationResult,
        violations: list[PolicyViolation],
        warnings: list[PolicyViolation],
        passed_rules: list[str],
    ) -> None:
        """Check period-specific compliance rules."""
        quarter = config.reporting_period.quarter
        year = config.reporting_period.year

        # Check if we're in definitive period (2026+)
        if year >= 2026:
            default_usage = stats.get("default_usage_percent", 0)
            if default_usage > 0:
                default_lines = [
                    r.line_id for r in calc_result.line_results
                    if r.method_direct == MethodType.DEFAULT
                ]
                violations.append(PolicyViolation(
                    rule_id="POL-002",
                    rule_name="Definitive Period Compliance",
                    severity=PolicyStatus.FAIL,
                    message=(
                        f"Default factors not allowed in definitive period (2026+). "
                        f"Found {len(default_lines)} lines using defaults."
                    ),
                    affected_lines=default_lines,
                    threshold=0,
                    actual_value=default_usage,
                    remediation=(
                        "All imports must have supplier-specific or verified "
                        "emission data for the CBAM definitive period."
                    ),
                ))
            else:
                passed_rules.append("POL-002: Definitive Period Compliance")
        else:
            passed_rules.append("POL-002: Definitive Period Compliance (transitional - relaxed)")

    def _check_authorization_threshold(
        self,
        total_annual_quantity_tonnes: float,
        config: CBAMConfig,
        calc_result: CalculationResult,
        violations: list[PolicyViolation],
        warnings: list[PolicyViolation],
        passed_rules: list[str],
    ) -> None:
        """Check if annual quantity exceeds authorization threshold."""
        threshold = self.config.authorization_threshold_tonnes

        if total_annual_quantity_tonnes > threshold:
            # Will need CBAM authorization in definitive period
            warnings.append(PolicyViolation(
                rule_id="POL-003",
                rule_name="Authorization Readiness",
                severity=PolicyStatus.WARN,
                message=(
                    f"Annual quantity ({total_annual_quantity_tonnes:.1f}t) exceeds "
                    f"authorization threshold ({threshold:.0f}t/year). "
                    "CBAM authorization will be required in definitive period."
                ),
                threshold=threshold,
                actual_value=total_annual_quantity_tonnes,
                remediation=(
                    "Prepare for CBAM authorized declarant registration. "
                    "Ensure financial guarantee capacity and "
                    "establish supplier verification processes."
                ),
            ))
        else:
            passed_rules.append("POL-003: Authorization Readiness")

    def _check_data_quality(
        self,
        calc_result: CalculationResult,
        violations: list[PolicyViolation],
        warnings: list[PolicyViolation],
        passed_rules: list[str],
    ) -> None:
        """Check data quality indicators."""
        stats = calc_result.statistics
        total_lines = stats.get("total_lines", 0)

        if total_lines == 0:
            violations.append(PolicyViolation(
                rule_id="POL-004",
                rule_name="Data Completeness",
                severity=PolicyStatus.FAIL,
                message="No import lines found in the data.",
                remediation="Ensure import ledger contains valid CBAM-regulated imports.",
            ))
        else:
            passed_rules.append("POL-004: Data Completeness")

        # Check for mixed methods (some supplier, some default)
        supplier_direct = stats.get("lines_with_supplier_direct_data", 0)
        if 0 < supplier_direct < total_lines:
            # Mixed data quality
            passed_rules.append("POL-005: Data Consistency (mixed methods allowed)")
        else:
            passed_rules.append("POL-005: Data Consistency")
